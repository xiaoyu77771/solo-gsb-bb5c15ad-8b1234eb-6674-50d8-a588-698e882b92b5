"""预约临时保留（hold）核心逻辑：短时占位与确认转正式预约。

临时保留用于“先占坑、稍后确认”的预约流程，与正式预约共用同一套
半开区间 ``[start, end)`` 冲突规则：

- **创建保留**（:func:`create_hold`）在单个 ``BEGIN IMMEDIATE`` 事务内，
  忽略已过期（``status='pending'`` 且 ``expires_at <= now``）的保留，
  对**进行中预约**（``reservation_participants`` 占用行）与**其他有效保留**
  （pending 且未过期）逐参与者做半开区间重叠检查；任一重叠即整体回滚，
  返回 :class:`HoldConflictError`（409），不写入任何行。
- **确认保留**（:func:`confirm_hold`）同样在单个 ``BEGIN IMMEDIATE``
  事务内：仅未过期的 pending 保留可确认；确认时把保留**原子转为正式预约**
  （插入 reservations / reservation_participants 并追加 ``created`` 哈希链
  审计事件），随后把保留标记为 ``confirmed``。重复确认（含过期后）幂等返回
  同一预约；过期未确认返回 :class:`HoldExpiredError`（410）。

保留行与保留参与者行在过期/确认后都**不删除**：有效性完全由
``status`` 与 ``expires_at`` 判定——过期保留的占用一律视为不存在，
确认后保留的参与者行与正式预约行并存，但确认行不再参与保留间冲突检查。

``hold_key`` 是客户端幂等键：同 key 的最新一条若是未过期 pending 保留，
相同规范化内容（起止时刻、去重排序后的参与者、ttl）重试 → 返回原保留；
内容不同 → :class:`HoldKeyConflictError`（409）。已过期或已确认的 key
可直接复用新建，不视为冲突（旧行保留作记录）。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from . import audit
from . import db
from .reservations import (
    RequestIdConflictError,
    Reservation,
    _row_to_reservation,
    normalize_participants,
)

# ttl 上下限（秒）：与 API 层 schema 的 1..900 保持一致
MIN_TTL_SECONDS = 1
MAX_TTL_SECONDS = 900

# 定宽 UTC 文本格式：与 reservations 表一致，字典序即时间序
_CANON_FMT = audit.CANON_FMT

_HOLD_COLUMNS = (
    "id, hold_key, start_utc, end_utc, participants_key, ttl_seconds,"
    " status, reservation_id, request_id, created_at, expires_at, confirmed_at"
)


class HoldNotFoundError(Exception):
    """指定 ID 的保留不存在。"""

    def __init__(self, hold_id: int) -> None:
        self.hold_id = hold_id
        super().__init__(f"reservation hold #{hold_id} not found")


class HoldExpiredError(Exception):
    """保留已过期（且未确认）：不可再确认。"""

    def __init__(self, hold_id: int) -> None:
        self.hold_id = hold_id
        super().__init__(f"reservation hold #{hold_id} has expired")


class HoldKeyConflictError(Exception):
    """hold_key 已被内容不同的未过期 pending 保留占用。"""

    def __init__(self, hold_id: int) -> None:
        self.hold_id = hold_id
        super().__init__(
            f"hold_key is already used by pending hold #{hold_id} "
            "with different content"
        )


class HoldConflictError(Exception):
    """参与者与进行中预约或其他有效保留存在半开区间重叠。"""

    def __init__(self, conflicts: list[dict[str, object]]) -> None:
        self.conflicts = conflicts  # [{"kind": "reservation"/"hold", ...}]
        pretty = ", ".join(
            (f"participant {c['participant_id']!r} overlaps reservation "
             f"#{c['reservation_id']}")
            if c["kind"] == "reservation"
            else (f"participant {c['participant_id']!r} overlaps hold "
                  f"#{c['hold_id']}")
            for c in conflicts
        )
        super().__init__(f"reservation hold conflict: {pretty}")


@dataclass(frozen=True)
class Hold:
    id: int
    hold_key: str
    start_utc: datetime
    end_utc: datetime
    participants: tuple[str, ...]
    ttl_seconds: int
    created_at: datetime
    expires_at: datetime
    status: str = "pending"  # pending / confirmed
    reservation_id: int | None = None
    request_id: str | None = None
    confirmed_at: datetime | None = None

    @property
    def is_expired(self) -> bool:
        """是否已过期（confirmed 保留永远不算过期）。"""
        return (self.status == "pending"
                and self.expires_at <= datetime.now(timezone.utc))


def _canon(dt: datetime) -> str:
    """统一成定宽 UTC 字符串，保证字典序与时间序一致。"""
    return dt.astimezone(timezone.utc).strftime(_CANON_FMT)


def _parse(text: str) -> datetime:
    return datetime.strptime(text, _CANON_FMT).replace(tzinfo=timezone.utc)


def _row_to_hold(row: sqlite3.Row) -> Hold:
    return Hold(
        id=row["id"],
        hold_key=row["hold_key"],
        start_utc=_parse(row["start_utc"]),
        end_utc=_parse(row["end_utc"]),
        participants=tuple(json.loads(row["participants_key"])),
        ttl_seconds=row["ttl_seconds"],
        created_at=_parse(row["created_at"]),
        expires_at=_parse(row["expires_at"]),
        status=row["status"],
        reservation_id=row["reservation_id"],
        request_id=row["request_id"],
        confirmed_at=_parse(row["confirmed_at"]) if row["confirmed_at"] else None,
    )


def _find_live_hold(conn: sqlite3.Connection, hold_key: str,
                   now_s: str) -> sqlite3.Row | None:
    """返回该 key 最新一条**未过期 pending** 保留；没有（含已过期/已确认）返回 None。"""
    return conn.execute(
        f"SELECT {_HOLD_COLUMNS} FROM reservation_holds"
        " WHERE hold_key = ? AND status = 'pending' AND expires_at > ?"
        " ORDER BY id DESC LIMIT 1",
        (hold_key, now_s),
    ).fetchone()


def _collect_conflicts(
    conn: sqlite3.Connection,
    participants: tuple[str, ...],
    start_s: str,
    end_s: str,
    now_s: str,
    exclude_hold_id: int | None = None,
) -> list[dict[str, object]]:
    """半开区间冲突检查：进行中预约 + 其他有效（pending 未过期）保留。

    相邻区间（``end == start``）因等号不成立而不命中；
    已过期与已确认的保留一律不参与检查。
    """
    conflicts: list[dict[str, object]] = []
    for pid in participants:
        # 进行中预约的占用（已取消预约的占用行在取消事务内被删除）
        for hit in conn.execute(
            "SELECT reservation_id, participant_id"
            " FROM reservation_participants"
            " WHERE participant_id = ? AND start_utc < ? AND ? < end_utc"
            " ORDER BY reservation_id",
            (pid, end_s, start_s),
        ):
            conflicts.append({
                "kind": "reservation",
                "reservation_id": hit["reservation_id"],
                "hold_id": None,
                "participant_id": hit["participant_id"],
            })
        # 其他有效保留：JOIN 过滤掉非 pending 与已过期行
        query = (
            "SELECT hp.hold_id, hp.participant_id"
            " FROM reservation_hold_participants hp"
            " JOIN reservation_holds h ON h.id = hp.hold_id"
            " WHERE hp.participant_id = ? AND hp.start_utc < ? AND ? < hp.end_utc"
            " AND h.status = 'pending' AND h.expires_at > ?"
        )
        params: list[object] = [pid, end_s, start_s, now_s]
        if exclude_hold_id is not None:
            # 确认时复检冲突：保留自身的占用不算冲突
            query += " AND hp.hold_id != ?"
            params.append(exclude_hold_id)
        query += " ORDER BY hp.hold_id"
        for hit in conn.execute(query, params):
            conflicts.append({
                "kind": "hold",
                "reservation_id": None,
                "hold_id": hit["hold_id"],
                "participant_id": hit["participant_id"],
            })
    return conflicts


def create_hold(
    hold_key: str,
    start_utc: datetime,
    end_utc: datetime,
    participant_ids: Iterable[str],
    ttl_seconds: int,
    db_path: str | None = None,
) -> tuple[Hold, bool]:
    """创建临时保留，返回 ``(保留, 是否新建)``。

    在单个 ``BEGIN IMMEDIATE`` 事务内：

    - 同 ``hold_key`` 的最新一条为未过期 pending 保留时：规范化内容
      （起止时刻、去重排序后的参与者、ttl）相同 → 返回原保留，
      ``created=False``（幂等重放，不新增记录）；内容不同
      → :class:`HoldKeyConflictError`（409）；
    - 对进行中预约与其他有效保留按半开区间检查冲突（忽略已过期保留）：
      任一参与者重叠 → :class:`HoldConflictError`（409 + 冲突明细），
      整体回滚，不留下任何部分写入。
    """
    if not (MIN_TTL_SECONDS <= int(ttl_seconds) <= MAX_TTL_SECONDS):
        # 防御性校验（API 层 Field 约束已覆盖直接 HTTP 调用）
        raise ValueError(
            f"ttl_seconds must be within {MIN_TTL_SECONDS}..{MAX_TTL_SECONDS}")

    participants = normalize_participants(participant_ids)
    start_s = _canon(start_utc)
    end_s = _canon(end_utc)
    participants_key = json.dumps(list(participants), ensure_ascii=False)

    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        # BEGIN IMMEDIATE：事务开始即持写锁，“幂等判定 + 冲突检查 + 写入”
        # 对并发请求原子（并发写者被串行化）。
        conn.execute("BEGIN IMMEDIATE")
        try:
            now = datetime.now(timezone.utc)
            now_s = now.strftime(_CANON_FMT)

            live = _find_live_hold(conn, hold_key, now_s)
            if live is not None:
                if (live["start_utc"] == start_s
                        and live["end_utc"] == end_s
                        and live["participants_key"] == participants_key
                        and live["ttl_seconds"] == ttl_seconds):
                    conn.execute("COMMIT")
                    return _row_to_hold(live), False
                raise HoldKeyConflictError(int(live["id"]))

            conflicts = _collect_conflicts(
                conn, participants, start_s, end_s, now_s)
            if conflicts:
                raise HoldConflictError(conflicts)

            expires = now + timedelta(seconds=ttl_seconds)
            expires_s = expires.strftime(_CANON_FMT)
            cursor = conn.execute(
                "INSERT INTO reservation_holds"
                " (hold_key, start_utc, end_utc, participants_key, ttl_seconds,"
                "  status, created_at, expires_at)"
                " VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
                (hold_key, start_s, end_s, participants_key,
                 ttl_seconds, now_s, expires_s),
            )
            hold_id = int(cursor.lastrowid)
            conn.executemany(
                "INSERT INTO reservation_hold_participants"
                " (hold_id, participant_id, start_utc, end_utc)"
                " VALUES (?, ?, ?, ?)",
                [(hold_id, pid, start_s, end_s) for pid in participants],
            )
            conn.execute("COMMIT")
            return Hold(
                id=hold_id,
                hold_key=hold_key,
                start_utc=_parse(start_s),
                end_utc=_parse(end_s),
                participants=participants,
                ttl_seconds=ttl_seconds,
                created_at=now,
                expires_at=expires,
            ), True
        except BaseException:
            # 无活动事务时 rollback 为 no-op；任何失败都保证整体回滚
            conn.rollback()
            raise
    finally:
        conn.close()


def confirm_hold(
    hold_id: int,
    request_id: str,
    db_path: str | None = None,
) -> tuple[Reservation, bool]:
    """确认保留：在同一事务转为正式预约，返回 ``(预约, 是否本次确认)``。

    整个流程在单个 ``BEGIN IMMEDIATE`` 事务内完成，任何一步失败都回滚：

    1. 保留不存在 → :class:`HoldNotFoundError`（404）；
    2. 已确认 → 幂等返回其指向的同一预约，``created=False``；
    3. pending 但 ``expires_at <= now`` → :class:`HoldExpiredError`（410），
       保留保持 pending（过期态），不产生任何写入；
    4. ``request_id`` 已被其他预约占用 → :class:`RequestIdConflictError`（409）；
    5. 复检与进行中预约及其他有效保留的半开区间冲突（创建保留后可能又有
       直接创建的正式预约或并发确认落入同一时段）→ :class:`HoldConflictError`
       （409 + 冲突明细）；
    6. 插入正式预约及参与者占用，同事务追加 ``created`` 审计事件，
       最后把保留置为 ``confirmed`` 并回填 ``reservation_id`` /
       ``request_id`` / ``confirmed_at``；返回预约，``created=True``。
    """
    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        # BEGIN IMMEDIATE：与创建路径一致，事务开始即持写锁，
        # “状态/有效期判定 + request_id + 冲突复检 + 转预约 + 标记确认”
        # 对并发请求原子。
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                f"SELECT {_HOLD_COLUMNS} FROM reservation_holds WHERE id = ?",
                (hold_id,),
            ).fetchone()
            if row is None:
                raise HoldNotFoundError(hold_id)

            if row["status"] == "confirmed":
                # 重复确认（含在确认之后、过期与否无关）：幂等返回同一预约
                reservation_row = conn.execute(
                    "SELECT * FROM reservations WHERE id = ?",
                    (row["reservation_id"],),
                ).fetchone()
                # 正常流程下 reservation_id 必然存在；保留行不删除，
                # 外部手工破坏数据时给出明确错误而非 500
                if reservation_row is None:
                    raise HoldNotFoundError(hold_id)
                conn.execute("COMMIT")
                return _row_to_reservation(reservation_row), False

            now = datetime.now(timezone.utc)
            now_s = now.strftime(_CANON_FMT)
            if row["expires_at"] <= now_s:
                # 过期未确认：终态，不可确认；不写入任何内容
                raise HoldExpiredError(hold_id)

            start_s = row["start_utc"]
            end_s = row["end_utc"]
            participants = tuple(json.loads(row["participants_key"]))

            # request_id 幂等：正常情况下不应与已有预约撞键（确认是产生预约
            # 的新途径）；撞键即按与直接创建一致的语义处理
            existing = conn.execute(
                "SELECT id, start_utc, end_utc, participants_key FROM reservations"
                " WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if existing is not None:
                if (existing["start_utc"] == start_s
                        and existing["end_utc"] == end_s
                        and existing["participants_key"] == row["participants_key"]):
                    # 同一 request_id + 同一内容的迟到确认：挂接既有预约并
                    # 幂等地把保留标记为 confirmed（不新增预约、不新增事件）
                    reservation_id = int(existing["id"])
                    conn.execute(
                        "UPDATE reservation_holds"
                        " SET status = 'confirmed', reservation_id = ?,"
                        "     request_id = ?, confirmed_at = ?"
                        " WHERE id = ?",
                        (reservation_id, request_id, now_s, hold_id),
                    )
                    reservation_row = conn.execute(
                        "SELECT * FROM reservations WHERE id = ?",
                        (reservation_id,),
                    ).fetchone()
                    conn.execute("COMMIT")
                    return _row_to_reservation(reservation_row), False
                raise RequestIdConflictError(int(existing["id"]))

            # 转正式预约前复检冲突：保留创建后可能已有直接创建的预约占用同一时段，
            # 或并发确认的另一个保留已先行落入该时段。
            # （排除自身 hold_id；其他已确认/已过期保留的占用不参与检查）
            conflicts = _collect_conflicts(
                conn, participants, start_s, end_s, now_s,
                exclude_hold_id=hold_id,
            )
            if conflicts:
                raise HoldConflictError(conflicts)

            cursor = conn.execute(
                "INSERT INTO reservations"
                " (request_id, start_utc, end_utc, participants_key, version, created_at)"
                " VALUES (?, ?, ?, ?, 1, ?)",
                (request_id, start_s, end_s, row["participants_key"], now_s),
            )
            reservation_id = int(cursor.lastrowid)
            conn.executemany(
                "INSERT INTO reservation_participants"
                " (reservation_id, participant_id, start_utc, end_utc)"
                " VALUES (?, ?, ?, ?)",
                [(reservation_id, pid, start_s, end_s) for pid in participants],
            )
            # 审计事件与“转正式 + 标记确认”同一事务：COMMIT 前任一失败整体回滚，
            # 不会出现“有预约无事件”或“已确认但无预约”
            snapshot = audit.build_snapshot(
                reservation_id=reservation_id,
                request_id=request_id,
                start_utc=start_s,
                end_utc=end_s,
                participants=participants,
                version=1,
                status="active",
                canceled_at=None,
                created_at=now_s,
            )
            audit.append_event(
                conn,
                reservation_id=reservation_id,
                action="created",
                timestamp=now_s,
                snapshot=snapshot,
            )
            conn.execute(
                "UPDATE reservation_holds"
                " SET status = 'confirmed', reservation_id = ?,"
                "     request_id = ?, confirmed_at = ?"
                " WHERE id = ?",
                (reservation_id, request_id, now_s, hold_id),
            )
            conn.execute("COMMIT")
            return Reservation(
                id=reservation_id,
                request_id=request_id,
                start_utc=_parse(start_s),
                end_utc=_parse(end_s),
                participants=participants,
                created_at=now,
                version=1,
            ), True
        except BaseException:
            # 无活动事务时 rollback 为 no-op；任何失败都保证整体回滚
            conn.rollback()
            raise
    finally:
        conn.close()
