"""预约临时保留（hold）接口测试：TTL 占位、过期忽略、冲突检测与确认转预约。"""

from __future__ import annotations

import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app import db as db_module
from app import holds as hold_core
from app import reservations as reservation_core
from app.main import app


@pytest.fixture()
def db_path(tmp_path, monkeypatch) -> str:
    """每个测试用独立的 SQLite 文件。"""
    path = str(tmp_path / "reservations.db")
    monkeypatch.setenv("RESERVATIONS_DB_PATH", path)
    db_module.init_db(path)
    return path


@pytest.fixture()
def client(db_path) -> TestClient:
    return TestClient(app)


def _hold_payload(**overrides):
    payload = {
        "hold_key": "hold-1",
        "start_utc": "2026-03-09T13:00:00Z",
        "end_utc": "2026-03-09T14:00:00Z",
        "participant_ids": ["alice", "bob"],
        "ttl_seconds": 120,
    }
    payload.update(overrides)
    return payload


def _reservation_payload(**overrides):
    payload = {
        "request_id": "req-1",
        "start_utc": "2026-03-09T13:00:00Z",
        "end_utc": "2026-03-09T14:00:00Z",
        "participant_ids": ["alice", "bob"],
    }
    payload.update(overrides)
    return payload


def _create_hold(client, **overrides):
    return client.post("/api/v1/reservation-holds",
                       json=_hold_payload(**overrides))


def _force_expire(db_path: str, hold_id: int) -> None:
    """直接把保留改成已过期（过期判定的时间来源是服务端时钟）。"""
    past = (datetime.now(timezone.utc) - timedelta(seconds=1)
            ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("UPDATE reservation_holds SET expires_at = ? WHERE id = ?",
                     (past, hold_id))
        conn.commit()
    finally:
        conn.close()


def _utc(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


# ------------------------------------------------------------------ 创建保留

def test_create_hold_returns_201_with_expiry(client):
    resp = _create_hold(client)
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["hold_id"] == 1
    assert data["hold_key"] == "hold-1"
    assert data["start_utc"] == "2026-03-09T13:00:00Z"
    assert data["end_utc"] == "2026-03-09T14:00:00Z"
    assert data["participants"] == ["alice", "bob"]
    assert data["ttl_seconds"] == 120
    assert data["status"] == "pending"
    assert data["reservation_id"] is None
    assert data["request_id"] is None
    assert data["confirmed_at"] is None
    assert data["created_at"]
    assert data["expires_at"]
    created = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
    expires = datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00"))
    assert (expires - created).total_seconds() == pytest.approx(120, abs=1)


def test_create_hold_normalizes_participants(client):
    resp = _create_hold(client, participant_ids=["bob", "alice", "bob"])
    assert resp.status_code == 201
    assert resp.json()["participants"] == ["alice", "bob"]


@pytest.mark.parametrize("ttl", [0, -1, 901, 10_000])
def test_ttl_out_of_range_returns_422(client, ttl):
    resp = _create_hold(client, ttl_seconds=ttl)
    assert resp.status_code == 422


@pytest.mark.parametrize("ttl", [1, 900])
def test_ttl_boundaries_accepted(client, ttl):
    resp = _create_hold(client, ttl_seconds=ttl)
    assert resp.status_code == 201, resp.text


def test_hold_reuses_reservation_window_validation(client):
    assert _create_hold(client, end_utc="2026-03-09T13:00:00Z").status_code == 422
    assert _create_hold(client, start_utc="2026-03-09T13:00:00").status_code == 422
    assert _create_hold(client, participant_ids=[]).status_code == 422
    assert _create_hold(client, hold_key="").status_code == 422


# ----------------------------------------------- 保留 vs 进行中预约的冲突检查

def test_hold_conflicts_with_active_reservation(client):
    assert client.post("/api/v1/reservations",
                       json=_reservation_payload()).status_code == 201
    resp = _create_hold(client)
    assert resp.status_code == 409
    conflicts = resp.json()["conflicts"]
    assert {"kind": "reservation", "reservation_id": 1,
            "hold_id": None, "participant_id": "alice"} in conflicts
    assert {"kind": "reservation", "reservation_id": 1,
            "hold_id": None, "participant_id": "bob"} in conflicts


def test_hold_conflict_reports_individual_overlapping_participants(client):
    # 进行中预约只占 alice；保留含 alice + carol → 仅 alice 冲突
    assert client.post("/api/v1/reservations",
                       json=_reservation_payload(participant_ids=["alice"])
                       ).status_code == 201
    resp = _create_hold(client, participant_ids=["carol", "alice"])
    assert resp.status_code == 409
    assert resp.json()["conflicts"] == [
        {"kind": "reservation", "reservation_id": 1,
         "hold_id": None, "participant_id": "alice"}
    ]


def test_hold_adjacent_to_active_reservation_ok(client):
    assert client.post("/api/v1/reservations",
                       json=_reservation_payload()).status_code == 201
    # 紧邻其后 / 其前（半开区间，end == start 不冲突）
    assert _create_hold(
        client, hold_key="h-after",
        start_utc="2026-03-09T14:00:00Z",
        end_utc="2026-03-09T15:00:00Z").status_code == 201
    assert _create_hold(
        client, hold_key="h-before",
        start_utc="2026-03-09T12:00:00Z",
        end_utc="2026-03-09T13:00:00Z").status_code == 201


def test_hold_disjoint_participants_same_slot_ok(client):
    assert client.post("/api/v1/reservations",
                       json=_reservation_payload()).status_code == 201
    resp = _create_hold(client, participant_ids=["carol", "dave"])
    assert resp.status_code == 201, resp.text


def test_canceled_reservation_does_not_block_hold(client):
    rid = client.post("/api/v1/reservations",
                      json=_reservation_payload()).json()["reservation_id"]
    assert client.delete(f"/api/v1/reservations/{rid}",
                         params={"expected_version": 1}).status_code == 200
    # 取消后占用同事务释放：保留可立即占位
    assert _create_hold(client).status_code == 201


def test_hold_conflict_leaves_no_writes(client, db_path):
    assert client.post("/api/v1/reservations",
                       json=_reservation_payload()).status_code == 201
    assert _create_hold(client).status_code == 409
    # 没有任何 hold / hold_participants 残留
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM reservation_holds").fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM reservation_hold_participants").fetchone()[0] == 0
    finally:
        conn.close()


# ------------------------------------------------------- 保留之间的冲突检查

def test_overlapping_holds_conflict(client):
    assert _create_hold(client, hold_key="h-1").status_code == 201
    resp = _create_hold(client, hold_key="h-2", participant_ids=["bob"])
    assert resp.status_code == 409
    assert resp.json()["conflicts"] == [
        {"kind": "hold", "reservation_id": None,
         "hold_id": 1, "participant_id": "bob"}
    ]


def test_overlapping_holds_disjoint_participants_ok(client):
    assert _create_hold(client, hold_key="h-1").status_code == 201
    assert _create_hold(client, hold_key="h-2",
                        participant_ids=["carol"]).status_code == 201


def test_adjacent_holds_do_not_conflict(client):
    assert _create_hold(client, hold_key="h-1").status_code == 201
    assert _create_hold(
        client, hold_key="h-2",
        start_utc="2026-03-09T14:00:00Z",
        end_utc="2026-03-09T15:00:00Z").status_code == 201


def test_expired_hold_is_ignored_in_conflict_check(client, db_path):
    first = _create_hold(client, hold_key="h-1").json()
    _force_expire(db_path, first["hold_id"])
    # 过期保留的占用被忽略：同参与者同时段可再次保留
    resp = _create_hold(client, hold_key="h-2")
    assert resp.status_code == 201, resp.text
    assert resp.json()["hold_id"] == 2


def test_confirmed_hold_does_not_block_new_hold_via_hold_table(client):
    # 确认后的保留以正式预约身份参与冲突检查（reservation_participants 占用）
    first = _create_hold(client, hold_key="h-1").json()
    confirm = client.post(
        f"/api/v1/reservation-holds/{first['hold_id']}/confirm",
        json={"request_id": "req-confirm"})
    assert confirm.status_code == 201
    resp = _create_hold(client, hold_key="h-2", participant_ids=["alice"])
    assert resp.status_code == 409
    assert resp.json()["conflicts"] == [
        {"kind": "reservation", "reservation_id": confirm.json()["reservation_id"],
         "hold_id": None, "participant_id": "alice"}
    ]


# ------------------------------------------------------------------ 幂等

def test_hold_idempotent_replay_returns_same_hold(client):
    first = _create_hold(client, participant_ids=["bob", "alice"])
    assert first.status_code == 201
    replay = _create_hold(client, participant_ids=["alice", "bob", "alice"])
    assert replay.status_code == 200
    assert replay.json() == first.json()


def test_hold_replay_with_different_window_returns_409(client):
    assert _create_hold(client).status_code == 201
    resp = _create_hold(client, end_utc="2026-03-09T15:00:00Z")
    assert resp.status_code == 409
    assert resp.json()["existing_hold_id"] == 1


def test_hold_replay_with_different_participants_returns_409(client):
    assert _create_hold(client).status_code == 201
    resp = _create_hold(client, participant_ids=["alice"])
    assert resp.status_code == 409
    assert resp.json()["existing_hold_id"] == 1


def test_hold_replay_with_different_ttl_returns_409(client):
    assert _create_hold(client, ttl_seconds=60).status_code == 201
    resp = _create_hold(client, ttl_seconds=120)
    assert resp.status_code == 409


def test_expired_hold_key_can_be_reused_as_new_hold(client, db_path):
    first = _create_hold(client).json()
    _force_expire(db_path, first["hold_id"])
    resp = _create_hold(client)  # 同 key 同内容，但旧保留已过期
    assert resp.status_code == 201, resp.text
    assert resp.json()["hold_id"] == 2
    # 旧行保留未被删除/覆盖
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, status FROM reservation_holds ORDER BY id").fetchall()
        assert rows == [(1, "pending"), (2, "pending")]
    finally:
        conn.close()


def test_confirmed_hold_key_can_be_reused(client):
    first = _create_hold(client).json()
    assert client.post(
        f"/api/v1/reservation-holds/{first['hold_id']}/confirm",
        json={"request_id": "req-c"}).status_code == 201
    # 同 key 再起一个不重叠的新保留（复用成功即视为可复用）
    resp = _create_hold(client,
                        start_utc="2026-03-10T13:00:00Z",
                        end_utc="2026-03-10T14:00:00Z")
    assert resp.status_code == 201, resp.text
    assert resp.json()["hold_id"] == 2
    assert resp.json()["status"] == "pending"


# ------------------------------------------------------------------ 确认

def test_confirm_pending_hold_creates_reservation(client):
    hold = _create_hold(client).json()
    resp = client.post(f"/api/v1/reservation-holds/{hold['hold_id']}/confirm",
                       json={"request_id": "req-confirm"})
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["reservation_id"] >= 1
    assert data["request_id"] == "req-confirm"
    assert data["start_utc"] == hold["start_utc"]
    assert data["end_utc"] == hold["end_utc"]
    assert data["participants"] == ["alice", "bob"]
    assert data["version"] == 1
    assert data["status"] == "active"
    assert data["canceled_at"] is None
    assert data["created_at"]


def test_confirm_marks_hold_confirmed(client):
    hold = _create_hold(client).json()
    confirmed = client.post(
        f"/api/v1/reservation-holds/{hold['hold_id']}/confirm",
        json={"request_id": "req-c"}).json()
    # 用另一 key 在同时段占位应被正式预约挡住，侧面证明占用已生效
    blocked = _create_hold(client, hold_key="h-other", participant_ids=["alice"])
    assert blocked.status_code == 409

    # 保留行已标记 confirmed（同 key 幂等重放可观察到状态）
    replay = _create_hold(client,
                          start_utc="2026-03-10T13:00:00Z",
                          end_utc="2026-03-10T14:00:00Z")
    assert replay.status_code == 201
    assert replay.json()["hold_id"] == 2
    # 直接检查第一条保留的终态
    conn = sqlite3.connect(os.environ["RESERVATIONS_DB_PATH"])
    try:
        row = conn.execute(
            "SELECT status, reservation_id, request_id, confirmed_at"
            " FROM reservation_holds WHERE id = 1").fetchone()
        assert row[0] == "confirmed"
        assert row[1] == confirmed["reservation_id"]
        assert row[2] == "req-c"
        assert row[3]
    finally:
        conn.close()


def test_confirm_appends_created_audit_event(client):
    hold = _create_hold(client).json()
    confirmed = client.post(
        f"/api/v1/reservation-holds/{hold['hold_id']}/confirm",
        json={"request_id": "req-audit"}).json()
    rid = confirmed["reservation_id"]
    history = client.get(f"/api/v1/reservations/{rid}/history")
    assert history.status_code == 200, history.text
    events = history.json()["events"]
    assert len(events) == 1
    event = events[0]
    assert event["seq"] == 1
    assert event["action"] == "created"
    assert event["snapshot"]["reservation_id"] == rid
    assert event["snapshot"]["request_id"] == "req-audit"
    assert event["snapshot"]["participants"] == ["alice", "bob"]
    assert event["snapshot"]["version"] == 1
    assert event["prev_hash"] == "0" * 64
    assert len(event["hash"]) == 64


def test_confirm_unknown_hold_returns_404(client):
    resp = client.post("/api/v1/reservation-holds/999/confirm",
                       json={"request_id": "req-x"})
    assert resp.status_code == 404
    assert "999" in resp.json()["detail"]


def test_confirm_expired_hold_returns_410(client, db_path):
    hold = _create_hold(client).json()
    _force_expire(db_path, hold["hold_id"])
    resp = client.post(f"/api/v1/reservation-holds/{hold['hold_id']}/confirm",
                       json={"request_id": "req-late"})
    assert resp.status_code == 410
    assert str(hold["hold_id"]) in resp.json()["detail"]

    # 过期未确认：没有产生预约/事件/占用
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM reservations").fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM reservation_events").fetchone()[0] == 0
        row = conn.execute(
            "SELECT status, reservation_id FROM reservation_holds WHERE id = 1"
        ).fetchone()
        assert row == ("pending", None)
    finally:
        conn.close()


def test_repeated_confirm_returns_same_reservation(client):
    hold = _create_hold(client).json()
    url = f"/api/v1/reservation-holds/{hold['hold_id']}/confirm"
    first = client.post(url, json={"request_id": "req-once"})
    assert first.status_code == 201
    # 重复确认（即使换 request_id）→ 幂等返回同一预约，200
    second = client.post(url, json={"request_id": "req-again"})
    assert second.status_code == 200
    assert second.json() == first.json()
    third = client.post(url, json={"request_id": "req-once"})
    assert third.status_code == 200
    assert third.json() == first.json()


def test_repeated_confirm_after_expiry_still_returns_same_reservation(
        client, db_path):
    hold = _create_hold(client, ttl_seconds=1).json()
    url = f"/api/v1/reservation-holds/{hold['hold_id']}/confirm"
    first = client.post(url, json={"request_id": "req-once"})
    assert first.status_code == 201
    # 确认之后保留再“过期”也不影响幂等：confirmed 优先级高于过期
    _force_expire(db_path, hold["hold_id"])
    replay = client.post(url, json={"request_id": "req-again"})
    assert replay.status_code == 200
    assert replay.json() == first.json()


def test_confirm_requires_nonempty_request_id(client):
    hold = _create_hold(client).json()
    resp = client.post(f"/api/v1/reservation-holds/{hold['hold_id']}/confirm",
                       json={"request_id": ""})
    assert resp.status_code == 422


def test_confirm_conflicts_with_reservation_created_after_hold(client):
    """保留创建后、确认前，直接创建的预约抢占了同一时段 → 确认 409。"""
    hold = _create_hold(client).json()
    assert client.post("/api/v1/reservations", json=_reservation_payload(
        request_id="req-racer")).status_code == 201
    resp = client.post(f"/api/v1/reservation-holds/{hold['hold_id']}/confirm",
                       json={"request_id": "req-confirm"})
    assert resp.status_code == 409
    assert resp.json()["conflicts"] == [
        {"kind": "reservation", "reservation_id": 1,
         "hold_id": None, "participant_id": "alice"},
        {"kind": "reservation", "reservation_id": 1,
         "hold_id": None, "participant_id": "bob"},
    ]
    # 确认失败：保留仍 pending，无预约产生
    conn = sqlite3.connect(os.environ["RESERVATIONS_DB_PATH"])
    try:
        assert conn.execute(
            "SELECT status FROM reservation_holds WHERE id = 1").fetchone()[0] \
            == "pending"
        # 只有直接创建的那一条预约
        assert conn.execute(
            "SELECT COUNT(*) FROM reservations").fetchone()[0] == 1
    finally:
        conn.close()


def test_confirm_request_id_clash_returns_409(client):
    hold = _create_hold(client).json()
    # request_id 先被内容不同的预约占用
    assert client.post("/api/v1/reservations", json=_reservation_payload(
        request_id="shared-req",
        start_utc="2026-03-20T09:00:00Z",
        end_utc="2026-03-20T10:00:00Z",
        participant_ids=["xenia"])).status_code == 201
    resp = client.post(f"/api/v1/reservation-holds/{hold['hold_id']}/confirm",
                       json={"request_id": "shared-req"})
    assert resp.status_code == 409
    assert resp.json()["existing_reservation_id"] == 1


def test_confirm_same_request_id_same_content_is_idempotent(client):
    """迟到确认撞上同内容的既有预约：挂接该预约并幂等标记 confirmed。"""
    hold = _create_hold(client).json()
    # 直接用同内容 + 同 request_id 先建预约（与保留时段相同，直接创建不检查保留）
    direct = client.post("/api/v1/reservations",
                         json=_reservation_payload(request_id="req-shared"))
    assert direct.status_code == 201
    rid = direct.json()["reservation_id"]

    resp = client.post(f"/api/v1/reservation-holds/{hold['hold_id']}/confirm",
                       json={"request_id": "req-shared"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["reservation_id"] == rid

    # 只产生了直接创建那一条 created 事件（确认没有新增预约/事件）
    history = client.get(f"/api/v1/reservations/{rid}/history").json()
    assert [e["action"] for e in history["events"]] == ["created"]


# ---------------------------------------------------------------------- 并发

def test_concurrent_holds_same_slot_single_winner(db_path):
    """8 个并发保留抢占同一参与者的同一时段：恰好一个成功，其余 409。"""
    barrier = threading.Barrier(8)

    def worker(i: int):
        barrier.wait()
        return hold_core.create_hold(
            hold_key=f"key-{i}",
            start_utc=_utc(2026, 3, 9, 13),
            end_utc=_utc(2026, 3, 9, 14),
            participant_ids=["alice"],
            ttl_seconds=60,
            db_path=db_path,
        )

    created, conflicts = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, i) for i in range(8)]
        for f in futures:
            try:
                created.append(f.result())
            except hold_core.HoldConflictError as exc:
                conflicts.append(exc)

    assert len(created) == 1
    assert len(conflicts) == 7
    winner = created[0][0]
    assert created[0][1] is True
    for exc in conflicts:
        assert exc.conflicts == [
            {"kind": "hold", "reservation_id": None,
             "hold_id": winner.id, "participant_id": "alice"}
        ]


def test_concurrent_confirms_single_reservation(db_path):
    """8 个并发确认同一保留：只建一条预约，全部拿到同一结果。"""
    hold, _ = hold_core.create_hold(
        hold_key="key-1",
        start_utc=_utc(2026, 3, 9, 13),
        end_utc=_utc(2026, 3, 9, 14),
        participant_ids=["alice"],
        ttl_seconds=300,
        db_path=db_path,
    )
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        return hold_core.confirm_hold(
            hold_id=hold.id, request_id="req-confirm", db_path=db_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda _: worker(), range(8)))

    ids = {r.id for r, _ in outcomes}
    assert len(ids) == 1                                    # 同一条预约
    assert sum(1 for _, created in outcomes if created) == 1  # 只新建一次
    # 保留参与者行仍在（不过期不删除），但已是 confirmed
    conn = sqlite3.connect(db_path)
    try:
        status = conn.execute(
            "SELECT status, reservation_id FROM reservation_holds WHERE id = ?",
            (hold.id,)).fetchone()
        assert status == ("confirmed", ids.pop())
    finally:
        conn.close()


def test_concurrent_holds_via_api(db_path):
    """API 层并发保留：同一参与者同一时段，恰好一个 201。"""

    def worker(i: int):
        c = TestClient(app)
        return c.post("/api/v1/reservation-holds", json=_hold_payload(
            hold_key=f"key-{i}", participant_ids=["alice"]))

    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(pool.map(worker, range(6)))

    assert sorted(r.status_code for r in responses) == (
        [201] + [409] * 5)


# ---------------------------------------------------------- 核心层过期边界

def test_hold_expiring_exactly_now_is_expired(db_path):
    """expires_at <= now 即过期（半开：等于也算过期）。"""
    hold, _ = hold_core.create_hold(
        hold_key="key-1",
        start_utc=_utc(2026, 3, 9, 13),
        end_utc=_utc(2026, 3, 9, 14),
        participant_ids=["alice"],
        ttl_seconds=60,
        db_path=db_path,
    )
    # 直接把 expires_at 改成当前时刻：<= now 应判定为过期
    now_s = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("UPDATE reservation_holds SET expires_at = ? WHERE id = ?",
                     (now_s, hold.id))
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(hold_core.HoldExpiredError):
        hold_core.confirm_hold(hold.id, "req-x", db_path=db_path)


def test_create_hold_ttl_validation_core(db_path):
    with pytest.raises(ValueError):
        hold_core.create_hold(
            hold_key="k", start_utc=_utc(2026, 3, 9, 13),
            end_utc=_utc(2026, 3, 9, 14), participant_ids=["a"],
            ttl_seconds=901, db_path=db_path)


# -------------------------------------------------- 既有数据库无损新增表

def test_startup_adds_hold_tables_to_existing_database(tmp_path, monkeypatch):
    """旧库（只有预约表）启动时无损新增保留表，既有预约数据不受影响。"""
    import sqlite3

    path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(path)
    try:
        # 最旧 schema：连 version 列都没有
        conn.executescript("""\
CREATE TABLE reservations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL UNIQUE,
    start_utc TEXT NOT NULL,
    end_utc TEXT NOT NULL,
    participants_key TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE reservation_participants (
    reservation_id INTEGER NOT NULL REFERENCES reservations(id) ON DELETE CASCADE,
    participant_id TEXT NOT NULL,
    start_utc TEXT NOT NULL,
    end_utc TEXT NOT NULL,
    PRIMARY KEY (reservation_id, participant_id)
);
""")
        conn.execute(
            "INSERT INTO reservations"
            " (request_id, start_utc, end_utc, participants_key, created_at)"
            " VALUES ('old-req', '2026-03-09T13:00:00.000000Z',"
            " '2026-03-09T14:00:00.000000Z', '[\"alice\"]',"
            " '2026-03-01T00:00:00.000000Z')")
        conn.execute(
            "INSERT INTO reservation_participants"
            " (reservation_id, participant_id, start_utc, end_utc)"
            " VALUES (1, 'alice', '2026-03-09T13:00:00.000000Z',"
            " '2026-03-09T14:00:00.000000Z')")
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setenv("RESERVATIONS_DB_PATH", path)
    db_module.init_db(path)
    db_module.init_db(path)  # 建表幂等

    # 既有预约无损可读
    got = reservation_core.create_reservation(
        request_id="old-req",
        start_utc=_utc(2026, 3, 9, 13),
        end_utc=_utc(2026, 3, 9, 14),
        participant_ids=["alice"],
        db_path=path,
    )[0]
    assert got.id == 1
    assert got.version == 1

    # 新表可用：保留与既有预约冲突
    with pytest.raises(hold_core.HoldConflictError):
        hold_core.create_hold(
            hold_key="new-key",
            start_utc=_utc(2026, 3, 9, 13, 30),
            end_utc=_utc(2026, 3, 9, 13, 45),
            participant_ids=["alice"],
            ttl_seconds=60,
            db_path=path,
        )
    # 不冲突的保留可创建并确认
    hold, created = hold_core.create_hold(
        hold_key="new-key",
        start_utc=_utc(2026, 3, 10, 13),
        end_utc=_utc(2026, 3, 10, 14),
        participant_ids=["alice"],
        ttl_seconds=60,
        db_path=path,
    )
    assert created is True
    reservation, did_confirm = hold_core.confirm_hold(
        hold.id, "new-req", db_path=path)
    assert did_confirm is True
    assert reservation.id == 2
