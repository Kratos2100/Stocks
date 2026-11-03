"""SQLite persistence (development). The schema is plain SQL so it ports
directly to PostgreSQL for production (see docs/DEPLOYMENT.md)."""
from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    ts TEXT,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_run_kind ON events(run_id, kind);
CREATE TABLE IF NOT EXISTS signals (
    signal_id TEXT, run_id TEXT, agent_id TEXT, contract_key TEXT, signal_type TEXT, status TEXT,
    signal_ts TEXT, entry_trigger REAL, reason TEXT, payload TEXT,
    PRIMARY KEY (run_id, signal_id)
);
CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT, run_id TEXT, agent_id TEXT, contract_key TEXT, option_type TEXT, entry_ts TEXT,
    exit_ts TEXT, entry_price REAL, exit_price REAL, qty INTEGER, net_pnl REAL, exit_reason TEXT, payload TEXT,
    PRIMARY KEY (run_id, trade_id)
);
"""


class Store:
    def __init__(self, path: str = ":memory:", run_id: str = "default"):
        self.path = path
        self.run_id = run_id
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(SCHEMA)

    def handle(self, kind: str, payload: dict[str, Any]) -> None:
        """EventBus subscriber."""
        js = json.dumps(payload, default=str)
        with self._lock:
            ts = payload.get("signal_ts") or payload.get("ts") or payload.get("exit_ts") or payload.get("updated_ts")
            self.conn.execute("INSERT INTO events(run_id, kind, ts, payload) VALUES (?,?,?,?)",
                              (self.run_id, kind, ts, js))
            if kind == "signal":
                self.conn.execute(
                    "INSERT OR REPLACE INTO signals VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (payload["signal_id"], self.run_id, payload["agent_id"], payload["contract_key"],
                     payload["signal_type"], payload["status"], payload["signal_ts"], payload.get("entry_trigger"),
                     payload.get("reason"), js))
            elif kind == "trade":
                self.conn.execute(
                    "INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (payload["trade_id"], self.run_id, payload["agent_id"], payload["contract_key"],
                     payload["option_type"], payload["entry_ts"], payload["exit_ts"], payload["entry_price"],
                     payload["exit_price"], payload["qty"], payload["net_pnl"], payload["exit_reason"], js))
            self.conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._lock:
            cur = self.conn.execute(sql, params)
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def trades(self, run_id: Optional[str] = None) -> list[dict[str, Any]]:
        return [json.loads(r["payload"]) for r in
                self.query("SELECT payload FROM trades WHERE run_id=? ORDER BY exit_ts", (run_id or self.run_id,))]

    def signals(self, run_id: Optional[str] = None, limit: int = 500) -> list[dict[str, Any]]:
        return [json.loads(r["payload"]) for r in self.query(
            "SELECT payload FROM signals WHERE run_id=? ORDER BY signal_ts DESC LIMIT ?",
            (run_id or self.run_id, limit))]

    def close(self) -> None:
        self.conn.close()
