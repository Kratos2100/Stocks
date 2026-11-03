from fastapi.testclient import TestClient

from options_mas.api.server import ReplayRuntime, create_app
from options_mas.config import AppConfig
from options_mas.data.synthetic import generate_synthetic_dataset


def test_dashboard_api_endpoints():
    rt = ReplayRuntime(AppConfig(), generate_synthetic_dataset(sessions=3, seed=1), speed=0)
    with TestClient(create_app(rt, autostart=False)) as c:
        for _ in range(120):
            rt.step()
        st = c.get("/api/status").json()
        assert "SYNTHETIC" in st["data_label"] and st["warnings"]
        assert {a["agent_id"] for a in st["agents"]} == {"A01", "A02", "A03"}
        key = c.get("/api/contracts").json()[0]["key"]
        ch = c.get(f"/api/chart/{key}").json()
        assert ch["candles"] and "ema50" in ch["lines"] and "ema9" in ch["lines"] and ch["delta"]
        assert ch["delta_estimated"] is True  # synthetic delta is never presented as bid/ask flow
        assert c.get("/api/chart/NOPE").status_code == 404
        assert isinstance(c.get("/api/signals").json(), list)
        assert "overall" in c.get("/api/performance").json()
        assert c.post("/api/emergency-stop").json()["emergency"] is True
        assert c.get("/").status_code == 200
        assert c.get("/static/vendor/lightweight-charts-4.1.3.js").status_code == 200
