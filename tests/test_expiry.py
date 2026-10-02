# test_expiry.py：漏月到期识别与到期赎回确认（2026-10-02 审计第 5 节「到期追补」）
# 覆盖：overdue 识别（不自动改状态）、正常到期标记、confirm_expiry 留痕/幂等/回滚/未到期拒绝。
import json
import os
import shutil
import sys
import unittest
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import live_portfolio as lp  # noqa: E402


class TestExpiry(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml",
                                f"_expiry_test_{os.getpid()}_{id(self)}")
        os.makedirs(self.tmp, exist_ok=True)
        self._orig = {k: getattr(lp, k) for k in ("LEDGER_DIR", "LEDGER_PATH", "STATE_PATH",
                                                  "EXECUTION_EVENTS_PATH", "PROJECT_ROOT")}
        lp.LEDGER_DIR = self.tmp
        lp.LEDGER_PATH = os.path.join(self.tmp, "portfolio_ledger.csv")
        lp.STATE_PATH = os.path.join(self.tmp, "portfolio_state.json")
        lp.EXECUTION_EVENTS_PATH = os.path.join(self.tmp, "execution_events.jsonl")
        lp.PROJECT_ROOT = self.tmp
        self.pf = lp.LivePortfolio(ledger_path=lp.LEDGER_PATH, state_path=lp.STATE_PATH)
        # 建仓 2026-02 → 到期月 2026-08（HOLD=6）
        self.pf.state["cohorts"]["2026-02"] = {
            "signal_date": "2026-02-27", "execution_date": "2026-03-02",
            "expire_month": "2026-08", "cohort_weight": 1 / 6, "n_codes": 2,
            "status": "active", "score_run": "RID_2026_02"}
        self.pf.ledger = pd.DataFrame([
            {"cohort_id": "2026-02", "fund_code": c, "weight_in_cohort": 1 / 12,
             "signal_date": "2026-02-27", "execution_date": "2026-03-02",
             "created_at": "2026-02-27T10:00:00", "expire_month": "2026-08",
             "status": "active"} for c in ("000001", "000002")])
        self.pf.save()

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(lp, k, v)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _events(self):
        p = lp.EXECUTION_EVENTS_PATH
        if not os.path.exists(p):
            return []
        with open(p, encoding="utf-8") as f:
            return [json.loads(x) for x in f if x.strip()]

    def _scores(self, as_of):
        return pd.DataFrame({"as_of": [as_of] * 60,
                             "fund_code": [f"1{i:05d}" for i in range(60)],
                             "confidence": ["main"] * 60,
                             "rank": list(range(1, 61)),
                             "score": [1.0 - i / 100 for i in range(60)]})

    def test_overdue_detected_without_status_change(self):
        """漏跑 2026-09：到 2026-10 运行时必须识别 overdue，但**不自动**改状态。"""
        action = self.pf.add_month(self._scores("2026-10-30"), score_run="RID_2026_10")
        self.assertEqual(action["overdue_cohorts"][0]["cohort"], "2026-02")
        self.assertEqual(action["overdue_cohorts"][0]["months_overdue"], 2)
        self.assertEqual(self.pf.state["cohorts"]["2026-02"]["status"], "active")   # 未自动 expired
        self.assertEqual(self.pf.state["overdue_cohorts"][0]["cohort"], "2026-02")

    def test_normal_expiry_mark_on_due_month(self):
        action = self.pf.add_month(self._scores("2026-08-31"), score_run="RID_2026_08B")
        self.assertIn("2026-02", action["expired_cohorts"])
        self.assertEqual(self.pf.state["cohorts"]["2026-02"]["status"], "expired")
        self.assertEqual(action["overdue_cohorts"], [])

    def test_confirm_expiry_records_event_and_is_idempotent(self):
        ev = self.pf.confirm_expiry("2026-02", reason="到期赎回确认（漏月补记）", operator="tester")
        self.assertEqual(ev["type"], "cohort_expiry")
        self.assertEqual(ev["expire_month"], "2026-08")
        self.assertEqual(ev["months_overdue"], 2)
        self.assertEqual(ev["operator"], "tester")
        self.assertEqual(self.pf.state["cohorts"]["2026-02"]["status"], "expired")
        self.assertEqual(len(self._events()), 1)
        self.assertTrue(all(s == "expired" for s in self.pf.ledger["status"]))
        ev2 = self.pf.confirm_expiry("2026-02", reason="重复确认")
        self.assertTrue(ev2.get("idempotent"))
        self.assertEqual(len(self._events()), 1)

    def test_confirm_expiry_rejects_not_due(self):
        self.pf.state["cohorts"]["2026-02"]["expire_month"] = "2027-03"
        self.pf.save()
        with self.assertRaises(ValueError) as cm:
            self.pf.confirm_expiry("2026-02", reason="提前确认")
        self.assertIn("尚未到期", str(cm.exception))
        self.assertEqual(self.pf.state["cohorts"]["2026-02"]["status"], "active")

    def test_confirm_expiry_rolls_back_on_event_failure(self):
        with mock.patch.object(self.pf, "_append_event", side_effect=OSError("disk full")):
            with self.assertRaises(RuntimeError):
                self.pf.confirm_expiry("2026-02", reason="故障注入")
        pf2 = lp.LivePortfolio(ledger_path=lp.LEDGER_PATH, state_path=lp.STATE_PATH)
        self.assertEqual(pf2.state["cohorts"]["2026-02"]["status"], "active")     # 已回滚
        self.assertEqual(self._events(), [])

    def test_confirm_expiry_rejects_planned(self):
        self.pf.state["cohorts"]["2026-02"]["status"] = "planned"
        self.pf.save()
        with self.assertRaises(ValueError):
            self.pf.confirm_expiry("2026-02", reason="计划状态不该确认到期")

    def test_confirm_expiry_requires_reason(self):
        with self.assertRaises(ValueError):
            self.pf.confirm_expiry("2026-02", reason="  ")


if __name__ == "__main__":
    unittest.main(verbosity=2)
