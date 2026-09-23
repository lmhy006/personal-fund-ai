# test_forward_runtime.py：前向纸面运行期测试（月末信号门禁 + 执行事件留痕）
# fixture/mock 隔离：不运行真实流水线、不改真实 ledger/snapshots。
import json
import os
import shutil
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import live_portfolio as lp  # noqa: E402
from production_pipeline import _is_month_end  # noqa: E402


class TestMonthEndGate(unittest.TestCase):
    """月末信号门禁近似判定（最后工作日近似）。"""

    def test_month_end_cases(self):
        self.assertFalse(_is_month_end(pd.Timestamp("2026-09-21")))   # 9-21 月中（9 月还有大量工作日）
        self.assertFalse(_is_month_end(pd.Timestamp("2026-09-14")))
        self.assertTrue(_is_month_end(pd.Timestamp("2026-09-30")))    # 9-30 周三 = 9 月最后工作日
        self.assertTrue(_is_month_end(pd.Timestamp("2026-12-31")))    # 12-31 周四
        self.assertTrue(_is_month_end(pd.Timestamp("2026-08-31")))    # 8-31 周一
        self.assertFalse(_is_month_end(pd.Timestamp("2026-10-15")))


class TestExecutionEvent(unittest.TestCase):
    """confirm_execution 纸面执行事件（不可变 jsonl + 状态哈希）。"""

    def setUp(self):
        self.tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml",
                                f"_fwd_test_{os.getpid()}_{id(self)}")
        os.makedirs(self.tmp, exist_ok=True)
        self._orig = {k: getattr(lp, k) for k in ("LEDGER_DIR", "LEDGER_PATH",
                                                   "STATE_PATH", "EXECUTION_EVENTS_PATH")}
        lp.LEDGER_DIR = self.tmp
        lp.LEDGER_PATH = os.path.join(self.tmp, "portfolio_ledger.csv")
        lp.STATE_PATH = os.path.join(self.tmp, "portfolio_state.json")
        lp.EXECUTION_EVENTS_PATH = os.path.join(self.tmp, "execution_events.jsonl")
        # 构造 planned 状态的 ledger（**显式传路径**：模块级默认参数在定义时绑定，
# monkeypatch 模块属性不会影响已绑定的默认值——必须显式传入，避免污染真实 ledger）
        self.pf = lp.LivePortfolio(ledger_path=lp.LEDGER_PATH, state_path=lp.STATE_PATH)
        self.pf.state["cohorts"]["2026-09"] = {
            "signal_date": "2026-09-30", "execution_date": None, "expire_month": "2027-03",
            "cohort_weight": 1 / 6, "n_codes": 2, "status": "planned",
            "score_run": "20260930_100000",
        }
        self.pf.ledger = pd.DataFrame([{
            "cohort_id": "2026-09", "fund_code": c, "weight_in_cohort": 1 / 12,
            "signal_date": "2026-09-30", "execution_date": None,
            "created_at": "2026-09-30T10:00:00", "expire_month": "2027-03",
            "status": "planned"} for c in ("000001", "000002")])
        self.pf.save()

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(lp, k, v)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_confirm_paper_event(self):
        ev = self.pf.confirm_execution("2026-09", "2026-10-09", exec_type="paper",
                                       operator="tester")
        self.assertEqual(ev["type"], "paper")
        self.assertEqual(ev["cohort"], "2026-09")
        self.assertEqual(ev["source_run_id"], "20260930_100000")
        self.assertEqual(ev["execution_date"], "2026-10-09")
        self.assertEqual(ev["operator"], "tester")
        self.assertTrue(ev["ledger_sha256"])
        self.assertTrue(ev["state_sha256"])
        # cohort 已 active 且事件落盘（append-only）
        self.assertEqual(self.pf.state["cohorts"]["2026-09"]["status"], "active")
        lines = open(lp.EXECUTION_EVENTS_PATH, encoding="utf-8").readlines()
        self.assertEqual(len(lines), 1)
        rec = json.loads(lines[0])
        self.assertEqual(rec, ev)
        # 再确认一次（幂等写事件，不抛错）
        ev2 = self.pf.confirm_execution("2026-09", "2026-10-09", operator="tester2")
        with open(lp.EXECUTION_EVENTS_PATH, encoding="utf-8") as f:
            self.assertEqual(len(f.readlines()), 2)
        self.assertEqual(ev2["operator"], "tester2")

    def test_actual_type_and_invalid(self):
        ev = self.pf.confirm_execution("2026-09", "2026-10-09", exec_type="actual")
        self.assertEqual(ev["type"], "actual")
        with self.assertRaises(ValueError):
            self.pf.confirm_execution("2026-09", "2026-10-09", exec_type="real")

    def test_unknown_cohort_rejected(self):
        with self.assertRaises(KeyError):
            self.pf.confirm_execution("2099-01", "2099-01-31")


if __name__ == "__main__":
    unittest.main(verbosity=2)