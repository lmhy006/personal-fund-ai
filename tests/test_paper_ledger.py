# test_paper_ledger.py：最小前向账本解析验证（先验计算正确性，合成数据 + 临时目录）
# 覆盖：建仓费用/份额解析解、迟发顺延与 lag_days、官方日增长率复权估值、现金计息、
#       不可变幂等、未确认执行拒绝、事件日期一致性、stale 记录。
import json
import os
import shutil
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import paper_ledger as pl  # noqa: E402


class TestPaperLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml",
                                f"_pledger_test_{os.getpid()}_{id(self)}")
        self.hist = os.path.join(self.tmp, "history")
        self.out = os.path.join(self.tmp, "ledgers")
        self.dry = os.path.join(self.tmp, "dryrun")
        os.makedirs(self.hist, exist_ok=True)
        # A00001：1.0 → +10% → +10%；A00002：2.0 恒定；A00003：执行日缺净值（10-09 才披露）
        self._fund("A00001", [("2026-09-30", 1.0, 0.0), ("2026-10-01", 1.1, 0.10),
                              ("2026-10-02", 1.21, 0.10)])
        self._fund("A00002", [("2026-09-30", 2.0, 0.0), ("2026-10-01", 2.0, 0.0),
                              ("2026-10-02", 2.0, 0.0)])
        self._fund("A00003", [("2026-10-09", 3.0, 0.0), ("2026-10-12", 3.3, 0.10)])
        self.inputs = os.path.join(self.tmp, "paper_inputs_TEST.json")
        self._inputs(["A00001", "A00002"])

    def _fund(self, code, rows):
        pd.DataFrame([{"date": d, "nav": n, "nav_acc": n, "daily_ret": r,
                       "suspicious_jump": False} for d, n, r in rows]) \
            .to_csv(os.path.join(self.hist, f"fund_{code}.csv"), index=False)

    def _inputs(self, codes, cohort="2026-09", signal="2026-09-30"):
        proto = {"initial_nav": 1.0, "cohort_weight": 1.0, "hold_months": 6,
                 "buy_fee": 0.0015, "sell_fee": 0.005, "cash_annual_rate": 0.02}
        w = 1.0 / len(codes)
        payload = {"paper_inputs_version": 1, "run_id": "RID_TEST",
                   "signal_date": signal, "cohort_id": cohort, "protocol": proto,
                   "top50": [{"fund_code": c, "rank": i + 1, "weight": w}
                             for i, c in enumerate(codes)]}
        with open(self.inputs, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _ledger(self):
        return pl.PaperLedger(self.inputs, self.hist, self.out, dryrun_dir=self.dry)

    # ---- 建仓解析解 ----
    def test_open_position_analytic(self):
        led = self._ledger()
        payload = led.build_position("2026-09-30", capital=1.0)
        self.assertEqual(payload["n_positions"], 2)
        self.assertAlmostEqual(payload["invested"], 1.0, places=12)
        self.assertAlmostEqual(payload["cash"], 0.0, places=12)
        self.assertAlmostEqual(payload["total_buy_fee"], 0.0015, places=12)
        p1, p2 = payload["positions"]
        # 每只金额 0.5；申购费 0.5×0.15% = 0.00075；净申购 0.49925
        self.assertAlmostEqual(p1["amount"], 0.5, places=12)
        self.assertAlmostEqual(p1["buy_fee"], 0.00075, places=12)
        self.assertAlmostEqual(p1["net_amount"], 0.49925, places=12)
        self.assertAlmostEqual(p1["shares"], 0.49925 / 1.0, places=12)   # nav 1.0
        self.assertAlmostEqual(p2["shares"], 0.49925 / 2.0, places=12)   # nav 2.0
        self.assertEqual(p1["lag_days"], 0)

    def test_nav_lag_shifts_to_next_disclosure(self):
        self._inputs(["A00003"])
        led = self._ledger()
        payload = led.build_position("2026-09-30", capital=1.0)
        p = payload["positions"][0]
        self.assertEqual(p["fund_code"], "A00003")
        self.assertEqual(p["nav_date"], "2026-10-09")     # 顺延到下一披露日
        self.assertEqual(p["lag_days"], 9)
        self.assertAlmostEqual(p["nav_at_open"], 3.0, places=12)
        self.assertEqual(payload["lag_funds"], ["A00003"])

    def test_value_uses_official_daily_growth(self):
        led = self._ledger()
        val = led.value("2026-10-02", ledger=led.build_position("2026-09-30", capital=1.0))
        by = {r["fund_code"]: r for r in val["positions"]}
        self.assertAlmostEqual(by["A00001"]["growth_factor"], 1.21, places=12)
        self.assertAlmostEqual(by["A00002"]["growth_factor"], 1.0, places=12)
        self.assertAlmostEqual(by["A00001"]["market_value"], 0.49925 * 1.21, places=12)
        self.assertAlmostEqual(by["A00002"]["market_value"], 0.49925, places=12)
        self.assertAlmostEqual(val["market_value"], 0.49925 * 1.21 + 0.49925, places=12)
        self.assertAlmostEqual(val["nav"], val["market_value"], places=12)   # capital=1.0
        self.assertAlmostEqual(val["return_since_open"], 0.49925 * 1.21 + 0.49925 - 1.0, places=12)
        self.assertEqual(val["days_held"], 2)
        self.assertEqual(val["stale_funds"], [])

    def test_cash_interest_analytic(self):
        # 365 天：(1 + 0.02/365)^365
        self.assertAlmostEqual(pl.cash_growth(100.0, 365, 0.02),
                               100.0 * (1 + 0.02 / 365) ** 365, places=10)
        self.assertAlmostEqual(pl.cash_growth(1.0, 0, 0.02), 1.0, places=12)
        # 账本估值里的现金计息同口径
        led = self._ledger()
        fake = {"cohort": "2026-09", "run_id": "RID_TEST", "execution_date": "2026-09-30",
                "capital": 1.0, "initial_nav": 1.0, "cash": 0.5,
                "positions": [{"fund_code": "A00002", "shares": 0.0, "nav_at_open": 2.0,
                               "nav_date": "2026-09-30"}]}
        val = led.value("2026-10-02", ledger=fake)
        self.assertAlmostEqual(val["cash_grown"], pl.cash_growth(0.5, 2, 0.02), places=12)
        self.assertAlmostEqual(val["total_value"], val["market_value"] + val["cash_grown"], places=12)

    def test_stale_funds_recorded(self):
        self._inputs(["A00001"])
        led = self._ledger()
        led_ = led.build_position("2026-09-30", capital=1.0)
        val = led.value("2026-10-05", ledger=led_)          # 10-03 起无新净值
        self.assertEqual(len(val["stale_funds"]), 1)
        self.assertEqual(val["stale_funds"][0]["last_nav_date"], "2026-10-02")
        self.assertEqual(val["stale_funds"][0]["stale_days"], 3)

    # ---- 不可变 / 执行事实 ----
    def test_open_is_immutable_and_idempotent(self):
        led = self._ledger()
        r1 = led.open_position("2026-09-30", capital=1.0, simulate=True)
        self.assertEqual(r1["status"], "opened")
        self.assertTrue(r1["ledger"]["simulated"])
        r2 = led.open_position("2026-09-30", capital=1.0, simulate=True)
        self.assertEqual(r2["status"], "existing")

    def test_requires_confirmed_execution(self):
        led = self._ledger()
        events = os.path.join(self.tmp, "events.jsonl")
        with self.assertRaises(RuntimeError):
            led.open_position("2026-09-30", capital=1.0, events_path=events)

    def test_execution_date_must_match_event(self):
        led = self._ledger()
        events = os.path.join(self.tmp, "events.jsonl")
        with open(events, "w", encoding="utf-8") as f:
            f.write(json.dumps({"event_id": "e1", "type": "paper", "cohort": "2026-09",
                                "execution_date": "2026-10-02"}) + "\n")
        with self.assertRaises(ValueError):
            led.open_position("2026-09-30", capital=1.0, events_path=events)
        res = led.open_position("2026-10-02", capital=1.0, events_path=events)
        self.assertEqual(res["status"], "opened")
        self.assertFalse(res["ledger"]["simulated"])

    def test_execution_fact_folds_correction(self):
        events = os.path.join(self.tmp, "events.jsonl")
        with open(events, "w", encoding="utf-8") as f:
            f.write(json.dumps({"event_id": "e1", "type": "paper", "cohort": "2026-09",
                                "execution_date": "2026-10-08"}) + "\n")
            f.write(json.dumps({"event_id": "e2", "type": "execution_correction",
                                "cohort": "2026-09", "execution_date": "2026-10-09",
                                "exec_type": "paper"}) + "\n")
        fact = pl.execution_fact("2026-09", events)
        self.assertEqual(fact["execution_date"], "2026-10-09")
        self.assertEqual(fact["event_id"], "e2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
