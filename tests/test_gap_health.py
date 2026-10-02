# test_gap_health.py：gap 例外机器化备案 + 基准新鲜度交易日口径（2026-10-02 审计第 5 节）
import json
import os
import shutil
import sys
import unittest
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import gap_exceptions as ge  # noqa: E402
import production_pipeline as pp  # noqa: E402
from data_health import BENCH_MAX_STALE_TRADING_DAYS, benchmark_stale_trading_days  # noqa: E402


class TestGapExceptions(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml",
                                f"_gap_test_{os.getpid()}_{id(self)}")
        os.makedirs(self.tmp, exist_ok=True)
        self.path = os.path.join(self.tmp, "gap_exceptions.json")
        self._orig = ge.EXCEPTIONS_PATH
        ge.EXCEPTIONS_PATH = self.path

    def tearDown(self):
        ge.EXCEPTIONS_PATH = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_record_then_covered(self):
        ge.record("2026-09-30", [{"fund_code": "002224", "last_date": "2026-09-24",
                                  "lag_trading_days": 3, "reason": "停更未清盘",
                                  "evidence": "重拉无效"}], "tester")
        cov = ge.coverage("2026-09-30", [{"fund_code": "002224", "last_date": "2026-09-24",
                                          "lag_trading_days": 3}])
        self.assertEqual(cov["n_gap"], 1)
        self.assertEqual(cov["missing"], [])
        self.assertEqual(cov["covered"][0]["fund_code"], "002224")
        self.assertEqual(cov["approved_by"], "tester")
        self.assertIn("恢复披露", cov["covered"][0]["expires_when"])

    def test_unregistered_is_missing(self):
        cov = ge.coverage("2026-09-30", [{"fund_code": "999999", "last_date": "2026-09-20",
                                          "lag_trading_days": 4}])
        self.assertEqual(len(cov["missing"]), 1)
        self.assertEqual(cov["missing"][0]["why"], "未备案")

    def test_last_date_drift_is_missing(self):
        ge.record("2026-09-30", [{"fund_code": "002224", "last_date": "2026-09-24",
                                  "lag_trading_days": 3, "reason": "停更"}], "tester")
        cov = ge.coverage("2026-09-30", [{"fund_code": "002224", "last_date": "2026-09-28",
                                          "lag_trading_days": 2}])
        self.assertEqual(len(cov["missing"]), 1)
        self.assertIn("不一致", cov["missing"][0]["why"])


class TestMonthEndGapGate(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml",
                                f"_gap_gate_{os.getpid()}_{id(self)}")
        os.makedirs(self.tmp, exist_ok=True)
        self.path = os.path.join(self.tmp, "gap_exceptions.json")
        self._orig = ge.EXCEPTIONS_PATH
        ge.EXCEPTIONS_PATH = self.path

    def tearDown(self):
        ge.EXCEPTIONS_PATH = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _health(self, gap_n=0, gap_funds="absent"):
        h = {"stale_n": 0, "gap_n": gap_n,
             "processed_dist": {"median": "2026-09-30"},
             "benchmark_stale_basis": "trading_days:0"}
        if gap_funds != "absent":
            h["gap_funds"] = gap_funds
        return h

    def test_unregistered_gap_blocks_month_end(self):
        h = self._health(1, [{"fund_code": "999999", "last_date": "2026-09-20",
                              "lag_trading_days": 4}])
        with self.assertRaises(RuntimeError) as cm:
            pp._check_month_end_data_ready(h, pd.Timestamp("2026-09-30"))
        self.assertIn("未备案", str(cm.exception))

    def test_missing_detail_blocks(self):
        h = self._health(3, "absent")            # 只有计数、无明细 → 无法逐只核销
        with self.assertRaises(RuntimeError) as cm:
            pp._check_month_end_data_ready(h, pd.Timestamp("2026-09-30"))
        self.assertIn("gap 明细", str(cm.exception))

    def test_registered_gap_passes_with_evidence(self):
        ge.record("2026-09-30", [{"fund_code": "999999", "last_date": "2026-09-20",
                                  "lag_trading_days": 4, "reason": "停更未清盘"}], "tester")
        h = self._health(1, [{"fund_code": "999999", "last_date": "2026-09-20",
                              "lag_trading_days": 4}])
        res = pp._check_month_end_data_ready(h, pd.Timestamp("2026-09-30"))
        self.assertEqual(res["gap_n"], 1)
        self.assertEqual(res["gap_exceptions"]["n_gap"], 1)
        self.assertEqual(res["gap_exceptions"]["missing"], [])
        self.assertEqual(res["gap_exceptions"]["covered"][0]["approved_by"], "tester")
        self.assertIn("逐只已备案", res["criteria"])

    def test_no_gap_passes_without_exceptions(self):
        res = pp._check_month_end_data_ready(self._health(0), pd.Timestamp("2026-09-30"))
        self.assertIsNone(res["gap_exceptions"])

    def test_stale_still_blocks(self):
        h = self._health(0)
        h["stale_n"] = 2
        with self.assertRaises(RuntimeError):
            pp._check_month_end_data_ready(h, pd.Timestamp("2026-09-30"))

    def test_processed_median_mismatch_blocks(self):
        h = self._health(0)
        h["processed_dist"] = {"median": "2026-09-29"}
        with self.assertRaises(RuntimeError):
            pp._check_month_end_data_ready(h, pd.Timestamp("2026-09-30"))


class TestBenchmarkStalenessTradingDays(unittest.TestCase):
    """基准新鲜度改用交易日口径：长假后不再误判 FAIL（审计第 5 节）。"""

    BENCH = pd.Timestamp("2026-09-30")

    def test_long_holiday_not_stale(self):
        # 10-8 早上（节后首个交易日、披露时点前）：其后已结束交易日数 = 0
        self.assertEqual(benchmark_stale_trading_days(self.BENCH, datetime(2026, 10, 8, 9, 0)), 0)
        # 10-8 21:00（已过披露时点）：当天计 1
        self.assertEqual(benchmark_stale_trading_days(self.BENCH, datetime(2026, 10, 8, 21, 0)), 1)
        # 10-9 早上：10-8 已结束 → 1
        self.assertEqual(benchmark_stale_trading_days(self.BENCH, datetime(2026, 10, 9, 9, 0)), 1)
        # 披露时点前不计当天，故 end = 昨天：
        # 10-12 早上 end=10-11 → 10-8、10-9 = 2
        self.assertEqual(benchmark_stale_trading_days(self.BENCH, datetime(2026, 10, 12, 9, 0)), 2)
        # 10-13 早上 end=10-12 → 3（等于阈值）
        self.assertEqual(benchmark_stale_trading_days(self.BENCH, datetime(2026, 10, 13, 9, 0)),
                         BENCH_MAX_STALE_TRADING_DAYS)
        # 10-14 早上 end=10-13 → 4 > 阈值 → 判定过旧
        self.assertGreater(benchmark_stale_trading_days(self.BENCH, datetime(2026, 10, 14, 9, 0)),
                           BENCH_MAX_STALE_TRADING_DAYS)

    def test_calendar_coverage_bounds(self):
        # 覆盖范围内工作日默认开市（节后无需逐日登记）
        self.assertEqual(benchmark_stale_trading_days(self.BENCH, datetime(2026, 10, 12, 9, 0)), 2)
        # 超出 coverage_end → None（调用方退回自然日口径）
        self.assertIsNone(benchmark_stale_trading_days(self.BENCH, datetime(2027, 3, 1, 9, 0)))

    def test_no_gap_when_bench_is_ahead(self):
        self.assertEqual(benchmark_stale_trading_days(pd.Timestamp("2026-09-30"),
                                                      datetime(2026, 9, 30, 9, 0)), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
