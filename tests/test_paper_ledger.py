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


class _LedgerFixture(unittest.TestCase):
    """共享 fixture（不直接含测试方法，避免子类重复执行父类用例）。"""

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

    def _inputs(self, codes, cohort="2026-09", signal="2026-09-30", cohort_weight=1.0):
        proto = {"initial_nav": 1.0, "cohort_weight": cohort_weight, "hold_months": 6,
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

class TestPaperLedger(_LedgerFixture):
    """基础解析验证（建仓费用/份额、迟发顺延、复权估值、现金计息、不可变、事件一致性）。"""

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
                "capital": 1.0, "initial_nav": 1.0, "cash": 0.5, "diagnostic": True,
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
                                "execution_date": "2026-10-02",
                                "source_run_id": "RID_TEST"}) + "\n")
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


class TestPaperLedgerHardening(_LedgerFixture):
    """2026-10-02 二次复核的四个缺口 + 组合层口径（cohort 净值 ≠ 组合净值）。"""

    def _events(self, run_id, execution_date="2026-09-30", path=None):
        path = path or os.path.join(self.tmp, "events.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"event_id": "e1", "type": "paper", "cohort": "2026-09",
                                "execution_date": execution_date,
                                "source_run_id": run_id}) + "\n")
        return path

    def test_source_run_id_binding(self):
        """P1：执行事件的 source_run_id 必须与封存输入 run_id 一致。"""
        led = self._ledger()
        ev = self._events("OTHER_RUN")                     # 封存输入是 RID_TEST
        with self.assertRaises(ValueError) as cm:
            led.open_position("2026-09-30", capital=1.0, events_path=ev)
        self.assertIn("不一致", str(cm.exception))
        # 一致时通过
        ev2 = self._events("RID_TEST")
        res = led.open_position("2026-09-30", capital=1.0, events_path=ev2)
        self.assertEqual(res["status"], "opened")
        self.assertEqual(res["ledger"]["execution_event"]["source_run_id"], "RID_TEST")

    def test_persist_false_is_pure_computation(self):
        """P2：persist=False 不写任何文件、不要求已确认执行。"""
        led = self._ledger()
        res = led.open_position("2026-09-30", capital=1.0, persist=False)
        self.assertEqual(res["status"], "computed")
        self.assertIsNone(res["path"])
        self.assertEqual(os.listdir(self.out) if os.path.exists(self.out) else [], [])
        self.assertEqual(os.listdir(self.dry) if os.path.exists(self.dry) else [], [])

    def test_value_rejects_nav_date_after_as_of(self):
        """P1：成交净值日晚于估值日 → 拒绝生成完整估值（不再出现 stale_days=-1）。"""
        self._inputs(["A00003"])                            # 执行日 9-30 无净值，10-09 才有
        led = self._ledger()
        payload = led.build_position("2026-09-30", capital=1.0)
        self.assertEqual(payload["positions"][0]["nav_date"], "2026-10-09")
        with self.assertRaises(ValueError) as cm:
            led.value("2026-10-01", ledger=payload)
        self.assertIn("尚未实际发生", str(cm.exception))
        # allow_pending：标记 pending，不计入市值，且 stale_days 不为负
        val = led.value("2026-10-01", ledger=payload, allow_pending=True)
        self.assertEqual(val["status"], "pending")
        self.assertEqual(len(val["pending_funds"]), 1)
        self.assertEqual(val["market_value"], 0.0)
        self.assertIsNone(val["positions"][0]["stale_days"])
        with self.assertRaises(RuntimeError):
            led.save_valuation(val)                         # pending 不得落盘

    def test_valuation_immutable_and_revision(self):
        """P1：同日估值重跑幂等；数据被改动后拒绝覆盖；显式 revision 留痕。

        这里用 build_position 的诊断账本 → 估值只能落 dryrun 目录。
        """
        led = self._ledger()
        payload = led.build_position("2026-09-30", capital=1.0)
        val = led.value("2026-10-02", ledger=payload)
        self.assertTrue(val["diagnostic"])
        r1 = led.save_valuation(val)
        self.assertEqual(r1["status"], "written")
        self.assertEqual(os.path.dirname(r1["path"]), self.dry)      # 诊断不得进正式目录
        r2 = led.save_valuation(led.value("2026-10-02", ledger=payload))
        self.assertEqual(r2["status"], "existing")          # 幂等
        # 改动净值数据（估值区间内）→ 切片指纹变 → 内容不同 → 拒绝
        self._fund("A00001", [("2026-09-30", 1.0, 0.0), ("2026-10-01", 1.2, 0.20),
                              ("2026-10-02", 1.44, 0.20)])
        led2 = self._ledger()
        val2 = led2.value("2026-10-02", ledger=led2.build_position("2026-09-30", capital=1.0))
        self.assertNotEqual(val["data_hashes"]["A00001"]["slice_sha256"],
                            val2["data_hashes"]["A00001"]["slice_sha256"])
        with self.assertRaises(RuntimeError) as cm:
            led2.save_valuation(val2)
        self.assertIn("不可变", str(cm.exception))
        r3 = led2.save_valuation(val2, revision=True, revision_reason="净值数据更正后重算")
        self.assertEqual(r3["status"], "revised")
        baks = [f for f in os.listdir(self.dry) if ".rev_" in f]
        self.assertEqual(len(baks), 1)
        with open(r3["path"], encoding="utf-8") as f:
            self.assertEqual(json.load(f)["revision"]["reason"], "净值数据更正后重算")

    def test_slice_fingerprint_ignores_later_appended_data(self):
        """P2：仅追加估值日**之后**的净值，不应影响估值幂等判定。"""
        led = self._ledger()
        payload = led.build_position("2026-09-30", capital=1.0)
        val1 = led.value("2026-10-02", ledger=payload)
        r1 = led.save_valuation(val1)
        self.assertEqual(r1["status"], "written")
        # 追加 10-05（估值日之后）→ 整文件哈希变、切片不变
        self._fund("A00001", [("2026-09-30", 1.0, 0.0), ("2026-10-01", 1.1, 0.10),
                              ("2026-10-02", 1.21, 0.10), ("2026-10-05", 1.331, 0.10)])
        led2 = self._ledger()
        val2 = led2.value("2026-10-02", ledger=led2.build_position("2026-09-30", capital=1.0))
        self.assertEqual(val1["data_hashes"]["A00001"]["slice_sha256"],
                         val2["data_hashes"]["A00001"]["slice_sha256"])
        self.assertNotEqual(val1["data_hashes"]["A00001"]["nav_file_sha256"],
                            val2["data_hashes"]["A00001"]["nav_file_sha256"])
        self.assertEqual(led2.save_valuation(val2)["status"], "existing")   # 不再被误拒

    def test_diagnostic_result_cannot_write_official_dir(self):
        """P1：诊断/预演产物只能写 dryrun 子树（含正式目录的**子目录**，四次复核补）。"""
        led = self._ledger()
        res = led.open_position("2026-09-30", capital=1.0, persist=False)
        self.assertTrue(res["ledger"]["diagnostic"])                 # 诊断身份
        val = led.value("2026-10-02", ledger=res["ledger"])
        self.assertTrue(val["diagnostic"])
        for bad in (self.out, os.path.join(self.out, "sub")):
            with self.assertRaises(RuntimeError) as cm:
                led.save_valuation(val, out_dir=bad)                 # 正式目录及其子目录 → 拒绝
            self.assertIn("只能写入 dryrun", str(cm.exception))
        r = led.save_valuation(val)                                  # 默认只能落 dryrun
        self.assertEqual(os.path.dirname(r["path"]), self.dry)
        r2 = led.save_valuation(val, out_dir=os.path.join(self.dry, "sub"))
        self.assertIn("sub", r2["path"])                             # dryrun 子目录允许

    def test_simulate_open_rejected_in_official_dir(self):
        """P1（四次复核）：simulate=True 指定正式目录/子目录 → 拒绝，避免污染正式建仓。"""
        led = self._ledger()
        for bad in (self.out, os.path.join(self.out, "sub")):
            with self.assertRaises(RuntimeError) as cm:
                led.open_position("2026-09-30", simulate=True, out_dir=bad)
            self.assertIn("只能写入 dryrun", str(cm.exception))
        self.assertFalse(os.path.exists(os.path.join(self.out, "paper_ledger_2026-09.json")))
        # dryrun 子树允许
        res = led.open_position("2026-09-30", simulate=True,
                                out_dir=os.path.join(self.dry, "sub"))
        self.assertEqual(res["status"], "opened")
        self.assertIn("sub", res["path"])

    def test_official_valuation_requires_execution_fact(self):
        """P1：正式（非诊断）账本估值/落盘必须能对上执行确认事实。"""
        led = self._ledger()
        payload = led.build_position("2026-09-30", capital=1.0)
        payload["diagnostic"] = False                                # 伪装成"正式"账本
        with self.assertRaises(RuntimeError) as cm:
            led.value("2026-10-02", ledger=payload)
        self.assertIn("无执行确认事件", str(cm.exception))
        with self.assertRaises(RuntimeError):
            led.save_valuation({**led.value("2026-10-02", ledger={**payload, "diagnostic": True}),
                                "diagnostic": False})

    def test_event_without_source_run_id_rejected(self):
        """P1：执行事件缺 source_run_id → 直接拒绝建仓（不得放行）。"""
        led = self._ledger()
        ev = os.path.join(self.tmp, "events_nosrc.jsonl")
        with open(ev, "w", encoding="utf-8") as f:
            f.write(json.dumps({"event_id": "e1", "type": "paper", "cohort": "2026-09",
                                "execution_date": "2026-09-30"}) + "\n")
        with self.assertRaises(ValueError) as cm:
            led.open_position("2026-09-30", capital=1.0, events_path=ev)
        self.assertIn("缺少 source_run_id", str(cm.exception))

    def _fake_ledger(self, cohort, exec_date, fund, nav, capital=1 / 6):
        return {"cohort": cohort, "run_id": f"R_{cohort}", "execution_date": exec_date,
                "capital": capital, "initial_nav": 1.0, "cash": 0.0,
                "positions": [{"fund_code": fund, "shares": capital * (1 - 0.0015) / nav,
                               "nav_at_open": nav, "nav_date": exec_date}],
                "simulated": True, "diagnostic": True}

    def test_portfolio_cash_segmented_interest(self):
        """P1：多 cohort 现金必须**逐段**计息（用户样本：相隔 32 天 → 1.000962429）。"""
        led = self._ledger()
        l1 = self._fake_ledger("2026-09", "2026-09-30", "A00002", 2.0)
        l2 = self._fake_ledger("2026-10", "2026-11-01", "A00002", 2.0)   # +32 天
        p1 = os.path.join(self.tmp, "paper_ledger_2026-09.json")
        p2 = os.path.join(self.tmp, "paper_ledger_2026-10.json")
        for p, l in ((p1, l1), (p2, l2)):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(l, f)
        pv = led.portfolio_value("2026-11-01", ledger_paths=[p1, p2], include_simulated=True)
        self.assertEqual(pv["n_cohorts"], 2)
        self.assertAlmostEqual(pv["cohorts"][0]["cohort_nav"], 1 - 0.0015, places=12)
        # 逐段计息解析解：[(5/6)·A − 1/6] + 2/6·(1−0.0015)，A=(1+0.02/365)^32
        a = (1 + 0.02 / 365) ** 32
        expected = ((5 / 6) * a - 1 / 6) + (2 / 6) * (1 - 0.0015)
        self.assertAlmostEqual(pv["portfolio_nav"], expected, places=12)
        self.assertAlmostEqual(pv["portfolio_nav"], 1.000962429, places=9)   # 用户样本
        # 反例：旧的"整段计息"会给 1.000669943，少计约 2.93bp
        naive = (2 / 6) * (1 - 0.0015) + (4 / 6) * a
        self.assertLess(naive, pv["portfolio_nav"])
        self.assertAlmostEqual(naive, 1.000669943, places=9)
        self.assertGreater(pv["portfolio_nav"] - naive, 2.9e-4)
        self.assertEqual(len(pv["cash_segments"]), 3)        # 投入、计息、投入

    def test_data_hashes_recorded_per_fund(self):
        led = self._ledger()
        val = led.value("2026-10-02", ledger=led.build_position("2026-09-30", capital=1.0))
        self.assertEqual(set(val["data_hashes"]), {"A00001", "A00002"})
        fp = val["data_hashes"]["A00001"]
        # 切片指纹（P2）：以实际参与估值的数据为准，整文件哈希仅作来源记录
        self.assertEqual(fp["slice_start"], "2026-09-30")
        self.assertEqual(fp["slice_end"], "2026-10-02")
        self.assertEqual(fp["slice_rows"], 2)          # (9-30, 10-02] → 10-01、10-02
        self.assertEqual(len(fp["slice_sha256"]), 64)
        self.assertEqual(len(fp["nav_file_sha256"]), 64)

    def test_portfolio_value_analytic(self):
        """组合层：建仓日无价格变化时应为 1/6×(1−0.0015) + 5/6 = 0.99975。"""
        self._inputs(["A00001", "A00002"], cohort_weight=1 / 6)
        led = self._ledger()
        res = led.open_position("2026-09-30", simulate=True)     # dryrun 目录
        pv = led.portfolio_value("2026-09-30", ledger_paths=[res["path"]],
                                 include_simulated=True)
        self.assertEqual(pv["n_cohorts"], 1)
        self.assertAlmostEqual(pv["cohorts"][0]["weight"], 1 / 6, places=12)
        self.assertAlmostEqual(pv["cohorts"][0]["cohort_nav"], 1 - 0.0015, places=12)
        self.assertAlmostEqual(pv["cash_weight"], 5 / 6, places=12)
        self.assertAlmostEqual(pv["portfolio_nav"], (1 / 6) * (1 - 0.0015) + 5 / 6, places=12)
        self.assertAlmostEqual(pv["portfolio_nav"], 0.99975, places=12)

    def test_portfolio_value_excludes_unopened_cohort(self):
        self._inputs(["A00001", "A00002"], cohort_weight=1 / 6)
        led = self._ledger()
        res = led.open_position("2026-09-30", simulate=True)
        pv = led.portfolio_value("2026-09-30", ledger_paths=[res["path"]],
                                 include_simulated=True)   # 同日：已建仓 → 计入
        self.assertEqual(pv["n_cohorts"], 1)
        # 模拟"该 cohort 的执行日晚于估值日" → 不计入，全部为现金
        with open(res["path"], encoding="utf-8") as f:
            fake = json.load(f)
        fake["execution_date"] = "2026-12-01"
        fake_path = os.path.join(self.tmp, "paper_ledger_fake.json")
        with open(fake_path, "w", encoding="utf-8") as f:
            json.dump(fake, f)
        pv2 = led.portfolio_value("2026-10-02", ledger_paths=[fake_path],
                                  include_simulated=True)
        self.assertEqual(pv2["n_cohorts"], 0)
        self.assertAlmostEqual(pv2["portfolio_nav"], 1.0, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
