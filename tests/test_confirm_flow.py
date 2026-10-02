# test_confirm_flow.py：首次 paper 确认流程测试（2026-10-02 审计 P0/P1 修复验证）
# 覆盖：P0-1 CSV 重载 dtype 往返、P0-2 诊断/no-save 不落盘、P1-1 新建 cohort 恒 planned、
#       P1-2 交易日历与来源快照完整性、P1-3 事件流最终事实/回滚/一致性。
# 全部用临时目录与 mock 隔离，不触碰真实 ledger/snapshots/scores。
import hashlib
import json
import os
import shutil
import sys
import unittest
from datetime import datetime as _real_datetime
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import live_portfolio as lp  # noqa: E402


def _sha(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _FrozenDatetime(_real_datetime):
    """冻结时钟（模拟节后 10-12 确认：10-8/10-9 已过去、10-10 是周末、10-12 未登记）。"""
    frozen = _real_datetime(2026, 10, 12, 10, 0, 0)

    @classmethod
    def now(cls, tz=None):
        return cls.frozen


class ConfirmFlowBase(unittest.TestCase):
    """通用 fixture：临时 ledger/state/events + 完整来源快照（9-30 月末信号）。"""

    SIGNAL_DATE = "2026-09-30"
    RUN_ID = "20260930_233855"
    CODES = ["000001", "000002", "000003"]

    def setUp(self):
        self.tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml",
                                f"_cf_test_{os.getpid()}_{id(self)}")
        os.makedirs(self.tmp, exist_ok=True)
        self._orig = {k: getattr(lp, k) for k in ("LEDGER_DIR", "LEDGER_PATH",
                                                  "STATE_PATH", "EXECUTION_EVENTS_PATH",
                                                  "PROJECT_ROOT")}
        lp.LEDGER_DIR = self.tmp
        lp.LEDGER_PATH = os.path.join(self.tmp, "portfolio_ledger.csv")
        lp.STATE_PATH = os.path.join(self.tmp, "portfolio_state.json")
        lp.EXECUTION_EVENTS_PATH = os.path.join(self.tmp, "execution_events.jsonl")
        lp.PROJECT_ROOT = self.tmp

        snap = os.path.join(self.tmp, "ml", "snapshots", self.RUN_ID)
        os.makedirs(snap, exist_ok=True)
        with open(os.path.join(snap, "COMPLETE"), "w", encoding="utf-8") as f:
            f.write(self.RUN_ID + "\n")
        score_csv = os.path.join(snap, "2026-09.csv")
        pd.DataFrame({"fund_code": self.CODES, "rank": [1, 2, 3]}).to_csv(score_csv, index=False)
        self._write_manifest(score_file_sha256=_sha(score_csv))

        self.pf = lp.LivePortfolio(ledger_path=lp.LEDGER_PATH, state_path=lp.STATE_PATH)
        self._write_cohort()
        self.pf.save()

    def _write_manifest(self, **over):
        snap = os.path.join(self.tmp, "ml", "snapshots", self.RUN_ID)
        os.makedirs(snap, exist_ok=True)
        mf = {"run_id": self.RUN_ID, "status": "complete", "signal_mode": "month_end",
              "signal_date": self.SIGNAL_DATE, "top50": list(self.CODES),
              "score_generated_at": "2026-09-30T23:44:30"}
        mf.update(over)
        with open(os.path.join(snap, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(mf, f, ensure_ascii=False)

    def _write_cohort(self, codes=None):
        codes = codes or self.CODES
        self.pf.state["cohorts"]["2026-09"] = {
            "signal_date": self.SIGNAL_DATE, "execution_date": None, "expire_month": "2027-03",
            "cohort_weight": 1 / 6, "n_codes": len(codes), "status": "planned",
            "score_run": self.RUN_ID,
        }
        self.pf.ledger = pd.DataFrame([{
            "cohort_id": "2026-09", "fund_code": c, "weight_in_cohort": (1 / 6) / len(codes),
            "signal_date": self.SIGNAL_DATE, "execution_date": None,
            "created_at": "2026-09-30T23:44:30", "expire_month": "2027-03",
            "status": "planned"} for c in codes])

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


class TestP01CsvRoundTrip(ConfirmFlowBase):
    """P0-1：真实 CSV 重载后首次确认必须成功（空 execution_date 列不得被推断为 float64）。"""

    def test_reload_dtype_not_float(self):
        # 审计事实：pandas 原生读取会把全空 execution_date 推断为 float64（P0-1 根因）；
        # 本模块用 _load_ledger 显式 nullable string，写入日期字符串不再报 TypeError。
        raw = pd.read_csv(lp.LEDGER_PATH, dtype={"fund_code": str})
        self.assertEqual(str(raw["execution_date"].dtype), "float64")
        pf = lp.LivePortfolio(ledger_path=lp.LEDGER_PATH, state_path=lp.STATE_PATH)
        self.assertTrue(str(pf.ledger["execution_date"].dtype).startswith("string"))

    def test_first_confirm_after_real_csv_reload(self):
        pf2 = lp.LivePortfolio(ledger_path=lp.LEDGER_PATH, state_path=lp.STATE_PATH)
        self.assertTrue(str(pf2.ledger["execution_date"].dtype).startswith("string"))
        with mock.patch.object(lp, "datetime", _FrozenDatetime):
            ev = pf2.confirm_execution("2026-09", "2026-10-08", exec_type="paper",
                                       operator="tester")
        self.assertEqual(ev["execution_date"], "2026-10-08")
        # 模拟成交日（10-8）与登记时刻（冻结的 10-12）分开记录，延迟显式留痕
        self.assertEqual(ev["execution_lag_days"], 4)
        self.assertIn("登记晚于所声称的模拟成交日", ev["note"])
        # 重载后仍 active、CSV 里能读回日期字符串
        pf3 = lp.LivePortfolio(ledger_path=lp.LEDGER_PATH, state_path=lp.STATE_PATH)
        self.assertEqual(pf3.state["cohorts"]["2026-09"]["status"], "active")
        self.assertEqual(set(pf3.ledger["execution_date"]), {"2026-10-08"})
        self.assertEqual(len(self._events()), 1)
        # 相同重试幂等（不追加事件）
        with mock.patch.object(lp, "datetime", _FrozenDatetime):
            ev2 = pf3.confirm_execution("2026-09", "2026-10-08", exec_type="paper")
        self.assertTrue(ev2.get("idempotent"))
        self.assertEqual(ev2["event_id"], ev["event_id"])
        self.assertEqual(len(self._events()), 1)

    def test_injected_failure_keeps_memory_and_disk_consistent(self):
        pf2 = lp.LivePortfolio(ledger_path=lp.LEDGER_PATH, state_path=lp.STATE_PATH)
        with mock.patch.object(pf2, "_append_event", side_effect=OSError("disk full")):
            with mock.patch.object(lp, "datetime", _FrozenDatetime):
                with self.assertRaises(RuntimeError):
                    pf2.confirm_execution("2026-09", "2026-10-08")
        pf3 = lp.LivePortfolio(ledger_path=lp.LEDGER_PATH, state_path=lp.STATE_PATH)
        self.assertEqual(pf3.state["cohorts"]["2026-09"]["status"], "planned")
        self.assertIsNone(pf3.state["cohorts"]["2026-09"].get("execution_date"))
        self.assertEqual(pf3.ledger["status"].tolist(), ["planned"] * len(self.CODES))
        self.assertEqual(self._events(), [])


class TestP11AlwaysPlanned(ConfirmFlowBase):
    """P1-1：新建 cohort 必须为 planned（即使日历已有信号日之后的交易日）。"""

    def test_new_cohort_stays_planned(self):
        scores = pd.DataFrame({
            "as_of": ["2026-08-31"] * 60,
            "fund_code": [f"1{i:05d}" for i in range(60)],
            "confidence": ["main"] * 60,
            "rank": list(range(1, 61)),
        })
        action = self.pf.add_month(scores, score_run="RUN_X")
        meta = self.pf.state["cohorts"]["2026-08"]
        self.assertEqual(meta["status"], "planned")          # 旧实现：日历有 9-1 → active
        self.assertEqual(action["cohort_status"], "planned")
        self.assertEqual(action["buy_weight_risk"], 0.0)     # planned 不投入风险权重
        self.assertIsNotNone(meta["execution_date"])         # 预计执行日仍记录
        self.assertTrue(meta["execution_date_is_estimate"])
        self.assertEqual(set(self.pf.ledger[self.pf.ledger["cohort_id"] == "2026-08"]["status"]),
                         {"planned"})
        self.assertEqual(self.pf.aggregate()["n_active"], 0)


class TestP02NoPersist(ConfirmFlowBase):
    """P0-2：诊断/--no-save 路径不得修改任何正式文件。"""

    def test_add_month_persist_false_touches_nothing(self):
        before = {p: _sha(p) for p in (lp.LEDGER_PATH, lp.STATE_PATH)}
        scores = pd.DataFrame({
            "as_of": ["2026-09-30"] * 60,
            "fund_code": [f"9{i:05d}" for i in range(60)],
            "confidence": ["main"] * 60,
            "rank": list(range(1, 61)),
        })
        action = self.pf.add_month(scores, score_run="TEST_RUN_NO_SAVE", persist=False)
        self.assertEqual(action["status"], "updated")        # 内存中确实算了
        after = {p: _sha(p) for p in (lp.LEDGER_PATH, lp.STATE_PATH)}
        self.assertEqual(before, after)                      # 但一个字节都没写
        # 内存也回到原样（不留诊断痕迹）
        self.assertEqual(self.pf.state["cohorts"]["2026-09"]["score_run"], self.RUN_ID)
        self.assertEqual(len(self.pf.ledger), len(self.CODES))

    def test_save_backup_unique_names(self):
        self.pf.save()
        self.pf.save()
        bak = os.path.join(self.tmp, ".bak")
        names = sorted(f for f in os.listdir(bak) if f.startswith("ledger_"))
        self.assertGreaterEqual(len(names), 2)               # 微秒时间戳：同秒多次 save 不覆盖
        self.assertEqual(len(names), len(set(names)))


class TestP12TradingDayAndSource(ConfirmFlowBase):
    """P1-2：交易日历 + 来源快照完整性校验。"""

    def test_holiday_and_weekend_rejected(self):
        with mock.patch.object(lp, "datetime", _FrozenDatetime):
            for bad in ("2026-10-01", "2026-10-10", "2026-10-02"):
                with self.assertRaises(ValueError) as cm:
                    self.pf.confirm_execution("2026-09", bad)
                self.assertIn("交易日", str(cm.exception))
        self.assertEqual(self.pf.state["cohorts"]["2026-09"]["status"], "planned")
        self.assertEqual(self._events(), [])

    def test_valid_trading_day_accepted(self):
        with mock.patch.object(lp, "datetime", _FrozenDatetime):
            ev = self.pf.confirm_execution("2026-09", "2026-10-08")
        self.assertEqual(ev["execution_date"], "2026-10-08")
        self.assertIn("calendar", ev["validation"])
        self.assertIn("override", ev["validation"]["calendar"]["source"])
        self.assertEqual(ev["validation"]["source_snapshot"]["snapshot"], self.RUN_ID)

    def test_source_manifest_drift_rejected(self):
        # 1) manifest status 非 complete
        self._write_manifest(status="aborted")
        with mock.patch.object(lp, "datetime", _FrozenDatetime):
            with self.assertRaises(ValueError):
                self.pf.confirm_execution("2026-09", "2026-10-08")
        # 2) signal_mode 非 month_end（月中观察不算正式来源）
        self._write_manifest(signal_mode="observation")
        with mock.patch.object(lp, "datetime", _FrozenDatetime):
            with self.assertRaises(ValueError):
                self.pf.confirm_execution("2026-09", "2026-10-08")
        # 3) signal_date 与 cohort 不一致
        self._write_manifest(signal_date="2026-09-29")
        with mock.patch.object(lp, "datetime", _FrozenDatetime):
            with self.assertRaises(ValueError):
                self.pf.confirm_execution("2026-09", "2026-10-08")
        # 4) 评分哈希不符
        self._write_manifest(score_file_sha256="deadbeef")
        with mock.patch.object(lp, "datetime", _FrozenDatetime):
            with self.assertRaises(ValueError):
                self.pf.confirm_execution("2026-09", "2026-10-08")

    def test_ledger_drift_rejected(self):
        # 台账里把一个代码改成 999999（来源 Top50 漂移）→ 拒绝
        self.pf.ledger.loc[self.pf.ledger.index[0], "fund_code"] = "999999"
        self.pf.save()
        with mock.patch.object(lp, "datetime", _FrozenDatetime):
            with self.assertRaises(ValueError) as cm:
                self.pf.confirm_execution("2026-09", "2026-10-08")
        self.assertIn("Top50", str(cm.exception))

    def test_weight_drift_rejected(self):
        self.pf.ledger.loc[self.pf.ledger.index[0], "weight_in_cohort"] = 0.5
        self.pf.save()
        with mock.patch.object(lp, "datetime", _FrozenDatetime):
            with self.assertRaises(ValueError) as cm:
                self.pf.confirm_execution("2026-09", "2026-10-08")
        self.assertIn("权重", str(cm.exception))


class TestP13EventConsistency(ConfirmFlowBase):
    """P1-3：事件流折叠最终事实 + 回滚 + recover 一致性。"""

    def test_correction_participates_in_idempotency(self):
        with mock.patch.object(lp, "datetime", _FrozenDatetime):
            self.pf.confirm_execution("2026-09", "2026-10-08")
            corr = self.pf.correct_execution("2026-09", "2026-10-09", "paper",
                                             reason="实际纸面成交晚一日")
            self.assertEqual(corr["execution_date"], "2026-10-09")
            # 按**最终事实**重试确认 → 幂等返回，不追加事件
            ev = self.pf.confirm_execution("2026-09", "2026-10-09", "paper")
            self.assertTrue(ev.get("idempotent"))
            self.assertEqual(ev["event_id"], corr["event_id"])
            self.assertEqual(len(self._events()), 2)
            # 旧日期（已被修正）再确认 → 拒绝
            with self.assertRaises(ValueError):
                self.pf.confirm_execution("2026-09", "2026-10-08", "paper")
            # 相同修正重试 → 幂等，不追加
            same = self.pf.correct_execution("2026-09", "2026-10-09", "paper", reason="重复修正")
            self.assertTrue(same.get("idempotent"))
            self.assertEqual(len(self._events()), 2)

    def test_correction_append_failure_rolls_back(self):
        with mock.patch.object(lp, "datetime", _FrozenDatetime):
            self.pf.confirm_execution("2026-09", "2026-10-08")
            with mock.patch.object(self.pf, "_append_event", side_effect=OSError("disk full")):
                with self.assertRaises(RuntimeError):
                    self.pf.correct_execution("2026-09", "2026-10-09", "paper", reason="故障注入")
        pf2 = lp.LivePortfolio(ledger_path=lp.LEDGER_PATH, state_path=lp.STATE_PATH)
        self.assertEqual(pf2.state["cohorts"]["2026-09"]["execution_date"], "2026-10-08")  # 未留 10-09
        self.assertEqual(len(self._events()), 1)

    def test_recover_detects_state_event_conflict(self):
        with mock.patch.object(lp, "datetime", _FrozenDatetime):
            self.pf.confirm_execution("2026-09", "2026-10-08")
        # 人为把状态改成 10-09（事件仍是 10-08）→ 矛盾必须被发现，而不是"事件存在即无需恢复"
        self.pf.state["cohorts"]["2026-09"]["execution_date"] = "2026-10-09"
        self.pf.save()
        with self.assertRaises(ValueError) as cm:
            self.pf.recover_execution("2026-09")
        self.assertIn("矛盾", str(cm.exception))

    def test_recover_planned_with_fact_rejected(self):
        with mock.patch.object(lp, "datetime", _FrozenDatetime):
            self.pf.confirm_execution("2026-09", "2026-10-08")
        self.pf.state["cohorts"]["2026-09"]["status"] = "planned"
        self.pf.save()
        with self.assertRaises(ValueError):
            self.pf.recover_execution("2026-09")


class TestOfficialFilesUntouched(unittest.TestCase):
    """P0-2 守卫：诊断路径不得改动真实正式产物（只读哈希断言）。"""

    def test_official_ledger_and_scores_unchanged_by_pure_compute(self):
        ledger, state = lp.LEDGER_PATH, lp.STATE_PATH
        scores = os.path.join(lp.PROJECT_ROOT, "ml", "scores", "2026-09.csv")
        if not (os.path.exists(ledger) and os.path.exists(scores)):
            self.skipTest("正式 ledger/scores 不存在（此守卫只在正式运行环境执行）")
        before = {p: _sha(p) for p in (ledger, state, scores)}
        pf = lp.LivePortfolio()                     # 真实路径（只读）
        df = pd.read_csv(scores, dtype={"fund_code": str})
        pf.add_month(df, score_run="GUARD_TEST_RUN", persist=False)   # 纯计算
        after = {p: _sha(p) for p in (ledger, state, scores)}
        self.assertEqual(before, after)


class TestPipelineTestModeIsolation(unittest.TestCase):
    """P0-2：流水线诊断模式写隔离目录、不推进正式 cohort（production_pipeline 层）。"""

    def setUp(self):
        import production_pipeline as pp
        self.pp = pp
        self.tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml",
                                f"_cf_pipe_{os.getpid()}_{id(self)}")
        os.makedirs(self.tmp, exist_ok=True)
        self._orig = {k: getattr(pp, k) for k in ("PROJECT_ROOT", "SCORES_DIR", "SNAPSHOTS_DIR")}
        pp.PROJECT_ROOT = self.tmp
        pp.SCORES_DIR = os.path.join(self.tmp, "ml", "scores")
        pp.SNAPSHOTS_DIR = os.path.join(self.tmp, "ml", "snapshots")

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(self.pp, k, v)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_step_live_score_persist_false_writes_isolated_dir(self):
        df = pd.DataFrame({"fund_code": ["000001"], "confidence": ["main"], "rank": [1],
                           "score": [1.0], "as_of": ["2026-09-30"]})
        with mock.patch("live_score.score_cross_section",
                        return_value=(df, {}, pd.Timestamp("2026-09-30"))), \
                mock.patch("live_score.verify_against_panel", side_effect=RuntimeError("panel 用时跳过")):
            _df, _t, path, _meta = self.pp.step_live_score(None, persist=False, run_id="RID1")
        self.assertIn("scores_test", path)
        self.assertTrue(os.path.exists(path))
        # 正式评分目录**没有**被写入
        self.assertFalse(os.path.exists(os.path.join(self.pp.SCORES_DIR, "2026-09.csv")))

    def test_step_portfolio_test_mode_passes_persist_false(self):
        fake = mock.MagicMock()
        fake.aggregate.return_value = {"n_active": 0, "active_cohorts": [], "n_funds": 0,
                                       "target_weights": {}, "cash_weight": 1.0, "sum_weights": 1.0}
        fake.add_month.return_value = {"status": "updated", "month": "2026-09"}
        with mock.patch("live_portfolio.LivePortfolio", return_value=fake):
            _pf, action, _agg = self.pp.step_portfolio(
                pd.DataFrame({"as_of": ["2026-09-30"]}), pd.Timestamp("2026-09-30"),
                "RID2", month_end=True, persist=False)
        self.assertIs(fake.add_month.call_args.kwargs.get("persist"), False)
        self.assertFalse(action["applied"])
        self.assertEqual(action["mode"], "test_no_persist")


if __name__ == "__main__":
    unittest.main(verbosity=2)
