# test_pipeline_isolation.py：生产隔离与失败恢复回归测试（2026-10-02 用户复核的三个残留缺口）
#   ① 测试模式影子评分隔离；② 重建/非前向来源被查询与确认入口拒绝；
#   ③ 快照失败后台账回滚到上一完整版本。
# 全部用临时目录与 mock 隔离，不触碰真实 ledger/scores/snapshots。
import json
import os
import shutil
import sys
import unittest
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import agent_tools as at  # noqa: E402
import live_portfolio as lp  # noqa: E402
import production_pipeline as pp  # noqa: E402
from datetime import datetime as _real_datetime  # noqa: E402


class _FrozenDatetime(_real_datetime):
    """冻结时钟（模拟节后 10-12 确认：10-8 已过去、10-10 周末、10-12 未登记）。"""
    frozen = _real_datetime(2026, 10, 12, 10, 0, 0)

    @classmethod
    def now(cls, tz=None):
        return cls.frozen


def _read_bytes(p):
    with open(p, "rb") as f:
        return f.read()


def _tmpdir(name):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml",
                     f"_iso_{name}_{os.getpid()}_{id(object())}")
    os.makedirs(p, exist_ok=True)
    return p


class TestShadowScoreIsolation(unittest.TestCase):
    """缺口 ①：因子 fresh 的诊断运行不得覆盖正式 ml/scores/shadow_*.csv。"""

    def setUp(self):
        self.tmp = _tmpdir("shadow")
        self._orig = {k: getattr(pp, k) for k in ("PROJECT_ROOT", "SCORES_DIR")}
        pp.PROJECT_ROOT = self.tmp
        pp.SCORES_DIR = os.path.join(self.tmp, "ml", "scores")

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(pp, k, v)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, persist):
        sh = pd.DataFrame({"fund_code": ["000001", "000002"], "score": [1.0, 0.9]})
        with mock.patch("shadow_score.shadow_scores",
                        return_value=(sh, sh, pd.Timestamp("2026-09-30"))):
            return pp.step_shadow_score({"shadow_stale": []}, pd.Timestamp("2026-09-30"),
                                        persist=persist, run_id="RID9")

    def test_persist_false_uses_isolated_dir(self):
        path, stale = self._run(persist=False)
        self.assertIsNone(stale)
        self.assertIn("scores_test", path)
        self.assertTrue(os.path.exists(path))
        # 正式影子文件**没有**被写入
        self.assertFalse(os.path.exists(os.path.join(pp.SCORES_DIR, "shadow_2026-09.csv")))

    def test_persist_true_uses_official_dir(self):
        path, _ = self._run(persist=True)
        self.assertEqual(os.path.dirname(path), pp.SCORES_DIR)
        self.assertTrue(os.path.exists(os.path.join(pp.SCORES_DIR, "shadow_2026-09.csv")))

    def test_stale_still_skips(self):
        path, stale = pp.step_shadow_score({"shadow_stale": ["sw_industry(最旧 2026-09-29)"]},
                                           pd.Timestamp("2026-09-30"), persist=False, run_id="RID9")
        self.assertIsNone(path)
        self.assertTrue(stale)


class TestReconstructionRejected(unittest.TestCase):
    """缺口 ②：重建/非前向快照不得被查询列为正式快照，也不得用于确认。"""

    RUNS = [("20260930_233855", {}),
            ("20261001_000000", {"is_reconstruction": True, "forward_eligible": False}),
            ("20261002_000000", {"forward_eligible": False}),
            ("20261003_000000", {"run_mode": "test"}),
            ("20261004_000000", {"run_mode": "observation"})]

    def setUp(self):
        self.tmp = _tmpdir("recon")
        self.snaps = os.path.join(self.tmp, "snapshots")
        for rid, extra in self.RUNS:
            d = os.path.join(self.snaps, rid)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "COMPLETE"), "w", encoding="utf-8") as f:
                f.write(rid + "\n")
            mf = {"run_id": rid, "status": "complete", "signal_date": "2026-09-30",
                  "score_generated_at": "2026-09-30T23:44:30"}
            mf.update(extra)
            with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(mf, f)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_query_layer_filters_reconstruction(self):
        snaps = at._complete_snapshots(self.snaps)
        self.assertEqual([s["run_id"] for s in snaps], ["20260930_233855"])

    def test_confirm_layer_rejects_reconstruction_source(self):
        # 直接构造 LivePortfolio fixture：cohort 指向一个"重建"快照 → 确认必须被拒
        tmp = _tmpdir("recon_lp")
        orig = {k: getattr(lp, k) for k in ("LEDGER_DIR", "LEDGER_PATH", "STATE_PATH",
                                            "EXECUTION_EVENTS_PATH", "PROJECT_ROOT")}
        try:
            lp.LEDGER_DIR = tmp
            lp.LEDGER_PATH = os.path.join(tmp, "ledger.csv")
            lp.STATE_PATH = os.path.join(tmp, "state.json")
            lp.EXECUTION_EVENTS_PATH = os.path.join(tmp, "events.jsonl")
            lp.PROJECT_ROOT = tmp
            src = "20261001_000000"
            d = os.path.join(tmp, "ml", "snapshots", src)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "COMPLETE"), "w", encoding="utf-8") as f:
                f.write(src + "\n")
            pd.DataFrame({"fund_code": ["000001", "000002"], "rank": [1, 2]}) \
                .to_csv(os.path.join(d, "2026-09.csv"), index=False)
            with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump({"run_id": src, "status": "complete", "signal_mode": "month_end",
                           "signal_date": "2026-09-30", "top50": ["000001", "000002"],
                           "score_file_sha256": lp._file_sha256(os.path.join(d, "2026-09.csv")),
                           "score_generated_at": "2026-09-30T23:44:30",
                           "is_reconstruction": True, "forward_eligible": False}, f)
            pf = lp.LivePortfolio(ledger_path=lp.LEDGER_PATH, state_path=lp.STATE_PATH)
            pf.state["cohorts"]["2026-09"] = {
                "signal_date": "2026-09-30", "execution_date": None, "expire_month": "2027-03",
                "cohort_weight": 1 / 6, "n_codes": 2, "status": "planned", "score_run": src}
            pf.ledger = pd.DataFrame([{
                "cohort_id": "2026-09", "fund_code": c, "weight_in_cohort": 1 / 12,
                "signal_date": "2026-09-30", "execution_date": None,
                "created_at": "2026-09-30T23:44:30", "expire_month": "2027-03",
                "status": "planned"} for c in ("000001", "000002")])
            pf.save()
            with mock.patch.object(lp, "datetime", _FrozenDatetime):
                with self.assertRaises(ValueError) as cm:
                    pf.confirm_execution("2026-09", "2026-10-08")
            self.assertIn("事后重建", str(cm.exception))
        finally:
            for k, v in orig.items():
                setattr(lp, k, v)
            shutil.rmtree(tmp, ignore_errors=True)


class TestSnapshotFailureRollback(unittest.TestCase):
    """缺口 ③：快照写入失败 → 正式台账回到运行前完整版本，且标记未完成来源。"""

    def setUp(self):
        self.tmp = _tmpdir("rollback")
        self.led = os.path.join(self.tmp, "ml", "ledger", "portfolio_ledger.csv")
        self.st = os.path.join(self.tmp, "ml", "ledger", "portfolio_state.json")
        os.makedirs(os.path.dirname(self.led), exist_ok=True)
        pd.DataFrame([{
            "cohort_id": "2026-09", "fund_code": "000001", "weight_in_cohort": 1 / 6,
            "signal_date": "2026-09-30", "execution_date": None,
            "created_at": "2026-09-30T23:44:30", "expire_month": "2027-03",
            "status": "planned"}]).to_csv(self.led, index=False, encoding="utf-8-sig")
        with open(self.st, "w", encoding="utf-8") as f:
            json.dump({"cohorts": {"2026-09": {
                "signal_date": "2026-09-30", "execution_date": None, "expire_month": "2027-03",
                "cohort_weight": 1 / 6, "n_codes": 1, "status": "planned",
                "score_run": "20260930_233855"}}, "last_month": "2026-09"}, f)

        self._orig_lp = {k: getattr(lp, k) for k in ("LEDGER_PATH", "STATE_PATH", "LEDGER_DIR")}
        self._orig_pp = {k: getattr(pp, k) for k in ("PROJECT_ROOT", "SCORES_DIR",
                                                     "SNAPSHOTS_DIR", "LOG_DIR")}
        lp.LEDGER_PATH, lp.STATE_PATH = self.led, self.st
        lp.LEDGER_DIR = os.path.dirname(self.led)
        pp.PROJECT_ROOT = self.tmp
        pp.SCORES_DIR = os.path.join(self.tmp, "ml", "scores")
        pp.SNAPSHOTS_DIR = os.path.join(self.tmp, "ml", "snapshots")
        pp.LOG_DIR = os.path.join(self.tmp, "logs")
        os.makedirs(pp.SCORES_DIR, exist_ok=True)

    def tearDown(self):
        for k, v in self._orig_lp.items():
            setattr(lp, k, v)
        for k, v in self._orig_pp.items():
            setattr(pp, k, v)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _health(self):
        return {"status": "PASS", "benchmark_latest_date": "2026-09-30",
                "raw_latest_date": "2026-09-30", "processed_latest_date": "2026-09-30",
                "processed_dist": {"median": "2026-09-30", "n_funds": 10},
                "stale_n": 0, "gap_n": 0, "shadow_factors": {}, "shadow_stale": [],
                "processed_funds": 10}

    def _scores(self):
        return pd.DataFrame({
            "as_of": ["2026-09-30"] * 60,
            "fund_code": [f"1{i:05d}" for i in range(60)],
            "confidence": ["main"] * 60,
            "rank": list(range(1, 61)),
            "score": [1.0 - i / 100 for i in range(60)]})

    def test_ledger_restored_when_snapshot_fails(self):
        before_led = _read_bytes(self.led)
        before_st = _read_bytes(self.st)
        scores = self._scores()
        scores_path = os.path.join(pp.SCORES_DIR, "2026-09.csv")
        scores.to_csv(scores_path, index=False)

        with mock.patch.object(pp, "step_benchmark"), \
                mock.patch.object(pp, "step_raw"), \
                mock.patch.object(pp, "step_clean"), \
                mock.patch.object(pp, "step_shadow_factors", return_value={"ok": 0}), \
                mock.patch.object(pp, "step_health", return_value=self._health()), \
                mock.patch.object(pp, "step_live_score",
                                  return_value=(scores, pd.Timestamp("2026-09-30"), scores_path,
                                                {"main": 60, "low": 0, "skip": {}})), \
                mock.patch.object(pp, "step_shadow_score", return_value=(None, None)), \
                mock.patch.object(pp, "step_snapshot", side_effect=RuntimeError("disk full")):
            with self.assertRaises(RuntimeError):
                pp.run_pipeline()

        # 台账已推进（score_run 变了）→ 但必须被回滚到运行前字节
        after_led = _read_bytes(self.led)
        after_st = _read_bytes(self.st)
        self.assertEqual(before_led, after_led, "快照失败后台账未回滚")
        self.assertEqual(before_st, after_st, "快照失败后 state 未回滚")
        self.assertIn("20260930_233855", after_st.decode("utf-8"))
        self.assertNotIn("2026-09-30T", json.loads(after_st)["cohorts"]["2026-09"].get("updated_at", "")
                         or "zzz")

    def test_aborted_snapshot_marked_not_complete(self):
        run_id = "20261008_000000"
        snap = os.path.join(pp.SNAPSHOTS_DIR, run_id)
        os.makedirs(snap, exist_ok=True)
        pp.mark_aborted_snapshot(run_id, "注入失败")
        self.assertTrue(os.path.exists(os.path.join(snap, "NOT_COMPLETE_ABORTED")))
        self.assertFalse(os.path.exists(os.path.join(snap, "COMPLETE")))
        # 未完成的目录不会被查询层列为正式快照
        self.assertEqual([s["run_id"] for s in at._complete_snapshots(pp.SNAPSHOTS_DIR)], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
