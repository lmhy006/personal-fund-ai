# test_snapshot_integrity.py：快照完整性链 + 运行锁（2026-10-02 审计第 5 节）
# 覆盖：产物清单哈希、manifest/COMPLETE 引用链、verify_snapshot 检出篡改、
#       目录拒绝覆盖、并发锁互斥与释放。
import json
import os
import shutil
import sys
import unittest
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import production_pipeline as pp  # noqa: E402


class TestSnapshotIntegrity(unittest.TestCase):
    RUN_ID = "20260930_233855"

    def setUp(self):
        self.tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml",
                                f"_snap_test_{os.getpid()}_{id(self)}")
        os.makedirs(self.tmp, exist_ok=True)
        self._orig = {k: getattr(pp, k) for k in ("SNAPSHOTS_DIR", "PROJECT_ROOT")}
        pp.SNAPSHOTS_DIR = os.path.join(self.tmp, "snapshots")
        pp.PROJECT_ROOT = self.tmp
        self.scores = os.path.join(self.tmp, "2026-09.csv")
        pd.DataFrame({"fund_code": ["000001", "000002"], "rank": [1, 2]}).to_csv(self.scores,
                                                                                index=False)
        self.ledger = os.path.join(self.tmp, "portfolio_ledger.csv")
        self.state = os.path.join(self.tmp, "portfolio_state.json")
        pd.DataFrame([{"cohort_id": "2026-09", "fund_code": "000001"}]).to_csv(self.ledger,
                                                                              index=False)
        with open(self.state, "w", encoding="utf-8") as f:
            json.dump({"cohorts": {}}, f)
        self.pf = mock.MagicMock()
        self.pf.ledger_path = self.ledger
        self.pf.state_path = self.state
        self.top50 = pd.DataFrame({"fund_code": ["000001", "000002"]})
        self.health = {"status": "PASS", "benchmark_latest_date": "2026-09-30",
                       "raw_latest_date": "2026-09-30", "processed_latest_date": "2026-09-30",
                       "processed_dist": {"median": "2026-09-30"}, "processed_funds": 2}
        # 输入侧留痕所需文件（基准 + processed 索引）
        os.makedirs(os.path.join(self.tmp, "data", "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.tmp, "data", "processed"), exist_ok=True)
        pd.DataFrame({"date": ["2026-09-30"], "close": [4357.616]}) \
            .to_csv(os.path.join(self.tmp, "data", "raw", "benchmark_hs300.csv"), index=False)
        pd.DataFrame({"fund_code": ["000001"], "source": ["alive"]}) \
            .to_csv(os.path.join(self.tmp, "data", "processed", "fund_history_index.csv"),
                    index=False)
        self.manifest = {"run_id": self.RUN_ID, "signal_date": "2026-09-30",
                         "score_generated_at": "2026-09-30T23:44:30", "data_cutoff": "2026-09-30"}

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(pp, k, v)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, mode="complete"):
        return pp.step_snapshot(self.RUN_ID, dict(self.manifest), self.scores, None,
                                self.health, self.pf, {"status": "updated"}, {"n_active": 0},
                                self.top50, mode=mode)

    def test_complete_chain_and_verify(self):
        snap = self._run()
        self.assertTrue(os.path.exists(os.path.join(snap, "files.json")))
        with open(os.path.join(snap, "manifest.json"), encoding="utf-8") as f:
            mf = json.load(f)
        self.assertIn("top50.csv", mf["output_files"])
        self.assertEqual(mf["output_files_sha256"],
                         pp.file_sha256(os.path.join(snap, "files.json")))
        self.assertIn("manifest_sha256=", open(os.path.join(snap, "COMPLETE"), encoding="utf-8").read())
        res = pp.verify_snapshot(snap)
        self.assertTrue(res["ok"], res)
        self.assertGreaterEqual(res["checked"], 4)     # 评分/ledger/state/health/top50
        # 输入侧留痕
        self.assertIn("input_data", mf)
        self.assertIn("benchmark_hs300.csv", mf["input_data"])

    def test_verify_detects_tampering(self):
        snap = self._run()
        with open(os.path.join(snap, "top50.csv"), "a", encoding="utf-8") as f:
            f.write("999999,3\n")
        res = pp.verify_snapshot(snap)
        self.assertFalse(res["ok"])
        self.assertTrue(any(m["file"] == "top50.csv" for m in res["mismatches"]))

    def test_manifest_tampering_detected(self):
        snap = self._run()
        with open(os.path.join(snap, "manifest.json"), encoding="utf-8") as f:
            mf = json.load(f)
        mf["top50"] = ["999999"]
        with open(os.path.join(snap, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(mf, f)
        res = pp.verify_snapshot(snap)
        self.assertFalse(res["ok"])
        self.assertTrue(any(m["file"] == "manifest.json" for m in res["mismatches"]))

    def test_refuses_to_overwrite_non_empty_dir(self):
        self._run()
        with self.assertRaises(RuntimeError) as cm:
            self._run()
        self.assertIn("拒绝覆盖", str(cm.exception))

    def test_test_and_observation_modes_also_listed(self):
        snap = self._run(mode="observation")
        self.assertTrue(os.path.exists(os.path.join(snap, "NOT_COMPLETE_OBSERVATION")))
        self.assertTrue(os.path.exists(os.path.join(snap, "files.json")))
        self.assertTrue(pp.verify_snapshot(snap)["ok"])


class TestRunLock(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml",
                                f"_lock_test_{os.getpid()}_{id(self)}")
        os.makedirs(self.tmp, exist_ok=True)
        self._orig = pp.SNAPSHOTS_DIR
        pp.SNAPSHOTS_DIR = os.path.join(self.tmp, "snapshots")

    def tearDown(self):
        pp.SNAPSHOTS_DIR = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_acquire_conflict_and_release(self):
        lock = pp._acquire_run_lock()
        self.assertTrue(os.path.exists(lock))
        with self.assertRaises(RuntimeError) as cm:
            pp._acquire_run_lock()
        self.assertIn("已有生产运行在进行", str(cm.exception))
        pp._release_run_lock(lock)
        self.assertFalse(os.path.exists(lock))
        lock2 = pp._acquire_run_lock()          # 释放后可重新获取
        pp._release_run_lock(lock2)

    def test_run_pipeline_releases_lock_on_failure(self):
        with mock.patch.object(pp, "_run_pipeline_inner", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                pp.run_pipeline()
        self.assertFalse(os.path.exists(os.path.join(pp.SNAPSHOTS_DIR, ".run.lock")))

    def test_test_run_does_not_take_lock(self):
        with mock.patch.object(pp, "_run_pipeline_inner", return_value={"status": "test"}):
            pp.run_pipeline(skip_refresh=True)
        self.assertFalse(os.path.exists(os.path.join(pp.SNAPSHOTS_DIR, ".run.lock")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
