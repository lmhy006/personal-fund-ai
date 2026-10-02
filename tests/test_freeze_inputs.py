# test_freeze_inputs.py：首次记账输入封存测试（不可变、来源过滤、哈希一致）
# 全部在临时目录内构造 fixture，不触碰真实 ml/paper 与 ml/snapshots。
import hashlib
import json
import os
import shutil
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import freeze_paper_inputs as fp  # noqa: E402


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TestFreezeInputs(unittest.TestCase):
    RUN_ID = "20260930_233855"
    SIGNAL = "2026-09-30"

    def setUp(self):
        self.tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml",
                                f"_fpi_test_{os.getpid()}_{id(self)}")
        self.snaps = os.path.join(self.tmp, "snapshots")
        self.out = os.path.join(self.tmp, "paper")
        os.makedirs(self.snaps, exist_ok=True)
        self.codes = [f"1{i:05d}" for i in range(50)]
        self._mk_snapshot(self.RUN_ID, self.SIGNAL, self.codes)

    def _mk_snapshot(self, run_id, signal_date, codes, extra=None):
        d = os.path.join(self.snaps, run_id)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "COMPLETE"), "w", encoding="utf-8") as f:
            f.write(run_id + "\n")
        rows = [{"fund_code": c, "rank": i + 1, "score": 1.0 - i * 0.01,
                 "as_of": signal_date, "confidence": "main"} for i, c in enumerate(codes)]
        rows.append({"fund_code": "900001", "rank": None, "score": 0.01,
                     "as_of": signal_date, "confidence": "low"})
        pd.DataFrame(rows).to_csv(os.path.join(d, f"{signal_date[:7]}.csv"), index=False)
        mf = {"run_id": run_id, "status": "complete", "signal_mode": "month_end",
              "signal_date": signal_date, "top50": list(codes), "data_cutoff": signal_date,
              "score_generated_at": signal_date + "T23:44:30", "stale_n": 0, "gap_n": 7}
        mf.update(extra or {})
        with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(mf, f)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_build_inputs_fields(self):
        p = fp.build_inputs(self.RUN_ID, self.snaps, self.out, write_main=False)
        self.assertEqual(p["run_id"], self.RUN_ID)
        self.assertEqual(p["signal_date"], self.SIGNAL)
        self.assertEqual(p["cohort_id"], "2026-09")
        self.assertEqual(p["n_codes"], 50)
        self.assertAlmostEqual(p["weight_per_fund"], (1 / 6) / 50)
        self.assertAlmostEqual(sum(x["weight"] for x in p["top50"]), 1 / 6)
        self.assertEqual(p["top50"][0]["fund_code"], self.codes[0])
        self.assertEqual(p["top50"][0]["rank"], 1)
        # main 全池对照集合：50 只 main（不含 low 那只）
        self.assertEqual(p["main_universe"]["count"], 50)
        # 协议参数固定（初始净资产/费用/现金计息/口径）
        pr = p["protocol"]
        self.assertEqual(pr["initial_nav"], 1.0)
        self.assertAlmostEqual(pr["buy_fee"], 0.0015)
        self.assertAlmostEqual(pr["sell_fee"], 0.005)
        self.assertEqual(pr["cash_annual_rate"], 0.02)
        self.assertEqual(pr["return_basis"], "official_daily_growth_rate")
        self.assertEqual(pr["dividend_policy"], "reinvest_via_daily_growth")
        self.assertIn("顺延", pr["nav_lag_policy"])
        self.assertTrue(pr["performance_not_computed"])
        # 来源哈希
        self.assertEqual(p["source_snapshot"]["score_file"], "2026-09.csv")
        self.assertEqual(p["hashes"]["manifest.json"],
                         _sha(os.path.join(self.snaps, self.RUN_ID, "manifest.json")))

    def test_reconstruction_source_excluded(self):
        rid = "20261001_000000"
        self._mk_snapshot(rid, self.SIGNAL, self.codes,
                          extra={"is_reconstruction": True, "forward_eligible": False})
        # 默认选取最新可作前向证据的快照 → 重建快照（生成时间更晚）必须被跳过
        p = fp.build_inputs(None, self.snaps, self.out, write_main=False)
        self.assertEqual(p["run_id"], self.RUN_ID)
        # 显式指定重建来源 → 拒绝
        with self.assertRaises(RuntimeError):
            fp.build_inputs(rid, self.snaps, self.out, write_main=False)

    def test_freeze_idempotent(self):
        r1 = fp.freeze(self.RUN_ID, self.snaps, self.out)
        self.assertEqual(r1["status"], "frozen")
        path = r1["path"]
        h1 = _sha(path)
        r2 = fp.freeze(self.RUN_ID, self.snaps, self.out)
        self.assertEqual(r2["status"], "existing")
        self.assertEqual(_sha(path), h1)          # 不可变：字节未改

    def test_changed_content_rejected_then_force_rebuild(self):
        r1 = fp.freeze(self.RUN_ID, self.snaps, self.out)
        # 改动来源快照（gap_n 变化）→ 内容不同 → 默认拒绝
        self._mk_snapshot(self.RUN_ID, self.SIGNAL, self.codes, extra={"gap_n": 5})
        with self.assertRaises(RuntimeError):
            fp.freeze(self.RUN_ID, self.snaps, self.out)
        # --force → 备份旧文件后重建
        r2 = fp.freeze(self.RUN_ID, self.snaps, self.out, force=True)
        self.assertEqual(r2["status"], "frozen")
        baks = [f for f in os.listdir(self.out) if ".bak_" in f]
        self.assertEqual(len(baks), 1)
        with open(r2["path"], encoding="utf-8") as f:
            self.assertEqual(json.load(f)["source_snapshot"]["gap_n"], 5)

    def test_main_universe_hash_matches_file(self):
        p = fp.build_inputs(self.RUN_ID, self.snaps, self.out, write_main=True)
        main_path = os.path.join(self.out, p["main_universe"]["file"])
        self.assertEqual(_sha(main_path), p["main_universe"]["sha256"])

    def test_no_forward_snapshot_raises(self):
        empty = os.path.join(self.tmp, "empty_snaps")
        os.makedirs(empty, exist_ok=True)
        with self.assertRaises(RuntimeError):
            fp.build_inputs(None, empty, self.out, write_main=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
