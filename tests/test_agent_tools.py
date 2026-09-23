# test_agent_tools.py：Phase 4 薄工具层测试（v1.1）
# fixture/mock 隔离：不运行真实生产流水线、不改真实 ledger/snapshots；
# 所有路径经 monkeypatch 指向工作区 ml/ 临时目录；_current_health 统一由 mock 提供。
import hashlib
import inspect
import json
import os
import shutil
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import agent_tools  # noqa: E402
import production_pipeline  # noqa: E402

RUN1 = "20260921_090000"
RUN2 = "20260922_100000"


def _make_snapshot(root, run_id, as_of, generated_at, commit="c0f7d91", has_shadow=False):
    d = os.path.join(root, "snapshots", run_id)
    os.makedirs(d, exist_ok=True)
    month = as_of[:7]
    import pandas as pd
    # 低置信度行放在 CSV 前部（回归 P0-2：历史排名必须读 rank/rank_lowconf 列而非行号）
    score = pd.DataFrame([
        {"fund_code": "000003", "rank": None, "rank_lowconf": 1, "score": 0.8,
         "fund_name": "低置信基金", "confidence": "low", "as_of": as_of},
        {"fund_code": "000001", "rank": 1, "rank_lowconf": None, "score": 2.0,
         "fund_name": "基金一号", "confidence": "main", "as_of": as_of},
        {"fund_code": "000002", "rank": 2, "rank_lowconf": None, "score": 1.5,
         "fund_name": "基金二号", "confidence": "main", "as_of": as_of},
    ])
    score.to_csv(os.path.join(d, f"{month}.csv"), index=False)
    if has_shadow:      # 快照内影子评分（v1.1.1：影子状态只认快照内文件）
        score[["fund_code", "score", "confidence"]].to_csv(
            os.path.join(d, f"shadow_{month}.csv"), index=False)
    pd.DataFrame([{"fund_code": "000001", "rank": 1}, {"fund_code": "000002", "rank": 2}]) \
        .to_csv(os.path.join(d, "top50.csv"), index=False)
    st = {"last_month": month, "cohorts": {
        month: {"signal_date": as_of, "execution_date": None, "expire_month": "2027-03",
                "cohort_weight": 1 / 6, "n_codes": 2, "created_at": generated_at,
                "status": "planned", "score_run": run_id}},
        "portfolio": {"n_active": 0, "cash_weight": 1.0, "n_funds": 0}}
    with open(os.path.join(d, "portfolio_state.json"), "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False)
    pd.DataFrame([{"cohort_id": month, "fund_code": c, "weight_in_cohort": 1 / 12,
                   "signal_date": as_of, "execution_date": None, "created_at": generated_at,
                   "expire_month": "2027-03", "status": "planned"}
                  for c in ("000001", "000002")]).to_csv(
        os.path.join(d, "portfolio_ledger.csv"), index=False)
    man = {"run_id": run_id, "git_commit_sha": commit, "strategy_version": "momentum-v1",
           "data_cutoff": as_of, "signal_date": as_of,
           "score_generated_at": generated_at, "execution_date": None,
           "eligible_size": 3, "main_size": 2, "low_size": 1,
           "status": "complete", "score_file_sha256": "abc123",
           "top50": ["000001", "000002"],
           "cohort_action": {"cohort_status": "planned", "score_run": run_id},
           "portfolio": {"n_active": 0, "cash_weight": 1.0}}
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False)
    with open(os.path.join(d, "COMPLETE"), "w", encoding="utf-8") as f:
        f.write(run_id + "\n")
    return d


class TestAgentTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._hc = mock.patch.object(agent_tools, "_current_health", return_value={
            "status": "PASS", "score_ready": True, "benchmark_latest_date": "2026-09-21",
            "gap_n": 0, "stale_n": 0, "shadow_stale": [], "shadow_factors": {}})
        cls._hc.start()

    @classmethod
    def tearDownClass(cls):
        cls._hc.stop()

    def setUp(self):
        self.tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml",
                                    f"_agt_test_{os.getpid()}_{id(self)}")
        os.makedirs(self.tmp_dir, exist_ok=True)
        root = self.tmp_dir
        snaps = os.path.join(root, "snapshots")
        os.makedirs(snaps, exist_ok=True)
        self.snap1 = _make_snapshot(root, RUN1, "2026-09-18", "2026-09-21T09:00:00")
        self.snap2 = _make_snapshot(root, RUN2, "2026-09-21", "2026-09-22T10:00:00",
                                    has_shadow=True)   # RUN2 最新快照内含影子 CSV
        partial = os.path.join(snaps, "20260922_120000")
        os.makedirs(partial, exist_ok=True)
        with open(os.path.join(partial, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"run_id": "20260922_120000", "status": "complete"}, f)

        led_dir = os.path.join(root, "ledger")
        os.makedirs(led_dir, exist_ok=True)
        led = {"last_month": "2026-09", "cohorts": {
            "2026-09": {"signal_date": "2026-09-21", "execution_date": None,
                        "expire_month": "2027-03", "cohort_weight": 1 / 6,
                        "n_codes": 2, "status": "planned", "score_run": RUN2}},
            "portfolio": {"n_active": 0, "cash_weight": 1.0, "n_funds": 0}}
        with open(os.path.join(led_dir, "portfolio_state.json"), "w", encoding="utf-8") as f:
            json.dump(led, f, ensure_ascii=False)

        vt = os.path.join(root, "vol_target_dev.txt")
        with open(vt, "w", encoding="utf-8") as f:
            f.write("  ② 波动目标动量      8.86%   16.56%     0.48   -38.04%     0.23\n")
            f.write("仓位统计（波动目标动量）：均值 0.75｜中位 0.77｜最低 0.31｜满仓月占比 23.7%\n")

        self._orig = {k: getattr(agent_tools, k) for k in (
            "SNAPSHOTS_DIR", "SCORES_DIR", "LEDGER_PATH", "VOLTARGET_REPORT", "AUDIT_DIR")}
        agent_tools.SNAPSHOTS_DIR = snaps
        agent_tools.SCORES_DIR = os.path.join(root, "scores")
        agent_tools.LEDGER_PATH = os.path.join(led_dir, "portfolio_state.json")
        agent_tools.VOLTARGET_REPORT = vt
        agent_tools.AUDIT_DIR = os.path.join(root, "agent_audit")
        os.makedirs(agent_tools.SCORES_DIR, exist_ok=True)
        # 可覆盖目录放一个旧影子文件，验证影子状态**不回退**（v1.1.1 P1）
        with open(os.path.join(agent_tools.SCORES_DIR, "shadow_2026-09.csv"), "w", encoding="utf-8") as f:
            f.write("fund_code,score,confidence\n000001,2.0,main\n")

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(agent_tools, k, v)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ---- T5-1 只读：10 个只读工具全部纳入树哈希验证 ----
    def test_readonly_no_modify(self):
        calls = [
            ("get_data_health", {}),
            ("get_latest_complete_snapshot", {}),
            ("get_score", {"month": "2026-09"}),
            ("get_score", {"run_id": RUN1}),
            ("get_top_funds", {"month": "2026-09", "top_n": 2}),
            ("get_fund_rank_history", {"fund_code": "000001"}),
            ("get_portfolio_state", {}),
            ("get_shadow_status", {}),
            ("compare_snapshots", {"run_a": RUN1, "run_b": RUN2}),
            ("get_risk_scenario_v1", {}),
            ("generate_research_report", {"topic": "data_status"}),
        ]
        self.assertEqual(sorted(agent_tools.READ_ONLY_TOOLS),
                         sorted({t for t, _ in calls if t != "get_data_health"} |
                                {"get_data_health"}), "READ_ONLY_TOOLS 应覆盖全部只读工具")

        def tree_hash(root):
            h = hashlib.sha256()
            for dp, _, fs in os.walk(root):
                for fn in sorted(fs):
                    p = os.path.join(dp, fn)
                    h.update(os.path.relpath(p, root).encode())
                    with open(p, "rb") as f:
                        h.update(f.read())
            return h.hexdigest()

        snaps_before = tree_hash(os.path.join(self.tmp_dir, "snapshots"))
        led_before = tree_hash(os.path.join(self.tmp_dir, "ledger"))
        for tool, kw in calls:
            out = agent_tools.call(tool, **kw)
            self.assertEqual(out["status"], "ok", f"{tool}: {out}")
        self.assertEqual(tree_hash(os.path.join(self.tmp_dir, "snapshots")), snaps_before,
                         "snapshots 被修改")
        self.assertEqual(tree_hash(os.path.join(self.tmp_dir, "ledger")), led_before,
                         "ledger 被修改")

    # ---- T5-12 health FAIL：当前查询停止、显式 run_id 可读（P0-1）----
    def test_health_fail_gate(self):
        with mock.patch.object(agent_tools, "_current_health", return_value={
                "status": "FAIL", "score_ready": False, "benchmark_latest_date": "2026-09-21",
                "shadow_stale": [], "shadow_factors": {}}):
            for tool, kw in (("get_score", {"month": "2026-09"}),
                             ("get_top_funds", {"month": "2026-09"}),
                             ("get_portfolio_state", {}),
                             ("generate_research_report", {"topic": "data_status"})):
                out = agent_tools.call(tool, **kw)
                self.assertEqual(out["status"], "unavailable", tool)
                self.assertEqual(out["errors"][0]["code"], "health_fail", tool)
            # 显式 run_id 历史审计读取不受阻，且必须标注 historical_audit
            out = agent_tools.call("get_score", run_id=RUN1)
            self.assertEqual(out["status"], "ok")
            self.assertTrue(any("historical_audit" in w for w in out["warnings"]))
            self.assertTrue(out["provenance"]["historical"])
            out = agent_tools.call("compare_snapshots", run_a=RUN1, run_b=RUN2)
            self.assertEqual(out["status"], "ok")
            self.assertTrue(any("historical_audit" in w for w in out["warnings"]))

    # ---- T5-13 低置信度在前时历史排名 = rank / rank_lowconf 列（P0-2）----
    def test_rank_uses_column_not_rowindex(self):
        out = agent_tools.call("get_fund_rank_history", fund_code="000002")
        self.assertEqual(out["status"], "ok")
        for mrow in out["detail"]["months"]:
            self.assertEqual(mrow["rank"], 2)          # rank 列，而非原始行号 3
            self.assertEqual(mrow["confidence"], "main")
            self.assertEqual(mrow["fund_name"], "基金二号")
            self.assertIsNotNone(mrow["score"])
            self.assertIsNotNone(mrow["month"])
        out3 = agent_tools.call("get_fund_rank_history", fund_code="000003")
        self.assertEqual(out3["detail"]["months"][0]["rank"], 1)     # rank_lowconf 列
        self.assertEqual(out3["detail"]["months"][0]["confidence"], "low")

    # ---- T5-14 shadow 权威判定（v1.1.1 P1：判 stale 用 shadow_stale、只认快照内影子）----
    def test_shadow_authoritative(self):
        def health(stale):
            return {"status": "WARN" if stale else "PASS", "score_ready": True,
                    "benchmark_latest_date": "2026-09-21",
                    "shadow_stale": stale, "shadow_factors": {}}
        shadow_csv = os.path.join(self.snap2, "shadow_2026-09.csv")
        mv = os.path.join(self.snap2, "shadow_2026-09.csv.bak")
        try:
            # (删除最新快照影子与否 × stale 之 None/风格/行业) → 期望 status/code
            cases = [
                (True, [], "ok", None),                                 # fresh + 快照有影子 → ok
                (False, [], "unavailable", "no_current_shadow_snapshot"),  # fresh + 无影子 → 不回退
                (True, ["style_index_sz399006.csv"], "unavailable", "shadow_factor_stale"),
                (True, ["sw_industry(最旧 2026-09-18)"], "unavailable", "shadow_factor_stale"),
            ]
            for has_file, stale, want, want_code in cases:
                if has_file and not os.path.exists(shadow_csv):
                    os.rename(mv, shadow_csv)
                elif not has_file and os.path.exists(shadow_csv):
                    os.rename(shadow_csv, mv)
                with mock.patch.object(agent_tools, "_current_health", return_value=health(stale)):
                    out = agent_tools.call("get_shadow_status")
                self.assertEqual(out["status"], want, (has_file, stale))
                if want == "unavailable":
                    self.assertEqual(out["errors"][0]["code"], want_code, (has_file, stale))
                # 快照引用必须带全（v1.1.1）：run_id/data_cutoff/source_snapshot
                self.assertEqual(out["run_id"], RUN2)
                self.assertEqual(out["data_cutoff"], "2026-09-21")
                self.assertTrue(out["source_snapshot"].endswith(RUN2))
                if want == "ok":
                    self.assertEqual(out["detail"]["last_score"], "shadow_2026-09.csv")
        finally:
            if os.path.exists(mv) and not os.path.exists(shadow_csv):
                os.rename(mv, shadow_csv)

    # ---- v1.1.1 P1：因子 fresh 但快照内无影子 → 不回退 ml/scores/（目录里有旧文件也不采用）----
    def test_shadow_no_fallback_to_scores_dir(self):
        shadow_csv = os.path.join(self.snap2, "shadow_2026-09.csv")
        mv = os.path.join(self.snap2, "shadow_2026-09.csv.bak")
        self.assertTrue(os.path.exists(os.path.join(agent_tools.SCORES_DIR, "shadow_2026-09.csv")))
        try:
            os.rename(shadow_csv, mv)
            out = agent_tools.call("get_shadow_status")     # fresh（setUpClass mock）
            self.assertEqual(out["status"], "unavailable")
            self.assertEqual(out["errors"][0]["code"], "no_current_shadow_snapshot")
            self.assertEqual(out["detail"]["last_score"], None)   # 未采用 ml/scores 旧文件
        finally:
            os.rename(mv, shadow_csv)

    # ---- v1.1.1 P1：报告传播底层 shadow 停止状态 ----
    def test_report_propagates_shadow_unavailable(self):
        shadow_csv = os.path.join(self.snap2, "shadow_2026-09.csv")
        mv = os.path.join(self.snap2, "shadow_2026-09.csv.bak")
        try:
            os.rename(shadow_csv, mv)
            out = agent_tools.call("generate_research_report", topic="shadow_status")
            self.assertEqual(out["status"], "unavailable")       # 不可包装成 ok
            self.assertEqual(out["errors"][0]["code"], "no_current_shadow_snapshot")
            self.assertEqual(out["run_id"], RUN2)
            self.assertTrue(out["source_snapshot"].endswith(RUN2))
            self.assertIn("shadow 状态", out["detail"]["report"])  # 报告正文仍保留
        finally:
            os.rename(mv, shadow_csv)

    # ---- v1.1.1 P1：top_n 必须为 1..200 的整数（负数/0/布尔/字符串/超限均拒绝）----
    def test_topn_interval_rejected(self):
        for bad in (-1, 0, True, 201, "50", 2.5):
            out = agent_tools.call("get_top_funds", month="2026-09", top_n=bad)
            self.assertEqual(out["status"], "not_executable", f"top_n={bad}")
            self.assertEqual(out["errors"][0]["code"], "research_boundary", f"top_n={bad}")
        for ok_n in (1, 200):
            out = agent_tools.call("get_top_funds", month="2026-09", top_n=ok_n)
            self.assertEqual(out["status"], "ok", f"top_n={ok_n}")

    # ---- T5-15 Top 名称/分数 与 组合溯源字段（P1-4/P1-5）----
    def test_top_details_and_portfolio_provenance(self):
        out = agent_tools.call("get_top_funds", month="2026-09", top_n=2)
        self.assertEqual(out["status"], "ok")
        for r in out["detail"]["top"]:
            self.assertIn("fund_name", r)
            self.assertIn("score", r)
        st = agent_tools.call("get_portfolio_state")
        self.assertEqual(st["status"], "ok")
        c = st["detail"]["cohorts"][0]
        self.assertEqual(c["score_run"], RUN2)                       # 溯源
        self.assertTrue(c["source_snapshot"].endswith(RUN2))          # 对应快照
        self.assertEqual(st["detail"]["reference_snapshot"], RUN2)
        self.assertEqual(st["data_cutoff"], "2026-09-21")

    # ---- T5-16 scenario=true 与仓位统计（P1-7）----
    def test_risk_scenario_v1(self):
        out = agent_tools.call("get_risk_scenario_v1")
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["detail"]["frozen_target_vol"], 0.15)
        self.assertIs(out["detail"]["scenario"], True)
        self.assertIn("position", out["detail"])
        self.assertAlmostEqual(out["detail"]["position"]["mean"], 0.75)

    # ---- T5-17 未注册工具与非法参数均写审计（P1-9）----
    def test_audit_for_rejected_calls(self):
        day = __import__("datetime").datetime.now().strftime("%Y%m%d")
        p = os.path.join(agent_tools.AUDIT_DIR, f"agent_audit_{day}.jsonl")
        before = sum(1 for _ in open(p, encoding="utf-8")) if os.path.exists(p) else 0
        agent_tools.call("confirm_execution", cohort_id="2026-09", execution_date="2026-10-01")
        agent_tools.call("get_top_funds", month="2026-09", hold_period=6)
        with open(p, encoding="utf-8") as _f:
            lines = _f.readlines()
        self.assertEqual(len(lines) - before, 2)
        last = json.loads(lines[-1])
        self.assertEqual(last["tool"], "get_top_funds")
        self.assertEqual(last["status"], "not_executable")
        self.assertEqual(last["caller"], "local_user")

    # ---- T5-18 mock run_pipeline：零参数成功 + 异常 → aborted ----
    def test_run_pipeline_mocked(self):
        import inspect as _inspect
        sig = _inspect.signature(agent_tools.run_production_pipeline)
        self.assertEqual(list(sig.parameters), [])
        man = {"run_id": "20260922_999999", "status": "complete", "data_cutoff": "2026-09-21",
               "shadow_stale": [], "eligible_size": 5118,
               "cohort_action": {"cohort_status": "planned"}}
        with mock.patch("production_pipeline.run_pipeline", return_value={
                "run_id": "20260922_999999", "snapshot": "ml/snapshots/20260922_999999",
                "manifest": man}):
            out = agent_tools.call("run_production_pipeline")
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["run_id"], "20260922_999999")
        with mock.patch("production_pipeline.run_pipeline",
                        side_effect=RuntimeError("step_clean 失败")):
            out = agent_tools.call("run_production_pipeline")
        self.assertEqual(out["status"], "aborted")
        self.assertEqual(out["errors"][0]["code"], "run_aborted")

    # ---- 既有契约检查（保留）----
    def test_month_latest_with_candidates(self):
        out = agent_tools.call("get_score", month="2026-09")
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["run_id"], RUN2)
        cand = [w for w in out["warnings"] if w.startswith("multiple_complete_for_month")]
        self.assertEqual(len(cand), 1)
        self.assertIn(RUN1, cand[0])
        self.assertTrue(out["provenance"]["score_file"].startswith(agent_tools.SNAPSHOTS_DIR))

    def test_runid_strict(self):
        out = agent_tools.call("get_score", run_id=RUN1)
        self.assertEqual(out["run_id"], RUN1)
        self.assertEqual(out["detail"]["as_of"], "2026-09-18")
        miss = agent_tools.call("get_score", run_id="20990101_000000")
        self.assertEqual(miss["status"], "unavailable")
        self.assertEqual(miss["errors"][0]["code"], "no_snapshot_for_run")

    def test_topn_too_large_rejected(self):
        out = agent_tools.call("get_top_funds", month="2026-09", top_n=201)
        self.assertEqual(out["status"], "not_executable")

    def test_strategy_params_rejected(self):
        for kw in ({"month": "2026-09", "hold_period": 6},
                   {"month": "2026-09", "top_n": 50, "fee_rate": 0.0015}):
            out = agent_tools.call("get_top_funds", **kw)
            self.assertEqual(out["status"], "not_executable", kw)
        self.assertEqual(agent_tools.call("run_production_pipeline", skip_refresh=True)["status"],
                         "not_executable")
        self.assertEqual(agent_tools.call("run_production_pipeline", as_of="2026-10-01")["status"],
                         "not_executable")

    def test_no_complete_unavailable(self):
        out = agent_tools.call("get_latest_complete_snapshot")
        self.assertEqual(out["status"], "ok")
        for s in (self.snap1, self.snap2):
            os.rename(os.path.join(s, "COMPLETE"), os.path.join(s, "COMPLETE.bak"))
        out2 = agent_tools.call("get_latest_complete_snapshot")
        self.assertEqual(out2["status"], "unavailable")
        for s in (self.snap1, self.snap2):
            os.rename(os.path.join(s, "COMPLETE.bak"), os.path.join(s, "COMPLETE"))

    def test_latest_snapshot_cohort_status_string(self):
        out = agent_tools.call("get_latest_complete_snapshot")
        self.assertEqual(out["provenance"]["cohort_status"], "planned")   # 字符串，非布尔

    def test_planned_warnings(self):
        out = agent_tools.call("get_portfolio_state")
        self.assertTrue(any("planned" in w for w in out["warnings"]))

    def test_unexposed_rejected(self):
        for tool in ("confirm_execution", "modify_strategy", "retrain", "promote_shadow", "trade"):
            out = agent_tools.call(tool)
            self.assertEqual(out["status"], "not_executable", tool)
            self.assertEqual(out["errors"][0]["code"], "unknown_tool", tool)

    def test_audit_written(self):
        day = __import__("datetime").datetime.now().strftime("%Y%m%d")
        p = os.path.join(agent_tools.AUDIT_DIR, f"agent_audit_{day}.jsonl")
        before = os.path.getsize(p) if os.path.exists(p) else 0
        agent_tools.call("get_score", month="2026-09")
        self.assertTrue(os.path.exists(p))
        self.assertGreater(os.path.getsize(p), before)

    def test_report_topic_enum(self):
        out = agent_tools.call("generate_research_report", topic="rebalance_search")
        self.assertEqual(out["status"], "not_executable")


if __name__ == "__main__":
    unittest.main(verbosity=2)