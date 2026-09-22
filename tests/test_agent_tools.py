# test_agent_tools.py：Phase 4 薄工具层 T5 测试（fixture/mock，**不运行真实生产流水线、
# 不改真实 ledger/snapshots**；所有路径通过 monkeypatch 指向临时目录）。
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import agent_tools  # noqa: E402
import production_pipeline  # noqa: E402   （仅确保可导入；测试不运行真实流水线）

RUN1 = "20260921_090000"
RUN2 = "20260922_100000"


def _score_rows(as_of):
    import pandas as pd
    return pd.DataFrame([
        {"fund_code": "000001", "rank": 1, "score": 2.0, "confidence": "main", "as_of": as_of},
        {"fund_code": "000002", "rank": 2, "score": 1.5, "confidence": "main", "as_of": as_of},
        {"fund_code": "000003", "rank": 3, "score": 1.0, "confidence": "low", "as_of": as_of},
    ])


def _make_snapshot(root, run_id, as_of, generated_at, commit="c0f7d91"):
    d = os.path.join(root, "snapshots", run_id)
    os.makedirs(d, exist_ok=True)
    month = as_of[:7]
    score = _score_rows(as_of)
    score.to_csv(os.path.join(d, f"{month}.csv"), index=False)
    pd = __import__("pandas")
    pd.DataFrame(score[score["confidence"] == "main"][["fund_code", "rank"]]
                 .to_dict("records")).sort_values("rank") \
        .to_csv(os.path.join(d, "top50.csv"), index=False)
    # ledger 副本（与 top50 一致的 2026-09 cohort，planned）
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
    def setUp(self):
        # 用工作区 ml/ 下临时目录（Windows 下 tempfile.mkdtemp 目录有 ACL 问题，改用 os.makedirs）
        self.tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml",
                                    f"_agt_test_{os.getpid()}_{id(self)}")
        os.makedirs(self.tmp_dir, exist_ok=True)
        root = self.tmp_dir
        snaps = os.path.join(root, "snapshots")
        os.makedirs(snaps, exist_ok=True)
        self.snap1 = _make_snapshot(root, RUN1, "2026-09-18", "2026-09-21T09:00:00")
        self.snap2 = _make_snapshot(root, RUN2, "2026-09-21", "2026-09-22T10:00:00")
        # 一个无 COMPLETE 的"半次运行"目录（不应进入候选）
        partial = os.path.join(snaps, "20260922_120000")
        os.makedirs(partial, exist_ok=True)
        with open(os.path.join(partial, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"run_id": "20260922_120000", "status": "complete"}, f)

        # ledger（当前状态：2026-09 planned）
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

        # monkeypatch 薄工具层路径 → fixture
        self._originals = {k: getattr(agent_tools, k) for k in (
            "SNAPSHOTS_DIR", "SCORES_DIR", "LEDGER_PATH", "VOLTARGET_REPORT", "AUDIT_DIR")}
        agent_tools.SNAPSHOTS_DIR = snaps
        agent_tools.SCORES_DIR = os.path.join(root, "scores")
        agent_tools.LEDGER_PATH = os.path.join(led_dir, "portfolio_state.json")
        agent_tools.VOLTARGET_REPORT = vt
        agent_tools.AUDIT_DIR = os.path.join(root, "agent_audit")
        os.makedirs(agent_tools.SCORES_DIR, exist_ok=True)

    def tearDown(self):
        for k, v in self._originals.items():
            setattr(agent_tools, k, v)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ---- T5-1 只读工具不修改 scores/ledger/snapshots ----
    def test_readonly_no_modify(self):
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
        for tool, kwargs in (("get_score", {"month": "2026-09"}),
                             ("get_top_funds", {"month": "2026-09", "top_n": 2}),
                             ("get_portfolio_state", {}),
                             ("get_latest_complete_snapshot", {}),
                             ("get_risk_scenario_v1", {}),
                             ("compare_snapshots", {"run_a": RUN1, "run_b": RUN2}),
                             ("get_fund_rank_history", {"fund_code": "000001"}),):
            out = agent_tools.call(tool, **kwargs)
            self.assertEqual(out["status"], "ok", f"{tool}: {out}")
        self.assertEqual(tree_hash(os.path.join(self.tmp_dir, "snapshots")), snaps_before,
                         "snapshots 被修改")
        self.assertEqual(tree_hash(os.path.join(self.tmp_dir, "ledger")), led_before,
                         "ledger 被修改")

    # ---- T5-2 月份查询：选最新完整快照 + 全部候选 + 警告 ----
    def test_month_latest_with_candidates(self):
        out = agent_tools.call("get_score", month="2026-09")
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["run_id"], RUN2)          # score_generated_at 最新
        self.assertEqual(out["detail"]["month"], "2026-09")
        cand_warn = [w for w in out["warnings"] if w.startswith("multiple_complete_for_month")]
        self.assertEqual(len(cand_warn), 1)
        self.assertIn(RUN1, cand_warn[0])
        self.assertIn(RUN2, cand_warn[0])
        # 评分必须来自快照内 CSV
        self.assertTrue(out["provenance"]["score_file"].startswith(agent_tools.SNAPSHOTS_DIR))

    # ---- T5-3 显式 run_id 严格读取 ----
    def test_runid_strict(self):
        out = agent_tools.call("get_score", run_id=RUN1)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["run_id"], RUN1)
        self.assertEqual(out["detail"]["as_of"], "2026-09-18")
        miss = agent_tools.call("get_score", run_id="20990101_000000")
        self.assertEqual(miss["status"], "unavailable")
        self.assertEqual(miss["errors"][0]["code"], "no_snapshot_for_run")

    # ---- T5-4 top_n 超限 → not_executable ----
    def test_topn_too_large_rejected(self):
        out = agent_tools.call("get_top_funds", month="2026-09", top_n=201)
        self.assertEqual(out["status"], "not_executable")
        self.assertEqual(out["errors"][0]["code"], "research_boundary")

    # ---- T5-5 策略参数/模式开关 → 签名拒绝（research_boundary / TypeError）----
    def test_strategy_params_rejected(self):
        out = agent_tools.call("get_top_funds", month="2026-09", hold_period=6)
        self.assertEqual(out["status"], "not_executable")
        out2 = agent_tools.call("run_production_pipeline", skip_refresh=True)
        self.assertEqual(out2["status"], "not_executable")
        out3 = agent_tools.call("run_production_pipeline", as_of="2026-10-01")
        self.assertEqual(out3["status"], "not_executable")
        self.assertIn("research_boundary", out3["errors"][0]["code"])

    # ---- T5-6 run_production_pipeline 签名无参数 ----
    def test_pipeline_signature_no_args(self):
        import inspect
        sig = inspect.signature(agent_tools.run_production_pipeline)
        self.assertEqual(list(sig.parameters), [])

    # ---- T5-7 无 COMPLETE 快照 → unavailable ----
    def test_no_complete_unavailable(self):
        out = agent_tools.call("get_latest_complete_snapshot")
        self.assertEqual(out["status"], "ok")             # fixture 有两个完整快照
        # 把 COMPLETE 全部移走 → 模拟无完整快照
        for s in (self.snap1, self.snap2):
            os.rename(os.path.join(s, "COMPLETE"), os.path.join(s, "COMPLETE.bak"))
        out2 = agent_tools.call("get_latest_complete_snapshot")
        self.assertEqual(out2["status"], "unavailable")
        for s in (self.snap1, self.snap2):
            os.rename(os.path.join(s, "COMPLETE.bak"), os.path.join(s, "COMPLETE"))

    # ---- T5-8 cohort planned → warnings（不得宣称已持仓）----
    def test_planned_warnings(self):
        out = agent_tools.call("get_portfolio_state")
        self.assertEqual(out["status"], "ok")
        self.assertTrue(any("planned" in w for w in out["warnings"]), out["warnings"])
        self.assertEqual(out["detail"]["n_active"], 0)

    # ---- T5-9 不注册不暴露操作 ----
    def test_unexposed_rejected(self):
        for tool in ("confirm_execution", "modify_strategy", "retrain", "promote_shadow", "trade"):
            out = agent_tools.call(tool)
            self.assertEqual(out["status"], "not_executable", tool)
            self.assertEqual(out["errors"][0]["code"], "unknown_tool", tool)

    # ---- T5-10 审计日志：调用后 agent_audit 增长（唯一可写路径）----
    def test_audit_written(self):
        p = os.path.join(agent_tools.AUDIT_DIR, f"agent_audit_{__import__('datetime').datetime.now().strftime('%Y%m%d')}.jsonl")
        before = os.path.getsize(p) if os.path.exists(p) else 0
        agent_tools.call("get_score", month="2026-09")
        self.assertTrue(os.path.exists(p))
        self.assertGreater(os.path.getsize(p), before)
        with open(p, encoding="utf-8") as f:
            last = json.loads(f.readlines()[-1])
        self.assertEqual(last["tool"], "get_score")
        self.assertEqual(last["status"], "ok")

    # ---- T5-11 风险情景工具无参数返回冻结 15% ----
    def test_risk_scenario_v1_noargs(self):
        out = agent_tools.call("get_risk_scenario_v1")
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["detail"]["frozen_target_vol"], 0.15)
        self.assertTrue(any("scenario=true" in w for w in out["warnings"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)