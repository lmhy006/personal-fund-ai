# agent_tools.py：Phase 4 薄工具层（按 docs/PHASE4_AGENT_SCHEMA.md v1.1 实现；v1.1 修复 2026-09-22）
#
# 契约要点（实现即约束）：
#   - 10 个只读工具 + 无参数 run_production_pipeline()；其余操作（confirm_execution / 参数修改 /
#     训练搜索 / 影子转正 / 交易）**不注册**；
#   - 只读工具**不得修改 scores/ledger/snapshots**，唯一可写路径是审计日志 logs/agent_audit/；
#   - 评分一律读**正式快照内 CSV**；月份查询固定选择 score_generated_at 最新完整快照 + 返回
#     全部候选 run_id + 警告；显式 run_id 严格读取指定快照并为历史审计；
#   - **当前健康门禁（v1.1 P0）**：data_health=FAIL 时，月份/无参"当前"查询与当前组合/状态报告
#     一律 unavailable(health_fail)；显式 run_id 的历史读取与快照比较不受阻，但必须标注
#     historical_audit、不得称为当前结果；
#   - run_production_pipeline 不接受任何参数（完整正式流程）。
import json
import os
import re
from datetime import datetime

import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# 路径（测试用 monkeypatch 覆盖这些模块属性即可指向临时 fixture）
SNAPSHOTS_DIR = os.path.join(PROJECT_ROOT, "ml", "snapshots")
SCORES_DIR = os.path.join(PROJECT_ROOT, "ml", "scores")
LEDGER_PATH = os.path.join(PROJECT_ROOT, "ml", "ledger", "portfolio_state.json")
VOLTARGET_REPORT = os.path.join(PROJECT_ROOT, "ml", "backtest", "vol_target_dev.txt")
VOLTARGET_MONTHLY = os.path.join(PROJECT_ROOT, "ml", "backtest", "vol_target_monthly.csv")
AUDIT_DIR = os.path.join(PROJECT_ROOT, "logs", "agent_audit")
STRATEGY_VERSION = "momentum-v1"
TOP_N_MAX = 200                        # top_n 仅查询展示，超上限或带策略参数 → not_executable
DEFAULT_CALLER = "local_user"          # 对话适配层注入调用者身份；当前阶段用 local_user

# ---------------------------------------------------------------- 基础设施
def _audit(tool: str, inputs: dict, status: str, ms: int, caller: str):
    """审计日志（薄工具层唯一可写路径：logs/agent_audit/YYYYMMDD.jsonl）。"""
    os.makedirs(AUDIT_DIR, exist_ok=True)
    day = datetime.now().strftime("%Y%m%d")
    rec = {"ts": datetime.now().isoformat(timespec="seconds"), "tool": tool,
           "caller": caller,
           "inputs": {k: v for k, v in inputs.items() if v is not None},
           "status": status, "duration_ms": ms}
    with open(os.path.join(AUDIT_DIR, f"agent_audit_{day}.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _env(detail=None, status="ok", run_id=None, data_cutoff=None, source_snapshot=None,
         warnings=(), errors=(), provenance=None):
    return {"status": status, "run_id": run_id, "data_cutoff": data_cutoff,
            "strategy_version": STRATEGY_VERSION, "source_snapshot": source_snapshot,
            "warnings": list(warnings), "errors": list(errors),
            "provenance": provenance, "detail": detail}


def _err(code, message):
    return {"code": code, "message": message}


def _hist_annotate(snap, warn: list, env_prov: dict) -> tuple:
    """历史审计标注：显式 run_id 读取时追加警告与 provenance.historical=True。"""
    warn.append(f"historical_audit：此结果来自 run_id={snap['run_id']}，为历史审计数据，不是当前结论")
    if env_prov is None:
        env_prov = {}
    env_prov["historical"] = True
    return warn, env_prov


def _current_health() -> dict:
    """现场计算当前数据健康（供当前门禁 / 影子状态判定；测试 monkeypatch 本函数）。"""
    from data_health import health_report
    try:
        return health_report()
    except Exception:  # noqa: BLE001
        return {"status": "FAIL", "benchmark_latest_date": None,
                "shadow_stale": None, "shadow_factors": {}}


def _health_gate(allow_current=True) -> dict | None:
    """当前查询的 health FAIL 门禁（v1.1 P0）。

    仅用于"当前结论"类查询；显式历史 run_id 读取不走此门禁（在 _resolve 分支处理）。
    :return: None=放行；否则为 unavailable envelope。
    """
    if not allow_current:
        return None
    h = _current_health()
    if h.get("status") == "FAIL":
        return _env(status="unavailable", data_cutoff=h.get("benchmark_latest_date"),
                    warnings=["data_health=FAIL：可读取历史快照（显式 run_id）进行审计，但不能当作当前结论"],
                    errors=[_err("health_fail", "数据健康 FAIL，禁止当前评分/组合结论")])
    return None


def _complete_snapshots(snaps_dir=None):
    """所有带 COMPLETE 且 status=complete 的快照，按 score_generated_at 升序。"""
    snaps_dir = snaps_dir or SNAPSHOTS_DIR
    out = []
    if not os.path.isdir(snaps_dir):
        return out
    for d in sorted(os.listdir(snaps_dir)):
        if not re.fullmatch(r"\d{8}_\d{6}", d):
            continue
        mf = os.path.join(snaps_dir, d, "manifest.json")
        if not os.path.exists(os.path.join(snaps_dir, d, "COMPLETE")) or not os.path.exists(mf):
            continue
        try:
            with open(mf, encoding="utf-8") as f:
                man = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        if man.get("status") != "complete":
            continue
        out.append({"run_id": d, "dir": os.path.join(snaps_dir, d), "manifest": man})
    out.sort(key=lambda x: str(x["manifest"].get("score_generated_at", "")))
    return out


def _resolve(run_id=None, month=None, snaps_dir=None):
    """解析 (snap, warnings, provenance)。

    run_id：严格读取指定完整快照（历史审计——不施加当前健康门禁，返回时调用方负责标注）；
    month：固定选 score_generated_at 最新完整快照 + 全部候选 run_id + 警告（**当前门禁**由调用方先执行）；
    无参：最新完整快照（当前门禁由调用方先执行）。
    """
    snaps = _complete_snapshots(snaps_dir)
    if run_id:
        for s in snaps:
            if s["run_id"] == run_id:
                return s, [], {}, None
        return None, None, None, _env(status="unavailable",
                                     warnings=["最近完整快照：" + (snaps[-1]["run_id"] if snaps else "无")],
                                     errors=[_err("no_snapshot_for_run", f"未找到 run_id={run_id} 的完整快照")])
    if month:
        cands = [s for s in snaps if str(s["manifest"].get("signal_date", ""))[:7] == month]
        if not cands:
            return None, None, None, _env(status="unavailable",
                                         warnings=["最近完整快照：" + (snaps[-1]["run_id"] if snaps else "无")],
                                         errors=[_err("no_snapshot_for_month", f"未找到 {month} 的正式快照")])
        latest = cands[-1]
        warn = ["multiple_complete_for_month：%s 存在 %d 个完整快照（%s），已取 score_generated_at 最新（%s）"
                % (month, len(cands), ", ".join(s["run_id"] for s in cands), latest["run_id"])] \
            if len(cands) > 1 else []
        return latest, warn, {}, None
    if not snaps:
        return None, None, None, _env(status="unavailable", warnings=["尚无正式快照"],
                                     errors=[_err("no_complete_snapshot", "暂无带 COMPLETE 的正式快照")])
    return snaps[-1], [], {}, None


def _score_csv(snap: dict) -> str | None:
    m = snap["manifest"]
    name = f"{str(m.get('signal_date', ''))[:7]}.csv"
    p = os.path.join(snap["dir"], name)
    return p if os.path.exists(p) else None


def _score(snap: dict) -> pd.DataFrame | None:
    csv_p = _score_csv(snap)
    if csv_p is None:
        return None
    return pd.read_csv(csv_p, dtype={"fund_code": str})


def _shadow_csv(snap: dict) -> str | None:
    m = snap["manifest"]
    name = f"shadow_{str(m.get('signal_date', ''))[:7]}.csv"
    p = os.path.join(snap["dir"], name)
    return p if os.path.exists(p) else None


# ---------------------------------------------------------------- 只读工具（10 个）
def get_data_health():
    rep = _current_health()
    prov = {"source": "现场计算（data_health.health_report()）"}
    p = os.path.join(SNAPSHOTS_DIR, "data_health_latest.json")
    if os.path.exists(p):
        prov["data_health_latest_json"] = p
    return _env(detail={"status": rep["status"], "score_ready": rep.get("score_ready"),
                        "benchmark": rep.get("benchmark_latest_date"),
                        "processed": rep.get("processed_dist"),
                        "gap_n": rep.get("gap_n"), "stale_n": rep.get("stale_n")},
                data_cutoff=rep.get("benchmark_latest_date"),
                provenance=prov,
                warnings=[] if rep.get("status") == "PASS" else
                [f"data_health={rep.get('status')}" + ("；shadow stale：" + ",".join(rep.get("shadow_stale") or [])
                                                      if rep.get("shadow_stale") else "")])


def get_latest_complete_snapshot():
    snaps = _complete_snapshots()
    if not snaps:
        return _env(status="unavailable",
                    errors=[_err("no_complete_snapshot", "暂无带 COMPLETE 标志的正式快照")])
    s = snaps[-1]
    m = s["manifest"]
    # cohort_status：读快照内 portfolio_state 的当月 cohort 状态（planned/active/expired）
    cohort_status = None
    st_path = os.path.join(s["dir"], "portfolio_state.json")
    if os.path.exists(st_path):
        try:
            with open(st_path, encoding="utf-8") as f:
                st = json.load(f)
            month = str(m.get("signal_date", ""))[:7]
            if month in st.get("cohorts", {}):
                cohort_status = st["cohorts"][month].get("status")
        except Exception:  # noqa: BLE001
            cohort_status = None
    if cohort_status is None:
        cohort_status = (m.get("cohort_action") or {}).get("cohort_status")
    return _env(detail={"manifest_status": m.get("status"),
                        "eligible": m.get("eligible_size"), "top50_count": len(m.get("top50") or []),
                        "signal_date": m.get("signal_date"),
                        "score_sha256": m.get("score_file_sha256")},
                run_id=s["run_id"], data_cutoff=m.get("data_cutoff"),
                source_snapshot=s["dir"],
                provenance={"git_commit_sha": m.get("git_commit_sha"),
                            "score_file_sha256": m.get("score_file_sha256"),
                            "cohort_status": cohort_status})


def get_score(run_id=None, month=None):
    gate = None
    if run_id is None:
        gate = _health_gate()          # 当前查询（month/最新）必须过健康门禁
        if gate:
            return gate
    snap, warn, prov, fail = _resolve(run_id, month)
    if snap is None:
        return fail
    if run_id:
        warn, prov = _hist_annotate(snap, list(warn), dict(prov))
    csv_p = _score_csv(snap)
    if csv_p is None:
        return _env(status="unavailable", run_id=snap["run_id"], source_snapshot=snap["dir"],
                    errors=[_err("missing_score_file", "快照内缺少评分 CSV")])
    df = pd.read_csv(csv_p, dtype={"fund_code": str})
    m = snap["manifest"]
    return _env(detail={"month": str(m.get("signal_date", ""))[:7], "eligible": int(len(df)),
                        "main": int((df["confidence"] == "main").sum()) if "confidence" in df else None,
                        "low": int((df["confidence"] == "low").sum()) if "confidence" in df else None,
                        "as_of": m.get("signal_date"), "score_file": csv_p,
                        "historical": bool(run_id)},
                run_id=snap["run_id"], data_cutoff=m.get("data_cutoff"),
                source_snapshot=snap["dir"], warnings=warn,
                provenance={**prov, "score_file_sha256": m.get("score_file_sha256"), "score_file": csv_p})


def get_top_funds(run_id=None, month=None, top_n=50):
    if not isinstance(top_n, int) or top_n > TOP_N_MAX:
        return _env(status="not_executable",
                    errors=[_err("research_boundary", f"top_n 仅用于查询展示（≤{TOP_N_MAX}），不改变冻结策略参数")])
    if run_id is None:
        gate = _health_gate()
        if gate:
            return gate
    snap, warn, prov, fail = _resolve(run_id, month)
    if snap is None:
        return fail
    if run_id:
        warn, prov = _hist_annotate(snap, list(warn), dict(prov))
    df = _score(snap)
    if df is None:
        return _env(status="unavailable", run_id=snap["run_id"],
                    errors=[_err("missing_score_file", "快照内缺少评分 CSV")])
    m = snap["manifest"]
    main = df[df["confidence"] == "main"] if "confidence" in df else df
    top = main.sort_values("rank").head(top_n) if "rank" in main.columns else \
        main.sort_values("score", ascending=False).head(top_n)
    rows = []
    for i, r in enumerate(top.to_dict("records")):
        rows.append({"rank": int(r.get("rank", i + 1)),
                     "fund_code": r["fund_code"],
                     "score": float(r.get("score")) if r.get("score") == r.get("score") else None,
                     "fund_name": r.get("fund_name")})
    warnings = list(warn) + ["top_n 仅用于展示，不改变冻结策略参数"]
    return _env(detail={"top": rows, "historical": bool(run_id)},
                run_id=snap["run_id"], data_cutoff=m.get("data_cutoff"),
                source_snapshot=snap["dir"], warnings=warnings,
                provenance={**prov, "score_file": _score_csv(snap)})


def get_fund_rank_history(fund_code):
    """历史排名（v1.1 P0-2 修复）：直接读取 `rank`（主策略）或 `rank_lowconf`（低置信度）列，
    不再使用原始行号；同时返回 score/fund_name/confidence/月份。"""
    snaps = _complete_snapshots()
    months = []
    for s in snaps:
        df = _score(s)
        if df is None or fund_code not in df["fund_code"].values:
            continue
        row = df[df["fund_code"] == fund_code].iloc[0]
        conf = str(row.get("confidence", "")).strip() if "confidence" in df else "main"
        rank = None
        if "rank" in df and conf == "main":
            rank = int(row["rank"]) if row["rank"] == row["rank"] else None
        elif "rank_lowconf" in df and conf == "low":
            rank = int(row["rank_lowconf"]) if row["rank_lowconf"] == row["rank_lowconf"] else None
        score = float(row["score"]) if "score" in df and row["score"] == row["score"] else None
        months.append({"run_id": s["run_id"], "month": str(s["manifest"].get("signal_date", ""))[:7],
                       "as_of": s["manifest"].get("signal_date"),
                       "confidence": conf if conf else None,
                       "rank": rank, "score": score,
                       "fund_name": row.get("fund_name") if "fund_name" in df else None,
                       "in_top50": bool(rank is not None and rank <= 50)})
    if not months:
        return _env(status="ok", detail={"fund_code": fund_code, "months": []},
                    warnings=["该基金未出现在任何正式快照评分中"], errors=[])
    return _env(status="ok", detail={"fund_code": fund_code, "months": months},
                warnings=["history 仅截至已完成的正式快照；2027-03 后才能对照 6 个月标签"], errors=[])


def get_portfolio_state():
    if not os.path.exists(LEDGER_PATH):
        return _env(status="unavailable", errors=[_err("no_ledger", "ledger 不存在（尚无正式组合状态）")])
    gate = _health_gate()
    if gate:
        return gate
    with open(LEDGER_PATH, encoding="utf-8") as f:
        st = json.load(f)
    agg = st.get("portfolio", {})
    # 权威快照（data_cutoff / score_run 溯源）：latest complete snapshot manifest
    snaps = _complete_snapshots()
    ref = None
    if snaps:
        ref = snaps[-1]["manifest"]
    cohorts_meta = []
    for c, m in st.get("cohorts", {}).items():
        score_run = m.get("score_run")
        src = os.path.join(SNAPSHOTS_DIR, score_run) if score_run else None
        cohorts_meta.append({"cohort": c, "status": m.get("status"),
                             "signal_date": m.get("signal_date"),
                             "execution_date": m.get("execution_date"),
                             "score_run": score_run,
                             "source_snapshot": src if (src and os.path.isdir(src)) else None})
    planned = [x for x in cohorts_meta if x["status"] == "planned"]
    warnings = []
    if planned:
        warnings.append(f"{len(planned)} 个 cohort 仍为 planned（成交待确认），不得解读为已持仓")
    return _env(status="ok",
                detail={"n_active": agg.get("n_active"), "cash_weight": agg.get("cash_weight"),
                        "n_funds": agg.get("n_funds"), "active_cohorts": agg.get("active_cohorts"),
                        "cohorts": cohorts_meta,
                        "data_cutoff": (ref or {}).get("data_cutoff"),
                        "reference_snapshot": (ref or {}).get("run_id")},
                data_cutoff=(ref or {}).get("data_cutoff"),
                source_snapshot=os.path.join(SNAPSHOTS_DIR, (ref or {}).get("run_id", "")) if ref else None,
                warnings=warnings,
                provenance={"ledger": LEDGER_PATH})


def get_shadow_status():
    """影子状态（v1.1 P1-3）：以 _current_health 的 shadow_stale/shadow_factors 为权威判定；
    最近影子以最新完整快照内 shadow CSV 为准（其次才看 ml/scores）。"""
    h = _current_health()
    fac = h.get("shadow_factors", {}) or {}
    stale = list(h.get("shadow_stale") or [])
    snaps = _complete_snapshots()
    last = None
    if snaps:
        p = _shadow_csv(snaps[-1])
        if p:
            last = os.path.basename(p)
    if last is None and os.path.isdir(SCORES_DIR):
        cands = sorted([f for f in os.listdir(SCORES_DIR) if f.startswith("shadow_")], reverse=True)
        if cands:
            last = cands[0]
    detail = {"factors": fac, "stale": stale, "last_score": last,
              "reference_snapshot": snaps[-1]["run_id"] if snaps else None,
              "skip_reason": ("因子 stale：" + ",".join(stale)) if stale else None}
    if stale:
        return _env(status="unavailable", detail=detail, warnings=[],
                    errors=[_err("shadow_factor_stale", "最近影子评分不存在（因子陈旧，已按规则跳过）")],
                    provenance={"factors": fac})
    return _env(status="ok", detail=detail, warnings=[], errors=[],
                provenance={"factors": fac})


def compare_snapshots(run_a, run_b):
    """历史快照比较：显式 run_id，不走当前健康门禁（历史审计）。"""
    snaps = _complete_snapshots()
    by_id = {s["run_id"]: s for s in snaps}
    if run_a not in by_id or run_b not in by_id:
        return _env(status="unavailable",
                    errors=[_err("no_snapshot_for_run", "至少一个 run_id 不是完整快照")])
    a, b = by_id[run_a]["manifest"], by_id[run_b]["manifest"]
    fields = ["data_cutoff", "signal_date", "eligible_size", "main_size", "low_size",
              "data_health_status", "score_file_sha256", "execution_date", "status"]
    diff = {}
    for f in fields:
        va, vb = a.get(f), b.get(f)
        if va != vb:
            diff[f] = [va, vb]
    return _env(status="ok", detail={"diff": diff, "historical": True},
                warnings=["historical_audit：快照比较属历史审计，不是当前结论"],
                provenance={"compared": [os.path.join(SNAPSHOTS_DIR, run_a),
                                         os.path.join(SNAPSHOTS_DIR, run_b)], "historical": True})


def get_risk_scenario_v1():
    """无参数：读已冻结的 15% 波动目标结果（vol_target 报告 + 仓位统计）。"""
    if not os.path.exists(VOLTARGET_REPORT):
        return _env(status="unavailable",
                    errors=[_err("missing_report", "vol_target 冻结结果报告缺失")])
    with open(VOLTARGET_REPORT, encoding="utf-8") as f:
        txt = f.read()
    m = re.search(r"波动目标动量\s+([+-]?\d+\.\d+)%\s+([+-]?\d+\.\d+)%\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)%", txt)
    pos = re.search(r"仓位统计（波动目标动量）：均值 ([0-9.]+)｜中位 ([0-9.]+)｜最低 ([0-9.]+)｜满仓月占比 ([0-9.]+)%", txt)
    if not m:
        return _env(status="unavailable", errors=[_err("parse_fail", "无法解析冻结 15% 结果")])
    detail = {"frozen_target_vol": 0.15, "ann_ret": float(m.group(1)) / 100,
              "vol": float(m.group(2)) / 100, "sharpe": float(m.group(3)),
              "mdd": float(m.group(4)) / 100, "scenario": True}
    if pos:
        detail["position"] = {"mean": float(pos.group(1)), "median": float(pos.group(2)),
                              "min": float(pos.group(3)),
                              "full_month_ratio": float(pos.group(4)) / 100}
    return _env(status="ok",
                warnings=["scenario=true：本结果仅作情景分析，不构成策略建议；vol_target 不转正"],
                detail=detail, provenance={"source": VOLTARGET_REPORT})


REPORT_TOPICS = ("momentum_excess", "data_status", "shadow_status")


def generate_research_report(topic="momentum_excess"):
    if topic not in REPORT_TOPICS:
        return _env(status="not_executable",
                    errors=[_err("research_boundary", f"topic 仅允许 {REPORT_TOPICS}")])
    lines = [f"# Agent 研究报告（topic={topic}）", "",
             f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}"]
    src = []
    if topic in ("momentum_excess", "data_status"):
        gate = _health_gate() if topic == "data_status" else None   # data_status 是"当前状态报告"
        if gate:
            return gate
        snaps = _complete_snapshots()
        if snaps:
            s = snaps[-1]
            m = s["manifest"]
            lines += [f"- 最新正式快照：{s['run_id']}（signal={m.get('signal_date')}，"
                      f"eligible={m.get('eligible_size')}，status={m.get('status')}）",
                      f"- git_commit_sha：{str(m.get('git_commit_sha'))[:12]}",
                      f"- score_file_sha256：{str(m.get('score_file_sha256'))[:12]}"]
            src.append(s["dir"])
        if topic == "data_status":
            h = _current_health()
            lines.append(f"- 当前数据健康：{h.get('status')}（score_ready={h.get('score_ready')}，"
                         f"stale={h.get('stale_n')}，gap={h.get('gap_n')}）")
        else:
            lines += ["- 说明：主策略为 ret_12m 动量 Top50（冻结）；费用后相对可执行全池超额未见可靠统计证据（Phase 3 冻结记录）。"]
    if topic == "shadow_status":
        st = get_shadow_status()
        lines.append(f"- shadow 状态：{st['status']}（{st.get('detail', {}).get('skip_reason') or '可用'}）")
    return _env(status="ok", detail={"topic": topic, "report": "\n".join(lines)},
                provenance={"sources": src} if src else None)


# ---------------------------------------------------------------- 写工具（仅 1 个，无参数）
def run_production_pipeline():
    """无参数：运行**完整**正式流水线（不允许 skip/as_of/策略参数）。"""
    from production_pipeline import run_pipeline
    try:
        res = run_pipeline()          # 完整路径：skip_refresh=False, skip_clean=False, as_of=None
    except Exception as e:            # noqa: BLE001
        return _env(status="aborted", warnings=[],
                    errors=[_err("run_aborted", f"关键步骤失败：{e}")],
                    provenance={"aborted_at": datetime.now().isoformat(timespec="seconds")})
    m = res["manifest"]
    return _env(status="ok", run_id=m.get("run_id"), data_cutoff=m.get("data_cutoff"),
                source_snapshot=res["snapshot"],
                warnings=["shadow 因子 stale，影子评分已跳过"] if m.get("shadow_stale") else [],
                provenance={"git_commit_sha": m.get("git_commit_sha"),
                            "script_hashes": m.get("script_hashes"),
                            "cohort_status": (m.get("cohort_action") or {}).get("cohort_status")},
                detail={"health": m.get("data_health_status"), "score": m.get("eligible_size"),
                        "shadow": "skipped(stale)" if m.get("shadow_stale") else "ok",
                        "portfolio": (m.get("cohort_action") or {}).get("cohort_status")})


TOOLS = {
    # 只读
    "get_data_health": get_data_health,
    "get_latest_complete_snapshot": get_latest_complete_snapshot,
    "get_score": get_score,
    "get_top_funds": get_top_funds,
    "get_fund_rank_history": get_fund_rank_history,
    "get_portfolio_state": get_portfolio_state,
    "get_shadow_status": get_shadow_status,
    "compare_snapshots": compare_snapshots,
    "get_risk_scenario_v1": get_risk_scenario_v1,
    "generate_research_report": generate_research_report,
    # 写（用户触发、无参数）
    "run_production_pipeline": run_production_pipeline,
}

# 只读工具清单（用于只读性测试与权限矩阵）
READ_ONLY_TOOLS = [t for t in TOOLS if t != "run_production_pipeline"]


def call(tool: str, caller: str = DEFAULT_CALLER, **inputs):
    """统一调用入口（薄工具层对外唯一接口）。
    caller：调用者身份（对话适配层注入；当前默认 local_user）。
    所有出口（含 unknown_tool / 非法参数）**统一进入审计**（v1.1 P1-9）。
    """
    t0 = datetime.now()
    if tool not in TOOLS:
        _audit(tool, inputs, "not_executable", 0, caller)
        return {"status": "not_executable", "run_id": None, "data_cutoff": None,
                "strategy_version": STRATEGY_VERSION, "source_snapshot": None,
                "warnings": [], "errors": [_err("unknown_tool", f"未注册的工具：{tool}")],
                "provenance": None, "detail": None}
    try:
        # 只允许 TOOLS 签名接受的参数（strategy 参数/模式开关一律 TypeError → research_boundary）
        out = TOOLS[tool](**inputs)
    except TypeError as e:
        out = _env(status="not_executable", warnings=[],
                   errors=[_err("research_boundary", f"参数不被接受（工具签名不含该参数）：{e}")],
                   provenance=None)
    except Exception as e:  # noqa: BLE001
        out = _env(status="error", warnings=[],
                   errors=[_err("tool_error", str(e)[:200])], provenance=None)
    _audit(tool, inputs, out.get("status"),
           int((datetime.now() - t0).total_seconds() * 1000), caller)
    return out


if __name__ == "__main__":
    import sys
    import json as _json
    tool = sys.argv[1] if len(sys.argv) > 1 else "get_data_health"
    kwargs = {}
    if len(sys.argv) > 2:
        kwargs = _json.loads(sys.argv[2])
    print(_json.dumps(call(tool, **kwargs), ensure_ascii=False, indent=2, default=str))