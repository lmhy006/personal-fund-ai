# production_pipeline.py：Phase 3.5 统一生产流水线（Research → Production）
#
# 目标：让"冻结的策略"（原动量 ret_12m Top50、6 个月持续、每月先收后调、申赎费口径）每天/每月
#       按与研究阶段一致的口径**稳定、可重复、幂等、失败显式**地运行。
# 本轮不新增任何历史策略搜索、不修改冻结参数（TOP_N/HOLD/费用/目标波动率等）。
#
# 顺序（任一步骤失败/FAIL → 后续评分必须中止，不拿旧数据静默输出）：
#   1. benchmark 更新（沪深300，data_loader 新浪接口）
#   2. raw 全市场更新（daily_update：1 次请求 + 按日追加；gap/stale/catch-up 统计）
#   3. full 模式刷新 fund_history（现存）+ 清盘基金目录刷新（fund_history_delisted）
#   4. data_health 检查（PASS/WARN/FAIL；**FAIL 必须中止**）
#   5. live_score（生成 ml/scores/YYYY-MM.csv，与历史口径自动对账）
#   6. shadow_score（**若风格/行业因子 stale 则跳过并记录原因**，不用旧因子静默评分）
#   7. live_portfolio（用本月 Top50 推进 6-cohort ledger）
#   8. 不可变 snapshot（ml/snapshots/{run_id}/：manifest + 评分/影子/ledger/健康报告副本 + 哈希）
#
# 时间语义（明确定义；无证据的细节列为 production limitation，见 README Phase 3.5）：
#   data_cutoff        评分用的数据截止日（= 基准最新交易日）
#   signal_date        信号产生日（= data_cutoff，用 ≤ 该日的净值计算）
#   score_generated_at 评分实际生成时刻（datetime.now）
#   execution_date     调仓可执行日（= 信号日后下一个基准交易日；**生产假设**，真实申购按基金
#                      公司/销售机构规则确认，属 limitation，不猜）
import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
SNAPSHOTS_DIR = os.path.join(PROJECT_ROOT, "ml", "snapshots")
SCORES_DIR = os.path.join(PROJECT_ROOT, "ml", "scores")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
STRATEGY_VERSION = "momentum-v1"          # 冻结主策略版本标识（ret_12m Top50 等权，6 月持）
TOP_N = 50
# 生产脚本清单：manifest 记录其 sha256，保证"即使工作区 dirty（代码未提交），快照也能复现
# 当时代码内容"（2026-09-21 用户 P0：HEAD SHA 在脏工作区不足以复现）
PROD_SCRIPTS = ["production_pipeline.py", "data_health.py", "live_portfolio.py",
                "daily_update.py", "live_score.py", "shadow_score.py",
                "backtest_strategy.py", "data_loader.py", "clean_nav.py", "panel_builder.py"]


# ---------------------------------------------------------------- 工具
def git_head_sha() -> str:
    """读取 .git/HEAD 解析当前 commit SHA（不 spawn git）。"""
    try:
        p = os.path.join(PROJECT_ROOT, ".git", "HEAD")
        with open(p, "r") as f:
            head = f.read().strip()
        if head.startswith("ref:"):
            ref = head[5:].strip()
            with open(os.path.join(PROJECT_ROOT, ".git", ref.replace("/", os.sep))) as f:
                return f.read().strip()[:40]
        return head
    except Exception:  # noqa: BLE001
        return "unknown"


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def script_hashes() -> dict:
    """生产脚本 sha256（复现依据：工作区 dirty 时以脚本哈希为准）。"""
    out = {}
    for fn in PROD_SCRIPTS:
        p = os.path.join(SRC_DIR, fn)
        if os.path.exists(p):
            out[fn] = file_sha256(p)
    return out


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[pipeline {ts}] {msg}", flush=True)


def bench_latest() -> pd.Timestamp | None:
    p = os.path.join(PROJECT_ROOT, "data", "raw", "benchmark_hs300.csv")
    try:
        return pd.Timestamp(pd.read_csv(p, parse_dates=["date"])["date"].max())
    except Exception:  # noqa: BLE001
        return None


def next_trading_day(d: pd.Timestamp) -> str:
    from panel_builder import BENCH_PATH
    bc = pd.read_csv(BENCH_PATH, parse_dates=["date"])["date"]
    after = bc[bc > d]
    return str(pd.Timestamp(after.iloc[0]).date()) if len(after) else None


# ---------------------------------------------------------------- 步骤
def step_benchmark():
    log("步骤1/8 更新沪深300基准…")
    from data_loader import load_benchmark_hs300
    df = load_benchmark_hs300(use_cache=False)
    if df is None:
        raise RuntimeError("benchmark 更新失败")
    log(f"  benchmark 最新日 {df['date'].max().date()}（{len(df)} 行）")


def step_raw():
    log("步骤2/8 更新全市场基金 raw（daily_update）…")
    from daily_update import run_daily_update
    stats = run_daily_update(dry_run=False)
    log(f"  {stats}")


def step_clean():
    log("步骤3/8 full 模式刷新 fund_history + 清盘基金目录…")
    from clean_nav import clean_all
    log("  （现存池 full 清洗，**用 fund_code_list.csv 限定**——raw 目录里还有清盘基金缓存，"
        "不能混入现存 fund_history；原子 swap，约 10-20 分钟）")
    clean_all(pool_file="fund_code_list.csv", window="full")
    # 清盘基金独立目录刷新（与现存池分离，用名单）
    from delisted_funds import DELISTED_HISTORY_DIR, FUND_PATH, clean_history
    if os.path.exists(FUND_PATH):
        codes = list(pd.read_csv(FUND_PATH, dtype={"fund_code": str})["fund_code"])
        clean_history(codes)
        log(f"  清盘池刷新完成：{DELISTED_HISTORY_DIR}")
    else:
        log("  ⚠️ 未找到清盘名单，跳过清盘池刷新（不影响主策略评分）")


def step_health() -> dict:
    log("步骤4/8 数据健康检查…")
    from data_health import health_report
    rep = health_report()
    log(f"  data_health 状态：{rep['status']}（score_ready={rep['score_ready']}）")
    return rep


def step_live_score(as_of: pd.Timestamp | None) -> tuple:
    log("步骤5/8 live_score（主策略 ret_12m Top50，分层/eligibility 与研究同口径）…")
    from live_score import score_cross_section, verify_against_panel
    df, skip, t_date = score_cross_section(as_of)
    n_main = int((df["confidence"] == "main").sum()) if len(df) else 0
    n_low = int((df["confidence"] == "low").sum()) if len(df) else 0
    log(f"  评分日 {t_date.date()} | 可评分 {len(df)}（主 {n_main} / 低置信度 {n_low}）"
        f" | 排除 {skip}")
    os.makedirs(SCORES_DIR, exist_ok=True)
    path = os.path.join(SCORES_DIR, f"{t_date.strftime('%Y-%m')}.csv")
    # 不可变性：覆盖既有评分前，先把旧版备份到 snapshots/legacy（保留"那时系统看见了什么"）
    if os.path.exists(path):
        bak = os.path.join(SNAPSHOTS_DIR, "legacy", os.path.basename(path))
        os.makedirs(os.path.dirname(bak), exist_ok=True)
        if not os.path.exists(bak):
            import shutil
            shutil.copy2(path, bak)
            log(f"  旧版评分已备份：{bak}")
    df.to_csv(path, index=False)
    log(f"  已落盘 {path}（自动对账见下）")
    verify = {}
    try:
        verify_against_panel(df, t_date)     # 打印对账；面板无同截面时打印 skip（记录原因）
        verify["note"] = "verify_against_panel called; panel 截面可能早于评分日（研究面板止于标签完整处）"
    except Exception as e:  # noqa: BLE001
        verify["error"] = str(e)[:200]
    return df, t_date, path, {"main": n_main, "low": n_low, "skip": skip, "verify": verify}


def step_shadow_score(health: dict, data_cutoff: pd.Timestamp) -> tuple | None:
    log("步骤6/8 shadow_score（两条中性化影子，仅记录不参与资金决策）…")
    stale = health.get("shadow_stale") or []
    if stale:
        log(f"  ⚠️ shadow 因子 stale：{stale} → **跳过影子评分**（绝不用旧因子静默出最新影子）")
        return None, stale
    from shadow_score import shadow_scores
    sh, main, t_date = shadow_scores()
    os.makedirs(SCORES_DIR, exist_ok=True)
    path = os.path.join(SCORES_DIR, f"shadow_{t_date.strftime('%Y-%m')}.csv")
    sh.to_csv(path, index=False)
    log(f"  影子评分已落盘 {path}（{len(sh)} 只）")
    return path, None


def step_portfolio(scores_df: pd.DataFrame, t_date: pd.Timestamp, run_id: str):
    log(f"步骤7/8 live_portfolio（6-cohort ledger，方案 A 逐步建仓；score_run={run_id}）…")
    from live_portfolio import LivePortfolio
    pf = LivePortfolio()
    # P0 修复（2026-09-22 用户审查）：正式调用必须传入本次 run_id 作为 score_run，
    # 使组合台账可追溯到权威 run_id（同月 planned 更新时不会保留旧来源）
    action = pf.add_month(scores_df, score_run=run_id)
    agg = pf.aggregate()
    log(f"  本月动作：{json.dumps(action, ensure_ascii=False)}")
    log(f"  当前：{json.dumps({k: agg[k] for k in ('n_active', 'cash_weight', 'n_funds')}, ensure_ascii=False)}")
    return pf, action, agg


def step_snapshot(run_id, manifest: dict, scores_path, shadow_path, health,
                  pf, action, agg, top50, test_mode: bool = False):
    log("步骤8/8 不可变 snapshot…")
    snap = os.path.join(SNAPSHOTS_DIR, run_id)
    os.makedirs(snap, exist_ok=True)
    # 文件副本
    import shutil
    shutil.copy2(scores_path, os.path.join(snap, os.path.basename(scores_path)))
    if shadow_path:
        shutil.copy2(shadow_path, os.path.join(snap, os.path.basename(shadow_path)))
    shutil.copy2(pf.ledger_path, os.path.join(snap, "portfolio_ledger.csv"))
    shutil.copy2(pf.state_path, os.path.join(snap, "portfolio_state.json"))
    with open(os.path.join(snap, "data_health.json"), "w", encoding="utf-8") as f:
        json.dump(health, f, ensure_ascii=False, indent=2, default=str)
    top50.to_csv(os.path.join(snap, "top50.csv"), index=False)
    manifest["score_file_sha256"] = file_sha256(scores_path)
    manifest["portfolio"] = agg
    manifest["cohort_action"] = action
    manifest["top50"] = top50["fund_code"].tolist()
    # 测试模式（含 skip 开关的诊断/重试运行）不得写 COMPLETE（v1.1 用户审查）：
    #   正式 forward 记录必须来自完整流程
    if test_mode:
        manifest["status"] = "test"
        with open(os.path.join(snap, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
        with open(os.path.join(snap, "NOT_COMPLETE_TEST"), "w", encoding="utf-8") as f:
            f.write(f"{manifest['run_id']} test_run_at={manifest['score_generated_at']}\n"
                    f"原因：本次运行含 skip 开关（诊断/重试），非正式 forward 记录\n")
        log(f"  ⚠️ snapshot 已写入 {snap}（**NOT_COMPLETE_TEST**：含 skip 开关，非正式记录）")
        return snap
    manifest["status"] = "complete"      # 只有全部步骤成功、快照落盘后才置 complete
    with open(os.path.join(snap, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    # COMPLETE 标志：与 manifest.status=complete 一致；缺失即"半次正式运行"，可用 .bak/legacy 回滚
    with open(os.path.join(snap, "COMPLETE"), "w", encoding="utf-8") as f:
        f.write(f"{manifest['run_id']} completed_at={manifest['score_generated_at']}\n")
    log(f"  snapshot 已写入 {snap}（COMPLETE 标志已落盘）")
    return snap


def run_pipeline(skip_refresh: bool = False, skip_clean: bool = False,
                 as_of: str | None = None) -> dict:
    """可编程生产入口（Phase 4 Agent 薄工具层只调用**完整路径**：skip_* 均 False、as_of=None）。

    :return: {"run_id", "snapshot", "manifest", "status"}；任一关键步骤失败抛异常（调用方
             捕获后返回 aborted）。skip_* 用于 CLI 诊断/重试，对应快照只写 NOT_COMPLETE_TEST。
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log(f"run_id={run_id}（commit={git_head_sha()[:8]}）")

    health = None
    scores_path = None
    if not skip_refresh:
        step_benchmark()
        step_raw()
        if not skip_clean:
            step_clean()
    else:
        log("（skip_refresh：跳过数据刷新）")

    health = step_health()
    if health["status"] == "FAIL":
        raise RuntimeError(f"data_health FAIL（{health.get('shadow_stale', '')}）→ 中止评分")

    as_of_ts = pd.Timestamp(as_of) if as_of else None
    scores_df, t_date, scores_path, score_meta = step_live_score(as_of_ts)
    data_cutoff = t_date
    shadow_path, shadow_stale = step_shadow_score(health, data_cutoff)
    pf, action, agg = step_portfolio(scores_df, t_date, run_id)

    main = scores_df[scores_df["confidence"] == "main"] if "confidence" in scores_df.columns else scores_df
    top50 = main.sort_values("rank").head(TOP_N) if "rank" in main.columns else \
        main.sort_values("score", ascending=False).head(TOP_N)
    manifest = {
        "run_id": run_id, "git_commit_sha": git_head_sha(),
        "strategy_version": STRATEGY_VERSION,
        "data_cutoff": str(data_cutoff.date()),
        "benchmark_cutoff": str(health["benchmark_latest_date"]),
        "factor_cutoff": health.get("shadow_factors"),
        "signal_date": str(data_cutoff.date()),
        "score_generated_at": datetime.now().isoformat(timespec="seconds"),
        # 基准日历止于 data_cutoff 时无下一交易日 → None（null/pending），不得退回 signal_date
        "execution_date": next_trading_day(data_cutoff),
        "script_hashes": script_hashes(),
        "universe_size": health.get("processed_funds") or health.get("raw_funds"),
        "eligible_size": int(len(scores_df)),
        "main_size": score_meta["main"], "low_size": score_meta["low"],
        "score_meta": score_meta, "shadow_stale": shadow_stale,
        "data_health_status": health["status"],
        "execution_semantics": "signal_date=评分日；execution_date=信号日后下一基准交易日；"
                               "基准日历止于 data_cutoff 时 execution_date=null（cohort 保持 "
                               "planned/pending，成交日确认后才 active，见 live_portfolio 的 "
                               "confirm_execution）",
    }
    snap = step_snapshot(run_id, manifest, scores_path, shadow_path, health, pf, action, agg, top50,
                         test_mode=bool(skip_refresh or skip_clean))
    with open(os.path.join(LOG_DIR, "production_pipeline_latest.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n")
    log("✅ 流水线完成；正式评分与快照已生成"
        if manifest["status"] == "complete" else
        "⚠️ 流水线完成，但本次为**测试运行**（含 skip 开关），快照已标 NOT_COMPLETE_TEST，非正式记录")
    log(f"  下一步：Agent 可读取 {snap}/manifest.json 复原'当时系统看见了什么'")
    return {"run_id": run_id, "snapshot": snap, "manifest": manifest,
            "status": manifest["status"]}


def main():
    ap = argparse.ArgumentParser(description="Phase 3.5 统一生产流水线（一条命令：数据刷新→正式评分→组合→快照）")
    ap.add_argument("--as-of", default=None, help="评分日 YYYY-MM-DD；缺省=净值最新交易日")
    ap.add_argument("--skip-refresh", action="store_true", help="跳过步骤1-3（只评分+组合+快照，测试/重试用）")
    ap.add_argument("--skip-clean", action="store_true", help="跳过 full 清洗（数据未变时的快速路径）")
    args = ap.parse_args()

    try:
        run_pipeline(args.skip_refresh, args.skip_clean, args.as_of)
    except Exception as e:  # noqa: BLE001
        log(f"❌ 流水线中止：{e}")
        traceback.print_exc()
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(os.path.join(LOG_DIR, "production_pipeline_abort.log"), "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} ABORT: {e}\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()