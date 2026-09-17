# holdout_final.py：主策略 holdout 终审（**一次性**——跑完即冻结主策略与评估口径）
#
# 纪律（2026-09-17 起冻结）：
#   主策略 = 近一年收益排序（ret_12m）；评估口径 = 对齐行（ret_12m 非缺失）+ NW(6) 判定；
#   publish_lag = 5 自然日；holdout = 末尾 12 个截面（2025-03~2026-02），dev 期从未触碰。
#   本脚本只允许运行一次；结果落盘留档，此后任何改动都记为"holdout 后变更"。
#
# 产出：① 逐月明细 ml/wf_splits/holdout_verdict_monthly.csv
#       ② 验收报告 ml/wf_splits/holdout_verdict.txt（含与 dev 段对照）
# 判定标准（事前定死）：holdout IC 方向是否与 dev 一致、是否出现负 IC 灾难性失效；
#   由于只有 12 个截面且标签重叠 5/6（有效样本≈3~5），显著性判定功效极低——
#   本终审的作用是"排查灾难性失效"，不是"确证有效性"。
import os

import numpy as np
import pandas as pd
from scipy import stats

from walk_forward_splitter import (WalkForwardSplitter, rank_ic, age_group,
                                   nw_tstat, MIN_TEST_ROWS_FOR_IC, LABEL_COL)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
PANEL_PATH = os.path.join(PROJECT_ROOT, "ml", "panel.parquet")
BENCH_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "benchmark_hs300.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "ml", "wf_splits")
DEV_BASELINE = os.path.join(OUT_DIR, "baseline_rankic.csv")
LABEL_HORIZON_TRADING = 126


def bench_future_6m() -> pd.Series:
    bench = pd.read_csv(BENCH_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    c = bench["close"].to_numpy(dtype=float)
    fr = np.full(len(c), np.nan)
    fr[:-LABEL_HORIZON_TRADING] = c[LABEL_HORIZON_TRADING:] / c[:-LABEL_HORIZON_TRADING] - 1.0
    return pd.Series(fr, index=bench["date"])


def run() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL_PATH)
    spl = WalkForwardSplitter(panel)
    bench = bench_future_6m()
    rows = []
    for fold in spl.folds("holdout"):
        t = fold.test
        scored = t[t["ret_12m"].notna()]          # 对齐行：主策略可评分（与研究评估口径一致）
        if len(scored) < MIN_TEST_ROWS_FOR_IC:
            continue
        g = scored["age_months"].map(age_group)
        ic, n = rank_ic(scored, "ret_12m")
        ic_sh, _ = rank_ic(scored, "sharpe_12m")
        ic_short, n_s = rank_ic(scored[g == "short"], "ret_12m")
        ic_full, n_f = rank_ic(scored[g == "full"], "ret_12m")
        k = max(1, int(round(len(scored) * 0.2)))
        top = scored.sort_values("ret_12m", ascending=False).head(k)
        top_ret = float(top[LABEL_COL].mean())
        pool = float(scored[LABEL_COL].mean())
        bot_ret = float(scored.sort_values("ret_12m").head(k)[LABEL_COL].mean())
        b6 = float(bench.reindex([fold.t]).iloc[0]) if fold.t in bench.index else float("nan")
        rows.append({
            "t_date": fold.t, "n": n, "n_short": n_s, "n_full": n_f,
            "ic_main": ic, "ic_sharpe": ic_sh, "ic_short": ic_short, "ic_full": ic_full,
            "top20_ret": top_ret, "bot20_ret": bot_ret, "pool_ret": pool, "bench_ret6m": b6,
            "top20_excess_hs300": top_ret - b6, "top20_excess_pool": top_ret - pool,
            "spread_top_bot": top_ret - bot_ret,
        })
    return pd.DataFrame(rows)


def verdict(df: pd.DataFrame) -> str:
    ic = df["ic_main"].dropna()
    n = len(ic)
    t_nw = nw_tstat(ic)
    p_nw = 2.0 * stats.t.sf(abs(t_nw), max(n - 1, 1)) if np.isfinite(t_nw) else float("nan")
    t_naive = ic.mean() / (ic.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")

    # dev 段对照（既有产物，不重跑）
    dev = pd.read_csv(DEV_BASELINE, parse_dates=["t_date"]) if os.path.exists(DEV_BASELINE) else None
    dev_ic = dev_all = dev_short = dev_full = float("nan")
    dev_months = 0
    if dev is not None:
        d_all = dev[(dev.feature == "ret_12m") & (dev.age_group == "all") & (dev.n >= 30)]
        dev_months = len(d_all)
        dev_all = float(d_all.ic.mean())
        dev_short = float(dev[(dev.feature == "ret_12m") & (dev.age_group == "short") & (dev.n >= 30)].ic.mean())
        dev_full = float(dev[(dev.feature == "ret_12m") & (dev.age_group == "full") & (dev.n >= 30)].ic.mean())

    ic_all = float(ic.mean())
    lines = []
    lines.append("=" * 74)
    lines.append("主策略 holdout 终审报告（一次性开启，2026-09-17）")
    lines.append("=" * 74)
    lines.append(f"主策略：近一年收益排序（ret_12m） | 评估口径：对齐行 + NW(6) | publish_lag=5天")
    lines.append(f"holdout 截面：{df.t_date.min().date()} ~ {df.t_date.max().date()}（{n} 个月末，dev 期从未触碰）")
    lines.append(f"每截面可评分基金：中位 {int(df.n.median())} 只")
    lines.append("")
    lines.append("【1】主策略 IC")
    lines.append(f"  holdout : mean IC = {ic_all:+.4f}  (std {ic.std():.4f}, IC>0 占比 {(ic > 0).mean():.2f}, {n} 个月)")
    lines.append(f"  dev 对照: mean IC = {dev_all:+.4f}  ({dev_months} 个月)")
    lines.append(f"  差异    : {ic_all - dev_all:+.4f}")
    lines.append(f"  显著性  : naive t = {t_naive:+.2f} | NW(6) t = {t_nw:+.2f} (p={p_nw:.3f})")
    lines.append(f"  参考    : 同段 sharpe 排序 IC = {df.ic_sharpe.mean():+.4f}")
    lines.append("")
    lines.append("【2】分年龄组（dev 对照：short +{:.4f} / full +{:.4f}）".format(dev_short, dev_full))
    lines.append(f"  short(12-36月): mean IC = {df.ic_short.mean():+.4f}（月度中位样本 {int(df.n_short.median())} 只）")
    lines.append(f"  full(≥36月)   : mean IC = {df.ic_full.mean():+.4f}（月度中位样本 {int(df.n_full.median())} 只）")
    lines.append("")
    lines.append("【3】Top20% 等权组合（6 个月收益，gross，标签重叠→统计偏乐观）")
    lines.append(f"  绝对收益   = {df.top20_ret.mean():.4%}   (全池均值 {df.pool_ret.mean():.4%} / 底20% {df.bot20_ret.mean():.4%})")
    lines.append(f"  相对沪深300 = {df.top20_excess_hs300.mean():+.4%}   (基准 {df.bench_ret6m.mean():.4%})")
    lines.append(f"  相对全池   = {df.top20_excess_pool.mean():+.4%}")
    lines.append(f"  Top−Bottom = {df.spread_top_bot.mean():+.4%}  (NW t={nw_tstat(df.spread_top_bot):+.2f})")
    lines.append(f"  dev 对照（对齐评估单因子 Ridge=动量）：绝对 9.41% / 相对300 +2.10% / 相对全池 +1.37%")
    lines.append("")
    lines.append("【4】判定（标准事前定死）")
    failure = (ic_all < 0) or (abs(ic_all) < abs(dev_all) * 0.3)
    same_sign = (ic_all > 0) == (dev_all > 0)
    lines.append(f"  方向一致性: {'一致（均为正）' if same_sign else '不一致——需警惕'}")
    lines.append(f"  灾难性失效: {'是——IC 转负或不足 dev 的 30%' if failure else '否'}")
    lines.append(f"  统计判定  : NW t={t_nw:+.2f} → " +
                 ("显著为正" if t_nw > 1.96 else ("显著为负" if t_nw < -1.96 else "不显著（证据不足，非失败）")))
    lines.append("")
    lines.append("【5】局限（必须随结论一起引用）")
    lines.append("  ① 仅 12 个截面且标签重叠 5/6 → 有效样本约 3~5，显著性判定的功效极低；")
    lines.append("     本终审用于排查灾难性失效，不能确证有效性；")
    lines.append("  ② 组合层标签重叠 → 收益统计偏乐观，未计费用/换手（属 Phase 3）；")
    lines.append("  ③ 仍是现存池条件性研究（无已清盘基金），不代表当年可执行策略；")
    lines.append("  ④ 本报告开启后，主策略与评估口径冻结：任何后续变更须标注为 holdout 后变更。")
    lines.append("=" * 74)
    report = "\n".join(lines)
    print(report)
    return report


def main():
    df = run()
    if not len(df):
        raise SystemExit("holdout 无可用折——检查切分器 holdout 配置")
    os.makedirs(OUT_DIR, exist_ok=True)
    mpath = os.path.join(OUT_DIR, "holdout_verdict_monthly.csv")
    df.to_csv(mpath, index=False)
    print(f"逐月明细已落盘: {mpath}（{len(df)} 折）")
    report = verdict(df)
    rpath = os.path.join(OUT_DIR, "holdout_verdict.txt")
    with open(rpath, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\n验收报告已留档: {rpath}")
    print("⚠️ holdout 已开启：主策略与评估口径就此冻结，不得再用 holdout 反复挑方案。")


if __name__ == "__main__":
    main()
