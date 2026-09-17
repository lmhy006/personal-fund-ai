# young_fund_check.py：不足一年基金方案——短窗口动量在年轻段的预测力检验
#
# 背景：主策略（近一年收益排序）要求过去 252 个交易日，因此成立 <12 个月的基金无法评分；
#   现行研究面板（panel_builder）的 eligibility 要求成立 ≥365 自然日，把这些样本整体排除，
#   所以"新基金能否评分"一直没有数据支撑。
#
# 本脚本独立构建"年轻基金样本"（不触碰已冻结的 ml/panel.parquet 与主策略）：
#   - 纳入下限：成立 ≥90 自然日（约 3 个月，可算 ret_3m）
#   - 特征：可得窗口动量 ret_1m/3m/6m/12m（窗口交易日数不足即 NaN，不凭空造历史）
#   - 覆盖门槛：窗口内有效披露率 ≥95%（分子为有效观测，分母为窗口交易日数）
#   - 标签：未来 126 交易日收益（与主口径一致，标签端 15 天内需有披露）
# 按年龄段分组算月度 Rank IC 并做 NW(6) 判定：
#   <6月 / 6-12月 / 12-36月 / ≥36月（对照）
#
# 结论用途：若 <12 月段某短窗口动量有显著正向 IC → 可给不足一年基金**低置信度**评分；
#   若无信号或样本不足 → 明确"证据不足、暂不评分"，不编造分数。
import os

import numpy as np
import pandas as pd
from scipy import stats

from panel_builder import (load_fund_series, month_end_trading_days, BENCH_PATH,
                           PROJECT_ROOT, MAX_GAP_DAYS, DAY_NS, MONTH_NS, LABEL_HORIZON)
from walk_forward_splitter import nw_tstat

OUT_DIR = os.path.join(PROJECT_ROOT, "ml", "experiments")
OUT_MONTHLY = os.path.join(OUT_DIR, "young_fund_check_monthly.csv")

WINDOWS = {"ret_1m": 21, "ret_3m": 63, "ret_6m": 126, "ret_12m": 252}
MIN_AGE_DAYS = 90        # 纳入下限（约 3 个月）
COVERAGE = 0.95          # 窗口内披露完整度门槛
MIN_MONTH_ROWS = 30      # 单个(月, 年龄组)横截面算 IC 的最小样本
MIN_MONTHS_FOR_NW = 12   # 至少要有这么多有效月才做 NW 判定


def age_bucket(age_months: float) -> str:
    if age_months < 6:
        return "1_lt6m"
    if age_months < 12:
        return "2_6to12m"
    if age_months < 36:
        return "3_12to36m"
    return "4_ge36m"


def build_young_samples(max_date: str = "2025-02-28") -> pd.DataFrame:
    """构建含年轻基金的样本长表（不使用 panel 的 eligibility）。

    max_date 默认 dev 段末尾（2025-02-28）：holdout 段已开启并冻结主策略，
    本检验的结论用于**上线评分规则**（属策略选择），不得用 holdout 段挑选。"""
    max_ns = np.datetime64(pd.Timestamp(max_date), "ns").astype("int64")
    bench = pd.read_csv(BENCH_PATH, parse_dates=["date"]).sort_values("date")
    bench_dates = bench["date"].to_numpy(dtype="datetime64[ns]").astype("int64")
    month_ends = [t for t in month_end_trading_days(bench_dates) if t <= max_ns]
    series = load_fund_series(bench_dates)
    print(f"全历史基金 {len(series)} 只 | 基准交易日 {len(bench_dates)} | "
          f"截面 {len(month_ends)}（截至 {max_date}，dev 段）")
    rows = []
    for t_ns in month_ends:
        i_t = int(np.searchsorted(bench_dates, t_ns, side="right")) - 1
        if i_t < max(WINDOWS.values()):
            continue
        i_end = i_t + LABEL_HORIZON
        if i_end >= len(bench_dates):
            continue  # 标签不可得（只保留有完整未来半年的截面，与面板一致）
        label_end_ns = int(bench_dates[i_end])
        for code, s in series.items():
            dates = s["dates"]
            first_ns = int(dates[0])
            age_days = t_ns - first_ns
            if age_days < MIN_AGE_DAYS * DAY_NS:
                continue  # 成立不足 3 个月：数据太少，不纳入
            j_t = int(np.searchsorted(dates, t_ns, side="right")) - 1
            if j_t < 0 or t_ns - int(dates[j_t]) > MAX_GAP_DAYS * DAY_NS:
                continue  # 当期无披露
            j_end = int(np.searchsorted(dates, label_end_ns, side="right")) - 1
            if j_end <= j_t or label_end_ns - int(dates[j_end]) > MAX_GAP_DAYS * DAY_NS:
                continue  # 标签期末端无披露
            i_first = int(np.searchsorted(bench_dates, first_ns, side="left"))
            row = {"fund_code": code, "t_date": pd.Timestamp(t_ns, unit="ns"),
                   "age_months": round(age_days / MONTH_NS, 1),
                   "age_bucket": age_bucket(age_days / MONTH_NS),
                   "y": float(s["wealth_al"][i_end] / s["wealth_al"][i_t] - 1.0)}
            for name, k in WINDOWS.items():
                if i_t - i_first + 1 < k + 1:      # 成立历史不足 k 个交易日 → 该窗口不可得
                    row[name] = np.nan
                    continue
                wr = s["r_al"][i_t - k:i_t + 1]
                n_eff = int(np.sum(~np.isnan(wr)))
                row[name] = (float(s["wealth_al"][i_t] / s["wealth_al"][i_t - k] - 1.0)
                             if n_eff / (k + 1) >= COVERAGE else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def monthly_ic(df: pd.DataFrame, feat: str) -> pd.Series:
    """按月的横截面 Rank IC（样本不足的月跳过）。"""
    out = {}
    for t, g in df.groupby("t_date"):
        sub = g[[feat, "y"]].dropna()
        if len(sub) >= MIN_MONTH_ROWS and sub[feat].nunique() > 1 and sub["y"].nunique() > 1:
            out[t] = stats.spearmanr(sub[feat], sub["y"]).statistic
    return pd.Series(out)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="不足一年基金：短窗口动量预测力检验（dev 段）")
    ap.add_argument("--max-date", default="2025-02-28",
                    help="样本截止日（默认 dev 段末尾；holdout 段不得用于规则选择）")
    args = ap.parse_args()
    df = build_young_samples(args.max_date)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_MONTHLY, index=False)
    print(f"样本长表已落盘: {OUT_MONTHLY}（{len(df)} 行）")
    print("\n=== 样本分布（每年龄组的行数与特征可得率）===")
    for b in ["1_lt6m", "2_6to12m", "3_12to36m", "4_ge36m"]:
        sub = df[df.age_bucket == b]
        if not len(sub):
            print(f"  {b:<12s} 无样本")
            continue
        avail = {f: f"{sub[f].notna().mean():.0%}" for f in WINDOWS}
        print(f"  {b:<12s} 行数 {len(sub):>6d} | 基金 {sub.fund_code.nunique():>4d} 只 | "
              f"动量可得率 1m/3m/6m/12m = {avail['ret_1m']}/{avail['ret_3m']}/{avail['ret_6m']}/{avail['ret_12m']}")

    print("\n=== 各年龄段的短窗口动量 IC（NW(6) 判定；单月样本≥30 才计入）===")
    print(f"{'年龄组':<12s}{'特征':<9s}{'有效月':>7s}{'mean_IC':>10s}{'NW t':>8s}")
    verdict_rows = []
    for b in ["1_lt6m", "2_6to12m", "3_12to36m", "4_ge36m"]:
        sub = df[df.age_bucket == b]
        if not len(sub):
            continue
        for feat in WINDOWS:
            ics = monthly_ic(sub, feat)
            if len(ics) < MIN_MONTHS_FOR_NW:
                print(f"{b:<12s}{feat:<9s}{len(ics):>7d}{'—':>10s}{'样本不足':>8s}")
                continue
            t = nw_tstat(ics)
            print(f"{b:<12s}{feat:<9s}{len(ics):>7d}{ics.mean():>+10.4f}{t:>+8.2f}")
            verdict_rows.append({"age_bucket": b, "feature": feat, "months": len(ics),
                                 "mean_ic": float(ics.mean()), "nw_t": t})

    print("\n=== 结论（用于不足一年基金是否评分的决策）===")
    young = [r for r in verdict_rows if r["age_bucket"] in ("1_lt6m", "2_6to12m")]
    usable = [r for r in young if r["nw_t"] > 1.96]
    if not young:
        print("  <12 月段无足够样本做判定——当前开发池（今天满三年的基金）在年轻时样本量不足；")
        print("  需用全量 universe（补拉完成后）重跑本检验，才能得出可用的结论。")
    elif usable:
        best = max(usable, key=lambda r: r["mean_ic"])
        print(f"  存在可用信号：{best['age_bucket']} × {best['feature']} "
              f"IC={best['mean_ic']:+.4f}（NW {best['nw_t']:+.2f}, {best['months']} 月）")
        print("  → 不足一年基金可用该短窗口动量给**低置信度**评分（须标注窗口与年龄）。")
    else:
        print("  <12 月段各短窗口动量均未达显著正向——**不足以支持给不足一年基金评分**；")
        print("  上线时应明确标注'证据不足、暂不评分'，而不是用更短的窗口硬凑分数。")
    print("\n（注：本检验为上线覆盖研究，不改动已冻结的研究面板与主策略；NW(6) 覆盖标签重叠）")


if __name__ == "__main__":
    main()
