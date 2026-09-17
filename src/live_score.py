# live_score.py：上线评分管道——问题④的最后缺口（无未来标签、不检查未来端点）
#
# 与研究管道的关键区别（README_FOR_HUMAN 问题④）：
#   panel_builder（研究）：样本需要未来 126 交易日标签 → 截面止于 2026-02；
#                          eligibility 还要检查"标签期末 15 天内有披露"（未来端点在研究数据库里是已知的）
#   live_score（上线）：只用**过去** 252 交易日   → 可评到净值最新日（2026-09）；
#                          eligibility 只看成立时长、当期披露、历史 coverage——**不触碰任何未来信息**
#
# 主策略（2026-09-18 拍板）= 近一年收益排序：score = W(t)/W(t-252) − 1（W 为财富指数）
# eligibility（无未来依赖版，与 panel 同门槛但去掉未来端点检查）：
#   成立 ≥ 365 自然日 + 评分日附近 15 天内有披露 + 过去 252 交易日 coverage ≥ 95%
# 输出：ml/scores/YYYY-MM.csv（README 评分留存约定，从第一次正式评分起积累向前验证记录）
#
# 自检：--as-of 传入面板已有截面 + --verify-panel，与面板 ret_12m 逐基金对账——
# 保证上线管道与研究管道口径不漂移。
import argparse
import os

import numpy as np
import pandas as pd

from clean_nav import load_fund_name_map
from panel_builder import (load_fund_series, FEATURE_LOOKBACK, MIN_AGE_NATURAL_DAYS,
                           MIN_COVERAGE, MAX_GAP_DAYS, DAY_NS, MONTH_NS,
                           BENCH_PATH, PROJECT_ROOT)

SCORES_DIR = os.path.join(PROJECT_ROOT, "ml", "scores")
PANEL_PATH = os.path.join(PROJECT_ROOT, "ml", "panel.parquet")
FULL_HISTORY_AGE = 36  # 与 panel_builder / walk_forward_splitter 的分层常量一致


def score_cross_section(as_of=None) -> tuple:
    """对指定交易日（默认净值最新日）算横截面动量分数。返回 (评分表, 排除统计, 评分日)。

    严格只用 ≤ as_of 的信息：不读标签、不检查未来端点。"""
    bench = pd.read_csv(BENCH_PATH, parse_dates=["date"]).sort_values("date")
    bench_dates = bench["date"].to_numpy(dtype="datetime64[ns]").astype("int64")
    series = load_fund_series(bench_dates)
    name_map = load_fund_name_map()

    if as_of is None:
        i_t = len(bench_dates) - 1
    else:
        t_probe = np.datetime64(pd.Timestamp(as_of), "ns").astype("int64")
        i_t = int(np.searchsorted(bench_dates, t_probe, side="right")) - 1
    t_ns = int(bench_dates[i_t])
    t_date = pd.Timestamp(t_ns, unit="ns")
    if i_t < FEATURE_LOOKBACK:
        raise SystemExit(f"评分日 {t_date.date()} 过早：基准日历不足 {FEATURE_LOOKBACK} 个交易日")

    w0 = i_t - FEATURE_LOOKBACK
    rows, skip = [], {"reject_young": 0, "reject_stale": 0, "reject_low_coverage": 0,
                      "reject_no_window": 0}
    for code, s in series.items():
        dates = s["dates"]
        # ① 成立 ≥ 365 自然日（无未来依赖）
        if dates[0] > t_ns - MIN_AGE_NATURAL_DAYS * DAY_NS:
            skip["reject_young"] += 1
            continue
        # ② 评分当期有披露（近 15 自然日）——注意：只看向过去，不看未来端点
        j_t = int(np.searchsorted(dates, t_ns, side="right")) - 1
        if j_t < 0 or t_ns - int(dates[j_t]) > MAX_GAP_DAYS * DAY_NS:
            skip["reject_stale"] += 1
            continue
        # ③ 过去 252 交易日 coverage ≥ 95%（缺失日=未披露，NaN）
        window_r = s["r_al"][w0:i_t + 1]
        n_eff = int(np.sum(~np.isnan(window_r)))
        cov = n_eff / (i_t + 1 - w0)
        if cov < MIN_COVERAGE:
            skip["reject_low_coverage"] += 1
            continue
        window_w = s["wealth_al"][w0:i_t + 1]
        score = float(window_w[-1] / window_w[-1 - FEATURE_LOOKBACK] - 1.0) \
            if n_eff - 1 >= FEATURE_LOOKBACK else np.nan
        if np.isnan(score):
            skip["reject_no_window"] += 1
            continue
        age_months = (t_ns - int(dates[0])) / MONTH_NS
        rows.append({
            "fund_code": code,
            "fund_name": name_map.get(code, ""),
            "score": score,
            "age_months": round(age_months, 1),
            "age_group": "full" if age_months >= FULL_HISTORY_AGE else "short",
            "coverage": round(cov, 4),
            "as_of": t_date,
        })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df, skip, t_date


def verify_against_panel(df: pd.DataFrame, t_date: pd.Timestamp) -> None:
    """与面板同一截面逐基金对账：score 应等于面板 ret_12m（同口径）；行数差异来自
    panel 额外的"标签期末披露"检查（研究需要未来端点，上线不需要）——不对账行数。"""
    try:
        panel = pd.read_parquet(PANEL_PATH)
    except Exception as e:
        print(f"对账跳过（面板不可读）: {e}")
        return
    ref = panel[panel["t_date"] == t_date][["fund_code", "ret_12m"]]
    if not len(ref):
        print(f"对账跳过：面板无 {t_date.date()} 截面（面板截面止于标签完整处）")
        return
    m = df.merge(ref, on="fund_code", how="inner", suffixes=("", "_panel"))
    diff = (m["score"] - m["ret_12m"]).abs()
    print(f"\n=== 上线管道 vs 面板口径对账（{t_date.date()}）===")
    print(f"交集基金 {len(m)} 只 | 最大绝对差 {diff.max():.3e} | 平均绝对差 {diff.mean():.3e}")
    print(f"上线评分 {len(df)} 只 vs 面板 {len(ref)} 只——差集来自研究端额外的"
          f"'标签期末披露'检查（上线无未来端点，覆盖更宽）")
    if diff.max() < 1e-9:
        print("✅ 口径一致：上线管道与研究管道逐基金同值")
    else:
        print("⚠️ 存在差异，需检查窗口口径")


def main():
    ap = argparse.ArgumentParser(description="上线评分管道（主策略：近一年收益排序）")
    ap.add_argument("--as-of", default=None, help="评分日 YYYY-MM-DD，缺省=净值最新交易日")
    ap.add_argument("--verify-panel", action="store_true", help="与面板同截面逐基金对账")
    ap.add_argument("--no-save", action="store_true", help="只打印不落盘（对账/试验用）")
    args = ap.parse_args()

    df, skip, t_date = score_cross_section(args.as_of)
    print(f"=== 上线评分（主策略=近一年收益排序）| 评分日 {t_date.date()} ===")
    print(f"可评分基金 {len(df)} 只 | 排除：成立不足12月 {skip['reject_young']}、"
          f"当期停披露 {skip['reject_stale']}、coverage不足 {skip['reject_low_coverage']}、"
          f"窗口不足 {skip['reject_no_window']}")
    if len(df):
        g = df["age_group"].value_counts().to_dict()
        print(f"年龄构成：short(12-36月) {g.get('short', 0)} / full(≥36月) {g.get('full', 0)}"
              f"（排行榜须分年龄段披露——E4 实测 Top 组 short 占比 +4.95pp）")
        print("\nTop10：")
        print(df.head(10)[["rank", "fund_code", "fund_name", "score", "age_months", "age_group"]]
              .to_string(index=False))
    if not args.no_save and len(df):
        os.makedirs(SCORES_DIR, exist_ok=True)
        path = os.path.join(SCORES_DIR, f"{t_date.strftime('%Y-%m')}.csv")
        df.to_csv(path, index=False)
        print(f"\n评分快照已落盘: {path}（向前验证记录，README 评分留存约定）")
    if args.verify_panel:
        verify_against_panel(df, t_date)


if __name__ == "__main__":
    main()
