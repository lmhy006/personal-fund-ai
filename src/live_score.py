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

# —— Windows 输出编码兜底（2026-09-24）：管道/重定向 stdout 默认 GBK，print ✅/⚠️ 等符号会崩
import sys
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

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

    rows, skip = [], {"reject_not_born": 0, "reject_stale": 0,
                      "reject_young_insufficient": 0, "reject_low_coverage": 0}
    SHORT_SIGNAL, SHORT_WINDOW = "ret_6m", 126   # 6-12月段信号（young_fund_check 实定）
    for code, s in series.items():
        dates = s["dates"]
        first_ns = int(dates[0])
        # ⓪ 评分日尚未成立（历史截面回测时大量出现，与"停披露"区分统计）
        if first_ns > t_ns:
            skip["reject_not_born"] += 1
            continue
        # ① 评分当期有披露（只看向过去，不检查未来端点）
        j_t = int(np.searchsorted(dates, t_ns, side="right")) - 1
        if j_t < 0 or t_ns - int(dates[j_t]) > MAX_GAP_DAYS * DAY_NS:
            skip["reject_stale"] += 1
            continue
        # ② 分层评分（2026-09-17 young_fund_check.py 研究结论，dev 段 NW 判定）：
        #    ≥12月（252交易日窗口）→ 主策略 ret_12m，confidence="main"
        #    6-12月（126交易日窗口）→ ret_6m，confidence="low"（该段 IC +0.120、NW +2.95）
        #    <6月 → 证据不足（唯一可用 ret_1m：IC +0.073、NW +1.30 不显著）→ 明确不评分
        i_first = int(np.searchsorted(bench_dates, first_ns, side="left"))
        trad_hist = i_t - i_first + 1
        signal = window = conf = None
        cov = np.nan
        for sig, k, c in (("ret_12m", FEATURE_LOOKBACK, "main"), (SHORT_SIGNAL, SHORT_WINDOW, "low")):
            if trad_hist < k + 1:      # 成立历史不足 k 个交易日 → 该窗口不可得
                continue
            n_eff = int(np.sum(~np.isnan(s["r_al"][i_t - k:i_t + 1])))
            c_ = n_eff / (k + 1)
            if c_ >= MIN_COVERAGE:
                signal, window, conf, cov = sig, k, c, c_
                break
        if signal is None:
            skip["reject_young_insufficient" if trad_hist < SHORT_WINDOW + 1
                 else "reject_low_coverage"] += 1
            continue
        score = float(s["wealth_al"][i_t] / s["wealth_al"][i_t - window] - 1.0)
        age_months = (t_ns - first_ns) / MONTH_NS
        rows.append({
            "fund_code": code,
            "fund_name": name_map.get(code, ""),
            "score": score,
            "signal": signal,
            "confidence": conf,
            "age_months": round(age_months, 1),
            "age_group": ("young" if age_months < 12 else
                          ("short" if age_months < FULL_HISTORY_AGE else "full")),
            "coverage": round(cov, 4),
            "as_of": t_date,
        })
    df = pd.DataFrame(rows)
    if len(df):
        # 主排行只在 confidence="main" 内排名（与 holdout 冻结口径一致）；
        # 低置信度组用**不同信号**（ret_6m），跨信号分数可比性未验证 → 只做组内排名、不与主排行混排
        df["rank"] = np.nan
        df["rank_lowconf"] = np.nan
        for mask, col in ((df["confidence"] == "main", "rank"),
                          (df["confidence"] == "low", "rank_lowconf")):
            idx = df.index[mask]
            if len(idx):
                order = df.loc[idx].sort_values("score", ascending=False).index
                df.loc[order, col] = np.arange(1, len(order) + 1)
        df = df.sort_values(["confidence", "score"], ascending=[True, False]).reset_index(drop=True)
        df = df[["rank", "rank_lowconf", "fund_code", "fund_name", "score", "signal",
                 "confidence", "age_months", "age_group", "coverage", "as_of"]]
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
    main = df[df["confidence"] == "main"] if "confidence" in df.columns else df
    m = main.merge(ref, on="fund_code", how="inner", suffixes=("", "_panel"))
    diff = (m["score"] - m["ret_12m"]).abs()
    print(f"\n=== 上线管道(主策略组) vs 面板口径对账（{t_date.date()}）===")
    print(f"交集基金 {len(m)} 只 | 最大绝对差 {diff.max():.3e} | 平均绝对差 {diff.mean():.3e}")
    print(f"主策略评分 {len(main)} 只 vs 面板 {len(ref)} 只——差集来自研究端额外的"
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
    n_main = int((df["confidence"] == "main").sum()) if len(df) else 0
    n_low = int((df["confidence"] == "low").sum()) if len(df) else 0
    print(f"=== 上线评分（分层：≥12月→主策略ret_12m / 6-12月→ret_6m低置信度 / <6月不评分）"
          f"| 评分日 {t_date.date()} ===")
    print(f"可评分 {len(df)} 只（主策略 {n_main} / 低置信度 {n_low}）| 排除："
          f"尚未成立 {skip['reject_not_born']}、当期停披露 {skip['reject_stale']}、"
          f"成立不足6月或窗口不足 {skip['reject_young_insufficient']}、"
          f"披露完整度不足 {skip['reject_low_coverage']}")
    if len(df):
        main_df = df[df["confidence"] == "main"]
        g = main_df["age_group"].value_counts().to_dict()
        print(f"主策略年龄构成：short(12-36月) {g.get('short', 0)} / full(≥36月) {g.get('full', 0)}"
              f"（排行榜须分年龄段披露——E4 实测 Top 组 short 占比 +4.95pp）")
        print("\nTop10（主策略）：")
        print(main_df.head(10)[["rank", "fund_code", "fund_name", "score", "age_months", "age_group"]]
              .to_string(index=False))
        if n_low:
            print(f"\n低置信度组（6-12月龄，信号 ret_6m，与主策略不同信号、不混排）前5：")
            print(df[df["confidence"] == "low"].head(5)[
                ["rank_lowconf", "fund_code", "fund_name", "score", "age_months"]].to_string(index=False))
    elif not n_main:
        print("（无可评分基金）")
    if not args.no_save and len(df):
        os.makedirs(SCORES_DIR, exist_ok=True)
        path = os.path.join(SCORES_DIR, f"{t_date.strftime('%Y-%m')}.csv")
        df.to_csv(path, index=False)
        print(f"\n评分快照已落盘: {path}（向前验证记录，README 评分留存约定）")
    if args.verify_panel:
        verify_against_panel(df, t_date)


if __name__ == "__main__":
    main()
