# shadow_score.py：两条中性化影子策略的正式评分留存（主策略不变，影子只记录、不用于资金决策）
#
# 背景（2026-09-19 用户拍板）：
#   第二组对照发现——规模风格中性（含清盘池后配对 NW +1.82、历史最长 218 月）与行业中性
#   （NW +1.73，波动/回撤略好）相对**原动量**方向为正但未达 5% 显著，且相对**可执行全池**
#   并不显著为正。因此**主策略仍为原动量（ret_12m）不更换**，这两条只作为**影子候选**从
#   本月起留存评分，等待向前积累数据（6 个月标签真正成熟需等到 2027-03 之后）。
#
# 口径（与 live_score 主策略完全一致的无未来端点规则）：
#   评分日 = 净值最新交易日；eligibility 只看成立时长 / 当期披露 / 过去 252 日 coverage；
#   风格暴露与中性化全部用**过去 252 交易日**回归，不触碰任何未来信息。
#   影子1 = 规模风格中性：ret_12m 对 [beta_mkt, beta_size] 做当月截面回归取残差
#   影子2 = 行业中性    ：ret_12m 对 [beta_mkt] + 6 个申万板块 beta 做当月截面回归取残差
# 输出：ml/scores/shadow_YYYY-MM.csv（fund_code/score_main/rank_main/风格中性残差与排名/
#        行业中性残差与排名/暴露/coverage/as_of）
#
# 未来评价规则（用户指定，写死在文档里）：影子要转正必须同时满足
#   ① 相对原动量有增量；② 相对可执行全池为正；③ 风险与换手没有明显恶化。
import argparse
import os

import numpy as np
import pandas as pd

from live_score import score_cross_section
from panel_builder import BENCH_PATH, PROJECT_ROOT, load_fund_series
from style_factors import SECTOR_KEYS, load_sector_factors, load_style_factors
from walk_forward_splitter import nw_tstat  # noqa: F401  （影子评价将来用，保持导入一致性）

SCORES_DIR = os.path.join(PROJECT_ROOT, "ml", "scores")
LOOKBACK = 252


def _betas(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """带 NaN 掩码的最小二乘（返回不含截距的系数）。"""
    mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    if int(mask.sum()) < X.shape[1] + 5:
        return np.full(X.shape[1], np.nan)
    beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
    return beta


def shadow_scores(as_of=None) -> tuple:
    """返回 (影子评分表, 主策略评分表, 评分日)。"""
    bench = pd.read_csv(BENCH_PATH, parse_dates=["date"]).sort_values("date")
    bench_dates = bench["date"].to_numpy(dtype="datetime64[ns]").astype("int64")
    df, _skip, t_date = score_cross_section(as_of)
    main = df[df["confidence"] == "main"].copy() if len(df) else pd.DataFrame()
    if not len(main):
        return pd.DataFrame(), main, t_date

    t_ns = np.datetime64(t_date, "ns").astype("int64")
    i_t = int(np.searchsorted(bench_dates, t_ns, side="right")) - 1
    print(f"加载净值与因子（评分日 {t_date.date()}，暴露窗口 {LOOKBACK} 交易日）…")
    series = load_fund_series(bench_dates)
    factors = {**load_style_factors(bench_dates)["factors"],
               **load_sector_factors(bench_dates)["factors"]}

    style_names = ["mkt", "size"]
    sector_names = ["mkt"] + SECTOR_KEYS
    rows = []
    w0 = i_t - LOOKBACK
    for code in main["fund_code"]:
        s = series.get(code)
        if s is None or w0 < 0:
            continue
        y = s["r_al"][w0:i_t + 1]
        rec = {"fund_code": code}
        B_style = np.column_stack([np.ones(len(y))] + [factors[k][w0:i_t + 1] for k in style_names])
        b1 = _betas(y, B_style)
        B_sec = np.column_stack([np.ones(len(y))] + [factors[k][w0:i_t + 1] for k in sector_names])
        b2 = _betas(y, B_sec)
        for j, nm in enumerate(style_names):
            rec[f"beta_{nm}"] = float(b1[j + 1])
        for j, nm in enumerate(sector_names[1:]):
            rec[f"beta_{nm}"] = float(b2[j + 1])
        rows.append(rec)
    ex = pd.DataFrame(rows)
    m = main.merge(ex, on="fund_code", how="left")

    # 当月截面残差（中性化）：ret_12m ~ 1 + 暴露
    def resid(df, cols):
        d = df[["score"] + cols].dropna()
        if len(d) < len(cols) + 10:
            return pd.Series(np.nan, index=df.index)
        X = np.column_stack([np.ones(len(d))] + [d[c].values for c in cols])
        beta, *_ = np.linalg.lstsq(X, d["score"].values, rcond=None)
        r = d["score"].values - X @ beta
        return pd.Series(r, index=d.index)

    m["style_neutral_score"] = resid(m, ["beta_mkt", "beta_size"])
    m["sector_neutral_score"] = resid(m, ["beta_mkt"] + [f"beta_{k}" for k in SECTOR_KEYS])
    m["rank_main"] = m["score"].rank(ascending=False)
    m["rank_style_neutral"] = m["style_neutral_score"].rank(ascending=False)
    m["rank_sector_neutral"] = m["sector_neutral_score"].rank(ascending=False)
    m = m.rename(columns={"score": "score_main"})
    cols = (["rank_main", "fund_code", "fund_name", "score_main", "rank_style_neutral",
             "style_neutral_score", "rank_sector_neutral", "sector_neutral_score",
             "beta_mkt", "beta_size"] + [f"beta_{k}" for k in SECTOR_KEYS] +
            ["age_months", "age_group", "coverage", "as_of"])
    cols = [c for c in cols if c in m.columns]
    return m[cols].sort_values("rank_main").reset_index(drop=True), main, t_date


def main():
    ap = argparse.ArgumentParser(description="两条中性化影子策略的评分留存（不改主策略）")
    ap.add_argument("--as-of", default=None, help="评分日 YYYY-MM-DD，缺省=净值最新交易日")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    sh, main, t_date = shadow_scores(args.as_of)
    if not len(sh):
        print("（无可评分基金）")
        return
    print(f"\n=== 影子评分（评分日 {t_date.date()}，主策略 = 原动量 ret_12m，未更换）===")
    print(f"主策略可评分 {len(main)} 只；影子表 {len(sh)} 行"
          f"（风格中性有效 {int(sh['style_neutral_score'].notna().sum())}、"
          f"行业中性有效 {int(sh['sector_neutral_score'].notna().sum())}）")
    ov_style = (len(set(sh.nsmallest(50, "rank_main")["fund_code"]) &
                    set(sh.nsmallest(50, "rank_style_neutral")["fund_code"])) / 50.0)
    ov_sector = (len(set(sh.nsmallest(50, "rank_main")["fund_code"]) &
                     set(sh.nsmallest(50, "rank_sector_neutral")["fund_code"])) / 50.0)
    print(f"与主策略 Top50 的重叠：风格中性 {ov_style:.0%} | 行业中性 {ov_sector:.0%}")
    print("\n影子候选 Top10（规模风格中性）：")
    print(sh.nsmallest(10, "rank_style_neutral")[
        ["rank_style_neutral", "fund_code", "fund_name", "score_main", "rank_main",
         "style_neutral_score"]].to_string(index=False))
    if not args.no_save:
        os.makedirs(SCORES_DIR, exist_ok=True)
        p = os.path.join(SCORES_DIR, f"shadow_{t_date.strftime('%Y-%m')}.csv")
        sh.to_csv(p, index=False)
        print(f"\n影子评分已落盘：{p}（影子仅记录，不用于资金决策；"
              f"转正需同时满足：相对原动量有增量 + 相对可执行全池为正 + 风险换手未恶化）")


if __name__ == "__main__":
    main()
