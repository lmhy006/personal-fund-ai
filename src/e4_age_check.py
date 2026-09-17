# e4_age_check.py：E4——主策略（动量基线）年龄适用性与跨段可比性检验
# 纯评估层，不训练任何模型（2026-09-17 拍板：只有发现分层确有增量空间，才训练年龄专用模型）
#
# 回答三个问题：
#   1. 适用性：动量 IC 在 short(12-36月)/full(≥36月) 组内是否都显著为正（NW 判定）
#   2. 跨段可比性：全池按动量排序是否被年龄污染——Top20 组的 short 占比 vs 全池 short 占比
#   3. 分层增量空间：组内排序 Top 收益 − 全池排序 Top 收益（NW 判定）——
#      只有显著 > 0 才值得年龄专用模型；不足一年基金不在本检验范围（面板 eligibility 已排除）
#
# 口径：主策略=近一年收益排序（ret_12m）；其 ret_12m 缺失行不评分（上线语义，缺失集中
# 于 12-16 月龄新基金）；IC 与 Top 全部在对齐行（ret_12m 非缺失）上算；一切显著性用 NW(6)。
import os

import numpy as np
import pandas as pd

from walk_forward_splitter import (WalkForwardSplitter, rank_ic, age_group,
                                    nw_tstat, MIN_TEST_ROWS_FOR_IC, LABEL_COL)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
PANEL_PATH = os.path.join(PROJECT_ROOT, "ml", "panel.parquet")
OUT_PATH = os.path.join(PROJECT_ROOT, "ml", "experiments", "e4_age_check_monthly.csv")


def top20_by(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    k = max(1, int(round(len(df) * 0.2)))
    return df.sort_values(score_col, ascending=False).head(k)


def run() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL_PATH)
    spl = WalkForwardSplitter(panel)
    rows = []
    for fold in spl.folds("dev"):
        scored = fold.test[fold.test["ret_12m"].notna()]  # 主策略可评分行（缺失=不评分）
        if len(scored) < MIN_TEST_ROWS_FOR_IC:
            continue
        g = scored["age_months"].map(age_group)
        short = scored[g == "short"]
        full = scored[g == "full"]

        ic_all, n = rank_ic(scored, "ret_12m")
        ic_short, n_s = rank_ic(short, "ret_12m")
        ic_full, n_f = rank_ic(full, "ret_12m")

        # Top20 组年龄构成（全池排序）
        top = top20_by(scored, "ret_12m")
        short_share_top = float((top["age_months"] < 36).mean())
        short_share_pool = float((scored["age_months"] < 36).mean())

        # 收益：全池排序 Top20 vs 组内排序（各组内各取 Top20% 合并等权）
        ret_pool_top = float(top[LABEL_COL].mean())
        segs = [top20_by(grp, "ret_12m")[LABEL_COL] for grp in (short, full) if len(grp) >= 5]
        ret_group_top = float(pd.concat(segs).mean()) if segs else np.nan

        rows.append({
            "t_date": fold.t, "n": n, "n_short": n_s, "n_full": n_f,
            "ic_all": ic_all, "ic_short": ic_short, "ic_full": ic_full,
            "short_share_top": short_share_top, "short_share_pool": short_share_pool,
            "ret_pool_top": ret_pool_top, "ret_group_top": ret_group_top,
            "ret_diff_group_minus_pool": ret_group_top - ret_pool_top,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"逐月明细落盘: {OUT_PATH}（{len(df)} 折）")

    print("\n=== E4：主策略（动量基线）年龄适用性与跨段可比性 ===")
    print("问题1 适用性（动量 IC，NW t vs 0）：")
    for col, name in [("ic_all", "全池"), ("ic_short", "short(12-36月)"), ("ic_full", "full(≥36月)")]:
        s = df[col].dropna()
        print(f"  {name:<15s} mean IC={s.mean():+.4f}  有效月={len(s)}  NW t={nw_tstat(s):+.2f}")
    d1 = (df["ic_full"] - df["ic_short"]).dropna()
    print(f"  组间差 full−short: {d1.mean():+.4f}  NW t={nw_tstat(d1):+.2f}")

    print("问题2 跨段可比性（Top20 组年龄构成偏差，>0=Top 组偏向新基金）:")
    d2 = (df["short_share_top"] - df["short_share_pool"]).dropna()
    print(f"  Top组short占比−全池short占比: 均值={d2.mean():+.4f}  NW t={nw_tstat(d2):+.2f}"
          f"{'  ← 分数被年龄系统性污染，跨年龄直接比较需谨慎' if abs(nw_tstat(d2)) > 1.96 else '  ← 无系统偏差，跨年龄排序可比'}")

    print("问题3 分层增量空间（组内排序Top − 全池排序Top 收益差，NW 判定）：")
    d3 = df["ret_diff_group_minus_pool"].dropna()
    t3 = nw_tstat(d3)
    print(f"  均值={d3.mean():+.4%}  naive t={d3.mean() / (d3.std() / np.sqrt(len(d3))):+.2f}  NW t={t3:+.2f}")
    if t3 > 1.96:
        print("  → 分层排序显著更优：存在年龄专用模型的增量空间，值得进入分层训练")
    elif t3 < -1.96:
        print("  → 分层排序显著更差：全池统一排序即可（且更稳）")
    else:
        print("  → 无显著差异：不训练年龄专用模型——主策略统一排行，各年龄段 IC 分列监控")
    print("（口径：对齐行=ret_12m 非缺失；组内方案=short/full 各取 Top20% 合并等权；NW(6) 修正标签重叠）")
    return df


if __name__ == "__main__":
    run()
