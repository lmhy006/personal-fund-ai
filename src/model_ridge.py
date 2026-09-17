# model_ridge.py：Phase 2B E1/E2/E1b——Ridge 模型实验 runner（含特征消融与标签模式）
#
# 标签口径（2026-09-17 拍板）：默认 demean（月内去均值——"实力=相对当期同伴的偏离"，
#   剥离市场+风格共同成分，免疫基准错配）；--label-mode rank 为月内秩变换（E1b 诊断：
#   压掉重尾、对齐 Spearman 目标）。评估层 IC 与任何月内平移口径等价，组合收益报
#   绝对 / 相对沪深300 / 相对全池 三口径。
#
# 实验结论（dev folds 228 折，2026-09-17，配对差 t 检验 vs 基线）：
#   E1  全10特征 demean:   IC 0.021，被 ret_12m 基线显著击败（t=-4.07）
#   E2  单因子 ret_12m:    IC 0.092 ≈ 基线（管线自证）；动量族 0.076 无显著差异
#   E2  +风险特征(sharpe/vol/mdd): IC -0.045 反号（t=-7.53）；+CAPM: 0.002；+age: 0.052
#   E1b rank标签:          全特征 0.011、动量+风险 -0.068——重尾损失不是病因；
#   ★  研究发现：控制动量后风险特征偏相关为负（同等过去收益下高波基金未来半年占优）
#      ——共线性下 MSE/Ridge 权重摊薄+反向污染，线性多因子无增量；E3 树模型待验证非线性
#
# 纪律（继承 Phase 2A，不重新谈判）：
#   - dev folds only，holdout 封存（--include-holdout 留最终验收，单独落盘）
#   - FoldPreprocessor fit 仅训练折；alpha 折内时序验证段选择（只用 ≤T 信息）
#   - 判定标准：同月 IC 配对差 t 检验，不比均值高低零点几
#
# Ridge 实现：手写闭式解 (X'X+λI)β=X'y（特征≤10 维，毫秒级），不引入 sklearn。
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from walk_forward_splitter import (WalkForwardSplitter, FoldPreprocessor, rank_ic,
                                    MIN_TEST_ROWS_FOR_IC, LABEL_COL)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
PANEL_PATH = os.path.join(PROJECT_ROOT, "ml", "panel.parquet")
BENCH_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "benchmark_hs300.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "ml", "experiments")

FEATURES = ["ret_1m", "ret_3m", "ret_6m", "ret_12m",
            "vol_12m", "sharpe_12m", "mdd_12m", "beta_12m", "alpha_12m", "age_months"]
# coverage 是数据质量指标而非基金属性，不入特征（面板 coverage≥0.95 门槛后近乎常数）
ALPHA_GRID = np.logspace(-3, 3, 13)   # Ridge 超参网格（标准化特征下的合理范围）
VALID_MONTHS_MAX = 24                  # 训练集内验证段上限（早期折自适应下探）
DEFAULT_ALPHA = 1.0                    # 折太小（验证不可行）时的回退值
LABEL_HORIZON_TRADING = 126            # 与 panel_builder 的 LABEL_HORIZON 一致（基准前瞻窗）


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """标准化设计矩阵下的 Ridge 闭式解 (X'X + αI)β = X'y"""
    A = X.T @ X + alpha * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def label_values(df: pd.DataFrame, mode: str = "demean", col: str = LABEL_COL) -> np.ndarray:
    """训练标签构造（IC/组合收益评估一律用原始 y）。
    demean: y' = y - 当月横截面均值（剥离市场+风格共同成分）
    rank:   y' = 月内 percentile rank（压掉重尾、与 Spearman 评估目标对齐——E1b）"""
    if mode == "rank":
        return df.groupby("t_date")[col].rank(pct=True).to_numpy()
    return (df[col] - df.groupby("t_date")[col].transform("mean")).to_numpy()


def bench_future_6m() -> pd.Series:
    """基准未来 126 交易日收益查找表（date → future_ret）。
    与面板 t_date 同一日历（同源 benchmark），供评估层报"相对300超额"三口径。"""
    bench = pd.read_csv(BENCH_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    c = bench["close"].to_numpy(dtype=float)
    fr = np.full(len(c), np.nan)
    fr[:-LABEL_HORIZON_TRADING] = c[LABEL_HORIZON_TRADING:] / c[:-LABEL_HORIZON_TRADING] - 1.0
    return pd.Series(fr, index=bench["date"])


def select_alpha(train: pd.DataFrame, features: list, label_mode: str = "demean") -> tuple:
    """训练集内部时序验证选 alpha（纪律：只用 ≤T 信息）。
    早段拟合（标准化统计也只见早段）→ 尾部验证段按月度 IC 均值选优 → 返回 (alpha, 审计信息)。"""
    months = np.sort(train["t_date"].unique())
    n_valid = min(VALID_MONTHS_MAX, max(6, len(months) // 3))
    valid_ts = set(months[-n_valid:])
    early = train[~train["t_date"].isin(valid_ts)]
    valid = train[train["t_date"].isin(valid_ts)]
    if len(early) < 60 or len(valid) < 20:
        return DEFAULT_ALPHA, {"valid_months": 0, "valid_ic": float("nan"),
                               "note": "折过小，回退默认alpha"}
    pre = FoldPreprocessor().fit(early, features)
    Xe = pre.transform(early).to_numpy()
    ye = label_values(early, label_mode)
    Xv = pre.transform(valid).to_numpy()
    # 逐月预分组（y 值），alpha 循环内只做线性预测+秩相关，避免重复 groupby
    month_groups = [g[LABEL_COL].to_numpy() for _, g in valid.groupby("t_date")]
    best_alpha, best_ic = DEFAULT_ALPHA, -np.inf
    for a in ALPHA_GRID:
        beta = ridge_fit(Xe, ye, float(a))
        pv = Xv @ beta
        # 按月切预测向量（groupby 顺序与 month_groups 对齐）
        ics = []
        pos = 0
        for y_arr in month_groups:
            seg = pv[pos:pos + len(y_arr)]
            pos += len(y_arr)
            if len(seg) >= 3 and pd.Series(seg).nunique() > 1 and pd.Series(y_arr).nunique() > 1:
                ics.append(stats.spearmanr(seg, y_arr).statistic)
        ic = float(np.nanmean(ics)) if ics else -np.inf
        if ic > best_ic:
            best_ic, best_alpha = ic, float(a)
    return best_alpha, {"valid_months": n_valid, "valid_ic": round(best_ic, 4)}


def paired_t(diff: pd.Series) -> tuple:
    """同月配对差的 t 检验（模型 IC - 基线 IC）。"""
    d = diff.dropna()
    n = len(d)
    if n < 2:
        return n, float("nan"), float("nan"), float("nan")
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    p = 2.0 * stats.t.sf(abs(t), n - 1)
    return n, float(d.mean()), float(t), float(p)


def run_experiment(panel: pd.DataFrame, phase: str = "dev", features: list = None,
                  label_mode: str = "demean") -> pd.DataFrame:
    features = features or FEATURES
    spl = WalkForwardSplitter(panel)
    bench = bench_future_6m()
    rows = []
    for fold in spl.folds(phase):
        alpha_sel, sel_info = select_alpha(fold.train, features, label_mode)
        # 最终拟合：全训练折（标准化统计含全部已揭晓训练行）
        pre = FoldPreprocessor().fit(fold.train, features)
        beta = ridge_fit(pre.transform(fold.train).to_numpy(),
                          label_values(fold.train, label_mode), alpha_sel)
        test = fold.test.assign(_pred=pre.transform(fold.test).to_numpy() @ beta)

        ic_model, n = rank_ic(test, "_pred")
        ic_base_r12, _ = rank_ic(fold.test, "ret_12m")
        ic_base_sh, _ = rank_ic(fold.test, "sharpe_12m")
        # Top20% 等权（gross；标签重叠5个月，t统计偏乐观，汇总注明）
        k = max(1, int(round(len(test) * 0.2)))
        order = test.sort_values("_pred", ascending=False)
        top20 = float(order.head(k)[LABEL_COL].mean())
        bot20 = float(order.tail(k)[LABEL_COL].mean())
        pool = float(test[LABEL_COL].mean())
        b6 = float(bench.reindex([fold.t]).iloc[0]) if fold.t in bench.index else float("nan")

        rows.append({
            "t_date": fold.t, "phase": fold.phase, "n_test": n, "alpha_sel": alpha_sel,
            "valid_ic": sel_info["valid_ic"],
            "ic_model": ic_model, "ic_base_ret12": ic_base_r12, "ic_base_sharpe": ic_base_sh,
            "top20_ret": top20, "bot20_ret": bot20, "pool_ret": pool, "bench_ret6m": b6,
            "top20_excess_hs300": top20 - b6, "top20_excess_pool": top20 - pool,
            "train_rows": fold.meta["train_rows"],
            "train_label_end_max": fold.meta["train_label_end_max"],
        })
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
    eff = df[df["n_test"] >= MIN_TEST_ROWS_FOR_IC]
    print(f"\n=== Ridge v1 vs 基线（dev folds，有效月 n≥{MIN_TEST_ROWS_FOR_IC}：{len(eff)} 个月）===")
    m_ic = eff["ic_model"].mean()
    print(f"模型  RankIC: mean={m_ic:.4f}  std={eff['ic_model'].std():.4f}  "
          f"IC>0占比={( eff['ic_model'] > 0).mean():.2f}")
    for base_col, base_name in [("ic_base_ret12", "ret_12m 基线"), ("ic_base_sharpe", "sharpe 基线")]:
        b_ic = eff[base_col].mean()
        n, dm, t, p = paired_t(eff["ic_model"] - eff[base_col])
        print(f"vs {base_name}: base_IC={b_ic:.4f} | 配对差={dm:+.4f}  t={t:+.2f}  p={p:.3f}"
              f"{'  ← 显著优于基线' if (t > 1.96 and p < 0.05) else ''}")
    # 组合层（三口径，6个月收益均值；标签重叠5个月→t统计偏乐观，仅作量级参考）
    top = eff["top20_ret"].mean()
    pool = eff["pool_ret"].mean()
    ex_b = eff["top20_excess_hs300"].mean()
    ex_p = eff["top20_excess_pool"].mean()
    bot = eff["bot20_ret"].mean()
    bench = eff["bench_ret6m"].mean()
    print(f"Top20%组合（6个月收益均值）: 绝对={top:.4%} | 相对沪深300={ex_b:+.4%} | "
          f"相对全池={ex_p:+.4%}  （全池均值 {pool:.4%} / 基准 {bench:.4%} / 底20% {bot:.4%}）")
    print("（注：相邻月标签重叠5个月，组合层统计偏乐观；正式换手/费用回测属 Phase 3）")


def main():
    ap = argparse.ArgumentParser(description="Phase 2B E1：Ridge v1 模型实验")
    ap.add_argument("--panel", default=PANEL_PATH)
    ap.add_argument("--features", default=",".join(FEATURES),
                    help="逗号分隔特征列（E2 消融诊断用），缺省=全特征")
    ap.add_argument("--label-mode", default="demean", choices=["demean", "rank"],
                    help="训练标签：demean=月内去均值（默认）；rank=月内秩变换（E1b）")
    ap.add_argument("--tag", default="v1", help="产物文件名标签（ridge_{tag}_monthly.csv）")
    ap.add_argument("--include-holdout", action="store_true",
                    help="打开最终 holdout（只允许最终验收跑一次；结果单独落盘）")
    args = ap.parse_args()

    features = [s.strip() for s in args.features.split(",") if s.strip()]
    bad = [f for f in features if f not in pd.read_parquet(args.panel).columns]
    if bad:
        raise SystemExit(f"特征列不存在: {bad}")
    panel = pd.read_parquet(args.panel)
    phase = "holdout" if args.include_holdout else "dev"
    if args.include_holdout:
        print("!!! 正在打开最终 holdout —— 只允许在全部实验拍板后的最终验收时运行一次 !!!")
    print(f"特征集({len(features)}): {features} | 标签模式: {args.label_mode}")
    df = run_experiment(panel, phase, features, args.label_mode)
    os.makedirs(OUT_DIR, exist_ok=True)
    fname = f"ridge_{args.tag}_{'holdout' if args.include_holdout else 'monthly'}.csv"
    fpath = os.path.join(OUT_DIR, fname)
    df.to_csv(fpath, index=False)
    print(f"逐月明细已落盘: {fpath}（{len(df)} 折）")
    print_summary(df)
    if not args.include_holdout:
        print("\nholdout 未触碰：全部实验拍板后加 --include-holdout 做最终验收")


if __name__ == "__main__":
    main()
