# model_gbdt.py：Phase 2B E3——LightGBM 树模型实验
#
# 复用 walk_forward_splitter 纪律框架与 model_ridge 的评估/对比口径（同月配对差 t 检验、
# Top20% 三口径收益、holdout 封存）。与 Ridge 的三点关键差异：
#   1. 特征不标准化、不填补：LightGBM 分裂原生处理 NaN（缺失进默认分支）——无拟合状态
#      = 无预处理泄漏面；ret_12m 的 1486 个新基金缺失行直接交给树学（Ridge 只能填中位数）。
#   2. 超参：固定保守组（防过拟合），复杂度控制靠折内**早停**——训练集内部再切一层时序
#      （尾部 24 个月揭晓段做验证集，只用 ≤T 信息），early_stopping(100)。
#   3. 早停模型直接作为该折模型（训练用早段，模型从未见过最近 24 个月的标签——
#      对时序外推更保守；不像 Ridge 闭式解便宜可全折重拟合）。
#
# 标签：同 model_ridge（--label-mode demean 默认 / rank 诊断）。
import argparse
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ConstantInputWarning

warnings.filterwarnings("ignore", category=ConstantInputWarning)  # 超浅树的常数预测无秩相关，无害

from walk_forward_splitter import (WalkForwardSplitter, rank_ic,
                                    MIN_TEST_ROWS_FOR_IC, LABEL_COL, nw_tstat)
from model_ridge import (FEATURES, label_values, bench_future_6m, paired_t,
                         VALID_MONTHS_MAX, OUT_DIR, PANEL_PATH)

LGB_PARAMS = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 50,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "seed": 42, "verbose": -1,
}
# 运行成本控制（2026-09-17）：IC 早停下轮数可达上千，2000 轮上限使全量跑需数小时；
# 改为 lr=0.1 / 上限 400 / 耐心 50（等价于 lr=0.05 下约 800 轮，对 10 特征面板足够收敛）
NUM_BOOST_ROUND = 400
EARLY_STOP_PATIENCE = 50


def split_train_valid(train: pd.DataFrame) -> tuple:
    """训练折内部时序切分：尾部 VALID_MONTHS_MAX 个月份做验证（≤T 已揭晓信息）。"""
    months = np.sort(train["t_date"].unique())
    n_valid = min(VALID_MONTHS_MAX, max(6, len(months) // 3))
    valid_ts = set(months[-n_valid:])
    early = train[~train["t_date"].isin(valid_ts)]
    valid = train[train["t_date"].isin(valid_ts)]
    return early, valid, n_valid


def train_fold_lgb(train: pd.DataFrame, features: list, label_mode: str,
                   early_stop_metric: str = "rmse") -> tuple:
    """折内训练：早段拟合 + 验证段早停。返回 (booster, n_rounds, 验证段月度IC)。

    early_stop_metric：
      rmse —— LightGBM 内置（E3 v1 实测：重尾 demean 标签下过于保守，med 仅 4 轮，
              4 棵浅树承载不了信号——不是树的公平审判）
      ic   —— 验证段**月度 Rank IC 均值**作早停准则（与排序评估目标对齐；feval 自定义，
              验证段是 ≤T 已揭晓信息，纪律合法）"""
    early, valid, _ = split_train_valid(train)
    if len(early) < 60 or len(valid) < 20:
        return None, 0, float("nan")
    # 标签按月变换在各自段内独立完成（月不重叠，与整体变换等价）
    dtrain = lgb.Dataset(early[features], label=label_values(early, label_mode))
    dvalid = lgb.Dataset(valid[features], label=label_values(valid, label_mode),
                         reference=dtrain)
    params = dict(LGB_PARAMS)
    feval = None
    if early_stop_metric == "ic":
        params["metric"] = "None"  # 禁用内置指标，只用 feval
        y_v = dvalid.get_label()
        mc = valid["t_date"].dt.year.to_numpy() * 100 + valid["t_date"].dt.month.to_numpy()
        idx_groups = [np.where(mc == m)[0] for m in np.unique(mc)]

        def feval(preds, _dataset):
            ics = []
            for idx in idx_groups:
                p, y = preds[idx], y_v[idx]
                if len(idx) >= 3 and pd.Series(p).nunique() > 1:
                    ics.append(stats.spearmanr(p, y).statistic)
            return "ic", float(np.nanmean(ics)) if ics else 0.0, True
    booster = lgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND,
                        valid_sets=[dvalid], feval=feval,
                        callbacks=[lgb.early_stopping(EARLY_STOP_PATIENCE, verbose=False)])
    pv = booster.predict(valid[features], num_iteration=booster.best_iteration)
    vd = valid.assign(_pred=pv)
    ics = [stats.spearmanr(g["_pred"], g[LABEL_COL]).statistic
           for _, g in vd.groupby("t_date") if len(g) >= 3]
    valid_ic = float(np.nanmean(ics)) if ics else float("nan")
    return booster, int(booster.best_iteration or NUM_BOOST_ROUND), valid_ic


def run_experiment(panel: pd.DataFrame, phase: str = "dev", features: list = None,
                   label_mode: str = "demean", limit: int = 0,
                   early_stop_metric: str = "rmse",
                   fold_start: int = 0, fold_end: int = 0) -> tuple:
    """返回 (逐月汇总, 逐基金预测留存)。fold_start/fold_end：折序号切片（分片并行；0=末尾）。

    评估行对齐（2026-09-17 审查落实）：IC/Top 与基线同在 ret_12m 非缺失行集上算；
    预测留存全体行（树可给缺失行评分——上线语义），口径可离线重算。"""
    features = features or FEATURES
    spl = WalkForwardSplitter(panel)
    bench = bench_future_6m()
    rows, pred_rows = [], []
    for i, fold in enumerate(spl.folds(phase)):
        if i < fold_start:
            continue
        if fold_end and i >= fold_end:
            break
        if limit and i >= limit:
            break
        booster, n_rounds, valid_ic = train_fold_lgb(fold.train, features, label_mode,
                                                     early_stop_metric)
        if booster is None:
            continue
        pred = booster.predict(fold.test[features], num_iteration=n_rounds)
        test = fold.test.assign(_pred=pred)

        # 逐基金预测留存（全体行）
        pred_rows.append(fold.test[["fund_code", "t_date", "age_months", "ret_12m",
                                    "sharpe_12m", LABEL_COL]].assign(
            pred=pred, phase=fold.phase))
        # —— 对齐行集：与基线同月同基金 ——
        test_al = test[test["ret_12m"].notna()]
        ic_model, n = rank_ic(test_al, "_pred")
        ic_base_r12, _ = rank_ic(test_al, "ret_12m")
        ic_base_sh, _ = rank_ic(test_al, "sharpe_12m")
        k = max(1, int(round(len(test_al) * 0.2)))
        order = test_al.sort_values("_pred", ascending=False)
        top20 = float(order.head(k)[LABEL_COL].mean())
        bot20 = float(order.tail(k)[LABEL_COL].mean())
        pool = float(test_al[LABEL_COL].mean())
        b6 = float(bench.reindex([fold.t]).iloc[0]) if fold.t in bench.index else float("nan")

        rows.append({
            "t_date": fold.t, "phase": fold.phase, "n_test": n,
            "n_rounds": n_rounds, "valid_ic": valid_ic,
            "ic_model": ic_model, "ic_base_ret12": ic_base_r12, "ic_base_sharpe": ic_base_sh,
            "top20_ret": top20, "bot20_ret": bot20, "pool_ret": pool, "bench_ret6m": b6,
            "top20_excess_hs300": top20 - b6, "top20_excess_pool": top20 - pool,
            "train_rows": fold.meta["train_rows"],
            "train_label_end_max": fold.meta["train_label_end_max"],
        })
    return pd.DataFrame(rows), (pd.concat(pred_rows, ignore_index=True)
                                if pred_rows else pd.DataFrame())


def print_summary(df: pd.DataFrame) -> None:
    eff = df[df["n_test"] >= MIN_TEST_ROWS_FOR_IC]
    print(f"\n=== LightGBM vs 基线（dev folds，有效月 n≥{MIN_TEST_ROWS_FOR_IC}：{len(eff)} 个月）===")
    m_ic = eff["ic_model"].mean()
    print(f"模型  RankIC: mean={m_ic:.4f}  std={eff['ic_model'].std():.4f}  "
          f"IC>0占比={(eff['ic_model'] > 0).mean():.2f}")
    for base_col, base_name in [("ic_base_ret12", "ret_12m 基线"), ("ic_base_sharpe", "sharpe 基线")]:
        b_ic = eff[base_col].mean()
        n, dm, t, p = paired_t(eff["ic_model"] - eff[base_col])
        t_nw = nw_tstat(eff["ic_model"] - eff[base_col])
        print(f"vs {base_name}: base_IC={b_ic:.4f} | 配对差={dm:+.4f}  naive t={t:+.2f}  "
              f"NW t={t_nw:+.2f}{'  ← 显著优于基线（NW 判定）' if t_nw > 1.96 else ''}")
    top = eff["top20_ret"].mean()
    ex_b = eff["top20_excess_hs300"].mean()
    ex_p = eff["top20_excess_pool"].mean()
    print(f"Top20%组合（6个月收益均值）: 绝对={top:.4%} | 相对沪深300={ex_b:+.4%} | "
          f"相对全池={ex_p:+.4%}")
    print(f"早停轮数: med={int(df['n_rounds'].median())} min={int(df['n_rounds'].min())} "
          f"max={int(df['n_rounds'].max())} | 验证段IC均值={df['valid_ic'].mean():.4f}")
    print("（注：相邻月标签重叠5个月，组合层统计偏乐观；正式换手/费用回测属 Phase 3）")


def main():
    ap = argparse.ArgumentParser(description="Phase 2B E3：LightGBM 树模型实验")
    ap.add_argument("--panel", default=PANEL_PATH)
    ap.add_argument("--features", default=",".join(FEATURES),
                    help="逗号分隔特征列，缺省=全特征")
    ap.add_argument("--label-mode", default="demean", choices=["demean", "rank"])
    ap.add_argument("--early-stop", default="rmse", choices=["rmse", "ic"],
                    help="早停准则：rmse=内置（E3 v1 实测过于保守）；ic=验证段月度RankIC（v2）")
    ap.add_argument("--tag", default="v1", help="产物文件名标签（gbdt_{tag}_monthly.csv）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 折（冒烟测试用）")
    ap.add_argument("--fold-start", type=int, default=0, help="折序号起点（分片并行）")
    ap.add_argument("--fold-end", type=int, default=0, help="折序号终点（0=到末尾；分片并行）")
    ap.add_argument("--include-holdout", action="store_true",
                    help="打开最终 holdout（只允许最终验收跑一次；结果单独落盘）")
    args = ap.parse_args()

    features = [s.strip() for s in args.features.split(",") if s.strip()]
    panel = pd.read_parquet(args.panel)
    bad = [f for f in features if f not in panel.columns]
    if bad:
        raise SystemExit(f"特征列不存在: {bad}")
    phase = "holdout" if args.include_holdout else "dev"
    if args.include_holdout:
        print("!!! 正在打开最终 holdout —— 只允许在全部实验拍板后的最终验收时运行一次 !!!")
    print(f"特征集({len(features)}): {features} | 标签模式: {args.label_mode} | 早停: {args.early_stop}"
          f"{' | 冒烟前' + str(args.limit) + '折' if args.limit else ''}")
    df, preds = run_experiment(panel, phase, features, args.label_mode, args.limit, args.early_stop,
                               args.fold_start, args.fold_end)
    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = "holdout" if args.include_holdout else "monthly"
    fpath = os.path.join(OUT_DIR, f"gbdt_{args.tag}_{suffix}.csv")
    df.to_csv(fpath, index=False)
    print(f"逐月明细已落盘: {fpath}（{len(df)} 折）")
    if len(preds):
        ppath = os.path.join(OUT_DIR, f"preds_gbdt_{args.tag}_{suffix}.csv")
        preds.to_csv(ppath, index=False)
        print(f"逐基金预测留存: {ppath}（{len(preds)} 行，全体行含 ret_12m 缺失）")
    print_summary(df)
    if not args.include_holdout:
        print("\nholdout 未触碰：全部实验拍板后加 --include-holdout 做最终验收")


if __name__ == "__main__":
    main()
