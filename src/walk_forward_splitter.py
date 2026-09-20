# walk_forward_splitter.py：Phase 2A——walk-forward 时序切分器（标签时间隔离的完整落地）
#
# 核心规则（README_FOR_HUMAN 问题④）：
#   预测截面 T（月末）的训练集 = { label_end_date + 公布滞后 < T 的行 }
#   ——只有"标签已走完且当时已公布"的历史行才可进入本轮训练；未来未揭晓的答案
#     （即使研究数据库里早已把 y 算好）一律不得入训练。
#   注意是 label_end_date < T - lag，不是 t_date < T：同一行的截面日与标签揭晓日相差约 6 个月。
#   逐月末滚动：每月重新确定当时已揭晓的训练世界，fit → 预测当月横截面。
#
# 交付边界（Phase 2A 全部，先于一切 ML）：
#   - 切分器本体：expanding / sliding 训练窗，逐月末滚动，末尾 holdout 隔离
#   - 防泄漏断言：每折强制校验时间边界，违反即 AssertionError（口径之争变代码级错误）
#   - FoldPreprocessor：折内预处理纪律（中位数填补 + 标准化，fit 仅允许在训练折），供 Phase 2B 复用
#   - 基线 Rank IC 入口：近一年收益排序 / 夏普排序能否预测未来半年——先于 XGBoost 的问题
#   - manifest 审计：每截面切分明细，含 train_label_end_max（该轮最晚揭晓的答案日期）
#
# 不做：模型训练（Phase 2B）、组合与回测（Phase 3）。
import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
PANEL_PATH = os.path.join(PROJECT_ROOT, "ml", "panel.parquet")
OUT_DIR = os.path.join(PROJECT_ROOT, "ml", "wf_splits")

LABEL_COL = "future_ret_6m"
DEFAULT_PUBLISH_LAG_DAYS = 5     # 标签期末净值公布滞后（自然日，保守缓冲；月末净值 T+1~T+3 公布）
DEFAULT_HOLDOUT_MONTHS = 12      # 末尾预留的最终验证截面数：开发期一切实验不得触碰
DEFAULT_MIN_TRAIN_ROWS = 300     # 单折训练行数门槛（探查实定：2005-06 训练 154 行 → 起步约 2006 年中）
MIN_TEST_ROWS_FOR_IC = 30       # IC 汇总计入的最小当月横截面行数（更小月份噪音主导，仅记录不计入）
# 年龄分层（与 panel_builder.py 常量保持一致）
SHORT_HISTORY_AGE = (12, 36)    # 12~36月：短历史段
FULL_HISTORY_AGE = 36           # ≥36月：长历史段


@dataclass
class Fold:
    t: pd.Timestamp              # 预测截面（月末）
    phase: str                    # dev / holdout
    train: pd.DataFrame          # 该轮可训练行（标签已揭晓且已公布）
    test: pd.DataFrame           # 当月横截面（预测对象）
    meta: dict                   # 审计信息（train_label_end_max = 该轮最晚可用的答案日期）


def nw_tstat(x, lags: int = 6) -> float:
    """Newey-West HAC t 检验（均值=0，Bartlett 核）。

    标签重叠修正（2026-09-17 审查确立）：6 个月标签逐月重叠 5 个月 → 月度 IC 序列
    lag1 自相关实测 0.6~0.8、n_eff≈40~50，naive t 高估约 2.3 倍——
    **一切正式显著性判定一律用本函数**（与 analyze_nav 的 HAC 口径同族），naive t 仅存档。"""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    if n < 5:
        return float("nan")
    e = x - x.mean()
    s = 0.0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1)
        s += 2.0 * w * float(np.dot(e[l:], e[:-l])) / n
    var = (float(np.dot(e, e)) / n + s) / n
    return float(x.mean() / np.sqrt(var)) if var > 0 else float("nan")


class WalkForwardSplitter:
    """逐月末 walk-forward 切分器。

    folds(phase) 每折：
      test  = t_date == T 的横截面
      train = label_end_date + publish_lag < T 的全部历史行（expanding；
              设 train_window_months=N 则只保留近 N 个月揭晓的标签，sliding）
    holdout 语义：末尾 holdout_months 个截面 phase='holdout'；
      folds('dev')（默认）绝不产出它们，从接口上防"开发期偷看最终验证段"。
    """

    REQUIRED_COLS = {"fund_code", "t_date", "age_months", "label_end_date", LABEL_COL}

    def __init__(self, panel: pd.DataFrame,
                 publish_lag_days: int = DEFAULT_PUBLISH_LAG_DAYS,
                 train_window_months: int | None = None,
                 min_train_rows: int = DEFAULT_MIN_TRAIN_ROWS,
                 holdout_months: int = DEFAULT_HOLDOUT_MONTHS):
        missing = self.REQUIRED_COLS - set(panel.columns)
        if missing:
            raise ValueError(f"面板缺列: {missing}")
        self.panel = panel.reset_index(drop=True)
        self.publish_lag = pd.Timedelta(days=publish_lag_days)
        self.train_window_months = train_window_months
        self.min_train_rows = min_train_rows
        # 预排序视图：每折 searchsorted O(log n) 定位切片，避免逐折全表布尔扫描
        # （pandas 3.0 的 datetime64 默认单位是 us，numpy searchsorted 与 Timestamp 直接混用会炸，
        #   统一显式转 ns——panel_builder 踩过同款坑）
        # **训练视图只含标签非空行**（2026-09-19 用户审查）：panel v3 里 `label_resolution="missing"`
        #   的行标签为 NaN（保留供审计），若混入训练会让模型把 NaN 当目标、或静默丢行破坏"同池比较"。
        #   测试/预测视图 `_by_t` 仍保留**全部行**（当月横截面要能对全体基金打分）。
        train_view = self.panel[self.panel[LABEL_COL].notna()]
        self._by_le = train_view.sort_values("label_end_date", kind="stable")
        self._le_vals = self._by_le["label_end_date"].to_numpy(dtype="datetime64[ns]")
        self._by_t = self.panel.sort_values("t_date", kind="stable")
        self._t_vals = self._by_t["t_date"].to_numpy(dtype="datetime64[ns]")
        self.month_ends = pd.DatetimeIndex(np.sort(self.panel["t_date"].unique()))
        n_hold = min(holdout_months, len(self.month_ends)) if holdout_months else 0
        self.holdout_ts = set(self.month_ends[-n_hold:]) if n_hold else set()

    def _train_slice(self, T: pd.Timestamp) -> pd.DataFrame:
        cutoff = T - self.publish_lag  # label_end < cutoff ⇔ label_end + lag < T
        hi = int(np.searchsorted(self._le_vals, cutoff.to_datetime64(), side="left"))
        lo = 0
        if self.train_window_months:
            lo_cutoff = T - pd.DateOffset(months=self.train_window_months) - self.publish_lag
            lo = int(np.searchsorted(self._le_vals, lo_cutoff.to_datetime64(), side="left"))
        return self._by_le.iloc[lo:hi]

    def _test_slice(self, T: pd.Timestamp) -> pd.DataFrame:
        t64 = T.to_datetime64()
        lo = int(np.searchsorted(self._t_vals, t64, side="left"))
        hi = int(np.searchsorted(self._t_vals, t64, side="right"))
        return self._by_t.iloc[lo:hi]

    def fold_index(self) -> list[dict]:
        """全部月末截面的切分审计行（含 skipped / holdout 轮），manifest 用。"""
        rows = []
        for T in self.month_ends:
            train = self._train_slice(T)
            test = self._test_slice(T)
            phase = "holdout" if T in self.holdout_ts else "dev"
            if phase == "holdout":
                status = "reserved_holdout"
            elif len(train) < self.min_train_rows:
                status = "skipped_small_train"
            elif len(test) == 0:
                status = "skipped_empty_test"
            else:
                status = "ok"
            rows.append({
                "t_date": T,
                "phase": phase,
                "status": status,
                "train_rows": len(train),
                "train_funds": train["fund_code"].nunique(),
                "train_t_min": train["t_date"].min() if len(train) else pd.NaT,
                "train_label_end_max": train["label_end_date"].max() if len(train) else pd.NaT,
                "test_rows": len(test),
                "test_funds": test["fund_code"].nunique(),
            })
        return rows

    def folds(self, phase: str = "dev"):
        """yield Fold。phase='dev'（默认）绝不产出 holdout 截面；'holdout' 只产出 holdout。"""
        for T in self.month_ends:
            in_holdout = T in self.holdout_ts
            if phase == "dev" and in_holdout:
                continue
            if phase == "holdout" and not in_holdout:
                continue
            train = self._train_slice(T)
            if len(train) < self.min_train_rows:
                continue
            test = self._test_slice(T)
            if len(test) == 0:
                continue
            # —— 防泄漏断言（每折强制）——
            cutoff = T - self.publish_lag
            assert train[LABEL_COL].notna().all(), f"训练集混入标签缺失行 @ {T.date()}"
            assert (train["label_end_date"] < cutoff).all(), f"训练集混入未揭晓标签 @ {T.date()}"
            assert (train["t_date"] < T).all(), f"训练集截面日不早于预测截面 @ {T.date()}"
            assert (test["t_date"] == T).all(), f"测试集非当月横截面 @ {T.date()}"
            meta = {
                "phase": phase,
                "train_rows": len(train),
                "train_funds": train["fund_code"].nunique(),
                "train_t_min": train["t_date"].min(),
                "train_t_max": train["t_date"].max(),
                "train_label_end_max": train["label_end_date"].max(),  # 该轮最晚可用的答案日期
                "test_rows": len(test),
                "test_funds": test["fund_code"].nunique(),
            }
            yield Fold(t=T, phase=phase, train=train, test=test, meta=meta)


class FoldPreprocessor:
    """折内预处理：中位数填补 + z-score 标准化。

    纪律内建：fit() 只允许在训练折上调用，transform() 用训练折统计量处理任意折
    ——防止"标准化/填补偷看测试集"。基线 Rank IC 不需要它（Spearman 对单调变换
    不变、缺失行剔除即可），留给 Phase 2B 线性/树模型直接复用。
    """

    def __init__(self):
        self.cols = None
        self._median = self._mu = self._sd = None

    def fit(self, train: pd.DataFrame, cols) -> "FoldPreprocessor":
        self.cols = list(cols)
        self._median = train[self.cols].median()
        self._mu = train[self.cols].mean()
        sd = train[self.cols].std(ddof=1)
        self._sd = sd.fillna(0.0).replace(0.0, 1.0)  # 全缺失/零方差列退化为不缩放
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._mu is not None, "FoldPreprocessor.fit 未调用（只允许在训练折上调用）"
        x = df[self.cols].fillna(self._median)
        return (x - self._mu) / self._sd


def age_group(age_months: float) -> str:
    """行级年龄分层（与 panel_builder 常量一致）：short 12~36月 / full ≥36月。

    under12 为死兜底：panel 的 age_months 已 round(...,1)（365 天 ≈ 11.99 → 12.0），
    实际不出现 under12 行，保留仅防 eligibility 口径变更（365 自然日 vs 12 个日历月）。
    """
    if age_months >= FULL_HISTORY_AGE:
        return "full"
    if age_months >= SHORT_HISTORY_AGE[0]:
        return "short"
    return "under12"


def rank_ic(df: pd.DataFrame, score_col: str, label_col: str = LABEL_COL):
    """横截面 Spearman(score, future_ret_6m)。缺失行剔除；n<3 或无变异返回 NaN。"""
    sub = df[[score_col, label_col]].dropna()
    n = len(sub)
    if n < 3 or sub[score_col].nunique() < 2 or sub[label_col].nunique() < 2:
        return float("nan"), n
    ic = stats.spearmanr(sub[score_col], sub[label_col]).statistic
    return float(ic), n


def run_baseline(spl: WalkForwardSplitter, features: list[str], phase: str) -> pd.DataFrame:
    """逐折计算基线 Rank IC 长表：每 (截面, feature, age_group) 一行。"""
    rows = []
    for fold in spl.folds(phase):
        gcol = fold.test["age_months"].map(age_group)
        for feat in features:
            for g in ("all", "short", "full", "under12"):
                gdf = fold.test if g == "all" else fold.test[gcol == g]
                ic, n = rank_ic(gdf, feat)
                rows.append({
                    "t_date": fold.t,
                    "phase": fold.phase,
                    "feature": feat,
                    "age_group": g,
                    "n": n,
                    "ic": ic,
                    "train_rows": fold.meta["train_rows"],
                    "train_label_end_max": fold.meta["train_label_end_max"],
                })
    return pd.DataFrame(rows)


def print_ic_summary(bdf: pd.DataFrame, features: list[str]) -> None:
    """按 (feature, age_group) 汇总：有效月（n≥MIN_TEST_ROWS_FOR_IC）的 IC 序列统计。

    **正式判定用 NW(6) t**（相邻月标签重叠 5 个月 → naive t 高估约 2.3 倍，见 nw_tstat docstring）；
    naive t 与 p 值一并列出仅供参考（2026-09-19 用户审查要求把 NW 值写进汇总）。
    """
    print(f"\n=== 基线 Rank IC 汇总（有效月=当月横截面 n≥{MIN_TEST_ROWS_FOR_IC}）===")
    header = (f"{'feature':<14s}{'group':<9s}{'months':>7s}{'mean_IC':>10s}{'std_IC':>9s}"
              f"{'naive_t':>9s}{'NW(6)t':>9s}{'p值':>9s}{'IC>0占比':>10s}")
    print(header)
    for feat in features:
        for g in ("all", "short", "full", "under12"):
            s = bdf[(bdf["feature"] == feat) & (bdf["age_group"] == g)
                    & (bdf["n"] >= MIN_TEST_ROWS_FOR_IC)]["ic"].dropna()
            if len(s) < 2:
                continue
            n = len(s)
            mean, sd = float(s.mean()), float(s.std(ddof=1))
            t = mean / (sd / np.sqrt(n))
            nw = nw_tstat(s)
            p = 2.0 * stats.t.sf(abs(t), n - 1)
            print(f"{feat:<14s}{g:<9s}{n:>7d}{mean:>10.4f}{sd:>9.4f}{t:>9.2f}{nw:>9.2f}"
                  f"{p:>9.3f}{(s > 0).mean():>10.2f}")
    print("（**正式判定看 NW(6)t**；naive_t/p 因标签重叠偏乐观；IC>0占比≈0.5 且 t 小 = 与随机无异）")


def main():
    ap = argparse.ArgumentParser(description="walk-forward 切分器 + 基线 Rank IC（Phase 2A）")
    ap.add_argument("--panel", default=PANEL_PATH, help="面板 parquet 路径")
    ap.add_argument("--baseline", default="", help="逗号分隔特征列（如 ret_12m,sharpe_12m）；缺省只出 manifest")
    ap.add_argument("--publish-lag-days", type=int, default=DEFAULT_PUBLISH_LAG_DAYS,
                    help="标签期末净值公布滞后（自然日）")
    ap.add_argument("--train-window-months", type=int, default=None,
                    help="滑动训练窗月数；缺省 expanding（用尽当时全部历史）")
    ap.add_argument("--min-train-rows", type=int, default=DEFAULT_MIN_TRAIN_ROWS)
    ap.add_argument("--holdout-months", type=int, default=DEFAULT_HOLDOUT_MONTHS)
    ap.add_argument("--include-holdout", action="store_true",
                    help="打开最终 holdout（只允许最终验收跑一次；结果单独落盘）")
    ap.add_argument("--tag", default="",
                    help="输出文件名后缀（如 _v3panel），避免覆盖既有 v2 产物；缺省与原来一致")
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    spl = WalkForwardSplitter(panel,
                              publish_lag_days=args.publish_lag_days,
                              train_window_months=args.train_window_months,
                              min_train_rows=args.min_train_rows,
                              holdout_months=args.holdout_months)
    os.makedirs(OUT_DIR, exist_ok=True)
    mdf = pd.DataFrame(spl.fold_index())
    mpath = os.path.join(OUT_DIR, f"wf_manifest{args.tag}.csv")
    mdf.to_csv(mpath, index=False)

    # —— manifest 审计打印 ——
    n_dev_ok = int(((mdf["phase"] == "dev") & (mdf["status"] == "ok")).sum())
    n_dev_skip = int(((mdf["phase"] == "dev") & (mdf["status"] != "ok")).sum())
    n_hold = int((mdf["phase"] == "holdout").sum())
    ok = mdf[mdf["status"] == "ok"]
    print("=== walk-forward 切分审计 ===")
    print(f"训练规则: label_end_date + {args.publish_lag_days}天 < 预测截面T"
          f"（{'sliding ' + str(args.train_window_months) + '个月' if args.train_window_months else 'expanding 全历史'}）")
    print(f"截面总数 {len(mdf)} | dev 可用 {n_dev_ok} | dev 跳过 {n_dev_skip} | holdout 预留 {n_hold}")
    if len(ok):
        print(f"第一折 {ok['t_date'].min().date()}（train {int(ok.iloc[0]['train_rows'])} 行）"
              f" ~ 末折 {ok['t_date'].max().date()}（train {int(ok.iloc[-1]['train_rows'])} 行）")
        print(f"每折训练规模: min={int(ok['train_rows'].min())} / med={int(ok['train_rows'].median())} / max={int(ok['train_rows'].max())} 行")
    print(f"manifest 已落盘: {mpath}")

    if not args.baseline:
        return
    features = [s.strip() for s in args.baseline.split(",") if s.strip()]
    bad = [f for f in features if f not in panel.columns]
    if bad:
        raise SystemExit(f"特征列不存在于面板: {bad}")
    phase = "holdout" if args.include_holdout else "dev"
    if args.include_holdout:
        print("\n!!! 正在打开最终 holdout —— 只允许在全部实验拍板后的最终验收时运行一次 !!!")
    bdf = run_baseline(spl, features, phase)
    bname = (f"baseline_rankic{args.tag}_holdout.csv" if args.include_holdout
             else f"baseline_rankic{args.tag}.csv")
    bpath = os.path.join(OUT_DIR, bname)
    bdf.to_csv(bpath, index=False)
    print(f"逐月明细已落盘: {bpath}（{len(bdf)} 行）")
    print_ic_summary(bdf, features)
    if not args.include_holdout and n_hold:
        print(f"\nholdout 未触碰：末尾 {n_hold} 个截面已预留，最终验收时加 --include-holdout 打开")


if __name__ == "__main__":
    main()
