# partial_ic_check.py：小范围"特征是否有独立信息"诊断（用户 2026-09-19 指定）
#
# 目的：只问一个问题——**在月内控制 ret_12m 之后，历史时点有效的 log(AUM) 与
#   flow_share_ratio 是否仍与未来 6 个月收益有秩相关**（partial Rank IC）。
#
# 严格边界（用户指定，不得扩张）：
#   - 只检验 **log(AUM)** 与 **flow_share_ratio** 两项（**不含费率**——当前费率回填历史属未来信息）；
#   - 只用**共同非缺失**基金（两项各自与 ret_12m、标签同时可得的横截面）；
#   - 不做新策略、不选股、不改权重、不生成新的回测规格；
#   - dev 段（≤ 2025-02-28），holdout 不触碰；
#   - 判定用 NW(6) + **Holm 校正两项**（总试验数 2）。
#
# partial Rank IC 定义：月内把 y（未来 6 月收益）、控制变量 c=ret_12m、目标特征 x 都转秩；
#   e_y = rank(y) 对 rank(c) 回归的残差；e_x = rank(x) 对 rank(c) 回归的残差；
#   partial IC = corr(e_y, e_x)（Spearman 偏相关）。
# 若两项都没有稳定正信息 → 正式关闭 AUM/资金流路线。
import argparse
import os
from math import erfc, sqrt

import numpy as np
import pandas as pd

from backtest_strategy import DEV_END
from walk_forward_splitter import nw_tstat

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC_DIR)
PANEL_V3 = os.path.join(ROOT, "ml", "panel_v3.parquet")
ATTRS = os.path.join(ROOT, "ml", "attrs", "fund_attrs_monthly.parquet")
OUT = os.path.join(ROOT, "ml", "backtest", "partial_ic_dev.txt")
TARGETS = [("log_aum", "历史时点 log(AUM)"), ("flow_share_ratio", "份额口径资金流")]
CONTROL = "ret_12m"
LABEL = "future_ret_6m"
MIN_ROWS = 30


def t_to_p(t):
    return float(erfc(abs(t) / sqrt(2))) if t == t else float("nan")


def holm_adjust(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: (pvals[i] if pvals[i] == pvals[i] else 1.0))
    adj = [float("nan")] * m
    running = 0.0
    for rank, i in enumerate(order):
        p = pvals[i] if pvals[i] == pvals[i] else 1.0
        running = max(running, (m - rank) * p)
        adj[i] = min(running, 1.0)
    return adj


def _resid_on(v: np.ndarray, c: np.ndarray) -> np.ndarray:
    """v 对 c（含截距）OLS 的残差。"""
    X = np.column_stack([np.ones(len(c)), c])
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    return v - X @ beta


def partial_ic_monthly(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for t, sub in df.groupby("t_date"):
        d = sub[[LABEL, CONTROL, target]].dropna()
        if len(d) < MIN_ROWS:
            continue
        ry = d[LABEL].rank().to_numpy(float)
        rc = d[CONTROL].rank().to_numpy(float)
        rx = d[target].rank().to_numpy(float)
        ey = _resid_on(ry, rc)
        ex = _resid_on(rx, rc)
        if ey.std() == 0 or ex.std() == 0:
            continue
        rows.append({"t_date": t, "n": len(d), "partial_ic": float(np.corrcoef(ey, ex)[0, 1])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="部分 Rank IC 诊断（控制 ret_12m；只验 AUM 与资金流）")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    panel = pd.read_parquet(PANEL_V3, columns=["fund_code", "t_date", LABEL, CONTROL])
    panel = panel[panel["t_date"] <= DEV_END]
    attrs = pd.read_parquet(ATTRS, columns=["fund_code", "t_date", "log_aum", "flow_share_ratio"])
    df = panel.merge(attrs, on=["fund_code", "t_date"], how="inner")
    print(f"面板 dev 行 {len(panel)} | 合并属性后 {len(df)} | 截面 {df.t_date.nunique()}")

    res, lines = {}, ["=" * 84,
                      "部分 Rank IC 诊断：控制 ret_12m 后，log(AUM) / 资金流是否仍有独立信息",
                      "=" * 84,
                      f"口径：月内共同非缺失（n≥{MIN_ROWS}）→ y/控制/特征全部转秩 → "
                      "双向对 ret_12m 秩回归取残差 → 残差相关（Spearman 偏相关）",
                      f"区间：dev 段（≤ {DEV_END}）｜研究池含清盘池｜**不使用当前费率**｜holdout 不触碰",
                      "",
                      f"  {'特征':<26}{'月数':>6}{'mean partial IC':>18}{'naive t':>10}{'NW(6) t':>10}"
                      f"{'p(单)':>9}{'p(Holm)':>9}{'IC>0':>8}"]
    ps = []
    for key, label in TARGETS:
        m = partial_ic_monthly(df, key)
        res[key] = m
        s = m["partial_ic"]
        n = len(s)
        mean, sd = float(s.mean()), float(s.std(ddof=1))
        nw = float(nw_tstat(s))
        ps.append(t_to_p(nw))
        res[key + "_stat"] = {"months": n, "mean": mean, "nw": nw}
    adj = holm_adjust(ps)
    for (key, label), p, a in zip(TARGETS, ps, adj):
        st = res[key + "_stat"]
        m = res[key]
        s = m["partial_ic"]
        nt = st["mean"] / (s.std(ddof=1) / np.sqrt(len(s)))
        lines.append(f"  {label:<24}{st['months']:>7}{st['mean']:>18.4f}{nt:>10.2f}"
                     f"{st['nw']:>10.2f}{p:>9.3f}{a:>9.3f}{(s > 0).mean():>8.2f}")
    lines += ["",
              "判定（预先写死）：两项在 **Holm 校正后**均无稳定正信息（NW 不显著为正或方向为负）",
              "→ **正式关闭 AUM / 资金流路线**，不再为其设计新规格；若有一项显著为正，也只作为",
              "「候选信息」记录，不据此修改主策略，需另行预登记才能进入策略层。"]
    report = "\n".join(lines)
    print("\n" + report)
    if not args.no_save:
        with open(OUT, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        for key, _ in TARGETS:
            res[key].to_csv(os.path.join(ROOT, "ml", "backtest", f"partial_ic_{key}_monthly.csv"),
                            index=False)
        print(f"\n报告已落盘：{OUT}")


if __name__ == "__main__":
    main()
