# excess_regime.py：诊断——"组合 − 可执行全池"超额的**状态分解**（不制定任何新规则）
#
# 用途（2026-09-19 用户指定）：把"动量组合相对可执行全池的超额"按市场状态拆开看，
#   纯粹是**诊断**，不用于设计择时/降仓规则——dev 段已被反复查看，看到状态差异后
#   再针对性设计规则会把历史偶然性写进策略（用户明确要求把择时排到后面）。
#
# 输入（都已有落盘产物，不重跑回测）：
#   ① 现存池版：ml/backtest/portfolio_nav_dev_top50.csv（backtest_strategy v2 输出）
#   ② 含清盘池版：ml/backtest/style_variants_{size,size_growth}_del_monthly.csv 的
#      variant="orig_all"（原动量、全台账池、含已清盘基金）
# 输出：ml/backtest/excess_regime_dev.txt
import argparse
import os

import numpy as np
import pandas as pd

from walk_forward_splitter import nw_tstat

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC_DIR)
BT = os.path.join(ROOT, "ml", "backtest")
REGIMES = ("bull", "mix", "bear")
REGIME_CN = {"bull": "上行(过去12月基准>+20%)", "mix": "震荡", "bear": "下行(<-20%)"}


def load_source(path: str, variant: str = None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["t_date"])
    if variant is not None:
        if "variant" not in df.columns:
            raise SystemExit(f"{path} 没有 variant 列")
        df = df[df["variant"] == variant].copy()
    cols = {"t_date": "t_date", "regime": "regime", "r_net": "r_net",
            "pool_ret_net": "pool_ret_net"}
    df = df[[c for c in cols if c in df.columns]].dropna(subset=["r_net", "pool_ret_net"])
    df["excess"] = df["r_net"] - df["pool_ret_net"]
    return df


def decompose(df: pd.DataFrame, label: str) -> list:
    out = [f"--- {label}（{df.t_date.min().date()} ~ {df.t_date.max().date()}，"
           f"{len(df)} 个月）---",
           f"  全期：月超额 {df.excess.mean():+.4%} | naive t "
           f"{df.excess.mean()/(df.excess.std(ddof=1)/np.sqrt(len(df))):+.2f} | "
           f"NW(6) t {nw_tstat(df.excess):+.2f} | 正超额月占比 {(df.excess > 0).mean():.1%}",
           f"  {'状态':<20}{'月数':>6}{'月均超额':>12}{'年化超额':>12}{'naive t':>10}{'NW(6) t':>10}{'胜率':>8}"]
    for reg in REGIMES:
        sub = df[df.regime == reg]
        if not len(sub):
            continue
        m = sub.excess.mean()
        ann = (1.0 + sub.excess).prod() ** (12.0 / len(sub)) - 1.0
        nt = m / (sub.excess.std(ddof=1) / np.sqrt(len(sub))) if len(sub) > 2 else np.nan
        out.append(f"  {REGIME_CN.get(reg, reg):<18}{len(sub):>6}{m:>12.4%}{ann:>12.2%}"
                   f"{nt:>10.2f}{nw_tstat(sub.excess):>10.2f}"
                   f"{(sub.excess > 0).mean():>8.1%}")
    # 状态之间的差（bull − bear）是否显著：合并序列检验用简单双样本 t（仅诊断）
    b, r = df[df.regime == "bull"].excess, df[df.regime == "bear"].excess
    if len(b) > 2 and len(r) > 2:
        se = np.sqrt(b.var(ddof=1) / len(b) + r.var(ddof=1) / len(r))
        out.append(f"  诊断：bull − bear 超额差 {b.mean() - r.mean():+.4%}"
                   f"（双样本 naive t {(b.mean() - r.mean())/se:+.2f}；"
                   f"状态为无前瞻标签，但**不据此设计规则**）")
    return out


def main():
    ap = argparse.ArgumentParser(description="组合−可执行全池超额的（诊断性）状态分解")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    lines = ["=" * 84,
             "诊断：动量组合相对可执行全池的超额 —— 按市场状态分解（不用于制定规则）",
             "=" * 84,
             "口径：超额 = 组合费用后月收益 − 可执行全池（同费率）月收益；状态 = 截至上月末的",
             "      过去12月沪深300累计（无前瞻，>+20% 上行 / <−20% 下行 / 其余震荡）",
             ""]
    srcs = [("现存池（backtest_strategy v2 Top50）",
             os.path.join(BT, "portfolio_nav_dev_top50.csv"), None)]
    for fp, tag in (("style_variants_size_del_monthly.csv", "含清盘池（mkt+size 因子集的原动量）"),
                    ("style_variants_size_growth_sector_del_monthly.csv",
                     "含清盘池（9 因子集的原动量）")):
        p = os.path.join(BT, fp)
        if os.path.exists(p):
            srcs.append((tag, p, "orig_all"))
    for label, path, variant in srcs:
        if not os.path.exists(path):
            lines.append(f"（缺 {os.path.basename(path)}，跳过）")
            continue
        lines += decompose(load_source(path, variant), label)
        lines.append("")
    report = "\n".join(lines)
    print(report)
    if not args.no_save:
        with open(os.path.join(BT, "excess_regime_dev.txt"), "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\n报告已落盘：{os.path.join(BT, 'excess_regime_dev.txt')}")


if __name__ == "__main__":
    main()
