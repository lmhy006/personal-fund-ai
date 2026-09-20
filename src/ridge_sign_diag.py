# ridge_sign_diag.py：Ridge 反号的定点诊断（用户 2026-09-19 指定）
#
# 背景：v3 面板上单因子 Ridge 有 40/237 个月出现 ic_model = −ic_base（预测与 ret_12m 完全反序）。
# 数学上单特征 Ridge 只有 ŷ = β·z(ret_12m)：β>0 → 排序同向；β<0 → 排序完全反向。
# 因此"反号"应当等价于"该折学出负系数"。但**折内最小化池化 MSE**与**评价用月度截面 Rank IC
# 再按月等权**是两个不同的目标：样本多、振幅大的月份对系数权重更大，扩展窗口的池化协方差
# 可能为负，即使平均月度 Rank IC 仍为正。
#
# 本脚本只做诊断（不改主策略、不改面板、不重抓数据），逐折记录并回答四问：
#   ① coef_ / X'y / 训练样本数 / ret_12m 缺失填补比例
#   ② 只用 ret_12m 原始非缺失训练行重新拟合 → 反号区间是否消失
#   ③ 仅诊断用的"月内百分位排名特征"版本（让训练输入更接近月度 Rank IC 的评价口径）
#   ④ 折内系数符号 ↔ 当月 ic_model/ic_base 符号是否一一对应
import argparse
import os

import numpy as np
import pandas as pd

from model_ridge import label_values, ridge_fit
from walk_forward_splitter import (MIN_TEST_ROWS_FOR_IC, FoldPreprocessor,
                                   WalkForwardSplitter, rank_ic)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC_DIR)
PANEL_V3 = os.path.join(ROOT, "ml", "panel_v3.parquet")
REF = os.path.join(ROOT, "ml", "experiments", "ridge_v3panel_r12_monthly.csv")
OUT = os.path.join(ROOT, "ml", "backtest", "ridge_sign_diag.txt")
FEAT = "ret_12m"
ALPHA = 0.001          # 实测该 tag 的 alpha_sel 全为 0.001，固定以复现


def _month_pct_rank(s: pd.Series, groups: pd.Series) -> pd.Series:
    """月内百分位排名（缺失保持 NaN）。"""
    return s.groupby(groups).rank(pct=True)


def fit_variant(fold, mode: str) -> dict:
    """三种口径拟合，返回该折的系数与预测统计。

    mode: 'base'  = 原口径（全训练行 + 中位数填补 + 全局 z-score）
          'clean' = 只用 ret_12m 非缺失训练行拟合（填补不再进入系数）
          'rank'  = 仅诊断：特征换成**月内百分位排名**（训练与测试各自按月内排名）
    """
    if mode == "clean":
        train = fold.train[fold.train[FEAT].notna()]
        if len(train) < 50:
            return None
    else:
        train = fold.train

    if mode == "rank":
        tr_x = _month_pct_rank(train[FEAT], train["t_date"]).fillna(0.5)
        te_x = _month_pct_rank(fold.test[FEAT], fold.test["t_date"]).fillna(0.5)
        X = ((tr_x - tr_x.mean()) / (tr_x.std(ddof=1) or 1.0)).to_numpy().reshape(-1, 1)
        Xte = ((te_x - tr_x.mean()) / (tr_x.std(ddof=1) or 1.0)).to_numpy().reshape(-1, 1)
    else:
        pre = FoldPreprocessor().fit(train, [FEAT])
        X = pre.transform(train).to_numpy()
        Xte = pre.transform(fold.test).to_numpy()

    y = label_values(train, "demean")
    ok = ~np.isnan(y)
    X, y = X[ok], y[ok]
    beta = ridge_fit(X, y, ALPHA)
    pred = Xte @ beta
    test = fold.test.assign(_pred=pred)
    al = test[test[FEAT].notna() & test["_pred"].notna()]
    ic_m, n = rank_ic(al, "_pred") if len(al) else (np.nan, 0)
    ic_b, _ = rank_ic(al, FEAT) if len(al) else (np.nan, np.nan)
    return {"beta": float(beta[0]), "Xty": float((X[:, 0] * y).sum()),
            "train_rows": len(train), "train_rows_used": int(ok.sum()),
            "fill_ratio": float(fold.train[FEAT].isna().mean()),
            "n_test": n, "ic_model": ic_m, "ic_base": ic_b}


def main():
    ap = argparse.ArgumentParser(description="Ridge 反号的定点诊断（只诊断）")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    panel = pd.read_parquet(PANEL_V3)
    spl = WalkForwardSplitter(panel)
    rows = []
    for fold in spl.folds("dev"):
        r = {"t_date": fold.t}
        for mode in ("base", "clean", "rank"):
            v = fit_variant(fold, mode)
            if v is None:
                continue
            for k, val in v.items():
                r[f"{mode}_{k}"] = val
        rows.append(r)
    df = pd.DataFrame(rows)
    print(f"折数 {len(df)}")

    # 对齐参考结果（已落盘的 model_ridge 逐月文件），核对符号对应关系
    ref = None
    if os.path.exists(REF):
        ref = pd.read_csv(REF, parse_dates=["t_date"])[
            ["t_date", "n_test", "ic_model", "ic_base_ret12", "alpha_sel"]].rename(
            columns={"ic_model": "ref_ic_model", "ic_base_ret12": "ref_ic_base",
                     "n_test": "ref_n_test", "alpha_sel": "ref_alpha"})
        df = df.merge(ref, on="t_date", how="left")
    df["sign_flip"] = np.sign(df["base_beta"]) < 0
    df["ic_opposite"] = np.sign(df["base_ic_model"]) != np.sign(df["base_ic_base"])
    if "ref_ic_model" in df.columns:
        df["ref_opposite"] = np.sign(df["ref_ic_model"]) != np.sign(df["ref_ic_base"])

    lines = ["=" * 88,
             "Ridge 反号定点诊断：池化 MSE 目标 vs 月度排序评价口径",
             "=" * 88,
             f"面板 ml/panel_v3.parquet | 单特征 {FEAT} | alpha={ALPHA}（实测该 tag 全为此值）",
             f"折数 {len(df)}",
             "",
             "【问题②③】三种拟合口径下，折内系数为负的月数与反号月数："]
    for mode, name in (("base", "原口径（全训练行 + 中位数填补）"),
                       ("clean", "只用 ret_12m 非缺失训练行"),
                       ("rank", "月内百分位排名特征（仅诊断）")):
        if f"{mode}_beta" not in df.columns:
            continue
        neg = int((df[f"{mode}_beta"] < 0).sum())
        opp = int((np.sign(df[f"{mode}_ic_model"]) != np.sign(df[f"{mode}_ic_base"])).sum())
        ic_m = df[f"{mode}_ic_model"].mean()
        lines.append(f"  {name:<34} 负系数月 {neg:>3} | 反号月 {opp:>3} | "
                     f"模型 IC 均值 {ic_m:+.4f}")

    lines += ["", "【问题④】折内系数符号 ↔ 当月 ic_model/ic_base 符号 的对应关系："]
    both = int((df["sign_flip"] & df["ic_opposite"]).sum())
    only_neg = int((df["sign_flip"] & ~df["ic_opposite"]).sum())
    only_opp = int((~df["sign_flip"] & df["ic_opposite"]).sum())
    lines.append(f"  负系数且反号 {both} | 负系数但不反号 {only_neg} | 正系数却反号 {only_opp}")
    if "ref_opposite" in df.columns:
        agree = int((df["ref_opposite"] == df["ic_opposite"]).sum())
        lines.append(f"  与已落盘 model_ridge 结果的反号判定一致月份：{agree}/{len(df)}")
    lines.append("")
    lines.append("诊断窗口（负系数 = 池化协方差为负）：")
    neg = df[df["sign_flip"]]["t_date"]
    if len(neg):
        runs, start, prev = [], None, None
        for t in neg:
            if start is None:
                start = prev = t
            elif (t - prev).days <= 40:
                prev = t
            else:
                runs.append((start, prev))
                start = prev = t
        if start is not None:
            runs.append((start, prev))
        for a, b in runs:
            lines.append(f"  {a.date()} ~ {b.date()}（{int(((df.t_date >= a) & (df.t_date <= b)).sum())} 折）")
    lines += ["",
              "判定（用户给定）：若负系数**完整解释**反号月 → 文档改称「池化 MSE 与月度排序目标不一致」，",
              "不再称为管线自证失败；若存在「系数为正却反号」的月份 → 继续查标准化/列顺序/预测落盘。"]
    report = "\n".join(lines)
    print("\n" + report)
    if not args.no_save:
        with open(OUT, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        df.to_csv(os.path.join(ROOT, "ml", "backtest", "ridge_sign_diag_monthly.csv"), index=False)
        print(f"\n报告已落盘：{OUT}")


if __name__ == "__main__":
    main()
