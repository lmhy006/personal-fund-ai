# vol_target.py：风险控制 v1 —— 事前波动目标（用户 2026-09-19 预登记草案的固定主规格）
#
# 固定口径（**不搜索参数**，用户指定）：
#   风险资产   ：含清盘池的**原动量 Top50**（台账截面选股、每月调仓持 6 月重叠，与原策略一致）
#   风险估计   ：最近 **12 个已完成月份**的原策略**费用后**收益的年化波动（未缩放，避免缩放收益反噬估计）
#   信息时点   ：月末 t 只用截至 t 已实现收益 → 决定 t+1 月仓位
#   目标波动   ：年化 15%
#   风险仓位   ：w_t = min(1, 15% / σ̂_t)，**不允许杠杆**
#   现金收益   ：年化 2%（项目既有口径）
#   调整频率   ：每月一次；暖机期 12 个月（所有对照同区间评价）
#   成本       ：组合内部换仓费用照旧；**总仓位变化另计**申购 0.15% / 赎回 0.5%
#   市场状态   ：**不使用** bull/bear 标签，不做择时
#   数据范围   ：仅 dev（≤ 2025-02-28）；holdout 不触碰
#
# 收益：r_VT,t+1 = w_t·r_momentum,t+1 + (1−w_t)·r_cash,t+1 − 仓位调整成本
#
# **四组对照**（用户要求，缺一不可——否则无法区分"动量被改善"与"任何风险资产降仓的机械结果"）：
#   ① 原动量 ② 波动目标动量 ③ 原全池等权（可执行费率） ④ 波动目标全池等权
#
# 验收门槛（同时满足才进影子运行）：
#   MDD 改善 ≥ 10pp｜年化 ≥ 原策略 75%｜Sharpe 或 Calmar 至少一个提高｜
#   年均新增费用 ≤ 0.5pp｜相对"同样波动控制的全池"没有进一步恶化
import argparse
import os

import numpy as np
import pandas as pd

from backtest_strategy import (BUY_FEE, DEV_END, HOLD, SELL_FEE, build_no_future_screens,
                               load_month_rets, pool_rebalance_cost,
                               regime_no_lookahead, run_strategy)
from panel_builder import BENCH_PATH, PROJECT_ROOT, load_fund_series
from termination_audit import build_inputs
from walk_forward_splitter import nw_tstat

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT_ROOT, "ml", "backtest")
TARGET_VOL = 0.15
VOL_WINDOW = 12
CASH_ANNUAL = 0.02
MIN_FUNDS = 50          # 与 backtest_strategy 默认一致（起点 2005-07）
TOP_N = 50
# 验收门槛
GATE_MDD_PP = 10.0      # 最大回撤改善（百分点）
GATE_RET_KEEP = 0.75    # 年化保留比例
GATE_FEE_PP = 0.5       # 年均新增费用上限（百分点）


def vol_target(r_net: np.ndarray, target=TARGET_VOL, window=VOL_WINDOW,
               cash_annual=CASH_ANNUAL, buy_fee=BUY_FEE, sell_fee=SELL_FEE):
    """对一条月度费用后收益序列施加事前波动目标。返回 (w, r_vt, cost, turnover)。

    w[m] 用**截至 m-1 的 12 个已实现月收益**（未缩放）估计 σ̂，然后作用于第 m 月。
    """
    n = len(r_net)
    r_cash_m = (1.0 + cash_annual) ** (1.0 / 12.0) - 1.0
    w = np.full(n, np.nan)
    for m in range(window, n):
        past = r_net[m - window:m]
        sd = float(np.std(past, ddof=1)) * np.sqrt(12.0)
        w[m] = min(1.0, target / sd) if sd > 0 else 1.0
    r_vt = np.full(n, np.nan)
    cost = np.zeros(n)
    turn = np.zeros(n)
    for m in range(window, n):
        prev_w = w[m - 1] if m - 1 >= window else 1.0     # 暖机结束首月：此前按满仓对待
        dw = w[m] - prev_w
        cost[m] = dw * buy_fee if dw > 0 else (-dw) * sell_fee
        turn[m] = abs(dw)
        r_vt[m] = w[m] * r_net[m] + (1.0 - w[m]) * r_cash_m - cost[m]
    return w, r_vt, cost, turn


def stats(r: np.ndarray, label: str) -> dict:
    r = r[~np.isnan(r)]
    nav = np.cumprod(1.0 + r)
    n = len(r)
    ann = nav[-1] ** (12.0 / n) - 1.0
    vol = float(np.std(r, ddof=1)) * np.sqrt(12.0)
    sharpe = (float(np.mean(r)) - CASH_ANNUAL / 12.0) / np.std(r, ddof=1) * np.sqrt(12.0)
    mdd = float(np.min(nav / np.maximum.accumulate(nav) - 1.0))
    return {"label": label, "months": n, "ann_ret": ann, "vol": vol, "sharpe": sharpe,
            "mdd": mdd, "calmar": ann / abs(mdd) if mdd else np.nan, "nav": nav}


def main():
    ap = argparse.ArgumentParser(description="风险控制 v1：事前波动目标（固定主规格 + 四组对照）")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    print("加载输入（含清盘池）…")
    inp = build_inputs(include_delisted=True)
    month_ts, month_idx, rets_map = inp["month_ts"], inp["month_idx"], inp["rets_map"]
    screens, series = inp["screens"], inp["series"]
    bench = pd.read_csv(BENCH_PATH, parse_dates=["date"]).sort_values("date")
    bc = bench["close"].to_numpy(dtype=float)
    bench_ret_m = bc[month_idx[1:]] / bc[month_idx[:-1]] - 1.0
    regime = regime_no_lookahead(bc, month_idx)

    # 全池等权（可执行费率）
    pool = np.array([np.mean([rets_map[c][m - 1] for c in [x[0] for x in screens[m - 1]]
                              if c in rets_map]) if screens[m - 1] else 0.0
                     for m in range(1, len(month_ts))])
    pool_cost = pool_rebalance_cost(screens, rets_map, pool)
    pool_net = (1.0 + pool) * (1.0 - pool_cost) - 1.0

    start = next(i for i, r in enumerate(screens) if len(r) >= MIN_FUNDS)
    df_mom = run_strategy(screens, month_ts, rets_map, bench_ret_m, pool, pool_net, regime,
                          TOP_N, BUY_FEE, SELL_FEE, start, stratify=False)
    off = start + 1                      # run_strategy 的第一行对应 month_ts[start+1]
    t_dates = df_mom["t_date"].to_numpy()
    r_mom = df_mom["r_net"].to_numpy(dtype=float)
    r_pool = df_mom["pool_ret_net"].to_numpy(dtype=float)

    # 四组
    w_m, vt_m, cost_m, turn_m = vol_target(r_mom)
    w_p, vt_p, cost_p, turn_p = vol_target(r_pool)
    res = {
        "① 原动量": stats(r_mom, "原动量"),
        "② 波动目标动量": stats(vt_m, "波动目标动量"),
        "③ 原全池等权": stats(r_pool, "原全池等权"),
        "④ 波动目标全池等权": stats(vt_p, "波动目标全池等权"),
    }
    warm = VOL_WINDOW                      # 各对照同区间：去掉暖机期
    mask = ~np.isnan(vt_m)
    # 暖机后同区间重算（对照可比）
    res_same = {
        "① 原动量": stats(r_mom[mask], "原动量"),
        "② 波动目标动量": stats(vt_m[mask], "波动目标动量"),
        "③ 原全池等权": stats(r_pool[mask], "原全池等权"),
        "④ 波动目标全池等权": stats(vt_p[mask], "波动目标全池等权"),
    }
    fee_m_pp = float(np.nansum(cost_m) / (mask.sum() / 12.0)) * 100.0
    fee_p_pp = float(np.nansum(cost_p) / (mask.sum() / 12.0)) * 100.0
    diff_vs_pool = pd.Series(vt_m[mask] - vt_p[mask]).dropna()
    diff_vs_mom = pd.Series(vt_m[mask] - r_mom[mask]).dropna()

    # —— 验收门槛 ——
    a, b = res_same["① 原动量"], res_same["② 波动目标动量"]
    gates = [
        ("最大回撤改善 ≥ 10pp", (b["mdd"] - a["mdd"]) * 100.0, GATE_MDD_PP,
         (b["mdd"] - a["mdd"]) * 100.0 >= GATE_MDD_PP),
        ("年化收益保留 ≥ 原策略 75%", b["ann_ret"] / a["ann_ret"], GATE_RET_KEEP,
         b["ann_ret"] / a["ann_ret"] >= GATE_RET_KEEP),
        ("Sharpe 或 Calmar 至少一个提高", max(b["sharpe"] - a["sharpe"], b["calmar"] - a["calmar"]),
         0.0, (b["sharpe"] > a["sharpe"]) or (b["calmar"] > a["calmar"])),
        ("年均新增费用 ≤ 0.5pp", fee_m_pp, GATE_FEE_PP, fee_m_pp <= GATE_FEE_PP),
        ("相对同样风控的全池未进一步恶化", float(diff_vs_pool.mean()), 0.0,
         bool(diff_vs_pool.mean() >= 0)),
    ]
    passed = all(g[3] for g in gates)

    lines = ["=" * 96,
             "风险控制 v1：事前波动目标（固定主规格，不搜索参数；四组对照）",
             "=" * 96,
             f"口径：风险资产=含清盘池原动量 Top{TOP_N}｜σ̂=最近 {VOL_WINDOW} 个已完成月份的年化波动"
             f"（未缩放）",
             f"      目标波动 {TARGET_VOL:.0%}｜w=min(1, 目标/σ̂)｜不允许杠杆｜现金 {CASH_ANNUAL:.0%}/年"
             f"｜月频｜暖机 {VOL_WINDOW} 月",
             f"      仓位调整成本 申购{BUY_FEE:.2%}+赎回{SELL_FEE:.2%}｜组合内部换仓费照旧｜"
             f"**不使用 bull/bear 状态**",
             f"区间：{pd.Timestamp(t_dates[mask][0]).date()} ~ {pd.Timestamp(t_dates[mask][-1]).date()}"
             f"（{int(mask.sum())} 个月，暖机后同区间对照）｜仅 dev，holdout 不触碰",
             "",
             f"{'方案':<20}{'年化':>9}{'波动':>9}{'Sharpe':>9}{'MDD':>10}{'Calmar':>9}"]
    for k in ("① 原动量", "② 波动目标动量", "③ 原全池等权", "④ 波动目标全池等权"):
        s = res_same[k]
        lines.append(f"{k:<20}{s['ann_ret']:>9.2%}{s['vol']:>9.2%}{s['sharpe']:>9.2f}"
                     f"{s['mdd']:>10.2%}{s['calmar']:>9.2f}")
    lines += ["",
              f"仓位统计（波动目标动量）：均值 {np.nanmean(w_m):.2f}｜中位 {np.nanmedian(w_m):.2f}"
              f"｜最低 {np.nanmin(w_m):.2f}｜满仓月占比 {float(np.nanmean(w_m >= 0.999)):.1%}",
              f"仓位调整成本：动量组年均 {fee_m_pp:.3f}pp｜全池组年均 {fee_p_pp:.3f}pp",
              "",
              "配对检验（NW(6)）：",
              f"  波动目标动量 − 原动量        ：月均 {diff_vs_mom.mean():+.4%}｜NW {nw_tstat(diff_vs_mom):+.2f}",
              f"  波动目标动量 − 波动目标全池  ：月均 {diff_vs_pool.mean():+.4%}｜NW {nw_tstat(diff_vs_pool):+.2f}"
              "   ← 判「是否只是任何风险资产降仓的机械结果」",
              "",
              "验收门槛（用户给定，需**同时满足**）："]
    for name, val, thr, ok in gates:
        unit = "pp" if "pp" in name else ("倍" if "保留" in name else "")
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name:<32} 实测 {val:+.4f}{unit}（阈值 {thr}）")
    lines += ["", f"→ 总判定：**{'达标，可进入影子运行' if passed else '未达标 → 停止，不遍历窗口与目标波动率'}**",
              "",
              "⚠️ 解读边界：本规格的目标是**改善回撤与持有体验**（事前波动目标），不是寻找 alpha；",
              "   因此 NW 配对收益检验继续报告但不作为唯一门槛。第④组的存在是为了排除「任何风险资产",
              "   降仓都会改善回撤」的机械解释——只有相对④不恶化，②的改善才算动量策略层面的改善。"]
    report = "\n".join(lines)
    print("\n" + report)
    if not args.no_save:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "vol_target_dev.txt"), "w", encoding="utf-8") as f:
            f.write(report + "\n")
        pd.DataFrame({
            "t_date": t_dates, "r_mom": r_mom, "r_pool": r_pool,
            "w_mom": w_m, "r_vt_mom": vt_m, "cost_vt_mom": cost_m, "turn_vt_mom": turn_m,
            "w_pool": w_p, "r_vt_pool": vt_p, "cost_vt_pool": cost_p, "turn_vt_pool": turn_p,
        }).to_csv(os.path.join(OUT_DIR, "vol_target_monthly.csv"), index=False)
        print(f"\n报告与逐月明细已落盘：{OUT_DIR}")


if __name__ == "__main__":
    main()
