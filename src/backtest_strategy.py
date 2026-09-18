# backtest_strategy.py：Phase 3——主策略（动量排序）组合回测引擎
#
# 口径（2026-09-18 拍板；v2 为审查修正版）：
#   组合构建：每月末按 ret_12m 排序取 Top N 等权（主配置 N=50；敏感性 20/100）
#   持有与调仓：每月调仓、持 6 个月、在持 6 期重叠；月度收益从净值财富指数现算
#   费用：申购 0.15% + 赎回 0.5%，费用前后双净值
#   v2 口径修正（2026-09-18 审查）：
#     ① 选股池用「无未来端点台账截面」（成立≥365天 + 当期披露 + 过去252日coverage≥95%，
#        与 live_score.py 同款 eligibility，**不含**"未来6月末仍披露"检查）——
#        用研究面板行选股会附带未来披露存活条件，美化回测（持有期内将终止的基金从不入选）；
#     ② 全池等权基准与组合**同入选时点**：基准 = 上月（组合入选时）的台账在座基金
#        在 [上月→本月] 的收益均值（此前用本月末基金池算上月收益，时点错位）；
#     ③ 市场状态标签**无前瞻**：截至上月末的过去12个月沪深300累计收益（不含当月）；
#        含当月则用作"当月空仓"决策即事后信息。
#   区间：dev 段（2006-03~2025-02，多轮牛熊）主研究；
#         holdout（2025-03~2026-02）**已看过**——仅作描述性对照，不再作为任何新设计的
#         "最终盲测"（新特征/状态过滤须在 dev 段滚动研究 + 积累新的向前验证期）。
# 输出：ml/backtest/portfolio_nav_{phase}_top{n}.csv + backtest_report_{phase}.txt
import argparse
import os

import numpy as np
import pandas as pd

from panel_builder import (load_fund_series, BENCH_PATH, PROJECT_ROOT, DAY_NS)
from walk_forward_splitter import LABEL_COL

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT_ROOT, "ml", "backtest")
PANEL_PATH = os.path.join(PROJECT_ROOT, "ml", "panel.parquet")

TOP_N_DEFAULT = 50
SENSITIVITY_N = [20, 50, 100]
BUY_FEE, SELL_FEE = 0.0015, 0.005
HOLD = 6
RF_ANNUAL = 0.02
BULL_TH, BEAR_TH = 0.20, -0.20
DEV_END = "2025-02-28"
# 无未来台账截面门槛（与 live_score.py 一致）
LOOKBACK = 252
MIN_AGE_DAYS = 365
MIN_COV = 0.95
MAX_GAP_DAYS = 15
HOLD_SEMANTICS = "每月调仓、持6月重叠"


def load_month_rets(series, bench_dates, month_ts):
    """每月末截面基金收益。返回（month_idx, rets_map）。
    rets_map[code][m] = code 从月末 m-1 到月末 m 的复利收益（wealth 前向保持）。"""
    month_idx = np.array(
        [int(np.searchsorted(bench_dates, np.datetime64(t, "ns").astype("int64"),
                             side="right")) - 1 for t in month_ts])
    rets_map = {}
    for code, s in series.items():
        v = s["wealth_al"][month_idx]
        rets_map[code] = v[1:] / v[:-1] - 1.0
    return month_idx, rets_map


def build_no_future_screens(series, bench_dates, month_idx, month_ts):
    """每月末的无未来端点台账截面（v2 修正①）。

    对每只基金、每月末：成立≥365自然日 + 当期15天内有披露 + 过去252交易日 coverage≥95%
    （缺失=未披露），ret_12m = W[i]/W[i-252]−1（窗口有效观测≥252 才可算）。
    不含任何未来披露端点检查；月集合来自研究面板的日期，仅用于时间轴。
    返回 screens[m] = list[(fund_code, ret_12m)]，m 与 month_ts 对齐（仅含上月已有数据索引）。
    """
    prep = {}
    for code, s in series.items():
        dates = s["dates"]
        first_ns = int(dates[0])
        i_first = int(np.searchsorted(bench_dates, first_ns, side="left"))
        nc = np.cumsum(~np.isnan(s["r_al"]))           # 日级非 NaN 累计（coverage O(1)）
        prep[code] = (first_ns, dates, i_first, nc, s["wealth_al"])
    screens = {}
    for m, i in enumerate(month_idx):
        t_ns = int(bench_dates[i])
        rows = []
        for code, (first_ns, dates, i_first, nc, w) in prep.items():
            if first_ns > t_ns - MIN_AGE_DAYS * DAY_NS:      # 成立不足 12 自然月
                continue
            j_t = int(np.searchsorted(dates, t_ns, side="right")) - 1
            if j_t < 0 or t_ns - int(dates[j_t]) > MAX_GAP_DAYS * DAY_NS:
                continue                                       # 当期无披露
            w0 = i - LOOKBACK
            if w0 < 0 or i - i_first + 1 < LOOKBACK + 1:
                continue                                       # 252 交易日窗口不足
            n_eff = int(nc[i + 1] - nc[w0])
            if n_eff / (LOOKBACK + 1) < MIN_COV:
                continue                                       # 披露完整度不足
            r12 = float(w[i] / w[w0] - 1.0)
            rows.append((code, r12))
        screens[m] = rows
    return screens


def regime_no_lookahead(bench_close, month_idx, lookback=LOOKBACK) -> list:
    """无前瞻市场状态（v2 修正③）：对区间 [m-1, m] 的标签用「截至 m-1 时刻」
    的过去 252 交易日沪深300累计收益——不含当月收益，用作当月持仓决策不构成事后信息。"""
    out = []
    for m in range(1, len(month_idx)):
        i_prev = month_idx[m - 1]
        if i_prev - lookback < 0:
            out.append("mix")
            continue
        c = float(bench_close[i_prev] / bench_close[i_prev - lookback] - 1.0)
        out.append("bull" if c > BULL_TH else ("bear" if c < BEAR_TH else "mix"))
    return out


def run_strategy(screens, month_ts, rets_map, bench_ret_m, pool_ret_m, regime_m,
                 top_n: int, buy_fee: float, sell_fee: float, start_idx: int = 0) -> pd.DataFrame:
    """逐月回测主循环（收益先算、调仓在后——只用 ≤t 信息）。"""
    rows = []
    active = []            # 在持队列：[[建仓月索引 m, codes], ...]
    nav_g = nav_n = 1.0
    for m in range(start_idx + 1, len(month_ts)):
        # ① 本月组合收益：建仓于 ≤m-1 的在持组合在 [m-1, m] 的表现
        part = []
        for bm, codes in active:
            vals = [rets_map[c][m - 1] for c in codes if c in rets_map]
            part.append(float(np.mean(vals)) if vals else 0.0)
        r_gross = float(np.mean(part)) if part else 0.0
        n_held = len(active)
        # ② 调仓（为下月）：到期移出 + 用本月台账截面选新仓
        expired = [a for a in active if m - a[0] >= HOLD]
        active = [a for a in active if m - a[0] < HOLD]
        top = sorted(screens[m], key=lambda x: x[1], reverse=True)[:top_n]
        active.append([m, [c for c, _ in top]])
        # ③ 费用（实际发生月）
        w = 1.0 / HOLD
        fee = w * sell_fee * (len(expired) > 0) + w * buy_fee
        r_net = (1.0 + r_gross) * (1.0 - fee) - 1.0
        nav_g *= (1.0 + r_gross)
        nav_n *= (1.0 + r_net)
        rows.append({
            "t_date": month_ts[m], "regime": regime_m[m - 1],
            "r_gross": r_gross, "r_net": r_net,
            "nav_gross": nav_g, "nav_net": nav_n,
            "bench_ret": bench_ret_m[m - 1], "pool_ret": pool_ret_m[m - 1],
            "n_held": n_held, "turnover_2sided": 2.0 * w,
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, label: str) -> dict:
    n = len(df)
    nav = df["nav_net"].values
    nav_g = df["nav_gross"].values
    rets = df["r_net"].values
    ann_ret = nav[-1] ** (12.0 / n) - 1.0
    ann_gross = nav_g[-1] ** (12.0 / n) - 1.0
    vol = float(np.std(rets, ddof=1)) * np.sqrt(12)
    sharpe = (np.mean(rets) - RF_ANNUAL / 12.0) / np.std(rets, ddof=1) * np.sqrt(12)
    peak = np.maximum.accumulate(nav)
    mdd = float(np.min(nav / peak - 1.0))
    b_nav = np.cumprod(1.0 + df["bench_ret"].values)
    b_ann = b_nav[-1] ** (12.0 / n) - 1.0
    p_nav = np.cumprod(1.0 + df["pool_ret"].values)
    p_ann = p_nav[-1] ** (12.0 / n) - 1.0
    return {
        "label": label, "n_months": n,
        "ann_ret": ann_ret, "ann_ret_gross": ann_gross,
        "fee_drag_pp": (ann_gross - ann_ret) * 100.0,
        "vol": vol, "sharpe": sharpe, "mdd": mdd,
        "calmar": ann_ret / abs(mdd) if mdd != 0 else np.nan,
        "bench_ann": b_ann, "pool_ann": p_ann,
        "monthly_turnover_2sided": float(df["turnover_2sided"].mean()),
    }


def main():
    ap = argparse.ArgumentParser(description="Phase 3 主策略组合回测（v2 审查修正）")
    ap.add_argument("--phase", default="dev", choices=["dev", "holdout", "all"])
    ap.add_argument("--top-n", type=int, default=TOP_N_DEFAULT)
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--min-funds", type=int, default=50,
                    help="回测起点：台账截面基金数至少该值")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    panel_all = pd.read_parquet(PANEL_PATH)
    if args.phase == "dev":
        panel = panel_all[panel_all["t_date"] <= DEV_END].copy()
    elif args.phase == "holdout":
        panel = panel_all[panel_all["t_date"] > DEV_END].copy()
    else:
        panel = panel_all
    month_ts = pd.DatetimeIndex(np.sort(panel["t_date"].unique()))

    bench = pd.read_csv(BENCH_PATH, parse_dates=["date"]).sort_values("date")
    bench_dates = bench["date"].to_numpy(dtype="datetime64[ns]").astype("int64")
    print("加载净值并构建台账截面（约 1-3 分钟）…")
    series = load_fund_series(bench_dates)
    month_idx, rets_map = load_month_rets(series, bench_dates, month_ts)
    screens = build_no_future_screens(series, bench_dates, month_idx, month_ts)
    print(f"截面 {len(month_ts)} 个月末；台账基金 {len(series)} 只；"
          f"首月可评 {len(screens[0])} 只 / 末月 {len(screens[len(month_ts) - 1])} 只")

    bc = bench["close"].to_numpy(dtype=float)
    bench_ret_m = bc[month_idx[1:]] / bc[month_idx[:-1]] - 1.0
    # 全池等权基准：与组合**同入选时点**——上月（m-1）台账在座基金在 [m-1, m] 的收益（v2 修正②）
    pool_ret_m = []
    for m in range(1, len(month_ts)):
        codes = [c for c, _ in screens[m - 1]]
        vals = [rets_map[c][m - 1] for c in codes if c in rets_map]
        pool_ret_m.append(float(np.mean(vals)) if vals else 0.0)
    pool_ret_m = np.array(pool_ret_m)
    regime_m = regime_no_lookahead(bc, month_idx)

    start_idx = 0
    for i, m in enumerate(range(len(month_ts))):
        if len(screens[i]) >= args.min_funds:
            start_idx = i
            break
    print(f"回测起点：{month_ts[start_idx].date()}（台账截面基金数 ≥ {args.min_funds}）")

    df = run_strategy(screens, month_ts, rets_map, bench_ret_m, pool_ret_m,
                      regime_m, args.top_n, BUY_FEE, SELL_FEE, start_idx)
    os.makedirs(OUT_DIR, exist_ok=True)
    if not args.no_save:
        df.to_csv(os.path.join(OUT_DIR, f"portfolio_nav_{args.phase}_top{args.top_n}.csv"),
                  index=False)
        print(f"逐月明细已落盘（{len(df)} 个月）")

    s = summarize(df, f"Top{args.top_n}（{args.phase} 段）")
    lines = ["=" * 72,
             f"Phase 3 主策略组合回测报告 v2（{args.phase} 段，费用后主口径）",
             f"组合：Top{args.top_n} 等权；{HOLD_SEMANTICS}；费用 申购{BUY_FEE:.2%}+赎回{SELL_FEE:.2%}",
             "口径：无未来端点台账截面选股 | 全池基准同入选时点 | 状态标签无前瞻",
             "=" * 72,
             f"区间       : {df.t_date.min().date()} ~ {df.t_date.max().date()}（{s['n_months']} 个月）",
             f"年化收益   : 费用后 {s['ann_ret']:.2%}（费用前 {s['ann_ret_gross']:.2%} → "
             f"侵蚀 {s['fee_drag_pp']:.2f} pp/年）",
             f"对比基准   : 沪深300 {s['bench_ann']:.2%} / 全池等权 {s['pool_ann']:.2%}",
             f"年化波动   : {s['vol']:.2%} | 夏普 {s['sharpe']:.2f} | MDD {s['mdd']:.2%} | Calmar {s['calmar']:.2f}",
             f"月双边换手 : {s['monthly_turnover_2sided']:.2%}（年化单边≈{s['monthly_turnover_2sided']*6:.0%}）"]
    lines.append("")
    lines.append("市场状态分解（费用后月收益，状态=截至上月末的过去12月基准累计，**无前瞻**）:")
    for reg in ["bull", "mix", "bear"]:
        sub = df[df.regime == reg]
        if len(sub):
            lines.append(f"  {reg:<5s}: {len(sub):>4d} 个月 | 月均 {sub.r_net.mean():+.4%} | "
                         f"年化≈{(1+sub.r_net).prod() ** (12/len(sub)) - 1:+.2%}")
    report = "\n".join(lines)
    print(report)

    if args.sensitivity:
        print("\n===== 敏感性：Top N（费用后） =====")
        sn = []
        for n in SENSITIVITY_N:
            d = run_strategy(screens, month_ts, rets_map, bench_ret_m, pool_ret_m,
                             regime_m, n, BUY_FEE, SELL_FEE, start_idx)
            ss = summarize(d, f"Top{n}")
            sn.append(ss)
            if not args.no_save:
                d.to_csv(os.path.join(OUT_DIR, f"portfolio_nav_{args.phase}_top{n}.csv"),
                         index=False)
        print(f"{'N':>5s}{'年化':>9s}{'波动':>9s}{'夏普':>7s}{'MDD':>9s}{'侵蚀pp':>8s}")
        for ss in sn:
            print(f"{ss['label'][3:]:>5s}{ss['ann_ret']:>9.2%}{ss['vol']:>9.2%}"
                  f"{ss['sharpe']:>7.2f}{ss['mdd']:>9.2%}{ss['fee_drag_pp']:>8.2f}")
    if not args.no_save:
        with open(os.path.join(OUT_DIR, f"backtest_report_{args.phase}.txt"),
                  "w", encoding="utf-8") as f:
            f.write(report + "\n")
    if args.phase == "holdout":
        print("\n⚠️ holdout 段（2025-03~2026-02）已被用于 holdout 终审与本轮回测——"
              "**已看过**。本结果仅作描述性对照，不得作为任何新设计的'最终盲测'。")
    print("\n（现存池条件性回测：未含已清盘基金；v2 已去除面板未来披露存活条件）")


if __name__ == "__main__":
    main()