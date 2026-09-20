# feature_group3.py：第三组特征（AUM / 费率 / 资金流）四规格对照
#
# 规格与判据**已预登记**在 ml/backtest/feature_group3_prereg.md（写在任何运行之前）：
#   S1 基线  = ret_12m
#   S2      = z(ret_12m) + z(log_aum)
#   S3      = z(ret_12m) + z(−fee_total)
#   S4（主）= z(ret_12m) + z(log_aum) + z(−fee_total) + z(flow_share_ratio)
#   合成固定为**当月横截面 z-score 后等权平均**；某分量缺失时该基金只用可得分量（不退化为 0）。
#
# 评价（预先固定）：① 相对原动量的费用后配对差（NW(6)）→ 非基线 3 规格做 **Holm 校正**、
#   报告总试验数 = 4；② 相对可执行全池的费用后超额（NW(6)）；③ 分 4 段年化方向；
#   ④ 换手 / MDD / 波动；⑤ 属性覆盖率（按年份）。
#
# 口径：台账截面（无未来端点）、Top50 等权、每月调仓持 6 月重叠、费用 申购0.15%+赎回0.5%；
#       研究池含清盘基金（--include-delisted 等价）；区间 dev（≤ 2025-02-28）；holdout 不参与。
import argparse
import os
from math import erfc, sqrt

import numpy as np
import pandas as pd

from backtest_strategy import (BUY_FEE, DELISTED_HISTORY_DIR, DEV_END, HOLD, OUT_DIR,
                               SELL_FEE, build_no_future_screens, load_month_rets)
from panel_builder import BENCH_PATH, PROJECT_ROOT, load_fund_series
from walk_forward_splitter import nw_tstat

ATTRS_PATH = os.path.join(PROJECT_ROOT, "ml", "attrs", "fund_attrs_monthly.parquet")
PANEL_PATH = os.path.join(PROJECT_ROOT, "ml", "panel.parquet")

# 预登记的 4 个规格（顺序固定，S4 为主规格）
SPECS = [
    ("S1_baseline", "动量（基线）", ["ret_12m"]),
    ("S2_aum", "动量 + log(AUM)", ["ret_12m", "log_aum"]),
    ("S3_fee", "动量 + 费率", ["ret_12m", "neg_fee"]),
    ("S4_main", "动量 + AUM + 费率 + 资金流", ["ret_12m", "log_aum", "neg_fee", "flow_share_ratio"]),
]
N_SEGMENTS = 4
MIN_FUNDS = 100


def t_to_p(t: float) -> float:
    """NW t → 双尾 p（正态近似；n 足够大时与 t 分布几乎一致）。"""
    return float(erfc(abs(t) / sqrt(2))) if t == t else float("nan")


def holm_adjust(pvals: list) -> list:
    """Holm step-down 校正：p_adj_(i) = max_{j<=i} (m-j+1)·p_(j)。"""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: (pvals[i] if pvals[i] == pvals[i] else 1.0))
    adj = [float("nan")] * m
    running = 0.0
    for rank, i in enumerate(order):
        p = pvals[i] if pvals[i] == pvals[i] else 1.0
        running = max(running, (m - rank) * p)
        adj[i] = min(running, 1.0)
    return adj


def load_inputs(include_delisted: bool = True):
    panel = pd.read_parquet(PANEL_PATH)
    month_ts = pd.DatetimeIndex(np.sort(panel[panel["t_date"] <= DEV_END]["t_date"].unique()))
    bench = pd.read_csv(BENCH_PATH, parse_dates=["date"]).sort_values("date")
    bench_dates = bench["date"].to_numpy(dtype="datetime64[ns]").astype("int64")
    extra = [DELISTED_HISTORY_DIR] if include_delisted else None
    series = load_fund_series(bench_dates, extra_dirs=extra)
    month_idx, rets_map = load_month_rets(series, bench_dates, month_ts)
    screens_map = build_no_future_screens(series, bench_dates, month_idx, month_ts)
    base = [screens_map[m] for m in range(len(month_ts))]
    return dict(month_ts=month_ts, month_idx=month_idx, rets_map=rets_map, base=base,
                series=series, bench_dates=bench_dates)


def load_attrs(month_ts) -> dict:
    a = pd.read_parquet(ATTRS_PATH)
    tmap = {pd.Timestamp(t): i for i, t in enumerate(month_ts)}
    out = {}
    for r in a.itertuples(index=False):
        m = tmap.get(pd.Timestamp(r.t_date))
        if m is None:
            continue
        out[(m, r.fund_code)] = {
            "log_aum": float(r.log_aum) if r.log_aum == r.log_aum else np.nan,
            "neg_fee": (-float(r.fee_total)) if r.fee_total == r.fee_total else np.nan,
            "flow_share_ratio": (float(r.flow_share_ratio)
                                 if r.flow_share_ratio == r.flow_share_ratio else np.nan),
        }
    return out


def _zscore(v: np.ndarray) -> np.ndarray:
    """横截面 z-score（NaN 保持 NaN；标准差为 0 时全 0）。"""
    m = ~np.isnan(v)
    if m.sum() < 3:
        return np.full(len(v), np.nan)
    mu, sd = np.nanmean(v), np.nanstd(v)
    if not sd or np.isnan(sd):
        return np.where(m, 0.0, np.nan)
    return np.where(m, (v - mu) / sd, np.nan)


def build_spec_screens(base, attrs, comps) -> list:
    """按规格合成当月截面分数（等权 z-score 平均；缺失分量不计入该基金）。"""
    out = []
    for m, rows in enumerate(base):
        codes = [c for c, _ in rows]
        r12 = np.array([v for _, v in rows], dtype=float)
        cols = {"ret_12m": _zscore(r12)}
        for comp in comps:
            if comp == "ret_12m":
                continue
            v = np.array([attrs.get((m, c), {}).get(comp, np.nan) for c in codes], dtype=float)
            if comp == "flow_share_ratio":
                # 资金流是份额口径，遇大额申赎/份额折算会出现 |流率| > 100% 的极端值；
                # 合成前先按**当月横截面 1%/99% 缩尾**（标准做法，避免极端值主导 z-score）
                if int(np.sum(~np.isnan(v))) >= 20:
                    lo, hi = np.nanquantile(v, [0.01, 0.99])
                    v = np.clip(v, lo, hi)
            cols[comp] = _zscore(v)
        Z = np.column_stack([cols[c] for c in comps])
        score = np.nanmean(Z, axis=1)          # 只用可得分量
        out.append([(c, float(s)) for c, s in zip(codes, score) if s == s])
    return out


def pool_returns(screens, rets_map, n_months):
    from backtest_strategy import pool_rebalance_cost
    pool = []
    for m in range(1, n_months):
        codes = [x[0] for x in screens[m - 1]]
        vals = [rets_map[c][m - 1] for c in codes if c in rets_map]
        pool.append(float(np.mean(vals)) if vals else 0.0)
    pool = np.array(pool)
    cost = pool_rebalance_cost(screens, rets_map, pool)
    return pool, (1.0 + pool) * (1.0 - cost) - 1.0


def missing_frac_top_vs_pool(screens, attrs, top_n, key):
    """该规格 Top50 中属性 `key` 缺失的比例 vs 全池缺失比例。

    为什么要报（2026-09-19 用户审查）：缺失分量的基金不会被删掉（保持同池比较），但它只有
    较少分量参与合成、分数方差更大，**更容易落到排名两端**——所以必须给出"Top50 缺失占比"，
    作为结果的边界条件，而不是把"入选"当成特征有效的证据。
    """
    top_m, pool_m = [], []
    for m, rows in enumerate(screens):
        if len(rows) < top_n:
            continue

        def miss(c):
            v = attrs.get((m, c), {}).get(key, np.nan)
            return v != v

        codes = [c for c, _ in rows]
        pm = float(np.mean([miss(c) for c in codes]))
        top = [c for c, _ in sorted(rows, key=lambda x: x[1], reverse=True)[:top_n]]
        top_m.append(float(np.mean([miss(c) for c in top])))
        pool_m.append(pm)
    if not top_m:
        return float("nan"), float("nan")
    return float(np.mean(top_m)), float(np.mean(pool_m))


def run_spec(inp, screens, top_n=50, start=None):
    from backtest_strategy import regime_no_lookahead, run_strategy, summarize
    month_ts, month_idx, rets_map = inp["month_ts"], inp["month_idx"], inp["rets_map"]
    bench = pd.read_csv(BENCH_PATH, parse_dates=["date"]).sort_values("date")
    bc = bench["close"].to_numpy(dtype=float)
    bench_ret_m = bc[month_idx[1:]] / bc[month_idx[:-1]] - 1.0
    pool, pool_net = pool_returns(screens, rets_map, len(month_ts))
    regime = regime_no_lookahead(bc, month_idx)
    if start is None:
        start = next(i for i, r in enumerate(screens) if len(r) >= MIN_FUNDS)
    df = run_strategy(screens, month_ts, rets_map, bench_ret_m, pool, pool_net, regime,
                      top_n, BUY_FEE, SELL_FEE, start, stratify=False)
    s = summarize(df, "spec")
    diff = (df["r_net"] - df["pool_ret_net"]).dropna()
    s["excess_mean"], s["excess_nw"] = float(diff.mean()), float(nw_tstat(diff))
    parts = np.array_split(np.arange(len(df)), N_SEGMENTS)
    s["segments"] = [float((1.0 + df["r_net"].iloc[p]).prod() ** (12.0 / len(p)) - 1.0)
                     for p in parts]
    s["start"] = str(month_ts[start].date())
    return df, s


def main():
    ap = argparse.ArgumentParser(description="第三组特征四规格对照（预登记见 feature_group3_prereg.md）")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(ATTRS_PATH):
        raise SystemExit(f"缺 {ATTRS_PATH}，请先运行 fund_attrs.py（--fetch both → --build-monthly）")
    print("加载输入（含清盘池）…")
    inp = load_inputs(include_delisted=True)
    attrs = load_attrs(inp["month_ts"])
    print(f"研究池 {len(inp['series'])} 只 | 属性记录 {len(attrs)} 条 | 截面 {len(inp['month_ts'])}")

    base = inp["base"]
    month_ts = inp["month_ts"]
    start = next(i for i, r in enumerate(base) if len(r) >= MIN_FUNDS)
    results, dfs = {}, {}
    for key, label, comps in SPECS:
        screens = build_spec_screens(base, attrs, comps)
        df, s = run_spec(inp, screens, args.top_n, start=start)
        results[key] = (label, comps, s)
        dfs[key] = df
        print(f"  {key:<12}{label:<28}年化 {s['ann_ret']:>7.2%} | 超额 {s['excess_mean']:+.4%}"
              f"（NW {s['excess_nw']:+.2f}）", flush=True)

    # 配对检验（各规格 vs S1 基线）+ Holm 校正
    base_df = dfs["S1_baseline"].set_index("t_date")["r_net"]
    pairs = {}
    for key, _label, _c in SPECS:
        if key == "S1_baseline":
            continue
        d = (dfs[key].set_index("t_date")["r_net"] - base_df).dropna()
        pairs[key] = {"mean": float(d.mean()), "naive_t": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))),
                      "nw_t": float(nw_tstat(d)), "p": t_to_p(float(nw_tstat(d))),
                      "win": float((d > 0).mean()), "cum": float((1.0 + d).prod() - 1.0),
                      "n": len(d)}
    keys = list(pairs)
    adj = holm_adjust([pairs[k]["p"] for k in keys])
    for k, a in zip(keys, adj):
        pairs[k]["p_holm"] = a

    # 属性覆盖率（按年份；**分母 = 当月台账在池基金**，不是属性表内部行数）
    cov_rows = []
    for m, rows_m in enumerate(base):
        codes = [c for c, _ in rows_m]
        if not codes:
            continue

        def frac(key):
            k = sum(1 for c in codes
                    if (m, c) in attrs and attrs[(m, c)][key] == attrs[(m, c)][key])
            return k / len(codes)

        cov_rows.append({"year": month_ts[m].year, "n": len(codes), "aum": frac("log_aum"),
                         "flow": frac("flow_share_ratio"), "fee": frac("neg_fee")})
    cov = pd.DataFrame(cov_rows)
    if len(cov):
        cov = cov.groupby("year").apply(
            lambda d: pd.Series({"n": float(np.median(d["n"])),
                                 "aum": float(np.average(d["aum"], weights=d["n"])),
                                 "flow": float(np.average(d["flow"], weights=d["n"])),
                                 "fee": float(np.average(d["fee"], weights=d["n"]))}),
            include_groups=False)

    lines = ["=" * 92,
             "第三组特征：AUM / 费率 / 资金流 —— 四规格对照（dev 段，含清盘池）",
             "=" * 92,
             "预登记：ml/backtest/feature_group3_prereg.md（4 规格、等权 z-score 合成、判据固定）",
             "口径：台账截面无未来端点｜Top50 等权｜每月调仓持 6 月重叠｜费用 申购0.15%+赎回0.5%",
             f"区间：{results['S1_baseline'][2]['start']} ~ {dfs['S1_baseline'].t_date.max().date()}"
             f"（{len(dfs['S1_baseline'])} 个月）｜研究池含已清盘基金",
             "",
             "⚠️ 费率口径（2026-09-19 用户审查）：S3/S4 用的是**当前费率回填到全部历史月份**",
             "   （历史近似）。基金此后的费率调整对早期月份属于未来信息，**预登记与 Holm 校正都消除不了**。",
             "   因此 S3/S4 的结论只能读作「在'当前费率作为历史近似'的等权合成规格下未检出增量」，",
             "   **不是**对历史费率信息的正式检验。S2 的 AUM 为历史时点口径（按法定披露时限对齐），基本成立。",
             "⚠️ S1 是**第三组共同区间内的基线**（MIN_FUNDS=100 → 起点 2006-11），与主策略历史默认口径",
             "   （MIN_FUNDS=50 → 2005-07）不同；两者绝对数字不可直接比较。",
             "⚠️ 波动差异的机制（2026-09-19 修正措辞）：结果与「动量相对权重下降」一致，但**机制未单独识别**",
             "   （排名对整体缩放不敏感；低费率本身也可能偏向低波动基金），不能断言是权重稀释造成的。",
             "",
             f"{'规格':<12}{'说明':<30}{'年化(费后)':>11}{'波动':>9}{'夏普':>7}{'MDD':>9}"
                 f"{'月超额(全池)':>13}{'NW t':>8}"]
    for key, label, _c in SPECS:
        s = results[key][2]
        lines.append(f"{key:<12}{label:<30}{s['ann_ret']:>11.2%}{s['vol']:>9.2%}{s['sharpe']:>7.2f}"
                     f"{s['mdd']:>9.2%}{s['excess_mean']:>13.4%}{s['excess_nw']:>8.2f}")
    lines += ["",
              "配对检验（各规格 vs S1 基线原动量；逐月费用后收益差，NW(6)，Holm 校正 m=3）：",
              f"  {'规格':<12}{'月均差':>10}{'naive t':>9}{'NW t':>8}{'p(单)':>9}{'p(Holm)':>9}"
              f"{'胜率':>8}{'累计差':>10}"]
    for key, label, _c in SPECS:
        if key == "S1_baseline":
            continue
        p = pairs[key]
        lines.append(f"  {key:<12}{p['mean']:>10.4%}{p['naive_t']:>9.2f}{p['nw_t']:>8.2f}"
                     f"{p['p']:>9.3f}{p['p_holm']:>9.3f}{p['win']:>8.1%}{p['cum']:>10.2%}")
    lines += ["", f"分{N_SEGMENTS}段年化（费用后）："]
    for key, label, _c in SPECS:
        lines.append(f"  {key:<12}" + " | ".join(f"{x:+.2%}"
                                                 for x in results[key][2]["segments"]))

    # 缺失分量边界（用户 2026-09-19 要求）：Top50 缺失占比 vs 全池
    lines += ["", "缺失分量边界（该规格 Top50 中属性缺失占比 vs 全池占比；"
                 "缺失基金只有较少分量参与合成、分数方差更大 → 更易落到排名两端）："]
    lines.append(f"  {'规格':<12}{'AUM(Top50/全池)':>22}{'资金流(Top50/全池)':>24}"
                 f"{'费率(Top50/全池)':>22}")
    for key, label, comps in SPECS:
        if key == "S1_baseline":
            continue
        sc = build_spec_screens(base, attrs, comps)
        cells = []
        for comp in ("log_aum", "flow_share_ratio", "neg_fee"):
            if comp in comps:
                t, p = missing_frac_top_vs_pool(sc, attrs, args.top_n, comp)
                cells.append(f"{t:.1%} / {p:.1%}")
            else:
                cells.append("—")
        lines.append(f"  {key:<12}{cells[0]:>22}{cells[1]:>24}{cells[2]:>22}")

    if len(cov):
        lines += ["", "属性覆盖率（按年份；分母=当年**台账在池基金**，非属性表行数）："]
        lines.append(f"  {'年份':<8}{'在池中位':>10}{'AUM':>10}{'资金流':>10}{'费率':>10}")
        for y, r in cov.iterrows():
            lines.append(f"  {int(y):<8}{int(r['n']):>10}{r['aum']:>10.1%}{r['flow']:>10.1%}"
                         f"{r['fee']:>10.1%}")
    lines += ["",
              "判定（预登记写死）：S4 主规格若 Holm 校正后仍显著为正（5%）**且**相对可执行全池为正、",
              "风险与换手未明显恶化 → 记为「第三组有增量」，进入影子策略；否则记为「未检出增量」。",
              "主策略仍为原动量，不因本表立即替换。"]
    report = "\n".join(lines)
    print("\n" + report)
    if not args.no_save:
        with open(os.path.join(OUT_DIR, "feature_group3_dev.txt"), "w", encoding="utf-8") as f:
            f.write(report + "\n")
        for key, df in dfs.items():
            df.to_csv(os.path.join(OUT_DIR, f"feature_group3_{key}_monthly.csv"), index=False)
        print(f"\n报告与逐月明细已落盘：{OUT_DIR}")


if __name__ == "__main__":
    main()
