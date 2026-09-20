# termination_audit.py：清盘基金在截面/组合中的参与度 + "终止处理规则"的敏感性检验
#
# 回答两个问题（2026-09-19 用户提出）：
#   ① 清盘基金实际参与了多少——曾进入可投资截面的只数、曾进入 Top50 的只数/次数、
#      在六个月持有期中终止的行数；
#   ② "终止后收益记零"（持有到终止，资金闲置）与"退出后资金重新分配"（提前赎回并把权重
#      分给同批其余基金）对组合结果差多少——若差异极小，说明终止处理规则不影响结论。
#
# 口径：dev 段（≤ 2025-02-28）、台账截面（无未来端点）、Top50 等权、每月调仓持 6 月重叠、
#       费用 申购0.15%+赎回0.50%（与 backtest_strategy v2 完全一致）；holdout 不参与。
import argparse
import os

import numpy as np
import pandas as pd

from backtest_strategy import (BUY_FEE, DELISTED_HISTORY_DIR, DEV_END, HOLD, OUT_DIR,
                               SELL_FEE, build_no_future_screens, load_month_rets)
from panel_builder import BENCH_PATH, PROJECT_ROOT, load_fund_series
from walk_forward_splitter import nw_tstat

PANEL_V3 = os.path.join(PROJECT_ROOT, "ml", "panel_v3.parquet")
DELISTED_LIST = os.path.join(PROJECT_ROOT, "data", "raw", "delisted_funds.csv")


def build_inputs(include_delisted=True):
    from backtest_strategy import PANEL_PATH
    panel = pd.read_parquet(PANEL_PATH)
    month_ts = pd.DatetimeIndex(np.sort(panel[panel["t_date"] <= DEV_END]["t_date"].unique()))
    bench = pd.read_csv(BENCH_PATH, parse_dates=["date"]).sort_values("date")
    bench_dates = bench["date"].to_numpy(dtype="datetime64[ns]").astype("int64")
    extra = [DELISTED_HISTORY_DIR] if include_delisted else None
    series = load_fund_series(bench_dates, extra_dirs=extra)
    month_idx, rets_map = load_month_rets(series, bench_dates, month_ts)
    screens_map = build_no_future_screens(series, bench_dates, month_idx, month_ts)
    screens = [screens_map[m] for m in range(len(month_ts))]
    # 每只基金最后净值日对应的基准索引（判断"某持仓月是否已终止"）
    last_idx = {}
    for code, s in series.items():
        last_idx[code] = int(np.searchsorted(bench_dates, int(s["dates"][-1]), side="right")) - 1
    delisted = set()
    if os.path.exists(DELISTED_LIST):
        delisted = set(pd.read_csv(DELISTED_LIST, dtype={"fund_code": str})["fund_code"])
    return dict(month_ts=month_ts, month_idx=month_idx, rets_map=rets_map, screens=screens,
                last_idx=last_idx, delisted=delisted, series=series)


def participation_stats(inp, top_n=50, min_funds=100):
    """清盘基金在台账截面与 Top50 中的参与度。"""
    screens, delisted, month_ts = inp["screens"], inp["delisted"], inp["month_ts"]
    start = next((i for i, r in enumerate(screens) if len(r) >= min_funds), None)
    rows = []
    top_hits = {}
    for m in range(start, len(screens)):
        codes = [c for c, _ in screens[m]]
        dl_in_pool = [c for c in codes if c in delisted]
        top = [c for c, _ in sorted(screens[m], key=lambda x: x[1], reverse=True)[:top_n]]
        dl_top = [c for c in top if c in delisted]
        for c in dl_top:
            top_hits[c] = top_hits.get(c, 0) + 1
        rows.append({"t_date": month_ts[m], "pool_n": len(codes), "delisted_in_pool": len(dl_in_pool),
                     "delisted_top": len(dl_top)})
    df = pd.DataFrame(rows)
    summary = {
        "回测区间": f"{df.t_date.min().date()} ~ {df.t_date.max().date()}（{len(df)} 月）",
        "台账在池中位": int(df.pool_n.median()),
        "含清盘基金月份数": int((df.delisted_in_pool > 0).sum()),
        "清盘基金进入截面的只数": len({c for m in range(start, len(screens))
                                     for c, _ in screens[m] if c in delisted}),
        "清盘基金进入 Top50 的只数": len(top_hits),
        "清盘基金进入 Top50 的月次": int(df.delisted_top.sum()),
        "单月最多清盘基金进 Top50": int(df.delisted_top.max()),
        "平均每月进 Top50 清盘只数": round(float(df.delisted_top.mean()), 3),
    }
    return summary, df, top_hits


def run_mode(inp, mode, top_n=50, hold=HOLD, min_funds=100, start_override=None):
    """组合模拟：mode='zero'（终止后记零）或 'reallocate'（终止后权重分给同批其余基金）。"""
    screens, rets_map, month_idx, last_idx = (inp["screens"], inp["rets_map"],
                                              inp["month_idx"], inp["last_idx"])
    month_ts = inp["month_ts"]
    start = start_override if start_override is not None else \
        next((i for i, r in enumerate(screens) if len(r) >= min_funds), None)
    active, rows = [], []
    nav = 1.0
    for m in range(start + 1, len(month_ts)):
        i_now = month_idx[m]                    # 本月月末的基准索引
        part = []
        for _bm, codes in active:
            vals = []
            for c in codes:
                r = rets_map[c][m - 1]
                ended = last_idx.get(c, 10 ** 12) < i_now     # 该月月末前已终止
                if ended and mode == "reallocate":
                    continue                                  # 权重分给其余在持基金
                vals.append(0.0 if ended else float(r))
            part.append(float(np.mean(vals)) if vals else 0.0)
        r_gross = float(np.mean(part)) if part else 0.0
        expired = [a for a in active if m - a[0] >= hold]
        active = [a for a in active if m - a[0] < hold]
        picks = [c for c, _ in sorted(screens[m], key=lambda x: x[1], reverse=True)[:top_n]]
        active.append([m, picks])
        w = 1.0 / hold
        fee = w * SELL_FEE * (len(expired) > 0) + w * BUY_FEE
        r_net = (1.0 + r_gross) * (1.0 - fee) - 1.0
        nav *= (1.0 + r_net)
        rows.append({"t_date": month_ts[m], "r_gross": r_gross, "r_net": r_net, "nav": nav,
                     "n_held": len(active)})
    df = pd.DataFrame(rows)
    n = len(df)
    ann = df["nav"].iloc[-1] ** (12.0 / n) - 1.0
    vol = float(df["r_net"].std(ddof=1)) * np.sqrt(12)
    nav_s = df["nav"].values
    mdd = float(np.min(nav_s / np.maximum.accumulate(nav_s) - 1.0))
    return df, {"mode": mode, "months": n, "ann_ret": ann, "vol": vol, "mdd": mdd}


def main():
    ap = argparse.ArgumentParser(description="清盘基金参与度与终止处理规则敏感性")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    print("加载输入（含清盘池）…")
    inp = build_inputs(include_delisted=True)
    print(f"研究池 {len(inp['series'])} 只 | 清盘名单 {len(inp['delisted'])} 只")

    summary, part_df, top_hits = participation_stats(inp, args.top_n)
    lines = ["=" * 78,
             "清盘基金参与度审计（dev 段，台账截面，含清盘池）",
             "=" * 78]
    for k, v in summary.items():
        lines.append(f"  {k:<26}{v}")
    lines.append("")
    lines.append(f"  进入 Top50 的清盘基金（按出现月次，最多 15 只）：")
    for c, k in sorted(top_hits.items(), key=lambda x: -x[1])[:15]:
        lines.append(f"    {c}  {k} 次")

    # panel v3 的终止标签统计
    if os.path.exists(PANEL_V3):
        p3 = pd.read_parquet(PANEL_V3)
        v3 = p3[p3["fund_code"].isin(inp["delisted"])]
        lines.append("")
        lines.append("panel v3 终止标签（仅清盘基金）：")
        lines.append(f"  行数 {len(v3)}（占 v3 全部 {len(p3)} 行的 {len(v3)/max(len(p3),1):.2%}）")
        for res, cnt in v3["label_resolution"].value_counts().items():
            lines.append(f"    {res:<18}{cnt}")
        lines.append(f"  terminated_in_label=True 的行数：{int(p3['terminated_in_label'].sum())}")
        by_year = (p3[p3["terminated_in_label"]]
                   .groupby(pd.to_datetime(p3.loc[p3["terminated_in_label"], "t_date"]).dt.year)
                   .size())
        lines.append("  按截面年份分布：" + " | ".join(f"{y}:{n}" for y, n in by_year.items()))

    # 终止处理规则敏感性
    lines.append("")
    lines.append("终止处理规则敏感性（同一组合，仅终止月份的收益分配方式不同）：")
    res = {}
    for mode in ("zero", "reallocate"):
        df, s = run_mode(inp, mode, args.top_n)
        res[mode] = (df, s)
        lines.append(f"  {mode:<12}月数 {s['months']:>4} | 年化 {s['ann_ret']:>7.2%} | "
                     f"波动 {s['vol']:>6.2%} | MDD {s['mdd']:>7.2%}")
    d = (res["reallocate"][0]["r_net"] - res["zero"][0]["r_net"]).dropna()
    lines.append(f"  两者逐月差：均值 {d.mean():+.5%} | naive t "
                 f"{d.mean()/(d.std(ddof=1)/np.sqrt(len(d))):+.2f} | NW(6) t {nw_tstat(d):+.2f}"
                 f" | 差不为 0 的月份数 {int((d.abs() > 1e-12).sum())}")

    report = "\n".join(lines)
    print("\n" + report)
    if not args.no_save:
        p = os.path.join(OUT_DIR, "termination_audit_dev.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        part_df.to_csv(os.path.join(OUT_DIR, "termination_participation_monthly.csv"), index=False)
        for mode in ("zero", "reallocate"):
            res[mode][0].to_csv(os.path.join(OUT_DIR, f"termination_mode_{mode}_monthly.csv"),
                                index=False)
        print(f"\n报告已落盘：{p}")


if __name__ == "__main__":
    main()
