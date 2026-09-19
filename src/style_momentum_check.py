# style_momentum_check.py：Phase 3 特征扩充第二组——滚动风格暴露 + 风格内动量
#
# 回答的问题（Phase 2 遗留局限）：过去一年收益的排序能力，有多少来自「风格延续」？
# 把 ret_12m 拆成 (a) 可由历史时点风格暴露解释的部分，(b) 风格之外的选基部分。
#
# 口径（与 backtest_strategy.py v2 完全同款，只换选股信号）：
#   - 选股池：无未来端点台账截面（成立≥365天 + 当期披露 + 过去252日 coverage≥95%）
#   - 风格暴露：每月末用**过去 252 交易日**把基金日收益回归到风格因子（只用 ≤t 信息）
#       mkt = 沪深300；size = 中证500−沪深300；growth = 创业板指−沪深300（2010-06 起）
#   - 三个方案在池集合**对齐为「暴露齐全」的同一子集**，差异只来自排序信号：
#       orig    ：按 ret_12m 排序 Top N（本对照的原动量基准）
#       neutral ：ret_12m 对 [beta_mkt, beta_size(, beta_growth)] 做**当月横截面回归**，
#                 取残差排序 Top N（风格中性动量）
#       group   ：按 beta_size(× beta_growth) 当月截面三分位分组，组内按 ret_12m 排序、
#                 按组内占比分配名额合并（风格内动量；与任务2第一组"类型分层"同一做法）
#   - 评价：费用后年化 / 波动 / 夏普 / MDD / 相对**可执行全池**月超额（NW(6) 判定）；
#           区间由因子可得性决定，**同一区间内**三方案对照，并附全台账池原动量参考行
#   - 区间：dev 段（t_date ≤ 2025-02-28）；holdout 段不参与（已看过，不得再作盲测）
# 输出：ml/backtest/style_exposure_monthly.csv（暴露长表）
#       ml/backtest/style_check_{phase}.txt（对照报告）
import argparse
import os

import numpy as np
import pandas as pd

from backtest_strategy import (BUY_FEE, DELISTED_HISTORY_DIR, DEV_END, HOLD, LOOKBACK,
                               MIN_COV, OUT_DIR, SELL_FEE, TOP_N_DEFAULT,
                               build_no_future_screens, load_month_rets,
                               pool_rebalance_cost, regime_no_lookahead, run_strategy,
                               select_by_group, summarize)
from panel_builder import BENCH_PATH, PROJECT_ROOT, load_fund_series
from style_factors import (SECTOR_FIRST_VALID, SECTOR_KEYS, SW_SECTORS,
                           load_sector_factors, load_style_factors)
from walk_forward_splitter import nw_tstat

PANEL_PATH = os.path.join(PROJECT_ROOT, "ml", "panel.parquet")
MIN_FUNDS_DEFAULT = 100      # 回测起点：台账在池基金数至少该值（太小的池子 Top50≈全池，无选择含义）
GROUP_QUANTILES = 3          # 风格分组档数（三分位）
N_SEGMENTS = 4               # 分阶段年化的段数（看增量是否稳定）
MIN_OBS_EXPOSURE = int(LOOKBACK * MIN_COV)   # 暴露回归最少有效观测（252×95%）


# ---------------------------------------------------------------- 风格暴露
def compute_exposures(series, bench_dates, month_pairs, factors, names, verbose=True):
    """在指定截面上用过去 252 交易日的基金日收益对风格因子做 OLS。

    :param month_pairs: [(m, bench_index), ...]，m 为 month_ts 的全局索引（可只覆盖因子可用区间）
    :return: dict[(m, code)] = {"beta_<factor>":..., "r2":..., "n_obs":...}
    """
    T = len(bench_dates)
    F = np.column_stack([np.ones(T)] + [factors[k] for k in names])
    fvalid = ~np.isnan(F).any(axis=1)
    out = {}
    codes = list(series.keys())
    for ci, code in enumerate(codes):
        r = series[code]["r_al"]
        rvalid = ~np.isnan(r)
        for m, i in month_pairs:
            w0 = i - LOOKBACK
            if w0 < 0:
                continue
            s = slice(w0, i + 1)
            mask = fvalid[s] & rvalid[s]
            n_obs = int(mask.sum())
            if n_obs < MIN_OBS_EXPOSURE:
                continue
            y = r[s][mask]
            X = F[s][mask]
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            sst = float(((y - y.mean()) ** 2).sum())
            rec = {"r2": 1.0 - float((resid ** 2).sum()) / sst if sst > 0 else np.nan,
                   "n_obs": n_obs}
            for j, nm in enumerate(names):
                rec[f"beta_{nm}"] = float(beta[1 + j])
            out[(m, code)] = rec
        if verbose and (ci + 1) % 500 == 0:
            print(f"  暴露计算进度 {ci+1}/{len(codes)} 只基金，累计记录 {len(out)} 条")
    return out


def _quantile_group(x, n_q=GROUP_QUANTILES):
    """当月截面分位分组（0..n_q-1）；样本过少时全部归 0。"""
    x = np.asarray(x, dtype=float)
    if len(x) < n_q * 2:
        return np.zeros(len(x), dtype=int)
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_q + 1)[1:-1]))
    if len(edges) == 0:
        return np.zeros(len(x), dtype=int)
    return np.searchsorted(edges, x, side="right")


def build_variant_screens(base_screens, exps, mode, control_names=None, names=None,
                          sector_keys=None):
    """构造选股截面；在池集合统一为「暴露齐全」子集。

    mode:
      'orig'    → (code, ret_12m)
      'neutral' → (code, 残差)：ret_12m 对当月截面暴露做 OLS 后的残差，
                  控制变量 = control_names（默认 = names 全部因子）
      'group'   → (code, ret_12m, 组标签)：给了 sector_keys 则按"最大板块 beta 归属"分组
                  （行业组内动量），否则按 size(×growth) 当月截面三分位分组（风格内动量）
    """
    names = names or []
    control_names = control_names or names
    out = []
    n_drop = 0
    for m, rows in enumerate(base_screens):
        data = []
        for code, r12 in rows:
            e = exps.get((m, code))
            if e is None:
                n_drop += 1
                continue
            data.append((code, r12, e))
        if not data:
            out.append([])
            continue
        if mode == "orig":
            out.append([(c, float(r)) for c, r, _ in data])
        elif mode == "neutral":
            y = np.array([r for _, r, _ in data], dtype=float)
            B = np.array([[e[f"beta_{nm}"] for nm in control_names] for _, _, e in data],
                         dtype=float)
            X = np.column_stack([np.ones(len(y)), B])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            out.append([(c, float(s)) for (c, _, _), s in zip(data, resid)])
        elif sector_keys:                       # 行业组内：按最大板块 beta 归属分组
            labels = []
            for _, _, e in data:
                vals = [e.get(f"beta_{k}", np.nan) for k in sector_keys]
                if all(v != v for v in vals):    # 全 NaN（板块暴露缺失）
                    labels.append("NA")
                else:
                    labels.append(sector_keys[int(np.nanargmax(vals))])
            out.append([(c, float(r), lab) for (c, r, _), lab in zip(data, labels)])
        else:                                   # 风格内：size(×growth) 三分位分组
            b_size = np.array([e.get("beta_size", np.nan) for _, _, e in data], dtype=float)
            q_size = _quantile_group(b_size)
            if "growth" in names:
                b_growth = np.array([e.get("beta_growth", np.nan) for _, _, e in data],
                                    dtype=float)
                q_growth = _quantile_group(b_growth)
                labels = [f"S{a}G{b}" for a, b in zip(q_size, q_growth)]
            else:
                labels = [f"S{a}" for a in q_size]
            out.append([(c, float(r), lab) for (c, r, _), lab in zip(data, labels)])
    return out, n_drop


# ---------------------------------------------------------------- 回测对照
def pool_returns(screens, rets_map, n_months):
    """全池等权（与组合同入选时点，v2 口径）：上月台账在座基金在 [m-1, m] 的收益均值。"""
    pool = []
    for m in range(1, n_months):
        codes = [x[0] for x in screens[m - 1]]
        vals = [rets_map[c][m - 1] for c in codes if c in rets_map]
        pool.append(float(np.mean(vals)) if vals else 0.0)
    return np.array(pool)


def find_start_idx(screens, min_funds):
    """第一个在池基金数 ≥ min_funds 的截面索引；找不到返回 None。"""
    for i, rows in enumerate(screens):
        if len(rows) >= min_funds:
            return i
    return None


def segment_annualized(df, n_seg=N_SEGMENTS):
    """把回测区间等分成 n_seg 段，各段费用后年化（看增量是否稳定，而非只看全期平均）。"""
    out = []
    for part in np.array_split(np.arange(len(df)), n_seg):
        sub = df.iloc[part]
        yrs = len(sub) / 12.0
        out.append(float((1.0 + sub.r_net).prod() ** (1.0 / yrs) - 1.0) if yrs > 0 else np.nan)
    return out


def paired_vs_orig(results, base="orig"):
    """各方案相对**原动量**的配对差检验（逐月费用后收益之差 → NW(6)）。

    为什么必须单独做：`excess_*` 检验的是「方案 vs **全池**」；用户要判的是
    「行业中性/风格中性是否**优于原动量**」——这是另一条配对差序列，t 值不可互推。
    """
    out = {}
    base_df = results[base][0].set_index("t_date")["r_net"]
    for kind, (df, _s) in results.items():
        if kind in ("orig", "orig_all"):
            continue
        d = (df.set_index("t_date")["r_net"] - base_df).dropna()
        n = len(d)
        out[kind] = {
            "mean": float(d.mean()),
            "naive_t": float(d.mean() / (d.std(ddof=1) / np.sqrt(n))) if n > 2 else float("nan"),
            "nw_t": float(nw_tstat(d)),
            "win_rate": float((d > 0).mean()) if n else float("nan"),
            "cum_diff": float((1.0 + d).prod() - 1.0) if n else float("nan"),
            "n_months": n,
        }
    return out


def run_variant(screens, month_ts, month_idx, rets_map, bc, top_n, start_idx, stratify, label):
    n_months = len(month_ts)
    pool = pool_returns(screens, rets_map, n_months)
    cost = pool_rebalance_cost(screens, rets_map, pool)
    pool_net = (1.0 + pool) * (1.0 - cost) - 1.0
    bench_ret_m = bc[month_idx[1:]] / bc[month_idx[:-1]] - 1.0
    regime = regime_no_lookahead(bc, month_idx)
    df = run_strategy(screens, month_ts, rets_map, bench_ret_m, pool, pool_net, regime,
                      top_n, BUY_FEE, SELL_FEE, start_idx, stratify=stratify)
    s = summarize(df, label)
    s["start"] = str(month_ts[start_idx].date())
    s["end"] = str(df.t_date.max().date())
    s["start_pool_n"] = len(screens[start_idx])
    diff = (df["r_net"] - df["pool_ret_net"]).dropna()
    s["excess_mean"] = float(diff.mean())
    s["excess_nw"] = float(nw_tstat(diff))
    s["regime_means"] = {reg: float(df[df.regime == reg].r_net.mean())
                         for reg in ("bull", "mix", "bear") if len(df[df.regime == reg])}
    s["regime_n"] = {reg: int(len(df[df.regime == reg])) for reg in ("bull", "mix", "bear")}
    s["pool_n_median"] = float(np.median([len(x) for x in screens[start_idx:]]))
    s["segments"] = segment_annualized(df)
    return df, s


# ---------------------------------------------------------------- 诊断
def exposure_cache_path(key: str) -> str:
    """因子集 key → 暴露缓存路径（size / size_growth / size_growth_sector）。"""
    return os.path.join(OUT_DIR, f"style_exposure_{key}_monthly.parquet")


def load_cached_exposures(path: str) -> dict:
    """读回暴露长表 → {(m, code): {beta_*, r2, n_obs}}（重跑对照时免去滚动回归）。

    注意：板块 beta 的列名是中文（beta_金融地产 等），pandas 的 itertuples 会重命名
    非标识符列名，因此这里用 to_dict("records") 保留原始列名。
    """
    df = pd.read_parquet(path)
    beta_cols = [c for c in df.columns if c.startswith("beta_")]
    out = {}
    for r in df.to_dict("records"):
        rec = {"r2": float(r["r2"]), "n_obs": int(r["n_obs"])}
        for col in beta_cols:
            v = r[col]
            if v == v:                         # 跳过 NaN
                rec[col] = float(v)
        out[(int(r["m"]), r["fund_code"])] = rec
    return out


def dump_exposures(exps, base_screens, month_ts, path):
    """只落盘台账在池记录（控制体积）；返回行数。"""
    in_pool = {(m, c) for m, rows in enumerate(base_screens) for c, _ in rows}
    recs = [{"t_date": month_ts[m], "m": m, "fund_code": code, **e}
            for (m, code), e in exps.items() if (m, code) in in_pool]
    pd.DataFrame(recs).sort_values(["t_date", "fund_code"]).to_parquet(path, index=False)
    return len(recs)


def order_shift_diagnostics(variants, month_ts, start, top_n, kinds):
    """排序改变程度：各变体的 Top N 与原动量 Top N 的平均重叠率（neutral 另报截面秩相关）。

    若重叠率接近 1，说明该方案几乎没有改变排序——"无增量"只是"没换过股"；
    若明显下降，则排序确实变了但组合结果没有改善。
    """
    out = {}
    for kind in kinds:
        is_group = kind.startswith("group")
        ov, rc = [], []
        for m in range(start, len(month_ts)):
            orig, alt = variants["orig"][m], variants[kind][m]
            if len(orig) < top_n or len(alt) < top_n:
                continue
            top_o = [c for c, *_ in sorted(orig, key=lambda x: x[1], reverse=True)[:top_n]]
            # 组内方案的选股是"组内排序 + 按组内占比分配名额"，不能用单一分数排序近似
            top_a = (select_by_group(alt, top_n) if is_group
                     else [c for c, *_ in sorted(alt, key=lambda x: x[1], reverse=True)[:top_n]])
            ov.append(len(set(top_o) & set(top_a)) / float(top_n))
            if not is_group:               # group 变体无连续分数可比，秩相关不适用
                o_map = {c: sc for c, sc, *_ in orig}
                a_map = {c: sc for c, sc, *_ in alt}
                codes = [c for c in o_map if c in a_map]
                s1 = pd.Series([o_map[c] for c in codes]).rank()
                s2 = pd.Series([a_map[c] for c in codes]).rank()
                rc.append(float(s1.corr(s2)))
        out[kind] = {"overlap_top": float(np.mean(ov)) if ov else np.nan,
                     "rank_corr": float(np.nanmean(rc)) if rc else np.nan,
                     "n_months": len(ov)}
    return out


def diagnose_momentum_vs_style(base_screens, exps, names, top_n, start=0):
    """动量与风格暴露的关系：① 截面 Spearman(ret_12m, beta)；② Top N 组风格偏离全池。"""
    rows = []
    for m, base in enumerate(base_screens):
        if m < start:
            continue
        data = [(c, r, exps[(m, c)]) for c, r in base if (m, c) in exps]
        if len(data) < 30:
            continue
        r12 = pd.Series([r for _, r, _ in data])
        rec = {"m": m, "n": len(data)}
        for nm in names:
            b = pd.Series([e[f"beta_{nm}"] for _, _, e in data])
            rec[f"spearman_{nm}"] = float(r12.rank().corr(b.rank()))
        picks = sorted(data, key=lambda x: x[1], reverse=True)[:top_n]
        for nm in names:
            allb = np.array([e[f"beta_{nm}"] for _, _, e in data], dtype=float)
            topb = np.array([e[f"beta_{nm}"] for _, _, e in picks], dtype=float)
            rec[f"top_minus_pool_{nm}"] = float(topb.mean() - allb.mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def fmt_report(title, lines_tables, extra=None):
    out = ["=" * 78, title, "=" * 78]
    out += lines_tables
    if extra:
        out += extra
    return "\n".join(out)


def variant_label(kind: str, groups: str) -> str:
    if kind == "orig_all":
        return "原动量(全池)"
    if kind == "orig":
        return "原动量(对齐)"
    if kind == "neutral":
        return "全因子中性" if groups == "sector" else "风格中性"
    if kind == "neutral_sector":
        return "行业中性"
    if kind == "group":
        return "行业组内" if groups == "sector" else "风格内"
    return kind


def factor_set_specs():
    """三套因子集（各区间不同：只在同一因子集内横向对照方案，不跨集比绝对数）：
       size               : mkt + size（2005-01 起）
       size_growth        : + growth（创业板指 2010-06 起）
       size_growth_sector : + 6 个板块价差（板块指数 2014-02 起 → 暴露自 2015-03 起）
    """
    return [
        dict(key="size", label="size only", groups="style", names=["mkt", "size"]),
        dict(key="size_growth", label="size+growth", groups="style",
             names=["mkt", "size", "growth"]),
        dict(key="size_growth_sector", label="size+growth+sector(6板块)", groups="sector",
             names=["mkt", "size", "growth"] + SECTOR_KEYS,
             extra_neutral=("neutral_sector", ["mkt"] + SECTOR_KEYS, "行业中性")),
    ]


def main():
    ap = argparse.ArgumentParser(description="Phase 3 第二组：滚动风格暴露 + 风格内动量对照")
    ap.add_argument("--top-n", type=int, default=TOP_N_DEFAULT)
    ap.add_argument("--min-funds", type=int, default=MIN_FUNDS_DEFAULT)
    ap.add_argument("--smoke", type=int, default=0, help="只跑前 N 个月末（验证用，不落盘报告）")
    ap.add_argument("--recompute-exposure", action="store_true",
                    help="忽略暴露缓存，强制重算滚动回归（默认优先读 ml/backtest/style_exposure_*.parquet）")
    ap.add_argument("--include-delisted", action="store_true",
                    help="把已清盘基金（data/processed/fund_history_delisted/）并入研究池"
                         "（幸存者偏差修复；暴露缓存 key 会加 _del 后缀，与现存池结果分开保存）")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    panel_all = pd.read_parquet(PANEL_PATH)
    panel = panel_all[panel_all["t_date"] <= DEV_END].copy()
    month_ts = pd.DatetimeIndex(np.sort(panel["t_date"].unique()))
    if args.smoke:
        month_ts = month_ts[-args.smoke:]      # 验证用：取最近的 N 个月末（暴露必定可算）
    print(f"dev 段截面：{len(month_ts)} 个月末（{month_ts[0].date()} ~ {month_ts[-1].date()}）")

    bench = pd.read_csv(BENCH_PATH, parse_dates=["date"]).sort_values("date")
    bench_dates = bench["date"].to_numpy(dtype="datetime64[ns]").astype("int64")
    bc = bench["close"].to_numpy(dtype=float)
    print("加载净值序列…")
    extra_dirs = [DELISTED_HISTORY_DIR] if args.include_delisted else None
    if args.include_delisted:
        print(f"含已清盘基金池：{DELISTED_HISTORY_DIR}")
    series = load_fund_series(bench_dates, extra_dirs=extra_dirs)
    print(f"研究池基金 {len(series)} 只")
    month_idx, rets_map = load_month_rets(series, bench_dates, month_ts)

    sf = load_style_factors(bench_dates)
    sec = load_sector_factors(bench_dates)
    factors = {**sf["factors"], **sec["factors"]}       # mkt/size/growth + 6 板块
    meta = sf["meta"]
    sector_meta = sec["meta"]
    print(f"板块因子：{len(SECTOR_KEYS)} 个（成分行业 {sum(len(v) for v in SW_SECTORS.values())} 个，"
          f"首个有效日 {SECTOR_FIRST_VALID}）")

    print("构建无未来端点台账截面…")
    base_map = build_no_future_screens(series, bench_dates, month_idx, month_ts)
    base_screens = [base_map[m] for m in range(len(month_ts))]

    def month_pairs_for(names):
        """该因子集可用的截面：窗口内要覆盖因子有效区间（板块因子 2014-02 起）。"""
        pairs = [(m, i) for m, i in enumerate(month_idx) if i - LOOKBACK >= 0]
        if any(k in names for k in SECTOR_KEYS):
            i_sec = int(np.searchsorted(
                bench_dates,
                np.datetime64(SECTOR_FIRST_VALID).astype("datetime64[ns]").astype("int64"),
                side="left"))
            pairs = [(m, i) for m, i in pairs if i - LOOKBACK >= i_sec]
        return pairs

    specs = factor_set_specs()
    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = "_del" if args.include_delisted else ""      # 含清盘池的结果与现存池分开存
    exps_by_key = {}
    for spec in specs:
        path = exposure_cache_path(spec["key"] + suffix)
        if (not args.recompute_exposure) and (not args.smoke) and os.path.exists(path):
            exps_by_key[spec["key"]] = load_cached_exposures(path)
            print(f"读取暴露缓存：{os.path.basename(path)}"
                  f"（{len(exps_by_key[spec['key']])} 条；--recompute-exposure 强制重算）")
            continue
        pairs = month_pairs_for(spec["names"])
        print(f"计算滚动暴露 [{spec['label']}]：{len(spec['names'])} 因子 × {len(pairs)} 个截面…")
        exps_by_key[spec["key"]] = compute_exposures(series, bench_dates, pairs, factors,
                                                     spec["names"])
        print(f"  暴露记录 {len(exps_by_key[spec['key']])} 条（基金×截面）")
        if not args.no_save:
            n = dump_exposures(exps_by_key[spec["key"]], base_screens, month_ts, path)
            print(f"暴露落盘：{os.path.basename(path)}（{n} 行，仅台账在池）")

    blocks = []
    for spec in specs:
        exps = exps_by_key[spec["key"]]
        names = spec["names"]
        groups = spec["groups"]
        variants = {
            # 全台账池原动量参考（不做暴露对齐，对应 v2 报告口径）
            "orig_all": [[(c, float(r)) for c, r in rows] for rows in base_screens],
        }
        variants["orig"], _ = build_variant_screens(base_screens, exps, "orig", names=names)
        variants["neutral"], _ = build_variant_screens(base_screens, exps, "neutral",
                                                       names=names, control_names=names)
        if groups == "sector":
            variants["group"], _ = build_variant_screens(base_screens, exps, "group",
                                                         names=names, sector_keys=SECTOR_KEYS)
        else:
            variants["group"], _ = build_variant_screens(base_screens, exps, "group",
                                                         names=names)
        if spec.get("extra_neutral"):
            vname, ctrl, _lbl = spec["extra_neutral"]
            variants[vname], _ = build_variant_screens(base_screens, exps, "neutral",
                                                       names=names, control_names=ctrl)
        start = find_start_idx(variants["orig"], args.min_funds)
        if start is None:
            print(f"⚠️ 因子集 [{spec['label']}] 无可用起点，跳过")
            continue
        results = {}
        for kind, sc in variants.items():
            df, s = run_variant(sc, month_ts, month_idx, rets_map, bc, args.top_n,
                                start, kind.startswith("group"), f"{kind}[{spec['key']}]")
            results[kind] = (df, s)
        diag = diagnose_momentum_vs_style(base_screens, exps, names, args.top_n, start=start)
        extra_kinds = [k for k in variants if k not in ("orig", "orig_all")]
        shift = order_shift_diagnostics(variants, month_ts, start, args.top_n, extra_kinds)
        paired = paired_vs_orig(results)
        blocks.append((spec, start, results, diag, shift, paired))

    if not args.no_save and not args.smoke:
        for spec, _start, results, _diag, _shift, _paired in blocks:
            rows = []
            for kind, (df, _s) in results.items():
                d = df[["t_date", "regime", "r_gross", "r_net", "pool_ret", "pool_ret_net",
                        "nav_gross", "nav_net"]].copy()
                d.insert(0, "variant", kind)
                d.insert(0, "factor_set", spec["key"])
                rows.append(d)
            out_df = pd.concat(rows, ignore_index=True)
            path = os.path.join(OUT_DIR, f"style_variants_{spec['key']}{suffix}_monthly.csv")
            out_df.to_csv(path, index=False)
            print(f"逐月收益已落盘：{os.path.basename(path)}（{len(out_df)} 行）")

    # ---------------- 报告
    lines = []
    lines.append(f"区间口径：dev 段（≤ {DEV_END}）；Top{args.top_n} 等权；每月调仓持 {HOLD} 月重叠；"
                 f"费用 申购{BUY_FEE:.2%}+赎回{SELL_FEE:.2%}")
    lines.append("风格因子：" + "；".join(
        f"{k} = {v[1]} − 沪深300（{v[2]} 起）" for k, v in meta.items()) + "；mkt = 沪深300")
    lines.append(f"行业板块因子（成分行业等权日收益 − 沪深300，首个有效日 {SECTOR_FIRST_VALID}）："
                 + "、".join(f"{k}({len(SW_SECTORS[k])})" for k in SECTOR_KEYS))
    lines.append("两类检验必须分开读：①『月超额(全池)/NW t』= 方案 vs **可执行全池**；"
                 "②『配对检验』= 方案 vs **原动量**（逐月费用后收益差）。两条序列不同。")
    lines.append("")
    for spec, start, results, diag, shift, paired in blocks:
        groups = spec["groups"]
        order = [k for k in ("orig_all", "orig", "neutral", "neutral_sector", "group")
                 if k in results]
        lines.append("-" * 78)
        lines.append(f"因子集 [{spec['label']}]：{' + '.join(spec['names'])}")
        lines.append(f"区间 {results['orig'][1]['start']} ~ {results['orig'][1]['end']}"
                     f"（{results['orig'][1]['n_months']} 个月）｜在池基金数：起点 "
                     f"{results['orig'][1]['start_pool_n']} 只 / 中位 "
                     f"{results['orig'][1]['pool_n_median']:.0f} 只（各方案对齐为暴露齐全子集；"
                     f"orig_all 为全台账池）")
        lines.append("-" * 78)
        lines.append(f"{'方案':<12}{'年化(费后)':>11}{'波动':>9}{'夏普':>7}{'MDD':>9}"
                     f"{'月超额(全池)':>13}{'NW t':>8}")
        for kind in order:
            _, s = results[kind]
            lines.append(f"{variant_label(kind, groups):<12}{s['ann_ret']:>11.2%}"
                         f"{s['vol']:>9.2%}{s['sharpe']:>7.2f}{s['mdd']:>9.2%}"
                         f"{s['excess_mean']:>10.4%}{s['excess_nw']:>8.2f}")
        lines.append("")
        lines.append("状态分解（费用后月均）：")
        for kind in order:
            if kind == "orig_all":
                continue
            s = results[kind][1]
            rm = s["regime_means"]
            lines.append(f"  {variant_label(kind, groups):<11}" + " | ".join(
                f"{k} {rm.get(k, float('nan')):+.2%}({s['regime_n'].get(k, 0)}月)"
                for k in ("bull", "mix", "bear")))
        lines.append("")
        lines.append(f"分{N_SEGMENTS}段年化（费用后，等长切分——看增量是否稳定）：")
        for kind in order:
            if kind == "orig_all":
                continue
            lines.append(f"  {variant_label(kind, groups):<11}"
                         + " | ".join(f"{x:+.2%}" for x in results[kind][1]["segments"]))
        lines.append("")
        lines.append("配对检验——各方案 vs **原动量**（同一在池集合、逐月费用后收益差，NW(6)）：")
        lines.append(f"  {'方案':<12}{'月均差':>10}{'naive t':>9}{'NW t':>8}{'胜率':>8}{'累计差':>10}")
        for kind in order:
            if kind in ("orig", "orig_all") or kind not in paired:
                continue
            p = paired[kind]
            lines.append(f"  {variant_label(kind, groups):<12}{p['mean']:>10.4%}"
                         f"{p['naive_t']:>9.2f}{p['nw_t']:>8.2f}{p['win_rate']:>8.1%}"
                         f"{p['cum_diff']:>10.2%}")
        lines.append("     （对照：上表『月超额 / NW t』的基准是**可执行全池**；本节基准是**原动量**——"
                     "两条配对序列不同，t 值不可互推）")
        lines.append("")
        lines.append("诊断——动量与暴露的关系（当月截面）：")
        for label, col in (("① Spearman(ret_12m, beta)", "spearman_"),
                           (f"② Top{args.top_n} 组 beta 偏离全池", "top_minus_pool_")):
            items = [f"{nm} {diag[col + nm].mean():+.4f}(NW {nw_tstat(diag[col + nm]):+.2f})"
                     for nm in spec["names"]]
            lines.append(f"  {label}：")
            for i in range(0, len(items), 4):
                lines.append("     " + " | ".join(items[i:i + 4]))

        def _shift_txt(k):
            sh = shift[k]
            rc_txt = "—" if not np.isfinite(sh["rank_corr"]) else f"{sh['rank_corr']:.3f}"
            return (f"{variant_label(k, groups)} Top{args.top_n}重叠 {sh['overlap_top']:.1%}"
                    f"、秩相关 {rc_txt}")

        n_shift = next(iter(shift.values()))["n_months"] if shift else 0
        lines.append(f"  ③ 排序改变程度（vs 原动量，{n_shift} 个月）：")
        for k in shift:
            lines.append("     " + _shift_txt(k))
        lines.append("")
    report = fmt_report("Phase 3 第二组：滚动风格暴露 + 行业维度（dev 段对照）", lines)
    if args.include_delisted:
        pool_note = ("② **已清盘基金池已并入**（`data/processed/fund_history_delisted/`，"
                     "EID 公告检索 → 拉净值 → 清洗；**只覆盖 2014 年后且公告库能查到的事件**，"
                     "2005-2013 清盘基金仍缺失——属**部分修复**，不等于幸存者偏差已消除；"
                     "持仓期内终止的基金按财富保持（该段收益记 0）计）")
    else:
        pool_note = "② 现存池条件性研究，未含已清盘基金（可加 `--include-delisted` 并入）"
    report += (f"\n\n边界：① dev 段内对照，holdout（2025-03~）已看过，不得作为新设计的盲测；"
               f"{pool_note}；③ 风格/行业指数均为价格指数；"
               "④ growth 因子自 2010-06、板块因子自 2014-02 才可得 → 各因子集区间不同，"
               "**只能在同一因子集内横向比较方案，不跨因子集比绝对数**；"
               "⑤ 板块划分为人为选择，2014 年前无法做行业中性化。\n")
    print("\n" + report)

    if not args.no_save and not args.smoke:
        out_name = f"style_check_dev{suffix}.txt"
        with open(os.path.join(OUT_DIR, out_name), "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n报告已落盘：{os.path.join(OUT_DIR, out_name)}")


if __name__ == "__main__":
    main()
