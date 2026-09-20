# panel_builder.py：phase2 面板数据集构建
# fund-month面板：X(i,t)=过去12个月窗口特征（仅≤t信息），y(i,t)=未来6个月收益
# eligibility 时变判定：is_eligible(fund, t)——成立满12个月 + 窗口coverage≥95%，不做整基金静态删除
import os
import numpy as np
import pandas as pd
from clean_nav import load_fund_name_map

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
HISTORY_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "fund_history")
PANEL_DIR = os.path.join(PROJECT_ROOT, "ml")
PANEL_PATH = os.path.join(PANEL_DIR, "panel.parquet")
BENCH_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "benchmark_hs300.csv")

# 面板口径常量
FEATURE_LOOKBACK = 252      # 特征回看窗口：12个月交易日
LABEL_HORIZON = 126         # 标签前瞻窗口：6个月交易日
MIN_AGE_NATURAL_DAYS = 365  # 截面时点要求成立≥12个自然月（365天，此前误用252自然日≈8.3个月）
MIN_COVERAGE = 0.95         # 时变eligibility：窗口内观测数/基准交易日数
MAX_GAP_DAYS = 15           # 截面/标签端点附近允许的最大披露缺口（自然日）
RET_BENCH_RATE = 0.02 / 252  # 日频无风险利率（夏普/alpha用）

# 短周期特征窗口（交易日）
SHORT_WINDOWS = {"ret_1m": 21, "ret_3m": 63, "ret_6m": 126, "ret_12m": 252}
DAY_NS = 86400 * 10**9  # 1天对应的int64纳秒
MONTH_NS = 30.4375 * DAY_NS  # 平均月长（用于age计算）
# 基金年龄分层（分层模型依据）：age_months 12~36 = 短历史段；≥36 = 长历史段
SHORT_HISTORY_AGE = (12, 36)
FULL_HISTORY_AGE = 36


def age_days_to_months(age_days_ns: int) -> float:
    """基金年龄（自然日纳秒数）→ 月数"""
    return age_days_ns / MONTH_NS


def load_fund_series(bench_dates: np.ndarray, extra_dirs=None):
    """
    读全历史清洗目录下全部基金csv，对齐到基准交易日日历
    对齐规则：基金净值映射到同日基准日索引；未披露日 r=NaN；财富指数 W=∏(1+r) 前向保持
    :param extra_dirs: 额外清洗目录（如 data/processed/fund_history_delisted/）——
        用于把**已清盘/终止基金**并回研究池（幸存者偏差修复，见 delisted_funds.py）；
        同名代码以主目录 HISTORY_DIR 优先
    :return: dict[fund_code] = {dates:int64, r_aligned:array(含NaN), wealth_aligned:array, first_ns:int64}
    """
    series = {}
    dirs = [HISTORY_DIR] + [d for d in (extra_dirs or []) if d and os.path.isdir(d)]
    for d in dirs:
        for fname in os.listdir(d):
            if not (fname.startswith("fund_") and fname.endswith(".csv")):
                continue
            code = fname.replace("fund_", "").replace(".csv", "")
            if code in series:          # 主目录优先
                continue
            df = pd.read_csv(os.path.join(d, fname), parse_dates=["date"])
            # 显式统一为纳秒时间戳（pandas 3.0 的 datetime64 默认单位是 us，直接 astype("int64") 会取成微秒）
            fdates = df["date"].to_numpy(dtype="datetime64[ns]").astype("int64")
            r = df["daily_ret"].astype(float).values
            # 对齐到基准日历：基金披露日 -> 基准日历位置（仅同日匹配，非交易日披露丢弃）
            pos = np.searchsorted(bench_dates, fdates)
            ok = (pos < len(bench_dates)) & (bench_dates[np.minimum(pos, len(bench_dates) - 1)] == fdates)
            r_al = np.full(len(bench_dates), np.nan)
            r_al[pos[ok]] = r[ok]
            # 财富指数：NaN日财富保持不变（相当于日收益0），有效日复利
            growth = np.where(np.isnan(r_al), 0.0, r_al)
            wealth_al = np.cumprod(1.0 + growth)
            series[code] = {"dates": fdates, "r_al": r_al, "wealth_al": wealth_al}
    return series


def month_end_trading_days(bench_dates_ns: np.ndarray) -> np.ndarray:
    """基准每月最后一个交易日，作为月度截面时点（输入输出均为int64纳秒）"""
    d = pd.to_datetime(bench_dates_ns, unit="ns")
    return pd.Series(d).groupby(d.to_period("M")).max().to_numpy(dtype="datetime64[ns]").astype("int64")


def fund_alpha_beta(rw: np.ndarray, w0: int, w1: int, bench_r_all: np.ndarray):
    """窗口内基金有效观测日与基准超额收益的OLS（截距=日alpha）"""
    mask = ~np.isnan(rw)
    if int(mask.sum()) < 2:
        return np.nan, np.nan
    x = bench_r_all[w0:w1 + 1] - RET_BENCH_RATE
    y = rw - RET_BENCH_RATE
    x, y = x[mask], y[mask]
    vx = x.var(ddof=1)
    if vx <= 0:
        return np.nan, np.nan
    beta = float(np.cov(y, x, ddof=1)[0, 1] / vx)
    alpha_d = float(y.mean() - beta * x.mean())
    return beta, float(alpha_d * 252)


def build_panel():
    """
    构建fund-month面板并输出 ml/panel.parquet
    每行：fund_code, t_date（月末截面交易日）, 特征（过去12m窗口）, y（未来6m收益）
    """
    os.makedirs(PANEL_DIR, exist_ok=True)
    bench = pd.read_csv(BENCH_PATH, parse_dates=["date"]).sort_values("date")
    bench_dates = bench["date"].to_numpy(dtype="datetime64[ns]").astype("int64")
    bench_r_all = bench["close"].pct_change().fillna(0.0).values  # 首日0（不参与回归窗口）
    month_ends = month_end_trading_days(bench_dates)
    series = load_fund_series(bench_dates)
    name_map = load_fund_name_map()
    print(f"全历史清洗基金：{len(series)} 只 | 基准交易日 {len(bench_dates)} 天 | 月末截面 {len(month_ends)} 个")

    rows = []
    n_months = 0
    for t_ns in month_ends:
        i_t = int(np.searchsorted(bench_dates, t_ns, side="right")) - 1  # 截面在基准日历中的索引
        if i_t < FEATURE_LOOKBACK:
            continue
        i_end = i_t + LABEL_HORIZON
        if i_end >= len(bench_dates):
            continue  # 标签期超出数据范围
        w0 = i_t - FEATURE_LOOKBACK
        w1 = i_t
        label_end_ns = bench_dates[i_end]
        for code, s in series.items():
            dates = s["dates"]
            # 时变eligibility 1：成立满12个月（基金首日 ≤ 截面 - 1年）
            if dates[0] > t_ns - MIN_AGE_NATURAL_DAYS * DAY_NS:
                continue
            # 截面当期有披露（t附近15天内有净值日）
            j_t = int(np.searchsorted(dates, t_ns, side="right")) - 1
            if j_t < 0 or t_ns - dates[j_t] > MAX_GAP_DAYS * DAY_NS:
                continue
            # 标签期末端有披露（t+h附近15天内有净值日），否则future return不可算
            j_end = int(np.searchsorted(dates, label_end_ns, side="right")) - 1
            if j_end <= j_t or label_end_ns - dates[j_end] > MAX_GAP_DAYS * DAY_NS:
                continue
            rw = s["r_al"][w0:w1 + 1]
            ww = s["wealth_al"][w0:w1 + 1]
            n_eff = int(np.sum(~np.isnan(rw)))
            cov = n_eff / (w1 - w0 + 1)
            # 时变eligibility：过去12m coverage≥95%
            if cov < MIN_COVERAGE:
                continue
            rv = rw[~np.isnan(rw)]
            age_days = t_ns - dates[0]  # 基金在该截面时的年龄（自然日）
            row = {"fund_code": code, "t_date": pd.Timestamp(t_ns, unit="ns"),
                   "fund_name": name_map.get(code, ""),
                   "age_months": round(age_days_to_months(age_days), 1),
                   "coverage": round(cov, 4)}
            # 复利区间收益：W(t)/W(t-k)-1
            for name, k in SHORT_WINDOWS.items():
                row[name] = float(ww[-1] / ww[-1 - k] - 1.0) if n_eff - 1 >= k else np.nan
            row["vol_12m"] = float(rv.std(ddof=1) * np.sqrt(252)) if n_eff > 2 else np.nan
            row["sharpe_12m"] = float((rv.mean() - RET_BENCH_RATE) / rv.std(ddof=1) * np.sqrt(252)) \
                if n_eff > 2 and rv.std(ddof=1) > 0 else np.nan
            peak = np.maximum.accumulate(ww)
            row["mdd_12m"] = float(np.nanmin(ww / peak - 1.0))
            beta, alpha = fund_alpha_beta(rw, w0, w1, bench_r_all)
            row["beta_12m"] = beta
            row["alpha_12m"] = alpha
            # 标签：未来6个月复利收益 W(t+h)/W(t)-1（对齐数组的财富在未披露日自然保持前值）
            row["future_ret_6m"] = float(s["wealth_al"][i_end] / s["wealth_al"][i_t] - 1.0)
            # 标签结束日：验证器据此判定"训练行的标签是否已到期能用于本次训练"
            row["label_end_date"] = pd.Timestamp(label_end_ns, unit="ns")
            rows.append(row)
        n_months += 1

    panel = pd.DataFrame(rows)
    panel.to_parquet(PANEL_PATH, index=False)
    print(f"面板构建完成：{PANEL_PATH}")
    print(f"面板规模：{len(panel)} 行（fund-month）| {panel['fund_code'].nunique()} 只基金 | {n_months} 个截面")
    print(f"截面范围：{panel['t_date'].min().date()} ~ {panel['t_date'].max().date()}")
    print(f"标签分布：future_ret_6m 中位={panel['future_ret_6m'].median():.2%} "
          f"均值={panel['future_ret_6m'].mean():.2%} std={panel['future_ret_6m'].std():.2%}")
    return panel


def build_panel_v3(out_path: str = None, extra_dirs=None):
    """panel v3：并入清盘基金 + 截面资格只看当时信息 + **终止标签按「实际收益 + 余下现金」**。

    与 v2（build_panel）的三处差异（用户 2026-09-19 指出 v2 会在训练面板上重新引入幸存者筛选）：
      1. **数据源**：fund_history + extra_dirs（`fund_history_delisted/`），后者是已清盘/终止基金；
      2. **资格判定不再要求「标签期末端有披露」**——v2 的这条检查等价于"事前知道该基金六个月后
         还活着"，会把标签期内终止的基金整行删掉（最差样本消失）；
      3. **标签三分类**（落盘 `label_resolution` 审计字段）：
         - `full`            ：标签期末端有披露（15 天容差内）→ W(t+h)/W(t)−1
         - `terminated_cash` ：**在清盘名单内且净值在标签期末前结束** → 用截至最后净值日的
                               实际收益 W(last)/W(t)−1，余下时间按现金持有（收益 0）计
         - `missing`         ：净值在标签期末前中断但**不在清盘名单内**（无法判断是终止还是
                               长期停披露）→ 标签保留为 NaN，不删行、不伪造
    审计字段：`terminated_in_label`、`last_nav_date`、`label_resolution`。

    注意：训练时应过滤 `label_resolution != "missing"`（或直接 dropna(标签)）；missing 行保留
    在面板里是为了**审计与覆盖率统计**，不代表可用样本。
    """
    from delisted_funds import FUND_PATH as DELISTED_LIST_PATH   # 清盘名单（代码 + 最早公告日）

    out_path = out_path or os.path.join(PANEL_DIR, "panel_v3.parquet")
    os.makedirs(PANEL_DIR, exist_ok=True)
    bench = pd.read_csv(BENCH_PATH, parse_dates=["date"]).sort_values("date")
    bench_dates = bench["date"].to_numpy(dtype="datetime64[ns]").astype("int64")
    bench_r_all = bench["close"].pct_change().fillna(0.0).values
    month_ends = month_end_trading_days(bench_dates)
    series = load_fund_series(bench_dates, extra_dirs=extra_dirs)
    name_map = dict(load_fund_name_map())
    delisted_codes = set()
    if os.path.exists(DELISTED_LIST_PATH):
        dl = pd.read_csv(DELISTED_LIST_PATH, dtype={"fund_code": str})
        delisted_codes = set(dl["fund_code"])
        if "fund_short_name" in dl.columns:       # 清盘基金名称（现存池名称表里没有它们）
            name_map.update({r["fund_code"]: r["fund_short_name"] for _, r in dl.iterrows()})
    print(f"[v3] 研究池 {len(series)} 只（含清盘 {len(set(series) & delisted_codes)} 只）| "
          f"基准 {len(bench_dates)} 天 | 月末截面 {len(month_ends)} 个")

    rows = []
    n_months = 0
    stat = {"full": 0, "terminated_cash": 0, "missing": 0, "rows_no_label_end": 0}
    for t_ns in month_ends:
        i_t = int(np.searchsorted(bench_dates, t_ns, side="right")) - 1
        if i_t < FEATURE_LOOKBACK:
            continue
        i_end = i_t + LABEL_HORIZON
        if i_end >= len(bench_dates):
            continue                    # 标签期超出数据范围（不是基金终止，是日历到头）
        w0 = i_t - FEATURE_LOOKBACK
        label_end_ns = bench_dates[i_end]
        for code, s in series.items():
            dates = s["dates"]
            if dates[0] > t_ns - MIN_AGE_NATURAL_DAYS * DAY_NS:
                continue
            j_t = int(np.searchsorted(dates, t_ns, side="right")) - 1
            if j_t < 0 or t_ns - int(dates[j_t]) > MAX_GAP_DAYS * DAY_NS:
                continue                # 截面当期无披露（当时的公开信息里它没有净值）
            rw = s["r_al"][w0:i_t + 1]
            n_eff = int(np.sum(~np.isnan(rw)))
            cov = n_eff / (i_t - w0 + 1)
            if cov < MIN_COVERAGE:
                continue
            last_nav_ns = int(dates[-1])
            terminated = code in delisted_codes and last_nav_ns < label_end_ns - MAX_GAP_DAYS * DAY_NS
            if terminated:
                # 标签期内终止：用截至最后净值日的实际收益，余下按现金（0 收益）
                i_last = int(np.searchsorted(bench_dates, last_nav_ns, side="right")) - 1
                if i_last <= i_t:
                    continue            # 终止发生在截面当期之前 → 当期本不该入选
                label = float(s["wealth_al"][i_last] / s["wealth_al"][i_t] - 1.0)
                res = "terminated_cash"
            else:
                j_end = int(np.searchsorted(dates, label_end_ns, side="right")) - 1
                if j_end > j_t and label_end_ns - int(dates[j_end]) <= MAX_GAP_DAYS * DAY_NS:
                    label = float(s["wealth_al"][i_end] / s["wealth_al"][i_t] - 1.0)
                    res = "full"
                else:
                    label = np.nan     # 无法判断（可能是长期停披露/终止但不在名单）
                    res = "missing"
            stat[res] += 1
            ww = s["wealth_al"][w0:i_t + 1]
            rv = rw[~np.isnan(rw)]
            age_days = t_ns - dates[0]
            row = {"fund_code": code, "t_date": pd.Timestamp(t_ns, unit="ns"),
                   "fund_name": name_map.get(code, ""),
                   "age_months": round(age_days_to_months(age_days), 1),
                   "coverage": round(cov, 4)}
            for name, k in SHORT_WINDOWS.items():
                row[name] = float(ww[-1] / ww[-1 - k] - 1.0) if n_eff - 1 >= k else np.nan
            row["vol_12m"] = float(rv.std(ddof=1) * np.sqrt(252)) if n_eff > 2 else np.nan
            row["sharpe_12m"] = float((rv.mean() - RET_BENCH_RATE) / rv.std(ddof=1) * np.sqrt(252)) \
                if n_eff > 2 and rv.std(ddof=1) > 0 else np.nan
            peak = np.maximum.accumulate(ww)
            row["mdd_12m"] = float(np.nanmin(ww / peak - 1.0))
            beta, alpha = fund_alpha_beta(rw, w0, i_t, bench_r_all)
            row["beta_12m"] = beta
            row["alpha_12m"] = alpha
            row["future_ret_6m"] = label
            row["label_end_date"] = pd.Timestamp(label_end_ns, unit="ns")
            row["terminated_in_label"] = bool(terminated)
            row["last_nav_date"] = pd.Timestamp(last_nav_ns, unit="ns")
            row["label_resolution"] = res
            rows.append(row)
        n_months += 1

    panel = pd.DataFrame(rows)
    panel.to_parquet(out_path, index=False)
    print(f"[v3] 面板构建完成：{out_path}")
    print(f"[v3] 规模 {len(panel)} 行 | {panel['fund_code'].nunique()} 只基金 | {n_months} 个截面"
          f" | 截面 {panel['t_date'].min().date()} ~ {panel['t_date'].max().date()}")
    print(f"[v3] 标签解析：full {stat['full']} | terminated_cash {stat['terminated_cash']} | "
          f"missing {stat['missing']}")
    ok = panel[panel["label_resolution"] != "missing"]
    print(f"[v3] 可用于训练的标签：full+terminated = {len(ok)} 行"
          f"（其中终止持有 {stat['terminated_cash']} 行）")
    print(f"[v3] 清盘基金参与：进入可投资截面 {panel[panel.fund_code.isin(delisted_codes)]['fund_code'].nunique()} 只"
          f" / 共 {len(set(panel[panel.fund_code.isin(delisted_codes)]['fund_code']))} 只名单内")
    return panel


def main():
    import argparse
    ap = argparse.ArgumentParser(description="面板构建（v2 现存池 / v3 并入清盘池 + 终止标签）")
    ap.add_argument("--v3", action="store_true", help="构建 panel v3（并入 fund_history_delisted）")
    ap.add_argument("--out", default=None, help="输出路径（默认 ml/panel.parquet 或 ml/panel_v3.parquet）")
    args = ap.parse_args()
    if args.v3:
        from delisted_funds import DELISTED_HISTORY_DIR
        build_panel_v3(out_path=args.out, extra_dirs=[DELISTED_HISTORY_DIR])
    else:
        build_panel()


if __name__ == "__main__":
    main()
