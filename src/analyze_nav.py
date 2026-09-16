import os
import re
import numpy as np
import pandas as pd
import akshare as ak
from scipy import stats as sps

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

CLEAN_NAV_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "fund_processed")
ANALYZE_DIR = os.path.join(PROJECT_ROOT, "data", "analyze")
SUMMARY_PATH = os.path.join(ANALYZE_DIR, "analysis_summary.csv")
CHART_DIR = os.path.join(ANALYZE_DIR, "charts")
BENCHMARK_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "benchmark_hs300.csv")
CLEAN_REPORT_PATH = os.path.join(CLEAN_NAV_DIR, "clean_report.csv")
FUND_META_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "fund_code_list.csv")


def load_fund_name_map() -> dict:
    """
    基金代码 -> 简称 映射，两级来源：
    1. 优先本批次 clean_report 的 fund_name 列（与processed同批次，快照一致）
    2. 回退全量元数据 fund_code_list.csv（任何池子都是其子集）
    """
    if os.path.exists(CLEAN_REPORT_PATH):
        rep = pd.read_csv(CLEAN_REPORT_PATH, dtype={"fund_code": str})
        if "fund_name" in rep.columns:
            m = dict(zip(rep["fund_code"], rep["fund_name"].fillna("")))
            if any(v for v in m.values()):
                return m
    if os.path.exists(FUND_META_PATH):
        df = pd.read_csv(FUND_META_PATH, dtype={"基金代码": str})
        return dict(zip(df["基金代码"], df["基金简称"].astype(str)))
    print(f"[警告] 名称来源缺失（clean_report与{FUND_META_PATH}均不可用），fund_name将为空")
    return {}

# 指标口径常量
TRADING_DAYS = 252      # 年化交易日
RISK_FREE_RATE = 0.02   # 年化无风险利率，夏普比率用
ANN_FACTOR = np.sqrt(TRADING_DAYS)
RF_DAILY = RISK_FREE_RATE / TRADING_DAYS        # 日频无风险利率
MIN_REGRESSION_DAYS = 60                        # alpha/beta回归的最小对齐样本天数
VAR_PCT = 0.05                                  # VaR/CVaR 置信水平（5%尾部）


def load_benchmark_returns() -> pd.Series:
    """
    加载沪深300基准日收益序列（读取时做防御处理：去重、排序、去NaN）
    :return: date为索引的日收益Series；基准文件缺失时返回None
    """
    if not os.path.exists(BENCHMARK_PATH):
        print(f"[警告] 基准文件不存在：{BENCHMARK_PATH}，alpha/beta将记为NaN")
        return None
    df = pd.read_csv(BENCHMARK_PATH, parse_dates=["date"])
    df = df.dropna(subset=["close"]).drop_duplicates(subset=["date"]).sort_values("date")
    return df.set_index("date")["close"].pct_change().dropna()


def analyze_fund(df: pd.DataFrame, fund_code: str, bench_r: pd.Series) -> dict:
    """
    单基金分析，收益类指标全部基于官方日增长率（daily_ret，真实日收益，已含分红调整）
    回撤/年化收益用 (1+daily_ret).cumprod() 重建的复权净值曲线计算
    注意：nav_acc不是复权净值，分红日对其pct_change会被稀释，故不再使用
    alpha/beta：基金日收益与沪深300日收益减去日频无风险利率后做OLS回归，
    beta为斜率，alpha为截距×252年化；样本不足MIN_REGRESSION_DAYS时记NaN
    标准误为Newey-West HAC（Bartlett核），p值用t分布
    口径说明：CAGR按真实自然年折算；波动/夏普/alpha年化按252交易日惯例
    :param df: 清洗后的净值df，列 date, nav, nav_acc, daily_ret
    :param fund_code: 基金代码
    :param bench_r: 基准日收益序列（date索引），可为None
    :return: 指标字典：基金代码、起止日期、年化收益、年化波动、夏普、最大回撤、
             最大回撤修复时间（谷底→重新站上前峰值，自然日）、是否已修复、
             年化alpha、beta、R²、alpha的t统计量与p值、回归样本数
    """
    # 日收益（官方口径，真实收益）；索引换成日期，便于和基准对齐
    r = df["daily_ret"].astype(float)
    r.index = df["date"].values
    n = len(r)
    if n < 2:
        raise ValueError(f"有效日收益数据不足：{n} 天")

    # 复权净值曲线（起点归1附近的相对水平），用于收益与回撤
    nav_adj = (1.0 + r).cumprod()
    dates = df["date"]

    # 年化收益：CAGR，按真实自然年折算（end-start天数/365.25）
    # 注意：不能用252/n观测数折算——实际每年披露数<252，会系统性高估CAGR
    # 其余年化因子（波动/夏普/下行偏差/alpha×252）保持行业252交易日惯例，口径差异见docstring
    years = float((dates.iloc[-1] - dates.iloc[0]).days / 365.25)
    if years <= 0:
        raise ValueError(f"窗口时间跨度异常：{years}年")
    ann_return = float(nav_adj.iloc[-1] ** (1.0 / years) - 1)

    # 年化波动：日收益标准差年化
    ann_vol = float(r.std() * ANN_FACTOR)

    # 夏普比率
    excess = r - RF_DAILY            # r 为日收益序列，RF_DAILY = 0.02/252
    sharpe = excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS)

    # 最大回撤（负数，基于复权净值）
    cummax = nav_adj.cummax()
    dd = nav_adj / cummax - 1
    mdd = float(dd.min())

    # 最大回撤修复时间（自然日）：从谷底到净值重新站上回撤前峰值的天数；至今未修复记NaN
    # drawdown_repaired 列显式标记是否已修复（False=未修复），避免NaN含义不明
    trough_pos = int(dd.values.argmin())
    if mdd == 0.0:
        recovery_days = np.nan
        repaired = True
    else:
        peak_val = float(nav_adj.iloc[:trough_pos + 1].max())
        after = nav_adj.iloc[trough_pos + 1:].values
        rec_pos = np.where(after >= peak_val)[0]
        if len(rec_pos) > 0:
            trough_date = dates.iloc[trough_pos]
            rec_date = dates.iloc[trough_pos + 1 + rec_pos[0]]
            recovery_days = float((rec_date - trough_date).days)
            repaired = True
        else:
            recovery_days = np.nan
            repaired = False

    # 尾部风险指标（历史法，基于日收益）
    # VaR95：日收益5%分位（负数）；CVaR95：VaR以下尾部均值；下行偏差：相对0收益的下行波动年化
    var_95 = float(np.percentile(r.values, VAR_PCT * 100))
    tail = r[r <= var_95]
    cvar_95 = float(tail.mean()) if len(tail) > 0 else var_95
    downside_dev = float(np.sqrt((np.minimum(r.values, 0.0) ** 2).mean()) * ANN_FACTOR)

    # 窗口内累计收益
    cum_return = float(nav_adj.iloc[-1] - 1.0)

    # alpha/beta：与沪深300日超额收益做最小二乘回归（CAPM）
    # 基准语义：alpha=相对沪深300的超额，行业/小盘基金存在基准错配风险，用r_squared辅助判断
    alpha_ann = np.nan
    beta = np.nan
    r_squared = np.nan
    alpha_t = np.nan
    alpha_p = np.nan
    reg_days = 0
    if bench_r is not None:
        merged = pd.concat([r.rename("r_f"), bench_r.rename("r_b")], axis=1, join="inner").dropna()
        reg_days = len(merged)
        if reg_days >= MIN_REGRESSION_DAYS:
            x = (merged["r_b"] - RF_DAILY).values
            y = (merged["r_f"] - RF_DAILY).values
            beta, alpha_d = np.polyfit(x, y, deg=1)
            # R² = 1 - SS_res/SS_tot（OLS口径，不受标准误修正影响）
            ss_res = float(np.sum((y - (beta * x + alpha_d)) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
            # alpha的t统计量：Newey-West HAC标准误（Bartlett核，修正异方差与自相关）
            X_mat = np.column_stack([np.ones(reg_days), x])
            u = y - (beta * x + alpha_d)  # OLS残差
            lag_max = int(np.floor(4.0 * (reg_days / 100.0) ** (2.0 / 9.0)))  # NW经验滞后
            XtX_inv = np.linalg.pinv(X_mat.T @ X_mat)
            ux = X_mat * u[:, None]
            S = ux.T @ ux
            for j in range(1, lag_max + 1):
                w = 1.0 - j / (lag_max + 1.0)
                g = ux[:-j].T @ ux[j:]
                S += w * (g + g.T)
            V = XtX_inv @ S @ XtX_inv
            se_alpha = float(np.sqrt(V[0, 0]))
            alpha_t = float(alpha_d / se_alpha) if se_alpha > 0 else np.nan
            # p值：t分布（df=reg_days-2）
            if np.isfinite(alpha_t):
                alpha_p = float(2.0 * sps.t.sf(abs(alpha_t), df=reg_days - 2))
            alpha_ann = float(alpha_d * TRADING_DAYS)
            beta = float(beta)

    return {
        "fund_code": fund_code,
        "date_start": dates.iloc[0].date(),
        "date_end": dates.iloc[-1].date(),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "drawdown_recovery_days": recovery_days,
        "drawdown_repaired": repaired,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "downside_dev": downside_dev,
        "cum_return": cum_return,
        "alpha_ann": alpha_ann,
        "beta": beta,
        "r_squared": r_squared,
        "alpha_t": alpha_t,
        "alpha_p": alpha_p,
        "reg_days": reg_days,
    }


def plot_fund(df: pd.DataFrame, fund_code: str, fund_name: str, bench_r: pd.Series) -> str:
    """
    单基金三联图：净值对比（vs沪深300归一化）/ 回撤曲线 / 日收益分布（含VaR、CVaR）
    :return: 图文件路径
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    r = df["daily_ret"].astype(float)
    r.index = df["date"].values
    nav_adj = (1.0 + r).cumprod()
    dd = nav_adj / nav_adj.cummax() - 1

    # 基准对齐到基金窗口期，同样复权归一
    bench_nav = None
    if bench_r is not None:
        b = bench_r.reindex(r.index).dropna()
        bench_nav = (1.0 + b).cumprod()

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False,
                             gridspec_kw={"height_ratios": [2, 1, 1]})
    title = f"{fund_name}({fund_code}) 窗口内表现"
    fig.suptitle(title, fontsize=13)

    # 1. 净值对比
    ax = axes[0]
    ax.plot(r.index, nav_adj.values, label="基金(复权)", color="tab:blue", lw=1.2)
    if bench_nav is not None:
        ax.plot(bench_nav.index, bench_nav.values, label="沪深300(归一)", color="tab:orange", lw=1.0)
    ax.set_ylabel("归一净值")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # 2. 回撤
    ax = axes[1]
    ax.fill_between(r.index, dd.values, 0, color="tab:red", alpha=0.35)
    ax.plot(r.index, dd.values, color="tab:red", lw=0.8)
    ax.set_ylabel("回撤")
    ax.grid(alpha=0.3)

    # 3. 日收益分布 + VaR/CVaR
    ax = axes[2]
    ax.hist(r.values, bins=60, color="tab:blue", alpha=0.6)
    var_95 = float(np.percentile(r.values, VAR_PCT * 100))
    tail = r[r <= var_95]
    cvar_95 = float(tail.mean()) if len(tail) > 0 else var_95
    ax.axvline(var_95, color="tab:orange", lw=1.5, label=f"VaR95={var_95:.2%}")
    ax.axvline(cvar_95, color="tab:red", lw=1.5, label=f"CVaR95={cvar_95:.2%}")
    ax.set_ylabel("频次")
    ax.set_xlabel("日收益")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    os.makedirs(CHART_DIR, exist_ok=True)
    out_path = os.path.join(CHART_DIR, f"fund_{fund_code}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def analyze_one(fund_code: str):
    """
    用户级接口：输入基金代码，输出该基金全部指标 + 三联图
    用法：python src/analyze_nav.py 000309
    """
    path = os.path.join(CLEAN_NAV_DIR, f"fund_{fund_code}.csv")
    if not os.path.exists(path):
        print(f"未找到基金 {fund_code} 的清洗数据：{path}")
        print("请先运行 clean_nav.py，或确认基金代码是否在当前窗口池内")
        return None

    bench_r = load_benchmark_returns()
    df = pd.read_csv(path, parse_dates=["date"])
    info = analyze_fund(df, fund_code, bench_r)

    # 名称来自元数据映射（批次快照优先，回退全量列表）
    fund_name = load_fund_name_map().get(fund_code, "")

    print(f"\n===== {fund_name}({fund_code}) =====")
    print(f"窗口：{info['date_start']} ~ {info['date_end']}")
    print(f"累计收益：{info['cum_return']:.2%} | 年化收益：{info['ann_return']:.2%}")
    print(f"年化波动：{info['ann_vol']:.2%} | 下行偏差：{info['downside_dev']:.2%}")
    print(f"夏普：{info['sharpe']:.2f} | 最大回撤：{info['max_drawdown']:.2%}")
    rec = info['drawdown_recovery_days']
    rec_txt = f"{rec:.0f}天" if np.isfinite(rec) else ("未修复" if not info['drawdown_repaired'] else "无回撤")
    print(f"回撤修复：{rec_txt} | VaR95：{info['var_95']:.2%} | CVaR95：{info['cvar_95']:.2%}")
    if np.isfinite(info["alpha_ann"]):
        print(f"alpha：{info['alpha_ann']:.2%}（t={info['alpha_t']:.2f}, p={info['alpha_p']:.3f}） | "
              f"beta：{info['beta']:.2f} | R²：{info['r_squared']:.2f} | 样本：{info['reg_days']}天")
    else:
        print(f"alpha/beta：不可用（回归样本{info['reg_days']}天不足{MIN_REGRESSION_DAYS}或基准缺失）")

    chart = plot_fund(df, fund_code, fund_name, bench_r)
    print(f"图已保存：{chart}")
    return info


def remove_stale_charts(success_codes: set[str]) -> int:
    """仅清理批量运行遗留的 fund_六位代码.png；先确认本轮图片完整。"""
    if not os.path.isdir(CHART_DIR):
        raise RuntimeError(f"图表目录不存在，跳过旧图清理：{CHART_DIR}")

    expected = {f"fund_{code}.png" for code in success_codes}
    chart_files = {
        entry.name: entry.path
        for entry in os.scandir(CHART_DIR)
        if entry.is_file(follow_symlinks=False)
        and re.fullmatch(r"fund_\d{6}\.png", entry.name)
    }
    missing = expected - chart_files.keys()
    empty = {name for name in expected & chart_files.keys()
             if os.path.getsize(chart_files[name]) == 0}
    if missing or empty:
        raise RuntimeError(
            f"本轮图表校验失败，未清理旧图：缺失{len(missing)}张，空文件{len(empty)}张"
        )

    stale = chart_files.keys() - expected
    for name in sorted(stale):
        os.remove(chart_files[name])

    remaining = {
        entry.name for entry in os.scandir(CHART_DIR)
        if entry.is_file(follow_symlinks=False)
        and re.fullmatch(r"fund_\d{6}\.png", entry.name)
    }
    if remaining != expected:
        raise RuntimeError("旧图清理后图表集合与本轮成功基金不一致")
    return len(stale)


def analyze_all():
    """
    批量分析：对 data/processed/fund_processed 下全部 fund_*.csv 做单基金分析
    输出 data/analyze/analysis_summary.csv + data/analyze/charts/ 下每基金一张三联图
    """
    os.makedirs(ANALYZE_DIR, exist_ok=True)
    bench_r = load_benchmark_returns()
    file_list = [f for f in os.listdir(CLEAN_NAV_DIR) if f.startswith("fund_") and f.endswith(".csv")]
    print(f"待分析基金数量：{len(file_list)}")

    # 名称来自元数据映射（批次快照优先，回退全量列表）
    name_map = load_fund_name_map()

    rows = []
    success_codes = set()
    for fname in file_list:
        fund_code = fname.replace("fund_", "").replace(".csv", "")
        try:
            df = pd.read_csv(os.path.join(CLEAN_NAV_DIR, fname), parse_dates=["date"])
            info = analyze_fund(df, fund_code, bench_r)
            info["fund_name"] = name_map.get(fund_code, "")
            plot_fund(df, fund_code, info["fund_name"], bench_r)
        except Exception as e:
            rows.append({"fund_code": fund_code, "error": str(e)})
            print(f"[{fund_code}] [FAIL] 分析失败 {str(e)[:80]}")
            continue
        rows.append(info)
        success_codes.add(fund_code)
        print(f"[{fund_code}] [OK] 年化:{info['ann_return']:.2%} 波动:{info['ann_vol']:.2%} "
              f"夏普:{info['sharpe']:.2f} 最大回撤:{info['max_drawdown']:.2%} "
              f"alpha:{info['alpha_ann']:.2%} beta:{info['beta']:.2f}")

    df_sum = pd.DataFrame(rows)
    # 列排序：代码、名称在前
    front_cols = ["fund_code", "fund_name"]
    other_cols = [c for c in df_sum.columns if c not in front_cols]
    df_sum = df_sum[front_cols + other_cols]
    df_sum.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    ok = df_sum.dropna(subset=["ann_return"])
    if len(success_codes) == len(file_list):
        removed = remove_stale_charts(success_codes)
        print(f"图表校验通过：本轮{len(success_codes)}张，清理旧图{removed}张")
    else:
        print(f"[警告] 本轮有{len(file_list) - len(success_codes)}只基金失败，跳过旧图清理")
    print(f"\n===== 分析完成 =====")
    print(f"成功 {len(ok)} 只 / 失败 {len(df_sum) - len(ok)} 只")
    print(f"汇总表输出至：{SUMMARY_PATH}")
    print(f"图表目录：{CHART_DIR}")
    print("\n指标分布（成功基金）：")
    print(ok[["ann_return", "ann_vol", "sharpe", "max_drawdown", "alpha_ann", "beta",
              "r_squared", "alpha_t"]].describe().round(4))
    low_r2 = int((ok["r_squared"] < 0.5).sum())
    print(f"\nR²<0.5（与沪深300相关性弱，alpha/beta解读需谨慎）：{low_r2} 只")
    sig_cnt = int((ok["alpha_t"].abs() > 2).sum())
    print(f"alpha |t|>2（常规显著水平）：{sig_cnt} 只")
    unrepaired = ok["drawdown_repaired"].eq(False).sum()
    print(f"\n最大回撤至今未修复的基金：{int(unrepaired)} 只（summary中drawdown_repaired=False）")
    return df_sum


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_one(sys.argv[1])
    else:
        analyze_all()
