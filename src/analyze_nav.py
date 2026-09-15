import os
import numpy as np
import pandas as pd
import akshare as ak

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

CLEAN_NAV_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "fund_processed")
ANALYZE_DIR = os.path.join(PROJECT_ROOT, "data", "analyze")
SUMMARY_PATH = os.path.join(ANALYZE_DIR, "analysis_summary.csv")
BENCHMARK_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "benchmark_hs300.csv")

# 指标口径常量
TRADING_DAYS = 252      # 年化交易日
RISK_FREE_RATE = 0.02   # 年化无风险利率，夏普比率用
ANN_FACTOR = np.sqrt(TRADING_DAYS)
RF_DAILY = RISK_FREE_RATE / TRADING_DAYS        # 日频无风险利率
MIN_REGRESSION_DAYS = 60                        # alpha/beta回归的最小对齐样本天数


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
    :param df: 清洗后的净值df，列 date, nav, nav_acc, daily_ret
    :param fund_code: 基金代码
    :param bench_r: 基准日收益序列（date索引），可为None
    :return: 指标字典：基金代码、起止日期、年化收益、年化波动、夏普、最大回撤、最大回撤修复时间、年化alpha、beta
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

    # 年化收益：CAGR，按交易日折算
    ann_return = float(nav_adj.iloc[-1] ** (TRADING_DAYS / n) - 1)

    # 年化波动：日收益标准差年化
    ann_vol = float(r.std() * ANN_FACTOR)

    # 夏普比率
    excess = r - RF_DAILY            # r 为日收益序列，RF_DAILY = 0.02/252
    sharpe = excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS)

    # 最大回撤（负数，基于复权净值）
    cummax = nav_adj.cummax()
    dd = nav_adj / cummax - 1
    mdd = float(dd.min())

    # 最大回撤修复时间（自然日）：谷底净值重新站上回撤前峰值所需天数；至今未修复记NaN
    trough_pos = int(dd.values.argmin())
    if mdd == 0.0:
        recovery_days = np.nan
    else:
        peak_val = float(nav_adj.iloc[:trough_pos + 1].max())
        after = nav_adj.iloc[trough_pos + 1:].values
        rec_pos = np.where(after >= peak_val)[0]
        if len(rec_pos) > 0:
            peak_pos = int(nav_adj.iloc[:trough_pos + 1].values.argmax())
            peak_date = dates.iloc[peak_pos]
            rec_date = dates.iloc[trough_pos + 1 + rec_pos[0]]
            recovery_days = float((rec_date - peak_date).days)
        else:
            recovery_days = np.nan

    # alpha/beta：与沪深300日超额收益做最小二乘回归（CAPM）
    alpha_ann = np.nan
    beta = np.nan
    if bench_r is not None:
        merged = pd.concat([r.rename("r_f"), bench_r.rename("r_b")], axis=1, join="inner").dropna()
        if len(merged) >= MIN_REGRESSION_DAYS:
            x = (merged["r_b"] - RF_DAILY).values
            y = (merged["r_f"] - RF_DAILY).values
            beta, alpha_d = np.polyfit(x, y, deg=1)
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
        "alpha_ann": alpha_ann,
        "beta": beta,
    }


def analyze_all():
    """
    批量分析：对 data/processed/fund_processed 下全部 fund_*.csv 做单基金分析
    输出 data/analyze/analysis_summary.csv，每基金一行指标
    """
    os.makedirs(ANALYZE_DIR, exist_ok=True)
    bench_r = load_benchmark_returns()
    file_list = [f for f in os.listdir(CLEAN_NAV_DIR) if f.startswith("fund_") and f.endswith(".csv")]
    print(f"待分析基金数量：{len(file_list)}")

    rows = []
    for fname in file_list:
        fund_code = fname.replace("fund_", "").replace(".csv", "")
        try:
            df = pd.read_csv(os.path.join(CLEAN_NAV_DIR, fname), parse_dates=["date"])
            info = analyze_fund(df, fund_code, bench_r)
        except Exception as e:
            rows.append({"fund_code": fund_code, "error": str(e)})
            print(f"[{fund_code}] [FAIL] 分析失败 {str(e)[:80]}")
            continue
        rows.append(info)
        print(f"[{fund_code}] [OK] 年化:{info['ann_return']:.2%} 波动:{info['ann_vol']:.2%} "
              f"夏普:{info['sharpe']:.2f} 最大回撤:{info['max_drawdown']:.2%} "
              f"alpha:{info['alpha_ann']:.2%} beta:{info['beta']:.2f}")

    df_sum = pd.DataFrame(rows)
    df_sum.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    ok = df_sum.dropna(subset=["ann_return"])
    print(f"\n===== 分析完成 =====")
    print(f"成功 {len(ok)} 只 / 失败 {len(df_sum) - len(ok)} 只")
    print(f"汇总表输出至：{SUMMARY_PATH}")
    print("\n指标分布（成功基金）：")
    print(ok[["ann_return", "ann_vol", "sharpe", "max_drawdown", "alpha_ann", "beta"]].describe().round(4))
    unrepaired = ok["drawdown_recovery_days"].isna().sum()
    print(f"\n最大回撤至今未修复的基金：{int(unrepaired)} 只")
    return df_sum


if __name__ == "__main__":
    analyze_all()
