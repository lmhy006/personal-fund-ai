# style_factors.py：Phase 3 特征扩充第二组——风格指数因子（大小盘 / 成长）
#
# 目的：把"过去一年高收益"拆成「风格延续」与「风格外选基」两部分，
#       回答 Phase 2 遗留的局限——月内去均值只去掉共同涨跌，**没有去掉风格差异**。
#
# 因子定义（全部只用历史可得数据，对齐沪深300交易日日历）：
#   mkt    = 沪深300 日收益
#   size   = 中证500 日收益 − 沪深300 日收益     （中小盘相对大盘；中证500 自 2005-01 起）
#   growth = 创业板指 日收益 − 沪深300 日收益     （成长相对大盘；创业板指自 2010-06 起）
#
# 口径与局限（写清以便复核）：
#   - size/growth 用「价差」而非正交化因子：正交化需要估计价差对市场的回归系数，
#     用全样本估计会给因子注入前视，用滚动估计又引入额外设计选择。价差的代价是
#     三者相关（大小盘与市场同涨同跌），回归系数解释需谨慎。
#   - 指数收盘价为**价格指数**（不含分红），与项目既有基准口径一致。
#   - 指数缺失日（与沪深300日历不一致）保留 NaN，不做前向填充——宁可少观测，
#     不用伪造价格。
#   - 行业维度未做（数据源已验证可用：`ak.index_hist_sw` 申万一级行业指数，
#     1999-12-30 起）——列入后续组。
#
# 缓存：data/raw/style_index_{symbol}.csv（注意 data/ 被 .gitignore 忽略，见 README 数据层约定）
import os
import time

import akshare as ak
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
BENCH_PATH = os.path.join(RAW_DIR, "benchmark_hs300.csv")

# 风格指数：key -> (新浪代码, 中文名)
STYLE_INDICES = {
    "size": ("sh000905", "中证500"),
    "growth": ("sz399006", "创业板指"),
}


def index_cache_path(symbol: str) -> str:
    return os.path.join(RAW_DIR, f"style_index_{symbol}.csv")


def fetch_index_daily(symbol: str, use_cache: bool = True, retries: int = 3) -> pd.DataFrame:
    """拉取/读取指数日线（新浪源，与项目沪深300基准同一接口）。返回 [date, close]。"""
    path = index_cache_path(symbol)
    if use_cache and os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])
    last_err = None
    for i in range(retries):
        try:
            df = ak.stock_zh_index_daily(symbol=symbol)
            if df is None or len(df) == 0:
                raise RuntimeError("接口返回空数据")
            df = df[["date", "close"]].copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            os.makedirs(RAW_DIR, exist_ok=True)
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"[风格指数] {symbol} 下载完成：{len(df)} 行 "
                  f"({df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()})")
            return df
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 2 + i * 2
            print(f"[警告] {symbol} 获取失败，重试 {i+1}/{retries}，等待 {wait}s，err={str(e)[:100]}")
            time.sleep(wait)
    raise RuntimeError(f"指数 {symbol} 多次请求失败：{last_err}")


def align_close_to_calendar(df: pd.DataFrame, bench_dates: np.ndarray) -> np.ndarray:
    """指数收盘价对齐到沪深300交易日日历；非同日（含无数据日）为 NaN。"""
    d = df["date"].to_numpy(dtype="datetime64[ns]").astype("int64")
    c = df["close"].astype(float).values
    out = np.full(len(bench_dates), np.nan)
    pos = np.searchsorted(bench_dates, d)
    ok = (pos < len(bench_dates)) & (bench_dates[np.minimum(pos, len(bench_dates) - 1)] == d)
    out[pos[ok]] = c[ok]
    return out


def _daily_rets(close_aligned: np.ndarray) -> np.ndarray:
    """对数无关的简单日收益；任一端缺失则为 NaN（首日 NaN）。"""
    r = np.full(len(close_aligned), np.nan)
    prev, cur = close_aligned[:-1], close_aligned[1:]
    with np.errstate(invalid="ignore"):
        r[1:] = cur / prev - 1.0
    return r


def load_style_factors(bench_dates: np.ndarray, use_cache: bool = True) -> dict:
    """构造风格因子日收益序列（对齐 bench_dates）。

    :return: {"mkt": arr, "size": arr, "growth": arr, "meta": {key: (symbol, name, first_valid_date)}}
    """
    bench = pd.read_csv(BENCH_PATH, parse_dates=["date"]).sort_values("date")
    bench_close = align_close_to_calendar(bench, bench_dates)
    mkt = _daily_rets(bench_close)
    facs = {"mkt": mkt}
    meta = {}
    for key, (symbol, name) in STYLE_INDICES.items():
        close_al = align_close_to_calendar(fetch_index_daily(symbol, use_cache=use_cache), bench_dates)
        spread = _daily_rets(close_al) - mkt
        facs[key] = spread
        valid = np.where(~np.isnan(spread))[0]
        first = bench_dates[valid[0]] if len(valid) else None
        meta[key] = (symbol, name, pd.Timestamp(first, unit="ns").date() if first is not None else None)
        print(f"[风格因子] {key} = {name} − 沪深300：可用 {len(valid)} 天，"
              f"首个有效日 {meta[key][2]}")
    return {"factors": facs, "meta": meta}


def factor_names_for(include_growth: bool) -> list:
    """因子集：始终含市场与规模；growth 可选（创业板指 2010-06 才有数据）。"""
    return ["mkt", "size", "growth"] if include_growth else ["mkt", "size"]


# ---------------------------------------------------------------- 行业维度（申万一级 → 6 板块）
# 申万一级行业 31 个（代码见 ak.sw_index_first_info），按 A 股常见风格聚成 6 个板块；
# 板块因子 = 成分行业指数**日收益等权平均** − 沪深300（相对市场的行业价差）。
# 起点限制（实测 2026-09-18）：申万行业体系 2014 重构，12 个行业指数 2014-02-21 才有数据
# （银行/非银/计算机/传媒/通信/电力设备/军工/机械/汽车/建材/建筑/煤炭），
# 另有 3 个 2021-12-13 才发布（美容护理 801980 / 石油石化 801960 / 环保 801970）——
# 这 3 个**暂不计入板块**（否则板块因子只能从 2021-12 起，区间过短），
# 因此板块因子（要求成分行业全部就绪）统一从 **2014-02-24** 起可用。
# 局限：板块划分是人为选择（非官方分类）；行业指数为价格指数；2014 年前无法做行业中性化。
SW_SECTORS = {
    "金融地产": ["801780", "801790", "801180"],                              # 银行/非银/房地产
    "消费": ["801120", "801110", "801130", "801140", "801200", "801210"],    # 食品饮料/家电/纺服/轻工/商贸/社服
    "医药": ["801150"],
    "科技": ["801080", "801750", "801760", "801770"],                         # 电子/计算机/传媒/通信
    "制造": ["801730", "801740", "801890", "801880"],                         # 电力设备/军工/机械/汽车
    "周期": ["801010", "801030", "801040", "801050", "801710", "801720",
             "801160", "801170", "801950", "801230"],                        # 农业/化工/钢铁/有色/建材/建筑/公用/交运/煤炭/综合
}
SECTOR_KEYS = list(SW_SECTORS.keys())
ALL_FACTOR_KEYS = ["mkt", "size", "growth"] + SECTOR_KEYS
SECTOR_FIRST_VALID = "2014-02-24"    # 板块因子首个有效日（= 最晚发布的成分行业上市次日）


def sw_industry_cache_path(symbol: str) -> str:
    return os.path.join(RAW_DIR, f"sw_industry_{symbol}.csv")


def fetch_sw_industry(symbol: str, use_cache: bool = True, retries: int = 3,
                      sleep_sec: float = 0.3) -> pd.DataFrame:
    """拉取/读取申万一级行业指数日线。返回 [date, close]（接口列为 代码/日期/收盘/...）。"""
    path = sw_industry_cache_path(symbol)
    if use_cache and os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])
    last_err = None
    for i in range(retries):
        try:
            df = ak.index_hist_sw(symbol=symbol, period="day")
            if df is None or len(df) == 0:
                raise RuntimeError("接口返回空数据")
            df = df.rename(columns={"日期": "date", "收盘": "close"})[["date", "close"]].copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            os.makedirs(RAW_DIR, exist_ok=True)
            df.to_csv(path, index=False, encoding="utf-8-sig")
            time.sleep(sleep_sec)
            return df
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 2 + i * 2
            print(f"[警告] 申万行业 {symbol} 获取失败，重试 {i+1}/{retries}，等待 {wait}s，"
                  f"err={str(e)[:100]}")
            time.sleep(wait)
    raise RuntimeError(f"申万行业 {symbol} 多次请求失败：{last_err}")


def load_sector_factors(bench_dates: np.ndarray, use_cache: bool = True) -> dict:
    """构造 6 个板块的「相对市场」价差因子（对齐沪深300日历）。

    :return: {"factors": {sector: arr}, "meta": {sector: (首个有效日, 成分行业数稳)},
              "industry_first": {code: 首个数据日}}
    """
    bench = pd.read_csv(BENCH_PATH, parse_dates=["date"]).sort_values("date")
    mkt = _daily_rets(align_close_to_calendar(bench, bench_dates))
    factors, meta, industry_first = {}, {}, {}
    for sector, codes in SW_SECTORS.items():
        rets, firsts = [], []
        for code in codes:
            df = fetch_sw_industry(code, use_cache=use_cache)
            industry_first[code] = pd.Timestamp(df["date"].iloc[0]).date()
            firsts.append(industry_first[code])
            rets.append(_daily_rets(align_close_to_calendar(df, bench_dates)))
        R = np.vstack(rets)                      # (n_industry, T)
        # 要求板块内全部成分行业当日均有数据，否则该日 NaN（避免早期"半板块"均值假象）
        board = np.where(np.isnan(R).any(axis=0), np.nan, R.mean(axis=0))
        spread = board - mkt
        factors[sector] = spread
        valid = np.where(~np.isnan(spread))[0]
        first_valid = (pd.Timestamp(bench_dates[valid[0]], unit="ns").date()
                       if len(valid) else None)
        meta[sector] = (first_valid, max(firsts))
    return {"factors": factors, "meta": meta, "industry_first": industry_first}
