import akshare as ak
import pandas as pd
import os
import time

# —— Windows 输出编码兜底（2026-09-24）：管道/重定向 stdout 默认 GBK，print ⚠️ 等符号会崩
import sys
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# 路径设置
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
RAW_DATA_DIR = os.path.join(PROJECT_DIR, "data", "raw")
NAV_DATA_DIR = os.path.join(RAW_DATA_DIR, "fund_nav")

# 缓存新鲜度容差：缓存末日期落后锚点超过该天数才视为过期重拉（容忍净值披露日不同步）
STALE_TOLERANCE_DAYS = 3

def load_single_fund(fund_code: str, use_cache: bool, stale_before=None) -> pd.DataFrame:
    """
    下载单只基金的净值数据（单位净值+累计净值）
    :param fund_code: 基金代码
    :param use_cache: 是否使用缓存
    :param stale_before: 最新交易日锚点（基准指数最新日期）
        None: 存在即用（断点续传语义）
        给定: 缓存末日期落后锚点超过STALE_TOLERANCE_DAYS天则视为过期重拉，否则跳过
    :return: 历史净值数据的DataFrame[date, nav, nav_acc]
    """
    cache_file = os.path.join(NAV_DATA_DIR, f"fund_{fund_code}.csv")

    # 缓存分支：存在即用，或按锚点判断新鲜度
    if use_cache and os.path.exists(cache_file):
        df = pd.read_csv(cache_file, parse_dates=["date"])
        df = df.set_index("date")
        if stale_before is None:
            return df
        if df.index.max() >= stale_before - pd.Timedelta(days=STALE_TOLERANCE_DAYS):
            print(f"[{fund_code}] 缓存新鲜（末日期{df.index.max().date()}），跳过")
            return df
        print(f"[{fund_code}] 缓存过期（末日期{df.index.max().date()} < 锚点{stale_before.date()}），重拉")

    max_retries = 3
    for retry in range(max_retries):
        try:
            # 获取单位净值
            df_nav = ak.fund_open_fund_info_em(
                symbol=fund_code, indicator="单位净值走势"
            )
            df_nav = df_nav.rename(columns={"净值日期": "date", "单位净值": "nav"})

            # 获取累计净值
            df_acc = ak.fund_open_fund_info_em(
                symbol=fund_code, indicator="累计净值走势"
            )
            df_acc = df_acc.rename(columns={"净值日期": "date", "累计净值": "nav_acc"})

            # 按日期合并两张表
            df_merge = pd.merge(df_nav, df_acc, on="date", how="inner")
            df_merge["date"] = pd.to_datetime(df_merge["date"])

            # 保存原始缓存
            os.makedirs(NAV_DATA_DIR, exist_ok=True)
            df_merge.to_csv(cache_file, index=False, encoding="utf-8-sig")

            df_merge = df_merge.set_index("date")
            return df_merge

        except Exception as e:
            wait_sec = 2 + retry * 2
            print(
                f"[警告] 基金{fund_code} 获取失败，重试 {retry+1}/{max_retries}，等待{wait_sec}s, err:{str(e)[:100]}"
            )
            time.sleep(wait_sec)

    print(f"[失败] 基金 {fund_code} 多次请求失败，跳过")
    return None

BENCHMARK_PATH = os.path.join(RAW_DATA_DIR, "benchmark_hs300.csv")

def load_benchmark_hs300(use_cache: bool = True) -> pd.DataFrame:
    """
    下载沪深300指数日线作为基准（注意：指数没有净值，用收盘价，不能复用load_single_fund）
    输出 data/raw/benchmark_hs300.csv，不写入fund_nav目录（避免被当成基金混进分析）
    :return: DataFrame[date, close]
    """
    if use_cache and os.path.exists(BENCHMARK_PATH):
        return pd.read_csv(BENCHMARK_PATH, parse_dates=["date"])

    max_retries = 3
    for retry in range(max_retries):
        try:
            # 新浪数据源
            df = ak.stock_zh_index_daily(symbol="sh000300")
            if df is None or len(df) == 0:
                raise RuntimeError("接口返回空数据")
            df = df[["date", "close"]].copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            df.to_csv(BENCHMARK_PATH, index=False, encoding="utf-8-sig")
            print(f"沪深300基准下载完成，行数：{len(df)}，保存至 {BENCHMARK_PATH}")
            return df
        except Exception as e:
            wait_sec = 2 + retry * 2
            print(
                f"[警告] 沪深300获取失败，重试 {retry+1}/{max_retries}，等待{wait_sec}s, err:{str(e)[:100]}"
            )
            time.sleep(wait_sec)

    print("[失败] 沪深300基准多次请求失败")
    return None

def batch_download_funds(sleep_sec: float = 1.0, force_refresh: bool = False,
                         pool_file: str = "fund_code_list.csv", backfill: bool = False):
    """
    批量下载净值：读取指定基金池（默认 fund_code_list.csv = 全量股票型+混合型）

    两种模式（2026-09-17 增补 backfill，区分"日常增量"与"长任务回填"）：
      日常增量（默认）：以基准最新日为锚点，缓存末日期落后超过 STALE_TOLERANCE_DAYS 天才重拉——
        容忍基金净值 T+1 披露与周末滞后，避免每天全量重拉。
      回填（backfill=True）：**已有缓存一律跳过，只为缺失缓存拉取**。
        为什么必须分开：锚点随日期前进，多天长任务（全量 universe 补拉 9661 只，需数小时至
        十几小时）若按新鲜度判断，前一天的进度会在 3 天后被判"过期"而重拉，长任务无法收敛。
        回填完成后如要让缓存跟上最新净值，再用日常增量模式跑一次即可。
    :param sleep_sec: 每只请求后休眠秒数，防止IP限流，建议1.0~3；全量拉取建议2.0
    :param force_refresh: True则无视缓存全部重拉（数据源异常回补时用）
    :param pool_file: data/raw 下的池文件名，phase0调试可用 "phase0_fund_pool.csv"
    :param backfill: 回填模式：只拉缺失缓存，已有缓存跳过（长任务断点续传）
    """
    pool_path = os.path.join(RAW_DATA_DIR, pool_file)
    if not os.path.exists(pool_path):
        raise FileNotFoundError(f"找不到基金池文件：{pool_path}，请先运行fund_list_loader.py")

    pool_df = pd.read_csv(pool_path, dtype={"基金代码": str})
    code_list = pool_df["基金代码"].astype(str).tolist()
    print(f"准备批量下载，基金总数：{len(code_list)}，force_refresh={force_refresh}，backfill={backfill}")

    # 先更新基准（一次请求，成本低），其最新日期作为缓存新鲜度锚点
    bench_df = load_benchmark_hs300(use_cache=False)
    stale_before = None
    if backfill:
        # 回填：只拉缺失——已有缓存一律跳过（不判新鲜度，避免长任务进度被锚点前进吞掉）
        missing = [c for c in code_list
                   if not os.path.exists(os.path.join(NAV_DATA_DIR, f"fund_{c}.csv"))]
        print(f"回填模式：已有缓存 {len(code_list) - len(missing)} 只跳过，"
              f"待拉缺失 {len(missing)} 只（断点续传：中断后重跑自动接着拉）")
        code_list = missing
    elif force_refresh:
        stale_before = None  # force模式下全部走强制下载，不需要锚点
    elif bench_df is not None:
        stale_before = bench_df["date"].max()
        print(f"缓存新鲜度锚点：{stale_before.date()}（落后超{STALE_TOLERANCE_DAYS}天自动重拉）")
    else:
        print("⚠️ 基准更新失败，退化为'缓存存在即用'模式，本次不会更新基金数据")

    success = 0
    fail = 0

    for code in code_list:
        df = load_single_fund(code, use_cache=not force_refresh, stale_before=stale_before)
        if df is not None:
            success += 1
        else:
            fail += 1

        time.sleep(sleep_sec)

    if bench_df is None:
        print("⚠️ 基准数据缺失，后续alpha/beta计算将不可用")
    print("\n===== 批量下载结束 =====")
    print(f"成功:{success}  失败:{fail}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="基金净值批量下载（日常增量 / 回填两种模式）")
    ap.add_argument("--pool-file", default="fund_code_list.csv", help="data/raw 下的池文件名")
    ap.add_argument("--sleep-sec", type=float, default=1.0, help="每只请求后休眠秒数（限流防护）")
    ap.add_argument("--force-refresh", action="store_true", help="无视缓存全部重拉")
    ap.add_argument("--backfill", action="store_true",
                    help="回填模式：只为缺失缓存拉取，已有缓存一律跳过（长任务断点续传）")
    args = ap.parse_args()
    batch_download_funds(sleep_sec=args.sleep_sec, force_refresh=args.force_refresh,
                         pool_file=args.pool_file, backfill=args.backfill)
