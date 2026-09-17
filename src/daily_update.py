# daily_update.py：每日增量更新——1 次请求拿全市场当日净值，按日追加到各基金缓存
#
# 为什么需要（2026-09-17）：data_loader.py 是"逐只整只全历史重拉"（9661 只 ≈ 8 小时），
#   只适合首次回填与长期停更后的补齐，**不能用于每日运行**。
#   东财 fund_open_fund_daily_em 一次请求返回全市场约 2.4 万只基金的**最近两个交易日**
#   净值 + 当日日增长率 → 每日更新 = 1 次请求（约 20 秒）+ 本地按日追加（秒级）。
#
# 分工：
#   首次回填 / 长期停更补齐 → `data_loader.py --backfill`（逐只全历史）
#   每日更新                → 本脚本（daily 接口 + 按日追加）
#
# 边界与口径：只追加**最新一个交易日**。若某基金缓存落后 ≥2 个交易日，接口只给最近
#   两天、且"日增长率"仅最新一天有值——不用"净值比"冒充官方日增长率（分红除权口径
#   可能不同），这类基金只报告，交由 data_loader 逐只补齐。
import argparse
import os
import time

import akshare as ak
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
NAV_DIR = os.path.join(RAW_DIR, "fund_nav")
DEFAULT_POOL = os.path.join(RAW_DIR, "fund_code_list.csv")


def fetch_daily() -> tuple:
    """当日全市场净值（一次请求）。返回 (以基金代码为索引的 DataFrame, 覆盖交易日升序列表)。"""
    df = ak.fund_open_fund_daily_em()
    nav_cols = [c for c in df.columns if str(c).endswith("-单位净值")]
    if not nav_cols:
        raise SystemExit("daily 接口未返回净值列——接口结构可能变更，请检查 akshare 版本")
    dates = sorted(str(c).replace("-单位净值", "") for c in nav_cols)
    return df.set_index("基金代码"), dates


def main():
    ap = argparse.ArgumentParser(description="每日增量更新（全市场当日净值 → 按日追加）")
    ap.add_argument("--pool", default=DEFAULT_POOL, help="基金名单/池文件路径")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写盘")
    args = ap.parse_args()

    t0 = time.time()
    daily, dates = fetch_daily()
    latest, prev = dates[-1], (dates[-2] if len(dates) > 1 else None)
    print(f"当日接口：{len(daily)} 只基金 | 覆盖交易日 {dates} | 请求耗时 {time.time() - t0:.1f}s")

    codes = pd.read_csv(args.pool, dtype={"基金代码": str})["基金代码"].astype(str).tolist()
    stat = {"updated": 0, "uptodate": 0, "stale": 0, "nocache": 0, "nodata": 0}
    stale_list = []
    for code in codes:
        path = os.path.join(NAV_DIR, f"fund_{code}.csv")
        if not os.path.exists(path):
            stat["nocache"] += 1
            continue
        if code not in daily.index:
            stat["nodata"] += 1
            continue
        row = daily.loc[code]
        nav_new = row.get(f"{latest}-单位净值")
        if pd.isna(nav_new):
            stat["nodata"] += 1
            continue
        cache = pd.read_csv(path, parse_dates=["date"])
        if not len(cache):
            stat["nocache"] += 1
            continue
        last = cache["date"].max()
        if last >= pd.Timestamp(latest):
            stat["uptodate"] += 1
            continue
        if prev is not None and last < pd.Timestamp(prev):
            stat["stale"] += 1
            stale_list.append((code, str(last.date())))
            continue
        if args.dry_run:
            stat["updated"] += 1
            continue
        new = pd.DataFrame([{
            "date": pd.Timestamp(latest),
            "nav": nav_new,
            "日增长率": row.get("日增长率"),
            "nav_acc": row.get(f"{latest}-累计净值"),
        }])
        out = (pd.concat([cache, new], ignore_index=True)
               .drop_duplicates("date", keep="last").sort_values("date"))
        tmp = path + ".tmp"
        out.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, path)  # 原子替换，避免半写文件
        stat["updated"] += 1

    print(f"\n===== 每日增量更新{'（dry-run，未写盘）' if args.dry_run else ''} =====")
    print(f"名单 {len(codes)} 只：追加最新交易日({latest}) {stat['updated']} 只 | "
          f"已最新 {stat['uptodate']} 只")
    print(f"名单内未缓存（需 --backfill 回填）{stat['nocache']} 只 | "
          f"接口无数据 {stat['nodata']} 只 | 落后≥2日需逐只补齐 {stat['stale']} 只")
    if stale_list:
        print("落后样例（代码, 缓存末日期）:", stale_list[:5])
    print(f"总耗时 {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()
