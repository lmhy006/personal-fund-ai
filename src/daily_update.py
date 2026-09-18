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
    ap.add_argument("--catch-up", action="store_true",
                    help="强制补齐全部落后基金（无视自动补齐阈值；漏跑多日后用，建议后台运行）")
    ap.add_argument("--auto-catchup-max", type=int, default=50,
                    help="缝隙区基金自动补齐的数量上限（默认 50；超过则只提示不自动跑）")
    ap.add_argument("--sleep-sec", type=float, default=1.5,
                    help="catch-up 逐只拉取的限流间隔（秒）")
    args = ap.parse_args()

    t0 = time.time()
    daily, dates = fetch_daily()
    latest, prev = dates[-1], (dates[-2] if len(dates) > 1 else None)
    print(f"当日接口：{len(daily)} 只基金 | 覆盖交易日 {dates} | 请求耗时 {time.time() - t0:.1f}s")

    codes = pd.read_csv(args.pool, dtype={"基金代码": str})["基金代码"].astype(str).tolist()
    stat = {"updated": 0, "uptodate": 0, "gap": 0, "nocache": 0, "nodata": 0}
    gap_list = []
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
            # 缝隙区：落后 ≥2 个交易日
            #   data_loader 的容差是 3 天（落后 ≤3 天它不重拉），而 daily 接口只给最近两天、
            #   缺中间交易日的**官方日增长率**——本脚本不猜（净值比在分红除权日口径不一致），
            #   因此列入 catch-up 队列（逐只全历史重拉，口径最干净）。
            stat["gap"] += 1
            gap_list.append((code, str(last.date()), int((pd.Timestamp(latest) - last).days)))
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
          f"接口无数据 {stat['nodata']} 只 | "
          f"缝隙区（落后≥2交易日，data_loader不拉+daily补不了）{stat['gap']} 只")
    if gap_list:
        print("缝隙区样例（代码, 缓存末日期, 落后自然日）:", gap_list[:5])

    # 缝隙区补齐：落后 ≥2 个交易日的基金既不在 data_loader 的 3 天容差内被重拉，
    # 也超出 daily 接口能力 → 默认自动逐只补齐（数量少时无感）；
    # 数量超阈值则只提示，避免"每日运行"意外变成数小时长任务。
    todo = gap_list
    auto = bool(todo) and (args.catch_up or len(todo) <= args.auto_catchup_max)
    if todo and args.dry_run:
        print(f"（dry-run：本次不执行 catch-up 补齐；缝隙区 {len(todo)} 只待补）")
    elif todo and auto:
        from data_loader import load_single_fund   # 延迟导入，避免无谓依赖
        print(f"\n>>> catch-up：逐只全历史重拉 {len(todo)} 只缝隙区基金"
              f"（预计 {len(todo) * (args.sleep_sec + 1) / 60:.1f} 分钟）...")
        ok = fail = 0
        for i, (code, _last, _days) in enumerate(todo, 1):
            df = load_single_fund(code, use_cache=False)   # 强制重拉，忽略缓存
            ok += df is not None
            fail += df is None
            time.sleep(args.sleep_sec)
            if i % 50 == 0:
                print(f"  进度 {i}/{len(todo)}（成功 {ok} / 失败 {fail}）")
        print(f"catch-up 完成：成功 {ok} / 失败 {fail}")
    elif todo and not auto:
        print(f"\n⚠️ 缝隙区有 {len(todo)} 只落后基金，超过自动补齐阈值 "
              f"{args.auto_catchup_max}——本次未自动执行。")
        print("   请运行：python src/daily_update.py --catch-up"
              "（或等 data_loader 的 3 天容差过期后自然接管）")
    print(f"总耗时 {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()
