# repair_daily_contamination.py：修复 2026-09-23 prev 增长率污染（P0-1，2026-09-23 用户审查）
#
# 背景：daily_update 补 prev（2026-09-22）时接口无日期专属日增长率列，曾回退写入属于 latest
#       （2026-09-23）的单列"日增长率"→ 9-22 行收益率错误（实测 8,668 只与 9-23 完全相同）。
#       clean_nav 直接把该字段作为正式 daily_ret → 污染已进入 processed/评分/影子。
#
# 修复（**纯重拉**，幂等）：load_single_fund(use_cache=False) 全历史覆盖缓存——
#   官方历史含正确 9-22 增长率，污染行自然消失；无需手工剔除（原 9-22 错误行随覆盖移除）。
#   另外把受污染原文件**原样**留档一份到 _repair_backup（审计证据，可选 --keep-backup），
#   完成后清单 → data/raw/daily_contamination_repair.csv。
# 后续：重新跑 production_pipeline（full 清洗 + 评分），并抽样核对 9-22 官方增长率。
import argparse
import csv
import os
import sys
import time

# —— Windows 输出编码兜底（2026-09-24）：管道/重定向 stdout 默认 GBK，print ⚠️ 等符号会崩
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from daily_update import NAV_DIR  # noqa: E402
from data_loader import load_single_fund  # noqa: E402

REPAIR_CSV = os.path.join(os.path.dirname(NAV_DIR), "daily_contamination_repair.csv")
DEFAULT_DROP_DATE = "2026-09-22"      # 被污染的日期（9-22 行增长率来自 latest 单列）


def affected_funds(drop_date: str) -> list:
    out = []
    for fn in sorted(os.listdir(NAV_DIR)):
        if not (fn.startswith("fund_") and fn.endswith(".csv")):
            continue
        p = os.path.join(NAV_DIR, fn)
        with open(p, "r", encoding="utf-8-sig", errors="replace") as f:
            f.readline()
            lines = f.readlines()
        if any(ln.split(",")[0] == drop_date for ln in lines):
            out.append((fn.replace("fund_", "").replace(".csv", ""), p))
    return out


def main():
    ap = argparse.ArgumentParser(description="修复 prev 日增长率污染（全历史重拉覆盖）")
    ap.add_argument("--drop-date", default=DEFAULT_DROP_DATE, help="被污染的日期（默认 2026-09-22）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 只（调试用；0=全部）")
    ap.add_argument("--sleep-sec", type=float, default=0.3, help="重拉限流间隔（秒）")
    ap.add_argument("--keep-backup", action="store_true",
                    help="把受污染原文件原样留档到 _repair_backup（审计证据）")
    ap.add_argument("--list-only", action="store_true", help="只输出受影响清单，不重拉")
    args = ap.parse_args()

    affected = affected_funds(args.drop_date)
    print(f"受影响基金（原文件含 {args.drop_date} 行）：{len(affected)} 只")
    if args.list_only:
        with open(REPAIR_CSV, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fund_code"])
            w.writerows((c,) for c, _ in affected)
        print(f"清单已输出：{REPAIR_CSV}")
        return

    bk_dir = os.path.join(NAV_DIR, "_repair_backup")
    if args.keep_backup:
        os.makedirs(bk_dir, exist_ok=True)
    todo = affected if not args.limit else affected[:args.limit]
    rows, ok, fail = [], 0, 0
    print(f"\n>>> 全历史重拉 {len(todo)} 只（预计 {len(todo) * (args.sleep_sec + 1) / 60:.1f} 分钟）…")
    t0 = time.time()
    for i, (code, p) in enumerate(todo, 1):
        try:
            if args.keep_backup:
                if not os.path.exists(os.path.join(bk_dir, os.path.basename(p))):
                    import shutil
                    shutil.copy2(p, os.path.join(bk_dir, os.path.basename(p)))
            df = load_single_fund(code, use_cache=False)      # 全历史覆盖（含正确 9-22）
            ok_ = df is not None
            err = ""
        except Exception as e:  # noqa: BLE001
            ok_, err = False, str(e)[:120]
        rows.append({"fund_code": code, "relaunch_ok": ok_, "err": err})
        ok += ok_; fail += (not ok_)
        time.sleep(args.sleep_sec)
        if i % 200 == 0:
            el = (time.time() - t0) / 60
            print(f"  {i}/{len(todo)}（成功 {ok} / 失败 {fail}，已用 {el:.1f} 分钟）")
    with open(REPAIR_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fund_code", "relaunch_ok", "err"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n完成：重拉成功 {ok} / 失败 {fail}；清单 → {REPAIR_CSV}")
    print("下一步：重新跑 production_pipeline（full 清洗 + 评分），并抽样核对 9-22 官方增长率。")
    if fail:
        print(f"⚠️ {fail} 只失败，可再次运行本脚本重试（幂等）。")


if __name__ == "__main__":
    main()