# data_health.py：Phase 3.5 生产数据健康检查（PASS / WARN / FAIL 三级）
#
# 用途：在任何**正式评分**之前调用（生产流水线 production_pipeline.py 的强制步骤），
#       保证不拿旧数据/过期因子/未刷新视图"静默评分"。
#
# 报告至少包含（用户 2026-09-19 要求）：
#   benchmark 最新日期 / raw 最新日期分布 / processed(fund_history) 最新日期 /
#   stale 数量 / gap 数量 / catch-up 成败（来自最近一次 pipeline manifest 或日志缺省）/
#   coverage 分布 / 全量基金数 / 可评分主策略数 / 低置信度数 / shadow 因子最新日期 /
#   是否满足正式评分条件 / 最终状态
#
# 判定规则（重要缺口必须 FAIL，不是打印 warning 后继续）：
#   - processed 视图落后 benchmark 超过容差      → FAIL（未刷新全历史清洗，禁止评分）
#   - benchmark 本身过旧（落后现状超过 7 天）      → FAIL
#   - raw 中 stale/gap 基金数超阈值（默认 50）    → FAIL（缺口未补齐，禁止评分）
#   - shadow 因子最新日期 < benchmark 日期        → 因子 stale：主策略可按自身状态决定，
#                                                   但 shadow 必须 FAIL（跳过并记录原因）
#   - 无任何 FAIL 但有 WARN                         → WARN
#   - 全部通过                                      → PASS
import argparse
import glob
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
NAV_DIR = os.path.join(RAW_DIR, "fund_nav")
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "fund_history")
BENCH_PATH = os.path.join(RAW_DIR, "benchmark_hs300.csv")
SCORES_DIR = os.path.join(PROJECT_ROOT, "ml", "scores")
SNAPSHOTS_DIR = os.path.join(PROJECT_ROOT, "ml", "snapshots")
STYLE_INDEX_FILES = ["style_index_sh000905.csv", "style_index_sz399006.csv"]
# 行业因子（申万一级 31 个）**平铺在 data/raw/sw_industry_*.csv**（2026-09-21 用户阻断点：
# 原实现误查 data/raw/sw_industry/ 目录，31 个行业文件从未进入 stale 判定）
SW_GLOB = os.path.join(RAW_DIR, "sw_industry_*.csv")
STALE_GAP_FAIL_N = 50      # raw 中 gap（落后 ≥2 交易日）基金数超过该值 → FAIL（对齐 daily_update 自动补齐阈值）
BENCH_MAX_AGE_DAYS = 7     # benchmark 落后现状超过该值 → FAIL
PROC_LAG_OK_DAYS = 3       # processed 落后 raw/benchmark 的容许天数


def latest_date_csv(path_col, path):
    """读取 CSV 的 date 列最大值；失败返回 None。path_col: 列名。"""
    try:
        df = pd.read_csv(path, parse_dates=[path_col])
        return pd.Timestamp(df[path_col].max())
    except Exception:  # noqa: BLE001
        return None


def raw_nav_latest():
    """扫描 data/raw/fund_nav 的末日期分布（只读文件名级别成本高——抽样即可？）。
    为控制开销：用 pandas 读取每个缓存会太慢（9661+1756 只）。改用轻量读取首末行。"""
    last, n = [], 0
    if not os.path.isdir(NAV_DIR):
        return None, 0
    for fn in os.listdir(NAV_DIR):
        if not (fn.startswith("fund_") and fn.endswith(".csv")):
            continue
        n += 1
        # 只读第一行与最后一行（文件按日期排序，见 data_loader/daily_update 写入约定）
        with open(os.path.join(NAV_DIR, fn), "r", encoding="utf-8-sig",
                  errors="replace") as f:
            header = f.readline()
            lines = f.readlines()
            if not lines:
                continue
            last_line = lines[-1].strip()
            parts = last_line.split(",")
            if parts:
                try:
                    last.append(pd.Timestamp(parts[0]))
                except Exception:  # noqa: BLE001
                    continue
    return (pd.Series(last).max() if last else None), n


def processed_latest():
    """返回 (分布 dict, 基金数)。分布含 median/p25/min/max/n_at_max——**最大日期不能代表整体
    新鲜度**（2026-09-21 用户 P1：一只最新基金可掩盖大面积未刷新）。"""
    if not os.path.isdir(PROC_DIR):
        return None, 0
    dates = []
    n = 0
    for fn in os.listdir(PROC_DIR):
        if not (fn.startswith("fund_") and fn.endswith(".csv")):
            continue
        n += 1
        try:
            df = pd.read_csv(os.path.join(PROC_DIR, fn), usecols=["date"],
                             parse_dates=["date"])
            dates.append(df["date"].max())
        except Exception:  # noqa: BLE001
            continue
    if not dates:
        return None, n
    s = pd.Series(dates)
    return {
        "median": pd.Timestamp(s.median()).date(),
        "p25": pd.Timestamp(s.quantile(0.25)).date(),
        "min": pd.Timestamp(s.min()).date(),
        "max": pd.Timestamp(s.max()).date(),
        "n_funds": n,
        "n_at_max": int((s == s.max()).sum()),
        "n_behind_max_gt1d": int(((s.max() - s) > pd.Timedelta(days=1)).sum()),
    }, n


def shadow_factor_dates():
    """风格/行业因子的最新日期（shadow_score 依赖）。

    **行业因子平铺在 data/raw/sw_industry_*.csv（31 个）**，逐个枚举并以**最旧日期**作为
    行业因子截止日（2026-09-21 用户 P0：原实现误查 sw_industry/ 目录，行业文件从未进入判定）。
    """
    out = {}
    for fn in STYLE_INDEX_FILES:
        p = os.path.join(RAW_DIR, fn)
        if os.path.exists(p):
            out[fn] = latest_date_csv("date", p)
    sw_files = sorted(glob.glob(SW_GLOB))
    if sw_files:
        dts = {os.path.basename(fn): latest_date_csv("date", fn) for fn in sw_files}
        valid = {k: v for k, v in dts.items() if v is not None}
        if valid:
            min_dt, max_dt = min(valid.values()), max(valid.values())
            out["sw_industry"] = {
                "n_files": len(sw_files),
                "min_date": str(min_dt.date()),            # ← 行业因子截止日（最旧为准）
                "max_date": str(max_dt.date()),
                "behind_max": sorted(os.path.basename(k) for k, v in valid.items() if v < max_dt),
            }
    return out


def last_scores_counts():
    """最近一次正式评分快照中的可信度计数（若存在）。"""
    if not os.path.isdir(SCORES_DIR):
        return None
    cands = sorted([f for f in os.listdir(SCORES_DIR) if re_fullmatch(f)],
                   reverse=True)
    for fn in cands:
        p = os.path.join(SCORES_DIR, fn)
        try:
            df = pd.read_csv(p, dtype={"fund_code": str})
            return {"file": fn, "as_of": str(df["as_of"].iloc[0]) if "as_of" in df else None,
                    "total": len(df),
                    "main": int((df["confidence"] == "main").sum()) if "confidence" in df else None,
                    "low": int((df["confidence"] == "low").sum()) if "confidence" in df else None}
        except Exception:  # noqa: BLE001
            continue
    return None


def re_fullmatch(name):
    import re
    return re.fullmatch(r"\d{4}-\d{2}\.csv", name) is not None


def check(report: dict) -> str:
    """根据指标字段汇总 PASS/WARN/FAIL。"""
    fails, warns = [], []
    bench = report["benchmark_latest"]
    if bench is None:
        fails.append("benchmark 缺失")
    else:
        age = (datetime.now() - bench.to_pydatetime()).days
        if age > BENCH_MAX_AGE_DAYS:
            fails.append(f"benchmark 过旧（{bench.date()}，距今天 {age} 天）")
    raw_l = report["raw_latest"]
    proc = report.get("processed_dist")
    if proc is None or raw_l is None:
        fails.append("processed 或 raw 视图缺失")
    else:
        proc_med = pd.Timestamp(proc["median"])
        if proc_med < raw_l - pd.Timedelta(days=PROC_LAG_OK_DAYS):
            fails.append(f"processed 中位日期({proc_med.date()}) 落后 raw({raw_l.date()}) 超过容差"
                         f"——未执行 full 清洗，禁止评分（{proc.get('n_at_max')}/{proc.get('n_funds')}"
                         f" 只到最新日）")
    gap_n = report.get("gap_n") or 0
    if gap_n > STALE_GAP_FAIL_N:
        fails.append(f"raw 缺口过大：gap（落后≥2交易日）{gap_n} > 阈值 {STALE_GAP_FAIL_N}")
    fac = report.get("shadow_factors", {})
    if bench is not None:
        stale_fac = []
        for k, v in fac.items():
            if isinstance(v, pd.Timestamp) and v < bench:
                stale_fac.append(k)
            elif isinstance(v, dict) and v.get("min_date"):
                if pd.Timestamp(v["min_date"]) < bench:
                    stale_fac.append(f"sw_industry(最旧 {v['min_date']})")
        report["shadow_stale"] = stale_fac
        if stale_fac:
            warns.append("shadow 因子 stale（影子评分应跳过）")
    return "FAIL" if fails else ("WARN" if warns else "PASS")


def health_report() -> dict:
    bench_date = latest_date_csv("date", BENCH_PATH)
    raw_date, raw_n = raw_nav_latest()
    proc_date, proc_n = processed_latest()

    # stale / gap（相对 benchmark）——**只对现存池**（清盘基金净值本就停止，不算缺口）
    stale_n = gap_n = None
    alive_codes = None
    idx_path = os.path.join(PROJECT_ROOT, "data", "processed", "fund_history_index.csv")
    if os.path.exists(idx_path):
        idx = pd.read_csv(idx_path, dtype={"fund_code": str})
        alive_codes = set(idx[idx["source"] == "alive"]["fund_code"])
    if bench_date is not None and os.path.isdir(NAV_DIR):
        stale_n = gap_n = 0
        bench_dt_arr = pd.to_datetime(pd.read_csv(BENCH_PATH, parse_dates=["date"])["date"]) \
            .to_numpy("datetime64[ns]")
        for fn in os.listdir(NAV_DIR):
            if not (fn.startswith("fund_") and fn.endswith(".csv")):
                continue
            code = fn.replace("fund_", "").replace(".csv", "")
            if alive_codes is not None and code not in alive_codes:
                continue                      # 清盘基金不计入缺口
            with open(os.path.join(NAV_DIR, fn), "r", encoding="utf-8-sig",
                      errors="replace") as f:
                f.readline()
                lines = f.readlines()
                if not lines:
                    continue
                d0 = lines[-1].split(",")[0]
                try:
                    d = pd.Timestamp(d0).to_datetime64()
                except Exception:  # noqa: BLE001
                    continue
                # 统一为**交易日**口径（2026-09-21 用户：与 daily_update 的"落后于前一交易日"
                # 缝隙定义一致，不再用自然日差）：
                #   落后 ≥2 交易日 → gap（daily_update 缝隙区，自动补齐阈值 50）
                #   落后 1 交易日   → stale（T+1 披露常见，可容忍）
                pos = int(np.searchsorted(bench_dt_arr, d, side="right")) - 1
                lag_td = len(bench_dt_arr) - 1 - pos
                if lag_td >= 2:
                    gap_n += 1
                elif lag_td == 1:
                    stale_n += 1

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark_latest": bench_date,
        "benchmark_latest_date": str(bench_date.date()) if bench_date else None,
        "raw_latest": raw_date, "raw_latest_date": str(raw_date.date()) if raw_date else None,
        "raw_funds": raw_n,
        "processed_dist": proc_date,                 # 分布 dict（median/p25/min/max/n_at_max）
        "processed_latest_date": (str(proc_date["median"]) if proc_date else None),  # 以中位数为代表
        "processed_funds": proc_n,
        "stale_n": stale_n, "gap_n": gap_n, "gap_scope": "alive 现存池，交易日口径（gap=落后≥2交易日）",
        "catchup": "见最近 pipeline manifest（data_health 独立运行不拉取）",
        "shadow_factors": shadow_factor_dates(),
        "last_scores": last_scores_counts(),
    }
    status = check(report)
    report["status"] = status
    report["score_ready"] = status != "FAIL"
    return report


def main():
    ap = argparse.ArgumentParser(description="Phase 3.5 生产数据健康检查")
    ap.add_argument("--json", action="store_true", help="输出 JSON 到 stdout")
    ap.add_argument("--out", default=os.path.join(SNAPSHOTS_DIR, "data_health_latest.json"),
                    help="落盘路径（默认 ml/snapshots/data_health_latest.json）")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    rep = health_report()
    text = json.dumps(rep, ensure_ascii=False, indent=2, default=str)
    print(f"== data_health 状态：{rep['status']} ==")
    if args.json:
        print(text)
    else:
        for k in ("benchmark_latest_date", "raw_latest_date", "raw_funds",
                  "processed_latest_date", "processed_funds", "stale_n", "gap_n",
                  "shadow_factors", "last_scores"):
            if k in rep:
                print(f"  {k:<22} {rep[k]}")
        if rep.get("shadow_stale"):
            print(f"  shadow_stale          {rep['shadow_stale']}")
    if not args.no_save:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"已落盘：{args.out}")
    return 0 if rep["status"] == "PASS" else (1 if rep["status"] == "FAIL" else 2)


if __name__ == "__main__":
    raise SystemExit(main())