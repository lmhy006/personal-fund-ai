# gap_exceptions.py：月末 gap 例外**机器化备案**（2026-10-02 审计第 5 节）
#
# 为什么需要：月末门禁原先只要求 stale_n==0 与 processed 中位数对齐，**gap 只由总量阈值把关**
#   （gap_n>50 才 FAIL）。于是"未备案的 gap=49"能直接通过，人工放行记录也只存在于文档里，
#   无法被机器核销。现在改为：凡是被判为 gap 的现存池基金，**逐只**必须有备案
#   （原因 / 证据 / 批准人 / 失效条件 / 备案时的末日），否则月末正式运行中止。
#
# 备案文件：ml/calendar/gap_exceptions.json（版本化、随运行证据一起留痕，允许人工编辑）
#   {
#     "version": 1,
#     "entries": {
#       "2026-09": {
#         "signal_date": "2026-09-30", "recorded_at": "...", "approved_by": "...",
#         "funds": {"002224": {"last_date": "...", "lag_trading_days": 3,
#                              "reason": "...", "evidence": "...", "expires_when": "..."}}
#       }
#     }
#   }
import json
import os
from datetime import datetime

from backtest_strategy import PROJECT_ROOT

CALENDAR_DIR = os.path.join(PROJECT_ROOT, "ml", "calendar")
EXCEPTIONS_PATH = os.path.join(CALENDAR_DIR, "gap_exceptions.json")


def load_exceptions(path: str | None = None) -> dict:
    p = path or EXCEPTIONS_PATH
    if not os.path.exists(p):
        return {"version": 1, "entries": {}}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("entries", {})
    return data


def coverage(signal_date: str, gap_funds: list, path: str | None = None) -> dict:
    """核对给定 gap 明细是否全部已备案。

    :param signal_date: 评分日（YYYY-MM-DD），按月份匹配 entries。
    :param gap_funds: [{fund_code, last_date, lag_trading_days}, ...]（来自 data_health.gap_funds）
    :return: {"month","signal_date","n_gap","covered":[...],"missing":[...],
              "approved_by","recorded_at","stale_entries":[...]}
             `missing`：未备案，或备案的末日与该基金**当前**末日不一致（数据变了 → 需重新核对）。
    """
    month = str(signal_date)[:7]
    exp = load_exceptions(path)
    entry = (exp.get("entries") or {}).get(month) or {}
    funds = entry.get("funds") or {}
    covered, missing, drifted = [], [], []
    for g in gap_funds or []:
        code = str(g.get("fund_code"))
        rec = funds.get(code)
        if not rec:
            missing.append({"fund_code": code, "last_date": g.get("last_date"),
                            "lag_trading_days": g.get("lag_trading_days"),
                            "why": "未备案"})
            continue
        if str(rec.get("last_date")) != str(g.get("last_date")):
            drifted.append({"fund_code": code, "current_last_date": g.get("last_date"),
                            "recorded_last_date": rec.get("last_date"),
                            "why": "备案末日与当前末日不一致（数据已变化，需重新核对）"})
            continue
        covered.append({"fund_code": code, "last_date": g.get("last_date"),
                        "lag_trading_days": g.get("lag_trading_days"),
                        "reason": rec.get("reason"), "approved_by": entry.get("approved_by"),
                        "expires_when": rec.get("expires_when")})
    return {"month": month, "signal_date": entry.get("signal_date", signal_date),
            "n_gap": len(gap_funds or []), "covered": covered,
            "missing": missing + drifted, "approved_by": entry.get("approved_by"),
            "recorded_at": entry.get("recorded_at")}


def record(signal_date: str, funds: list, approved_by: str, path: str | None = None) -> dict:
    """登记/更新某月的 gap 例外（人工核销后调用；保留既有条目）。

    :param funds: [{fund_code, last_date, lag_trading_days, reason, evidence?, expires_when?}]
    """
    p = path or EXCEPTIONS_PATH
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    exp = load_exceptions(p)
    month = str(signal_date)[:7]
    entry = exp["entries"].setdefault(month, {"signal_date": signal_date, "funds": {}})
    entry["signal_date"] = signal_date
    entry["recorded_at"] = datetime.now().isoformat(timespec="seconds")
    entry["approved_by"] = approved_by
    for f in funds:
        entry["funds"][str(f["fund_code"])] = {
            "last_date": f["last_date"],
            "lag_trading_days": f.get("lag_trading_days"),
            "reason": f.get("reason"),
            "evidence": f.get("evidence"),
            "expires_when": f.get("expires_when",
                                  "该基金恢复披露后即失效；每次月末运行前重新核对"),
        }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(exp, f, ensure_ascii=False, indent=2)
    return {"month": month, "n_funds": len(entry["funds"]), "path": p,
            "approved_by": approved_by, "recorded_at": entry["recorded_at"]}
