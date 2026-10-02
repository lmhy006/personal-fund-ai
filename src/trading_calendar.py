# trading_calendar.py：版本化交易日历（首次 paper 确认前的 P1-2 校验依赖）
#
# 为什么需要（2026-10-02 审计 P1-2）：确认入口原先只检查日期格式/晚于信号日/不在未来，
#   于是 10-1（国庆休市）与 10-10（周六）都会被接受；基准 CSV 又止于 9-30，无法覆盖
#   10-8 之后的执行日。因此引入一份**显式、版本化、可追溯来源**的交易日历：
#     - 基准日历（benchmark_hs300.csv）是历史交易日的权威来源（有行情即有交易）；
#     - 长假与临时休市按交易所公告登记为 holidays；个别需要显式说明的交易日登记为 overrides；
#     - 基准日历之后的**覆盖范围内工作日默认开市**（宽松于逐日登记，避免节后每天都要登记），
#       但日历的 holiday 登记仍优先；超出 coverage_end 或早于基准起点 → 直接拒绝，不猜。
# 日历文件改动即为版本变更（calendar_version 递增），确认事件里记录所依据的版本与判定来源。
import json
import os
from datetime import date as _date
from datetime import datetime

import pandas as pd

from backtest_strategy import PROJECT_ROOT
from panel_builder import BENCH_PATH

CALENDAR_DIR = os.path.join(PROJECT_ROOT, "ml", "calendar")
CALENDAR_PATH = os.path.join(CALENDAR_DIR, "trading_calendar.json")


class UnknownTradingDayError(ValueError):
    """日期不在版本化日历覆盖范围内、且未登记 → 必须人工登记后才能用于执行确认。"""


def load_calendar(path: str = CALENDAR_PATH) -> dict:
    if not os.path.exists(path):
        raise UnknownTradingDayError(
            f"交易日历文件缺失：{path}——执行确认必须依据版本化日历，请先登记")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _bench_days(bench_path: str = BENCH_PATH) -> pd.DatetimeIndex:
    bc = pd.read_csv(bench_path, parse_dates=["date"])["date"]
    return pd.DatetimeIndex(bc).normalize()


def describe_trading_day(d, calendar_path: str = CALENDAR_PATH,
                         bench_path: str = BENCH_PATH) -> dict:
    """判定 d 是否为交易日，并给出**可追溯的判定来源**（供执行事件留痕）。

    :return: {"date","is_trading_day","source","calendar_version"}
    :raises UnknownTradingDayError: 无法判定（覆盖范围外且未登记）
    """
    cal = load_calendar(calendar_path)
    ts = pd.Timestamp(d).normalize()
    key = ts.strftime("%Y-%m-%d")
    ver = cal.get("calendar_version")
    overrides = cal.get("overrides") or {}
    holidays = set(cal.get("holidays") or [])

    if key in overrides:
        ov = overrides[key]
        return {"date": key, "is_trading_day": bool(ov.get("trading")),
                "source": f"override：{ov.get('source', '未注明')}（登记于 {ov.get('recorded_at', '?')}）",
                "calendar_version": ver}
    if key in holidays:
        return {"date": key, "is_trading_day": False,
                "source": f"holiday 登记：{cal.get('holiday_source', '交易所休市安排')}",
                "calendar_version": ver}

    bc = _bench_days(bench_path)
    if ts.weekday() >= 5:
        return {"date": key, "is_trading_day": False,
                "source": "周末（周一至周五之外）", "calendar_version": ver}
    if len(bc) and bc.min() <= ts <= bc.max():
        hit = bool((bc == ts).any())
        return {"date": key, "is_trading_day": hit,
                "source": ("基准日历命中（有沪深300行情）" if hit else "基准日历范围内缺失（非交易日）"),
                "calendar_version": ver}

    # 超出基准日历（未来日期）：**覆盖范围内的工作日按交易所安排默认开市**——长假与临时休市
    # 一律登记在 holidays/overrides 里；超出 coverage_end 则拒绝判定（不猜）。
    coverage_end = cal.get("coverage_end")
    if coverage_end and ts <= pd.Timestamp(coverage_end).normalize():
        return {"date": key, "is_trading_day": True,
                "source": f"覆盖范围内工作日（默认开市；节假日/休市须在日历登记，版本 {ver}）",
                "calendar_version": ver}
    raise UnknownTradingDayError(
        f"{key} 超出基准日历覆盖（{bc.max().date() if len(bc) else '?'}）"
        f"{f'与日历 coverage_end={coverage_end}' if coverage_end else ''}"
        f"——请按交易所公告在 {calendar_path} 登记后再确认（当前日历版本 {ver}）")


def is_trading_day(d, calendar_path: str = CALENDAR_PATH, bench_path: str = BENCH_PATH) -> bool:
    return describe_trading_day(d, calendar_path, bench_path)["is_trading_day"]


def next_trading_day(d, calendar_path: str = CALENDAR_PATH,
                     bench_path: str = BENCH_PATH, max_scan_days: int = 30):
    """d 之后的下一个交易日（依据版本化日历）；无法在扫描窗口内确定则返回 None。"""
    ts = pd.Timestamp(d).normalize()
    for i in range(1, max_scan_days + 1):
        cand = ts + pd.Timedelta(days=i)
        try:
            if is_trading_day(cand, calendar_path, bench_path):
                return cand
        except UnknownTradingDayError:
            return None            # 超出日历覆盖 → 不猜（调用方保持 planned/pending）
    return None
