# fund_attrs.py：第三组特征的属性数据（规模历史 → AUM / 资金流；费率）
#
# 数据源（2026-09-19 实测）：
#   ① 规模历史：`https://fundf10.eastmoney.com/FundArchivesDatas.aspx?type=gmbd&code={code}`
#      返回 `var gmbd_apidata={ content:"<table>…</table>", … }`，表列＝
#      日期（报告期）/期间申购(亿份)/期间赎回(亿份)/期末总份额(亿份)/期末净资产(亿元)/净资产变动率
#      → **AUM = 期末净资产**；**资金流 = 期间申购 − 期间赎回**（份额口径，与用户给的
#      「本期规模 − 上期规模×(1+收益)」互为校验）
#   ② 费率：`ak.fund_fee_em(symbol=code, indicator="运作费用")` → 管理费率/托管费率/销售服务费率
#      （**当前值**；费率历史上很少变动，作为历史近似需标注局限——与"基金类型取当前标签"同类）
#
# ⚠️ 时间口径（用户 2026-09-19 要求）：规模按**报告期**披露，不能把季度末当成当时已知。
#   东财页面只给报告期、不给公告日，因此本模块用**法定披露时限**做保守滞后（宁可晚用）：
#     一季报(03-31)/三季报(09-30)/半年报(06-30)/年报(12-31) 按法定时限推后并取整到月末
#   该滞后为**规则化近似**（真实公告日通常更早；要精确可用 EID 逐只取定期报告公告日），
#   字段 `available_date` 即"当时可得日"。
#
# 缓存（data/raw 只增不改）：
#   data/raw/fund_scale/fund_{code}.csv    每只基金全部报告期的规模变动
#   data/raw/fund_fee.csv                  费率汇总（逐只追加，含抓取时间）
import argparse
import os
import re
import time

import numpy as np
import pandas as pd
import requests

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
SCALE_DIR = os.path.join(RAW_DIR, "fund_scale")
FEE_PATH = os.path.join(RAW_DIR, "fund_fee.csv")
INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "fund_history_index.csv")

GMBD_URL = "https://fundf10.eastmoney.com/FundArchivesDatas.aspx"
UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
      "Referer": "http://fundf10.eastmoney.com/gmbd_000001.html"}
# 法定披露时限（《公开募集证券投资基金信息披露管理办法》：季报=季度结束后 15 个工作日、
# 中期报告=上半年结束后 2 个月、年度报告=年度结束后 3 个月；见证监会现行办法
# https://www.csrc.gov.cn/csrc/c106256/c1653985/content.shtml）。
# 实现：按法定时限推后，再**向上取整到该月月末**——本项目只在月末截面使用属性数据，
# 这样天然对齐截面，并比法定时限再晚 0~10 天（不存在"用了尚未公布数据"的风险）。
# 措辞注意（2026-09-19 用户审查）：这不是"严格保守"的任意缓冲，而是"法定期限 + 取整到月末"。
LAG_MONTHS = {3: 1, 6: 2, 9: 1, 12: 3}


# 非季末月份：清盘/终止基金在东财表里会给出清盘期间的**高频规模记录**（如 505888 的
# 2019-08-14、180002 的 2020-04-03）——这些是清盘公告期的临时披露，按 +7 天滞后；
# 构造特征时只取季末报告期（month ∈ {3,6,9,12}）
LAG_DAYS_OTHER = 7


def available_date(report_date: pd.Timestamp) -> pd.Timestamp:
    """报告期 → 该数据"当时可得"的最晚日（法定时限，向上取整到月末）。"""
    m = report_date.month
    if m in LAG_MONTHS:
        return report_date + pd.DateOffset(months=LAG_MONTHS[m]) + pd.offsets.MonthEnd(0)
    return report_date + pd.Timedelta(days=LAG_DAYS_OTHER)


def is_quarter_end(report_date: pd.Timestamp) -> bool:
    """是否为季末报告期（AUM/资金流特征只用这些记录）。"""
    return report_date.month in (3, 6, 9, 12)


def _to_float(x):
    if x is None:
        return float("nan")
    s = str(x).strip().replace(",", "")
    if s in ("", "-", "--", "---"):
        return float("nan")
    m = re.match(r"^-?\d+(\.\d+)?$", s)
    return float(s) if m else float("nan")


def parse_gmbd(text: str) -> pd.DataFrame:
    """解析 FundArchivesDatas.aspx 返回的 gmbd_apidata.content → 规模变动表。"""
    m = re.search(r'content:"(.*?)"\s*,\s*arryear', text, re.S)
    content = m.group(1) if m else text
    rows = []
    for tr in re.findall(r"<tr>(.*?)</tr>", content, re.S):
        tds = [re.sub(r"<[^>]+>", "", t).strip()
               for t in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(tds) >= 5 and re.match(r"^\d{4}-\d{2}-\d{2}$", tds[0]):
            rows.append({"report_date": tds[0], "purchase": _to_float(tds[1]),
                         "redemption": _to_float(tds[2]), "total_shares": _to_float(tds[3]),
                         "net_assets": _to_float(tds[4]),
                         "asset_change": tds[5] if len(tds) > 5 else None})
    df = pd.DataFrame(rows)
    if len(df):
        df["report_date"] = pd.to_datetime(df["report_date"])
        df["available_date"] = df["report_date"].map(available_date)
        # 资金流（份额口径）：申购 − 赎回
        df["net_flow_shares"] = df["purchase"] - df["redemption"]
        df = df.sort_values("report_date").reset_index(drop=True)
    return df


def fetch_scale(code: str, use_cache: bool = True, retries: int = 2) -> bool:
    """拉取单只基金的规模历史并落盘；返回是否成功。"""
    path = os.path.join(SCALE_DIR, f"fund_{code}.csv")
    if use_cache and os.path.exists(path):
        return True
    for k in range(retries):
        try:
            r = requests.get(GMBD_URL, params={"type": "gmbd", "code": code},
                             headers=UA, timeout=30)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            r.encoding = "utf-8"
            df = parse_gmbd(r.text)
            if not len(df):
                return False
            os.makedirs(SCALE_DIR, exist_ok=True)
            df.to_csv(path, index=False, encoding="utf-8-sig")
            return True
        except Exception:  # noqa: BLE001
            time.sleep(1.0 + k)
    return False


def fetch_fee(code: str) -> dict:
    """拉取当前运作费率（管理/托管/销售服务）。返回 dict（可能为空）。"""
    import akshare as ak
    try:
        df = ak.fund_fee_em(symbol=code, indicator="运作费用")
    except Exception:  # noqa: BLE001
        return {}
    if df is None or not len(df):
        return {}
    row = df.iloc[0]
    out = {}
    vals = list(row.values)
    for i in range(0, len(vals) - 1, 2):
        k = str(vals[i]).strip()
        v = str(vals[i + 1]).strip()
        m = re.search(r"([\d.]+)\s*%", v)
        if k in ("管理费率", "托管费率", "销售服务费率"):
            out[k] = float(m.group(1)) if m else float("nan")
    return out


def load_pool() -> list:
    """需要拉取属性的基金：清洗后研究池（现存 + 清盘）来自 fund_history_index.csv。"""
    if os.path.exists(INDEX_PATH):
        return sorted(pd.read_csv(INDEX_PATH, dtype={"fund_code": str})["fund_code"])
    codes = pd.read_csv(os.path.join(RAW_DIR, "fund_code_list.csv"),
                        dtype={"基金代码": str})["基金代码"].tolist()
    if os.path.exists(os.path.join(RAW_DIR, "delisted_funds.csv")):
        codes += pd.read_csv(os.path.join(RAW_DIR, "delisted_funds.csv"),
                             dtype={"fund_code": str})["fund_code"].tolist()
    return sorted(set(codes))


def build_attrs_monthly(out_path: str = None) -> pd.DataFrame:
    """把规模/费率对齐到**每月末截面**（只用 `available_date ≤ t` 的最新季末报告期）。

    产出长表：fund_code / t_date / aum（亿元）/ log_aum / flow_share_ratio / fee_* /
              report_date / available_date / n_reports
    - AUM = 期末净资产（亿元）；log_aum = ln(aum)
    - 资金流（**份额口径**，页面原生字段，避免用收益推算引入误差）：
        flow_share_ratio = (期间申购 − 期间赎回) / 期末总份额
    - 费率：当前值（管理/托管/销售服务），历史近似（局限见预登记文档）
    - 只取季末报告期（month ∈ {3,6,9,12}）且 net_assets 有效的记录
    """
    out_path = out_path or os.path.join(PROJECT_ROOT, "ml", "attrs", "fund_attrs_monthly.parquet")
    panel_path = os.path.join(PROJECT_ROOT, "ml", "panel_v3.parquet")
    if not os.path.exists(panel_path):
        panel_path = os.path.join(PROJECT_ROOT, "ml", "panel.parquet")
    panel = pd.read_parquet(panel_path, columns=["t_date"])
    month_ts = np.array(sorted(pd.to_datetime(panel["t_date"].unique())))
    t_ns = month_ts.astype("datetime64[ns]").astype("int64")

    fee_map = {}
    if os.path.exists(FEE_PATH):
        fd = pd.read_csv(FEE_PATH, dtype={"fund_code": str})
        fee_map = fd.set_index("fund_code").to_dict("index")

    codes = sorted({f.replace("fund_", "").replace(".csv", "")
                    for f in os.listdir(SCALE_DIR)}) if os.path.isdir(SCALE_DIR) else []
    print(f"对齐 {len(codes)} 只基金的规模历史到 {len(month_ts)} 个月末截面…")
    rows = []
    for i, code in enumerate(codes, 1):
        try:
            df = pd.read_csv(os.path.join(SCALE_DIR, f"fund_{code}.csv"), parse_dates=["report_date", "available_date"])
        except Exception:  # noqa: BLE001
            continue
        q = df[df["report_date"].dt.month.isin([3, 6, 9, 12]) & df["net_assets"].notna()]
        q = q.sort_values("available_date")
        if not len(q):
            continue
        av = q["available_date"].to_numpy("datetime64[ns]").astype("int64")
        idx = np.searchsorted(av, t_ns, side="right") - 1     # 每个 t 对应的最新已可得报告期
        fee = fee_map.get(code, {})
        for j, m in enumerate(idx):
            if m < 0:
                continue
            r = q.iloc[m]
            aum = float(r["net_assets"])
            ts = float(r["total_shares"]) if pd.notna(r["total_shares"]) else float("nan")
            nf = float(r["net_flow_shares"]) if pd.notna(r["net_flow_shares"]) else float("nan")
            rows.append({
                "fund_code": code, "t_date": month_ts[j],
                "aum": aum, "log_aum": float(np.log(aum)) if aum > 0 else np.nan,
                "flow_share_ratio": (nf / ts) if (ts and ts > 0 and nf == nf) else np.nan,
                "fee_mgmt": fee.get("管理费率", np.nan),
                "fee_cust": fee.get("托管费率", np.nan),
                "fee_total": (float(fee.get("管理费率", 0) or 0)
                              + float(fee.get("托管费率", 0) or 0)) if fee else np.nan,
                "report_date": r["report_date"], "available_date": r["available_date"],
                "n_reports": len(q),
            })
        if i % 500 == 0:
            print(f"  进度 {i}/{len(codes)}，累计 {len(rows)} 行", flush=True)
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"属性特征已落盘：{out_path}（{len(out)} 行，{out['fund_code'].nunique()} 只基金）")
    print(f"  覆盖率：log_aum {out['log_aum'].notna().mean():.1%} | "
          f"资金流 {out['flow_share_ratio'].notna().mean():.1%} | "
          f"费率 {out['fee_total'].notna().mean():.1%}")
    return out


def main():
    ap = argparse.ArgumentParser(description="第三组特征属性数据：规模历史 + 费率")
    ap.add_argument("--fetch", choices=["scale", "fee", "both", "none"], default="both")
    ap.add_argument("--build-monthly", action="store_true", help="把规模/费率对齐到月末截面并落盘")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 只（测试用）")
    ap.add_argument("--codes", default=None, help="指定代码，逗号分隔（测试用）")
    ap.add_argument("--sleep", type=float, default=0.35, help="每只请求后的限流间隔")
    ap.add_argument("--fee-workers", type=int, default=4,
                    help="费率阶段的并发线程数（实测 fund_fee_em 线程安全：4 线程结果与单线程逐只一致、"
                         "约 5 只/秒；与净值拉取不同——后者内部 V8 不可多线程）")
    args = ap.parse_args()

    if args.build_monthly:
        build_attrs_monthly()
        return

    codes = ([c.strip() for c in args.codes.split(",")] if args.codes else load_pool())
    if args.limit:
        codes = codes[: args.limit]
    print(f"待处理 {len(codes)} 只 | 模式 {args.fetch} | 限流 {args.sleep}s")

    if args.fetch in ("scale", "both"):
        t0 = time.time()
        ok = fail = skip = 0
        fails = []
        for i, c in enumerate(codes, 1):
            if os.path.exists(os.path.join(SCALE_DIR, f"fund_{c}.csv")):
                skip += 1
                continue
            if fetch_scale(c):
                ok += 1
            else:
                fail += 1
                fails.append(c)
            time.sleep(args.sleep)
            if i % 200 == 0 or i == len(codes):
                rate = i / max(time.time() - t0, 1e-6)
                print(f"  [规模] {i}/{len(codes)} 新增 {ok} / 失败 {fail} / 已有 {skip}"
                      f"（{rate:.1f} 只/秒）", flush=True)
        if fails:
            pd.Series(fails, name="fund_code").to_csv(
                os.path.join(RAW_DIR, "fund_scale_failures.csv"), index=False,
                encoding="utf-8-sig")
        print(f"[规模] 完成：新增 {ok} / 失败 {fail} / 已有 {skip}")

    if args.fetch in ("fee", "both"):
        t0 = time.time()
        done = set()
        if os.path.exists(FEE_PATH):
            done = set(pd.read_csv(FEE_PATH, dtype={"fund_code": str})["fund_code"])
        todo = [c for c in codes if c not in done]
        workers = max(1, args.fee_workers)
        print(f"[费率] 待拉 {len(todo)} / 已有 {len(done)}｜{workers} 线程")
        rows, fail = [], 0

        def _one(c):
            d = fetch_fee(c)
            time.sleep(args.sleep)
            return c, d

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, (c, d) in enumerate(ex.map(_one, todo), 1):
                if d:
                    rows.append({"fund_code": c, "管理费率": d.get("管理费率"),
                                 "托管费率": d.get("托管费率"),
                                 "销售服务费率": d.get("销售服务费率")})
                else:
                    fail += 1
                if i % 200 == 0 or i == len(todo):
                    rate = i / max(time.time() - t0, 1e-6)
                    print(f"  [费率] {i}/{len(todo)} 失败 {fail}（{rate:.1f} 只/秒）", flush=True)
        if rows:
            new = pd.DataFrame(rows)
            old = pd.read_csv(FEE_PATH, dtype={"fund_code": str}) if os.path.exists(FEE_PATH) else None
            out = pd.concat([old, new], ignore_index=True) if old is not None else new
            out = out.drop_duplicates("fund_code", keep="last")
            out.to_csv(FEE_PATH, index=False, encoding="utf-8-sig")
        print(f"[费率] 完成：新增 {len(rows)} / 失败 {fail} → {FEE_PATH}")


if __name__ == "__main__":
    main()
