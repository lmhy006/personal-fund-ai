# delisted_funds.py：已清盘/终止公募基金名单抓取（证监会 EID 公告检索）
#
# 为什么走 EID（2026-09-18 实测结论）：
#   - AKShare 全部列表类接口只返回**存续**基金（无退市标的）；
#   - 巨潮 cninfo `column=fund` 只覆盖**场内**基金（实测公告代码前缀 15/16/18），
#     场外清盘基金（00xxxx/01xxxx 等）在其公告库里查不到；
#   - **证监会基金电子披露平台（eid.csrc.gov.cn）是公募基金法定披露渠道**，按公告类型
#     检索覆盖场外 ✓（实测 2023-01 的 FC030030 返回 000049 / 005247 / 012677 / 005429 等场外代码）。
#
# 服务端限制与对策：跨度 >1 个月时必须提供 6 位基金代码（"仅支持根据6位基金代码查询
#   1个月以上数据！"）→ 因此**按月扫描**（每月窗口 ≤31 天）。
#
# 公告类型（取自 EID 字典接口 upload_inforeport_query.do 的 reportTypeList，共 127 类）：
#   FC030010 基金合同终止公告｜FC030030 基金清算公告｜FB060010 清算报告｜
#   FC050050 基金合并运作公告｜FC260040 基金终止上市交易公告
#
# 输出（原始层，只增不改；data/ 被 .gitignore 忽略，见 README 数据层约定）：
#   data/raw/delisted_announcements.csv —— 逐条公告（year/type/code/name/title/date/uploadInfoId）
#   data/raw/delisted_funds.csv        —— 去重名单（code/name/est_delist_date/first_ann/last_ann/types/n_ann）
#
# 口径提醒：est_delist_date 取**最早公告日**（FC030010 合同终止公告或 FC030030 清算公告），
#   是"终止事件被披露"的日期近似，不等于清盘完成日；如需精确终止日可用
#   `--verify-eastmoney` 抓东财详情页的"终止日期：YYYY-MM-DD"做交叉校验（反爬较慢，抽样用）。
import argparse
import os
import re
import time

# —— Windows 输出编码兜底（2026-09-24）：管道/重定向 stdout 默认 GBK，print ⚠️ 等符号会崩
import sys
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import pandas as pd
import requests

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
ANN_PATH = os.path.join(RAW_DIR, "delisted_announcements.csv")
FUND_PATH = os.path.join(RAW_DIR, "delisted_funds.csv")

EID_QUERY_URL = "http://eid.csrc.gov.cn/fund/disclose/advanced_search_report.do"
EID_HOME = "http://eid.csrc.gov.cn/fund/disclose/advanced_search.html"
UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
      "X-Requested-With": "XMLHttpRequest",
      "Referer": EID_HOME}
# 清算/终止相关公告类型（key → 中文名）
ANNOUNCE_TYPES = {
    "FC030010": "基金合同终止公告",
    "FC030030": "基金清算公告",
    "FB060010": "清算报告",
    "FC050050": "基金合并运作公告",
}
PAGE_LEN = 30


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(UA)
    s.get(EID_HOME, timeout=30)          # 建立会话（cookie）
    return s


def _ao(report_type: str, start: str, end: str, i_start: int, length: int) -> str:
    """DataTables aoData（字段顺序与 report_query.js 的 retrieveData 一致）。"""
    cols = ["fundCode", "fundId", "reportName", "organName", "reportDesp", "reportSendDate"]
    ao = [{"name": "sEcho", "value": 1}, {"name": "iColumns", "value": len(cols)},
          {"name": "sColumns", "value": ""},
          {"name": "iDisplayStart", "value": i_start},
          {"name": "iDisplayLength", "value": length}]
    ao += [{"name": f"mDataProp_{i}", "value": c} for i, c in enumerate(cols)]
    ao += [{"name": "sSearch", "value": ""}, {"name": "bRegex", "value": False},
           {"name": "iSortCol_0", "value": 5}, {"name": "sSortDir_0", "value": "desc"},
           {"name": "iSortingCols", "value": 1},
           {"name": "fundType", "value": ""}, {"name": "reportType", "value": report_type},
           {"name": "reportYear", "value": ""},
           {"name": "fundCompanyShortName", "value": ""}, {"name": "fundCode", "value": ""},
           {"name": "fundShortName", "value": ""},
           {"name": "startUploadDate", "value": start},
           {"name": "endUploadDate", "value": end}]
    import json
    return json.dumps(ao)


def fetch_page(sess, report_type: str, start: str, end: str, i_start: int, length: int,
               retries: int = 3) -> dict:
    last = None
    for k in range(retries):
        try:
            r = sess.get(EID_QUERY_URL, params={"aoData": _ao(report_type, start, end,
                                                             i_start, length)}, timeout=40)
            js = r.json()
            if js.get("success") is False:
                msg = js.get("message", "")
                raise RuntimeError(f"EID 拒绝：{msg}")
            return js
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 + 2 * k)
    raise RuntimeError(f"EID 查询失败（{report_type} {start}~{end}）：{last}")


def fetch_range(sess, report_type: str, start: str, end: str, sleep_sec: float = 0.8) -> list:
    """按月窗口拉取某类型公告的**全部页**。"""
    rows, i = [], 0
    while True:
        js = fetch_page(sess, report_type, start, end, i, PAGE_LEN)
        batch = js.get("aaData") or []
        rows.extend(batch)
        total = int(js.get("iTotalRecords") or 0)
        i += PAGE_LEN
        if not batch or i >= total:
            break
        time.sleep(sleep_sec)
    return rows


def month_windows(start_year: int, end_year: int, end_month: str = None) -> list:
    """(start, end) 月度窗口；服务端要求跨度 ≤1 个月。"""
    out = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            first = f"{y}-{m:02d}-01"
            last_day = pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)
            last = last_day.strftime("%Y-%m-%d")
            if end_month and last > end_month:
                last = end_month
            if first > last:
                continue
            out.append((first, last))
    return out


def scan(start_year: int, end_year: int, types: list, sleep_sec: float = 0.8,
         limit_months: int = 0, verbose: bool = True) -> pd.DataFrame:
    sess = make_session()
    rows, n_req_windows = [], 0
    for rt in types:
        for (s, e) in month_windows(start_year, end_year):
            if limit_months and n_req_windows >= limit_months:
                break
            n_req_windows += 1
            try:
                batch = fetch_range(sess, rt, s, e, sleep_sec)
            except Exception as ex:  # noqa: BLE001
                print(f"⚠️ {rt} {s}~{e} 失败：{str(ex)[:120]}")
                continue
            for b in batch:
                rows.append({
                    "year": int(s[:4]), "month": s[:7], "report_type": rt,
                    "type_name": ANNOUNCE_TYPES.get(rt, rt),
                    "fund_code": str(b.get("fundCode") or "").strip(),
                    "fund_short_name": b.get("fundShortName"),
                    "title": b.get("reportName"),
                    "report_send_date": b.get("reportSendDate"),
                    "organ_name": b.get("organName"),
                    "upload_info_id": b.get("uploadInfoId"),
                })
            if verbose and n_req_windows % 24 == 0:
                print(f"  进度 {rt} {s} → 累计公告 {len(rows)} 条")
            time.sleep(sleep_sec)
        if limit_months and n_req_windows >= limit_months:
            break
    df = pd.DataFrame(rows)
    if len(df):
        df["report_send_date"] = pd.to_datetime(df["report_send_date"], errors="coerce")
    return df


def build_list(ann: pd.DataFrame) -> pd.DataFrame:
    """公告逐条 → 去重基金名单（含最早/最晚公告日与出现过的类型）。"""
    if not len(ann):
        return pd.DataFrame()
    df = ann[ann["fund_code"].str.len() == 6].copy()
    # 只保留 6 位数字代码（场外 0/1/2/3 开头 + 场内 5/15/16 开头）
    df = df[df["fund_code"].str.fullmatch(r"\d{6}")]
    g = df.groupby("fund_code", as_index=False).agg(
        fund_short_name=("fund_short_name", "first"),
        first_ann=("report_send_date", "min"),
        last_ann=("report_send_date", "max"),
        n_ann=("report_send_date", "size"),
        types=("type_name", lambda x: "/".join(sorted(set(x)))),
        organ_name=("organ_name", "first"),
    )
    g["est_delist_date"] = g["first_ann"].dt.strftime("%Y-%m-%d")
    g["is_onsite"] = g["fund_code"].str.startswith(("15", "16", "18", "50", "51", "52",
                                                    "56", "58"))
    return g.sort_values("fund_code")


def year_coverage(funds: pd.DataFrame, year: int = None) -> pd.DataFrame:
    """按年份统计名单覆盖（发现只数、其中场内/场外、最早公告年份分布）。"""
    if not len(funds):
        return pd.DataFrame()
    f = funds.copy()
    f["first_year"] = pd.to_datetime(f["first_ann"]).dt.year
    g = f.groupby("first_year").agg(
        n_funds=("fund_code", "size"),
        n_onsite=("is_onsite", "sum"),
    )
    g["n_offsite"] = g["n_funds"] - g["n_onsite"]
    return g


HISTORY_MAIN_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "fund_history")
INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "fund_history_index.csv")


def build_history_index(force: bool = False) -> pd.DataFrame:
    """扫描现存池 + 清盘池清洗目录 → (fund_code, source, first_date, last_date, n_rows)。

    用途：① 覆盖率报告的分母用**全量历史池**（比开发池 panel 准确）；
          ② 观察历史池规模随年份的变化。结果缓存到 data/processed/fund_history_index.csv。
    """
    if os.path.exists(INDEX_PATH) and not force:
        return pd.read_csv(INDEX_PATH, parse_dates=["first_date", "last_date"])
    rows = []
    for src, d in (("alive", HISTORY_MAIN_DIR), ("delisted", DELISTED_HISTORY_DIR)):
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not (fn.startswith("fund_") and fn.endswith(".csv")):
                continue
            code = fn.replace("fund_", "").replace(".csv", "")
            try:
                s = pd.read_csv(os.path.join(d, fn), usecols=["date"], parse_dates=["date"])["date"]
                rows.append({"fund_code": code, "source": src, "first_date": s.min(),
                             "last_date": s.max(), "n_rows": len(s)})
            except Exception:  # noqa: BLE001
                continue
    df = pd.DataFrame(rows).sort_values("fund_code").reset_index(drop=True)
    df.to_csv(INDEX_PATH, index=False, encoding="utf-8-sig")
    print(f"历史池索引已落盘：{INDEX_PATH}（{len(df)} 只；"
          f"现存 {int((df['source'] == 'alive').sum())} / 清盘 {int((df['source'] == 'delisted').sum())}）")
    return df


def coverage_report(panel_path: str = None) -> pd.DataFrame:
    """按年份报告清盘名单覆盖率（**诚实口径**：只反映 EID 公告库里能查到的事件）。

    列：n_funds（该年最早出现终止/清算公告的基金数）、n_onsite/n_offsite、
        n_in_history（已清洗进 fund_history_delisted 的只数）、
        pool_alive（**全量历史池**中该年末仍存续的基金数，来自 fund_history_index）、
        delist_share_est（清盘数 /（当年存续 + 清盘数），**下限估计**）。
    """
    if not os.path.exists(FUND_PATH):
        raise SystemExit(f"找不到名单 {FUND_PATH}")
    panel_path = panel_path or os.path.join(PROJECT_ROOT, "ml", "panel.parquet")
    f = pd.read_csv(FUND_PATH, dtype={"fund_code": str})
    f["first_year"] = pd.to_datetime(f["first_ann"]).dt.year
    have = set()
    if os.path.isdir(DELISTED_HISTORY_DIR):
        have = {fn.replace("fund_", "").replace(".csv", "")
                for fn in os.listdir(DELISTED_HISTORY_DIR)
                if fn.startswith("fund_") and fn.endswith(".csv")}
    f["in_history"] = f["fund_code"].isin(have)
    g = f.groupby("first_year").agg(
        n_funds=("fund_code", "size"),
        n_onsite=("is_onsite", "sum"),
        n_in_history=("in_history", "sum"),
    )
    g["n_offsite"] = g["n_funds"] - g["n_onsite"]
    # 分母：全量历史池（现存 + 清盘）中该年末仍存续的只数
    try:
        idx = build_history_index()
        data_end = idx["last_date"].max()
        alive, partial = {}, {}
        for y in g.index:
            year_end = pd.Timestamp(f"{int(y)}-12-31")
            cutoff = min(year_end, data_end)          # 当年未完则用数据末日
            alive[y] = int(((idx["first_date"] <= cutoff) & (idx["last_date"] >= cutoff)).sum())
            partial[y] = bool(year_end > data_end)
        g["pool_alive"] = pd.Series(alive)
        g["delist_share_est"] = (g["n_funds"] / (g["n_funds"] + g["pool_alive"])).round(4)
        g["partial_year"] = pd.Series(partial)        # 当年未完（占比未年化，勿直接比较）
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 历史池索引不可用（{str(e)[:80]}），分母退化为 panel 参照")
        if os.path.exists(panel_path):
            panel = pd.read_parquet(panel_path, columns=["t_date", "fund_code"])
            a = panel.groupby(panel["t_date"].dt.year)["fund_code"].nunique()
            g["pool_alive"] = a.reindex(g.index)
            g["delist_share_est"] = (g["n_funds"] / (g["n_funds"] + g["pool_alive"])).round(4)
    return g


NAV_DIR = os.path.join(RAW_DIR, "fund_nav")
DELISTED_HISTORY_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "fund_history_delisted")


def _fetch_one_light(code: str) -> bool:
    """轻量拉取单只清盘基金净值：只调「单位净值走势」（含**官方日增长率**）。

    为什么不用 data_loader.load_single_fund：
      - 它对每只调两次接口（单位净值 + 累计净值），而清盘基金里相当一部分在东财返回
        404 HTML（akshare 解析报 SyntaxError），会触发 3 次重试（等待 2+4+6 秒）→ 实测
        约 15 秒/只，1798 只要 7 小时以上；
      - 本函数**单接口 + 解析失败快速返回**（不重试），实测成功 0.3s / 失败 0.4s。
    nav_acc 用单位净值占位：清洗与收益都只用「日增长率」（wealth 由 daily_ret 现算），
    nav_acc 仅参与"非 0/非 NaN"行校验，因此占位不影响任何计算（README 指标口径）。
    """
    import akshare as ak
    try:
        df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
    except Exception:  # noqa: BLE001  （404 HTML / 解析失败：直接判失败，不重试）
        return False
    if df is None or not len(df) or "日增长率" not in df.columns:
        return False
    try:
        out = df.rename(columns={"净值日期": "date", "单位净值": "nav"})
        out["date"] = pd.to_datetime(out["date"])
        out["nav_acc"] = out["nav"]
        out = out[["date", "nav", "日增长率", "nav_acc"]].sort_values("date")
        path = os.path.join(NAV_DIR, f"fund_{code}.csv")
        tmp = path + ".tmp"
        out.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, path)          # 原子替换，避免半写文件
        return True
    except Exception:  # noqa: BLE001
        return False


def fetch_navs(codes: list, sleep_sec: float = 0.15, workers: int = 1) -> tuple:
    """拉取清盘基金净值（已有缓存跳过；失败清单单独落盘）。

    ⚠️ 并发不可用（2026-09-18 实测）：akshare 的 `fund_open_fund_info_em` 内部用
    py_mini_racer（V8）解析东财 JS，**多线程会在同一进程重复初始化 V8 → 进程崩溃**
    （`partition_address_space.cc Check failed: !IsConfigurablePoolInitialized()`）。
    因此默认单线程；提速靠「单接口 + 解析失败不重试」（实测 15s/只 → 约 0.5s/只）。
    """
    todo = [c for c in codes if not os.path.exists(os.path.join(NAV_DIR, f"fund_{c}.csv"))]
    skip = len(codes) - len(todo)
    print(f"待拉 {len(todo)} 只 / 已有缓存 {skip} 只（单线程，每只限流 {sleep_sec}s）")
    os.makedirs(NAV_DIR, exist_ok=True)
    ok, fails = 0, []
    t0 = time.time()
    for i, c in enumerate(todo, 1):
        if _fetch_one_light(c):
            ok += 1
        else:
            fails.append(c)
        time.sleep(sleep_sec)
        if i % 200 == 0 or i == len(todo):
            rate = i / max(time.time() - t0, 1e-6)
            print(f"  进度 {i}/{len(todo)}（成功 {ok} / 失败 {len(fails)}，"
                  f"{rate:.1f} 只/秒）", flush=True)
    return ok, len(fails), skip, sorted(fails)


def clean_history(codes: list) -> pd.DataFrame:
    """把清盘基金净值清洗到**独立目录**（不触碰现有 fund_history/，无需 swap/backup）。

    复用 clean_nav.clean_fund 的行级清洗（官方日增长率转小数、去重、剔除无效行），
    只对名单内代码处理；样本过短（<30 行）的剔除并记录。
    """
    from clean_nav import clean_fund
    os.makedirs(DELISTED_HISTORY_DIR, exist_ok=True)
    want = {f"fund_{c}.csv" for c in codes}
    files = sorted(f for f in os.listdir(NAV_DIR)
                   if f.startswith("fund_") and f.endswith(".csv") and f in want)
    rows = []
    for fname in files:
        code = fname.replace("fund_", "").replace(".csv", "")
        try:
            raw = pd.read_csv(os.path.join(NAV_DIR, fname), parse_dates=["date"])
            df, stat = clean_fund(raw)
            if len(df) < 30:
                rows.append({"fund_code": code, "status": "reject_too_short", **stat})
                continue
            df.to_csv(os.path.join(DELISTED_HISTORY_DIR, fname), index=False,
                      encoding="utf-8-sig")
            rows.append({"fund_code": code, "status": "ok", **stat})
        except Exception as e:  # noqa: BLE001
            rows.append({"fund_code": code, "status": f"error_{type(e).__name__}"})
    rep = pd.DataFrame(rows)
    rep.to_csv(os.path.join(DELISTED_HISTORY_DIR, "clean_delisted_report.csv"),
               index=False, encoding="utf-8-sig")
    n_ok = int((rep["status"] == "ok").sum()) if len(rep) else 0
    print(f"清洗完成：{n_ok}/{len(files)} 只写入 {DELISTED_HISTORY_DIR}")
    if len(rep):
        print(rep["status"].value_counts().to_string())
    return rep if len(rep) else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description="已清盘/终止基金名单抓取（证监会 EID 公告检索）")
    ap.add_argument("--start-year", type=int, default=2005)
    ap.add_argument("--end-year", type=int, default=2026)
    ap.add_argument("--types", default=",".join(ANNOUNCE_TYPES))
    ap.add_argument("--sleep-sec", type=float, default=0.8)
    ap.add_argument("--limit-months", type=int, default=0, help="测试用：只扫前 N 个月窗口")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--fetch-nav", action="store_true",
                    help="对名单中尚无缓存的基金拉取全历史净值（写 data/raw/fund_nav/）")
    ap.add_argument("--nav-sleep-sec", type=float, default=0.15, help="每只拉取后的限流间隔")
    ap.add_argument("--nav-workers", type=int, default=1,
                    help="并发线程数；**必须为 1**（akshare 内部 V8 不支持同进程多线程，>1 会崩）")
    ap.add_argument("--clean-history", action="store_true",
                    help="把清盘基金清洗到 data/processed/fund_history_delisted/（独立目录，不触碰 fund_history/）")
    ap.add_argument("--coverage-report", action="store_true",
                    help="按年份报告名单覆盖率（含已清洗入库数与现存池参照）后退出")
    args = ap.parse_args()

    if args.coverage_report:
        rep = coverage_report()
        print("清盘名单按年份覆盖率（**只反映 EID 公告库里能查到的事件**；"
              "2005-2013 公告库无数据 → 早年清盘基金必然缺失）：")
        print(rep.to_string())
        print(f"\n合计：{int(rep['n_funds'].sum())} 只（场外 {int(rep['n_offsite'].sum())}），"
              f"已清洗入库 {int(rep['n_in_history'].sum())} 只")
        return

    # 模式 B/C：不重新扫描，直接用已落盘名单
    if args.fetch_nav or args.clean_history:
        if not os.path.exists(FUND_PATH):
            raise SystemExit(f"找不到名单 {FUND_PATH}，请先运行扫描（不加 --fetch-nav/--clean-history）")
        funds = pd.read_csv(FUND_PATH, dtype={"fund_code": str})
        codes = funds["fund_code"].tolist()
        print(f"名单 {len(codes)} 只（场内 {int(funds['is_onsite'].sum())} / "
              f"场外 {int((~funds['is_onsite']).sum())}）")
        if args.fetch_nav:
            print(f"\n拉取净值（{args.nav_workers} 线程，限流 {args.nav_sleep_sec}s/任务；已有缓存跳过）…")
            ok, fail, skip, fails = fetch_navs(codes, args.nav_sleep_sec, args.nav_workers)
            print(f"净值拉取完成：新增 {ok} / 失败 {fail} / 已有 {skip}")
            if fails:
                pd.Series(fails, name="fund_code").to_csv(
                    os.path.join(RAW_DIR, "delisted_nav_failures.csv"), index=False,
                    encoding="utf-8-sig")
                print(f"失败清单已落盘（{len(fails)} 只）")
        if args.clean_history:
            print("\n清洗到独立目录…")
            rep = clean_history(codes)
            print(rep.to_string())
        return

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    print(f"扫描 {args.start_year}-{args.end_year}，类型 {types}，月度窗口，"
          f"限流 {args.sleep_sec}s/请求")
    t0 = time.time()
    ann = scan(args.start_year, args.end_year, types, args.sleep_sec, args.limit_months)
    print(f"扫描完成：公告 {len(ann)} 条，耗时 {time.time() - t0:.0f}s")
    if not len(ann):
        print("⚠️ 未取到任何公告，请检查网络/接口")
        return
    funds = build_list(ann)
    print(f"去重名单：{len(funds)} 只基金"
          f"（场内 {int(funds['is_onsite'].sum())} / 场外 {int((~funds['is_onsite']).sum())}）")
    print("\n按最早公告年份的覆盖（**只反映公告库里能查到的事件**，"
          "2005-2013 EID 无公告数据 → 早年清盘基金必然缺失）：")
    print(year_coverage(funds).to_string())

    if not args.no_save:
        os.makedirs(RAW_DIR, exist_ok=True)
        ann.to_csv(ANN_PATH, index=False, encoding="utf-8-sig")
        funds.to_csv(FUND_PATH, index=False, encoding="utf-8-sig")
        print(f"\n已落盘：{ANN_PATH}\n         {FUND_PATH}")


if __name__ == "__main__":
    main()
