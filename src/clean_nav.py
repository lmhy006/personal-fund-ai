import os
import re
import shutil
import pandas as pd
import numpy as np

# —— Windows 输出编码兜底（2026-09-24）：管道/重定向 stdout 默认 GBK，print ❌ 等符号会崩
import sys
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
RAW_FUND_NAV_DIR = os.path.join(RAW_DATA_DIR, "fund_nav")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "fund_processed")
TMP_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "fund_processed_tmp")
BACKUP_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "fund_processed_backup")
CLEAN_REPORT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "fund_processed", "clean_report.csv")
# 全历史模式（ML数据源）输出路径
HISTORY_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "fund_history")
HISTORY_TMP_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "fund_history_tmp")
HISTORY_BACKUP_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "fund_history_backup")
HISTORY_REPORT_PATH = os.path.join(HISTORY_DIR, "clean_history_report.csv")
# 基金名称元数据：全量列表（任何池子都是其子集），不绑定本轮池文件
FUND_META_PATH = os.path.join(RAW_DATA_DIR, "fund_code_list.csv")


def load_fund_name_map() -> dict:
    """基金代码 -> 简称 映射，来源为全量元数据列表；缺失文件返回空映射并警告"""
    if not os.path.exists(FUND_META_PATH):
        print(f"[警告] 基金元数据缺失：{FUND_META_PATH}，名称映射将不完整")
        return {}
    df = pd.read_csv(FUND_META_PATH, dtype={"基金代码": str})
    return dict(zip(df["基金代码"], df["基金简称"].astype(str)))

# 统一分析窗口：近 WINDOW_YEARS 年，锚点=全部基金最新净值日期（全局统一，避免各取"自己最后三年"造成窗口错位）
WINDOW_YEARS = 3
# 窗口边缘容差：起点后/终点前容差天数，覆盖净值披露日不同步的情况
EDGE_TOLERANCE_DAYS = 30
# 可疑跳变阈值：单日涨跌幅绝对值
JUMP_THRESHOLD = 0.20
# 披露覆盖率门槛：窗口内有效净值观测数/同期基准交易日数，低于阈值视为低频披露（周频/月频）剔除
COVERAGE_THRESHOLD = 0.95
# 份额后缀正则：基金简称尾部单个份额字母（A/C/D/E/H/I/M），用于同基金不同份额聚类
SHARE_SUFFIX_RE = re.compile(r"[ACDEHIM]$")
# 基准数据路径（coverage分母：同期基准交易日数）
BENCHMARK_PATH = os.path.join(RAW_DATA_DIR, "benchmark_hs300.csv")

# 创建输出目录
os.makedirs(PROCESSED_DIR, exist_ok=True)


def clean_fund(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    单只基金清洗逻辑
    :param df: 原始df，列 date, nav, 日增长率, nav_acc
    :return: (df_clean, stat)
        df_clean: 列 date, nav, nav_acc, daily_ret，新增 suspicious_jump 标记列
            daily_ret: 官方"日增长率"转小数，真实日收益（已含分红除权调整）
                      注意：nav_acc不是复权净值，分红日对其pct_change会被稀释，不能当收益用
        stat: 统计字典：原始行数、去重删除行数、无效行删除数、剩余行数、可疑跳变数量
    """
    stat = {
        "rows_original": len(df),
        "rows_removed_dup": 0,
        "rows_removed_nan_zero": 0,
        "rows_after_clean": 0,
        "suspicious_jump_cnt": 0,
    }

    # 1. 保留四列；官方日增长率转小数（接口返回百分数，如0.0815表示0.0815%）
    df = df[["date", "nav", "nav_acc", "日增长率"]].copy()
    df = df.rename(columns={"日增长率": "daily_ret"})
    df["daily_ret"] = pd.to_numeric(df["daily_ret"], errors="coerce") / 100

    # 2. 日期转datetime & 按日期排序
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 3. 去重：同一日期保留最后一条
    dup_mask = df.duplicated(subset=["date"], keep="last")
    rows_dup = int(dup_mask.sum())
    stat["rows_removed_dup"] = rows_dup
    df = df[~dup_mask].reset_index(drop=True)

    # 4. 删除无效行：nav/nav_acc为NaN或0；daily_ret为NaN（未公布）
    #    注意daily_ret=0保留：新发建仓期净值真实不动
    bad_row_mask = (
        df["nav"].isna() | (df["nav"] == 0)
        | df["nav_acc"].isna() | (df["nav_acc"] == 0)
        | df["daily_ret"].isna()
    )
    rows_bad = int(bad_row_mask.sum())
    stat["rows_removed_nan_zero"] = rows_bad
    df = df[~bad_row_mask].reset_index(drop=True)

    # 5. 基于官方日收益率标记可疑跳变，只标记，不删除（官方口径下不再有分红假跳变）
    df["suspicious_jump"] = df["daily_ret"].abs() > JUMP_THRESHOLD
    stat["suspicious_jump_cnt"] = int(df["suspicious_jump"].sum())

    # 6. 零收益计数（QA报警指标，不作删除标准：低波动基金天然有真0收益日）
    stat["zero_ret_cnt"] = int((df["daily_ret"] == 0).sum())

    stat["rows_after_clean"] = len(df)
    return df, stat


def calc_window_anchor(file_list: list) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    扫描全部raw文件的最新净值日期，确定全局统一窗口 [window_start, window_end]
    锚点必须全局统一，各基金不能各自取"自己的最后N年"，否则窗口错位失去可比性
    """
    window_end = None
    for fname in file_list:
        df = pd.read_csv(os.path.join(RAW_FUND_NAV_DIR, fname), usecols=["date"], parse_dates=["date"])
        dmax = df["date"].max()
        if window_end is None or dmax > window_end:
            window_end = dmax
    window_start = window_end - pd.DateOffset(years=WINDOW_YEARS)
    return window_start, window_end


def share_cluster_key(name: str) -> str:
    """基金简称去掉尾部份额字母，得到同基金不同份额的聚类key"""
    return SHARE_SUFFIX_RE.sub("", str(name).strip())


def dedup_shares(passed: list, report_list: list, name_map: dict) -> list:
    """
    份额去重（批量层）：同一基金不同份额只保留一个代表份额
    代表规则：簇内保留窗口内跨度最长者，并列时优先A类（简称尾部A），再按代码升序
    必须在行级清洗和窗口校验之后执行：先各自过质量关，再选代表，避免误杀整簇
    :param passed: [(fund_code, df_win, stat, span), ...] 已通过校验的基金
    :param report_list: 清洗报告列表，非代表份额在此追加 reject_duplicate_share 记录
    :param name_map: 基金代码->简称 映射（由clean_all从全量元数据构建）
    :return: 代表份额列表 [(fund_code, df_win, stat, span), ...]
    """
    # 缺失名称的基金各自独立成簇（code本身做key，不会误合并）
    clusters = {}
    for fund_code, df_win, stat, span in passed:
        name = name_map.get(fund_code, "")
        key = share_cluster_key(name) if name else f"__code_{fund_code}"
        clusters.setdefault(key, []).append((fund_code, df_win, stat, span, name))

    kept = []
    for members in clusters.values():
        members_sorted = sorted(
            members,
            key=lambda m: (-m[3], 0 if m[4].endswith("A") else 1, m[0])
        )
        keep = members_sorted[0]
        kept.append(keep[:4])
        for fund_code, df_win, stat, span, name in members_sorted[1:]:
            report_list.append({
                "fund_code": fund_code,
                "status": "reject_duplicate_share",
                "reason": f"与代表份额{keep[0]}({keep[4]})为同一基金",
                **stat,
                "clean_date_span_days": span
            })
            print(f"[{fund_code}] 与 {keep[0]}({keep[4]}) 同基金不同份额，保留代表份额")
    return kept


def clean_all(pool_file: str = None, window: str = "current"):
    """
    批量清洗基金净值
    :param pool_file: data/raw下的池文件名（如 fund_dev_pool.csv）；None=清洗raw目录下全部基金
    :param window: "current"=三年严格共同窗口（Phase1分析视图，fund_processed/）；
                   "full"=完整历史无窗口（Phase2面板数据源，fund_history/）
    流程：
    1. 【幂等】tmp目录就绪
    2. 读取raw/fund_nav下面所有fund_*.csv（若指定pool_file则只清洗池内基金）
    3. 调用clean_fund做行层面清洗，捕获清洗阶段异常
    4. 【批量层】统一窗口校验：窗口起点覆盖不足 / 窗口尾部缺数据 的基金整只剔除
    5. 截断到统一窗口（近WINDOW_YEARS年），保证跨基金可比
    6. 【批量层】份额去重：同基金不同份额只保留代表份额（历史最长，并列优先A类）
    7. 通过的基金写入临时目录，报告也写进临时目录；
       全部完成后用backup让位法原子替换：正式目录先rename让位，tmp上位，失败则回滚
       （任何时刻磁盘上都保留一份完整批次；成功后删除backup）
    8. 输出clean_report.csv（随tmp一起交换，报告永远与processed同一批次）

    window="full"（ML数据源模式）：
    - 保留完整历史，不做起止校验、coverage硬门槛、共同窗口截断（eligibility下沉到panel层时变判定）
    - coverage_ratio按基金自身起止范围相对基准交易日数记录（只记不剔）
    - 输出 data/processed/fund_history/ + clean_history_report.csv，供PanelBuilder使用
    """
    if pool_file is not None:
        pool_path = os.path.join(RAW_DATA_DIR, pool_file)
        if not os.path.exists(pool_path):
            raise FileNotFoundError(f"找不到池文件：{pool_path}")
        pool_codes = set(pd.read_csv(pool_path, dtype={"基金代码": str})["基金代码"].astype(str))
        print(f"清洗范围限定为池文件 {pool_file}（{len(pool_codes)} 只）")
    else:
        pool_codes = None

    # 按模式选择输出路径：current=Phase1三年分析视图；full=ML全历史数据源
    if window == "full":
        out_dir = HISTORY_DIR
        tmp_dir = HISTORY_TMP_DIR
        backup_dir = HISTORY_BACKUP_DIR
        report_path = HISTORY_REPORT_PATH
    elif window == "current":
        out_dir = PROCESSED_DIR
        tmp_dir = TMP_DIR
        backup_dir = BACKUP_DIR
        report_path = CLEAN_REPORT_PATH
    else:
        raise ValueError(f"未知window模式：{window}（可选 current / full）")

    # 幂等：清理上轮swap失败可能残留的临时目录，本轮先写tmp
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"临时目录就绪：{tmp_dir}\n")

    report_list = []
    name_map = load_fund_name_map()  # 名称来自全量元数据，与本轮池文件无关
    file_list = [f for f in os.listdir(RAW_FUND_NAV_DIR) if f.startswith("fund_") and f.endswith(".csv")]
    if pool_codes is not None:
        file_list = [f for f in file_list
                     if f.replace("fund_", "").replace(".csv", "") in pool_codes]
    print(f"待清洗基金总数：{len(file_list)}")

    # 统一分析窗口锚点（全局一致，保证跨基金可比；full模式无窗口概念，跳过）
    if window == "current":
        window_start, window_end = calc_window_anchor(file_list)
        print(f"统一分析窗口：{window_start.date()} ~ {window_end.date()}（近{WINDOW_YEARS}年，边缘容差{EDGE_TOLERANCE_DAYS}天）\n")

    passed = []  # 暂存通过校验的基金，去重后统一写盘

    for fname in file_list:
        fund_code = fname.replace("fund_", "").replace(".csv", "")
        raw_path = os.path.join(RAW_FUND_NAV_DIR, fname)

        try:
            df_raw = pd.read_csv(raw_path, parse_dates=["date"])
        except Exception as e:
            report_list.append({
                "fund_code": fund_code,
                "status": "error_read",
                "reason": f"文件读取失败:{str(e)}",
                "rows_original": 0,
                "rows_removed_dup": 0,
                "rows_removed_nan_zero": 0,
                "rows_after_clean": 0,
                "suspicious_jump_cnt": 0,
                "clean_date_span_days": 0
            })
            print(f"[{fund_code}] ❌读取失败 {str(e)[:80]}")
            continue

        # 读取成功，但清洗过程可能报错（如缺失nav_acc列 KeyError）
        try:
            df_clean, stat = clean_fund(df_raw)
        except Exception as e:
            report_list.append({
                "fund_code": fund_code,
                "status": "error_clean",
                "reason": f"清洗阶段异常:{str(e)}",
                "rows_original": len(df_raw),
                "rows_removed_dup": 0,
                "rows_removed_nan_zero": 0,
                "rows_after_clean": 0,
                "suspicious_jump_cnt": 0,
                "clean_date_span_days": 0
            })
            print(f"[{fund_code}] ❌清洗异常 {str(e)[:80]}")
            continue

        # 清洗后如果数据为空
        if len(df_clean) == 0:
            report_list.append({
                "fund_code": fund_code,
                "status": "reject_empty",
                "reason": "清洗之后无有效数据",
                **stat,
                "clean_date_span_days": 0
            })
            print(f"[{fund_code}] [WARN] 清洗后为空，整只剔除")
            continue

        # 批量层校验1/2（仅current模式）：窗口起止校验；full模式保留完整历史，eligibility下沉到panel层
        if window == "current":
            first_date = df_clean["date"].min()
            if first_date > window_start + pd.Timedelta(days=EDGE_TOLERANCE_DAYS):
                report_list.append({
                    "fund_code": fund_code,
                    "status": "reject_short_history",
                    "reason": f"窗口覆盖不足：首个净值日{first_date.date()} 晚于窗口起点{window_start.date()}+{EDGE_TOLERANCE_DAYS}天",
                    **stat,
                    "clean_date_span_days": 0
                })
                print(f"[{fund_code}] [WARN] 覆盖不足{WINDOW_YEARS}年窗口（首日{first_date.date()}），整只剔除")
                continue

            last_date = df_clean["date"].max()
            if last_date < window_end - pd.Timedelta(days=EDGE_TOLERANCE_DAYS):
                report_list.append({
                    "fund_code": fund_code,
                    "status": "reject_stale_data",
                    "reason": f"窗口尾部缺数据：最后净值日{last_date.date()} 早于窗口终点{window_end.date()}-{EDGE_TOLERANCE_DAYS}天",
                    **stat,
                    "clean_date_span_days": 0
                })
                print(f"[{fund_code}] [WARN] 窗口尾部缺数据（最后{last_date.date()}），整只剔除")
                continue

            # 截断到统一窗口后暂存
            df_out = df_clean[(df_clean["date"] >= window_start) & (df_clean["date"] <= window_end)].reset_index(drop=True)
        else:
            # full模式：完整历史，不截断
            df_out = df_clean
        clean_span = (df_out["date"].max() - df_out["date"].min()).days
        passed.append((fund_code, df_out, stat, clean_span))

    # 批量层校验3：披露覆盖率——起止合规但中间稀疏的低频披露基金（周频/月频/长期停披露）必须挡掉
    # 否则其"日收益"实际是跨多日区间收益，波动/夏普等年化指标全部失真
    if window == "current":
        # 分母：同期基准交易日数（基准文件缺失时回退全池披露日并集）；先剔低频再用幸存者定义共同窗口
        bench_days = None
        if os.path.exists(BENCHMARK_PATH):
            bench = pd.read_csv(BENCHMARK_PATH, parse_dates=["date"])
            bench_days = int(bench[(bench["date"] >= window_start) & (bench["date"] <= window_end)]["date"].nunique())
        if bench_days is not None and bench_days > 2:
            cov_total = bench_days
        elif passed:
            cov_total = int(pd.concat([df["date"] for _, df, _, _ in passed]).drop_duplicates().shape[0])
        else:
            cov_total = 1
        filtered = []
        for fund_code, df_win, stat, clean_span in passed:
            cov_ratio = len(df_win) / cov_total
            stat["coverage_ratio"] = cov_ratio  # QA指标：覆盖率随报告留存
            # zero_ret_cnt 统一为窗口内口径（clean_fund里是全历史计数，避免两QA指标口径不一致）
            stat["zero_ret_cnt"] = int((df_win["daily_ret"] == 0).sum())
            if cov_ratio < COVERAGE_THRESHOLD:
                report_list.append({
                    "fund_code": fund_code,
                    "status": "reject_low_coverage",
                    "reason": f"披露覆盖率过低：{cov_ratio:.1%} < {COVERAGE_THRESHOLD:.0%}（窗口内{len(df_win)}条/基准{cov_total}个交易日），疑似周频/月频披露或长期停披露",
                    **stat,
                    "clean_date_span_days": clean_span
                })
                print(f"[{fund_code}] [WARN] 披露覆盖率 {cov_ratio:.1%} 过低（{len(df_win)}条/{cov_total}日），整只剔除")
                continue
            filtered.append((fund_code, df_win, stat, clean_span))
        passed = filtered
    else:
        # full模式：coverage按基金自身起止范围相对基准交易日数记录，只记不剔（eligibility在panel层滚动判定）
        bench_dates = None
        if os.path.exists(BENCHMARK_PATH):
            bench_dates = pd.read_csv(BENCHMARK_PATH, parse_dates=["date"])["date"].sort_values().reset_index(drop=True)
        for fund_code, df_out, stat, clean_span in passed:
            f0, f1 = df_out["date"].min(), df_out["date"].max()
            if bench_dates is not None:
                cov_total_fund = int(((bench_dates >= f0) & (bench_dates <= f1)).sum())
            else:
                cov_total_fund = max(len(df_out), 1)
            stat["coverage_ratio"] = len(df_out) / max(cov_total_fund, 1)
            stat["zero_ret_cnt"] = int((df_out["daily_ret"] == 0).sum())

    # 严格统一窗口（仅current模式）：取所有合格基金的共同起止日，二次截断
    # 保证横向排名时每只基金的窗口完全一致（0天差异），而不是依赖30天容差；full模式不做共同窗口
    if window == "current" and passed:
        common_start = max(df["date"].min() for _, df, _, _ in passed)
        common_end = min(df["date"].max() for _, df, _, _ in passed)
        if (common_end - common_start).days < 1000:
            print(f"[WARN] 共同窗口跨度异常：{common_start.date()} ~ {common_end.date()}，请人工检查数据")
        print(f"严格共同窗口：{common_start.date()} ~ {common_end.date()}（全体基金一致）")
        aligned = []
        for fund_code, df_win, stat, _ in passed:
            df_c = df_win[(df_win["date"] >= common_start) & (df_win["date"] <= common_end)].reset_index(drop=True)
            span_c = (df_c["date"].max() - df_c["date"].min()).days
            aligned.append((fund_code, df_c, stat, span_c))
        passed = aligned

    # 份额去重：同基金不同份额只保留代表份额
    kept = dedup_shares(passed, report_list, name_map)

    # 写入临时目录 + ok记录
    for fund_code, df_win, stat, clean_span in kept:
        out_path = os.path.join(tmp_dir, f"fund_{fund_code}.csv")
        df_win.to_csv(out_path, index=False, encoding="utf-8-sig")
        report_list.append({
            "fund_code": fund_code,
            "status": "ok",
            "reason": "success",
            **stat,
            "clean_date_span_days": clean_span
        })
        print(f"[{fund_code}] [OK] ok |原始:{stat['rows_original']} 清洗后:{stat['rows_after_clean']} 跨度:{clean_span}d 可疑跳变:{stat['suspicious_jump_cnt']}")

    # 统一填充基金名称（来自全量元数据），报告与processed同批次携带名称快照
    for rec in report_list:
        rec.setdefault("fund_name", name_map.get(rec["fund_code"], ""))

    # 报告写进临时目录，随swap一起生效，保证与processed同一批次
    df_report = pd.DataFrame(report_list)
    df_report.to_csv(os.path.join(tmp_dir, os.path.basename(report_path)), index=False, encoding="utf-8-sig")

    # 原子替换（backup让位法）：任何时刻磁盘上都保留一份完整批次
    # 1) 正式目录原子让位为backup 2) tmp上位；失败则回滚让位（旧批次复位） 3) 成功后删backup
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)  # 清理上轮异常残留
    if not os.path.exists(out_dir):
        os.rename(tmp_dir, out_dir)
    else:
        os.rename(out_dir, backup_dir)
        try:
            os.rename(tmp_dir, out_dir)
        except Exception:
            os.rename(backup_dir, out_dir)  # 回滚：旧批次完整复位
            raise
        shutil.rmtree(backup_dir)
    print(f"\nprocessed目录已替换：{out_dir}（{len(kept)} 只，报告随目录交换）")

    cnt_ok = (df_report["status"] == "ok").sum()
    cnt_err_read = (df_report["status"] == "error_read").sum()
    cnt_err_clean = (df_report["status"] == "error_clean").sum()
    cnt_reject_empty = (df_report["status"] == "reject_empty").sum()
    cnt_reject_short = (df_report["status"] == "reject_short_history").sum()
    cnt_reject_stale = (df_report["status"] == "reject_stale_data").sum()
    cnt_reject_cov = (df_report["status"] == "reject_low_coverage").sum()
    cnt_reject_dup = (df_report["status"] == "reject_duplicate_share").sum()

    print(f"\n=====清洗完成=====")
    print(f"成功保留基金: {cnt_ok} 只（已含份额去重）")
    print(f"读取异常: {cnt_err_read} 只 | 清洗异常: {cnt_err_clean} 只")
    print(f"清洗后为空剔除: {cnt_reject_empty} 只")
    print(f"窗口覆盖不足剔除: {cnt_reject_short} 只 | 窗口尾部缺数据剔除: {cnt_reject_stale} 只")
    print(f"披露覆盖率过低剔除: {cnt_reject_cov} 只 | 同基金重复份额剔除: {cnt_reject_dup} 只")
    print(f"清洗报告输出至：{report_path}")
    return df_report


if __name__ == "__main__":
    clean_all()
