import os
import pandas as pd
import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
FUND_NAV_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "fund_nav")
OUTPUT_REPORT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "eda_probe_report.csv")

def probe_fund(csv_path: str) -> dict:
    """
    输入单只基金csv文件路径，只读，输出体检信息字典
    csv列：date, nav, nav_acc
    返回字段：
        fund_code: 基金代码
        file_name: 文件名
        rows_total: 总行数
        date_start: 最早日期
        date_end: 最晚日期
        date_span_days: 时间跨度总天数(自然日)
        nav_zero_cnt: nav等于0的数量
        nav_nan_cnt: nav NaN数量
        nav_acc_zero_cnt: nav_acc等于0数量
        nav_acc_nan_cnt: nav_acc NaN数量
        dup_date_cnt: 重复(date)的行数
        jump_suspicious_cnt: 可疑跳变行数(单日绝对值涨跌幅>20%)
    """
    file_name = os.path.basename(csv_path)
    fund_code = file_name.replace("fund_", "").replace(".csv", "")

    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
    except Exception as e:
        return {
            "fund_code": fund_code,
            "file_name": file_name,
            "rows_total": 0,
            "date_start": None,
            "date_end": None,
            "date_span_days": 0,
            "nav_zero_cnt": 0,
            "nav_nan_cnt": 0,
            "nav_acc_zero_cnt": 0,
            "nav_acc_nan_cnt": 0,
            "dup_date_cnt": 0,
            "jump_suspicious_cnt": 0,
            "error": str(e)
        }
    
    rows_total = len(df)

    if rows_total == 0:
        return {
            "fund_code": fund_code,
            "file_name": file_name,
            "rows_total": 0,
            "date_start": None,
            "date_end": None,
            "date_span_days": 0,
            "nav_zero_cnt": 0,
            "nav_nan_cnt": 0,
            "nav_acc_zero_cnt": 0,
            "nav_acc_nan_cnt": 0,
            "dup_date_cnt": 0,
            "jump_suspicious_cnt": 0,
        }

    date_start = df["date"].min()
    date_end = df["date"].max()
    date_span_days = (date_end - date_start).days

    nav_zero_cnt = (df["nav"] == 0).sum()
    nav_nan_cnt = df["nav"].isna().sum()

    nav_acc_zero_cnt = (df["nav_acc"] == 0).sum()
    nav_acc_nan_cnt = df["nav_acc"].isna().sum()

    # 重复日期计数：同一date出现>1条的记录总条数
    dup_mask = df.duplicated(subset=["date"], keep=False)
    dup_date_cnt = dup_mask.sum()

    # 可疑跳变：官方"日增长率"口径（与清洗口径一致，已含分红调整），|r|>20%只计数
    if "日增长率" in df.columns:
        dr = pd.to_numeric(df["日增长率"], errors="coerce") / 100.0
        jump_suspicious_cnt = int((dr.abs() > 0.20).sum())
    else:
        jump_suspicious_cnt = 0

    return {
        "fund_code": fund_code,
        "file_name": file_name,
        "rows_total": rows_total,
        "date_start": date_start.date() if pd.notna(date_start) else None,
        "date_end": date_end.date() if pd.notna(date_end) else None,
        "date_span_days": date_span_days,
        "nav_zero_cnt": int(nav_zero_cnt),
        "nav_nan_cnt": int(nav_nan_cnt),
        "nav_acc_zero_cnt": int(nav_acc_zero_cnt),
        "nav_acc_nan_cnt": int(nav_acc_nan_cnt),
        "dup_date_cnt": int(dup_date_cnt),
        "jump_suspicious_cnt": int(jump_suspicious_cnt),
    }

def probe_all():
    """
    批量探查全部基金csv，汇总报告，打印并保存 eda_probe_report.csv
    """
    collect = []
    file_list = [f for f in os.listdir(FUND_NAV_DIR) if f.startswith("fund_") and f.endswith(".csv")]
    print(f"找到待探查基金文件数量：{len(file_list)}")

    for fname in file_list:
        full_path = os.path.join(FUND_NAV_DIR, fname)
        info = probe_fund(full_path)
        collect.append(info)
        if info.get("error", ""):
            print(f"[{info['fund_code']}] ❌ ERROR: {info['error']}")
        else:
            print(f"[{info['fund_code']}] rows:{info['rows_total']} span_days:{info['date_span_days']} jump:{info['jump_suspicious_cnt']}")

    df_report = pd.DataFrame(collect)
    os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
    df_report.to_csv(OUTPUT_REPORT_PATH, index=False, encoding="utf-8-sig")
    error_count =df_report["error"].notna().sum() if "error" in df_report.columns else 0
    print(f"\n✅探查报告已输出：{OUTPUT_REPORT_PATH}，坏文件数量：{error_count}")
    print(df_report.head(10))
    return df_report

if __name__ == "__main__":
    probe_all()