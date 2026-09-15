import os
import pandas as pd
import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

RAW_FUND_NAV_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "fund_nav")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "fund_processed")
CLEAN_REPORT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "clean_report.csv")

# 时间跨度门槛：自然日，低于该值整个基金直接剔除
MIN_DATE_SPAN_DAYS = 730
# 可疑跳变阈值：单日涨跌幅绝对值
JUMP_THRESHOLD = 0.20

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

    stat["rows_after_clean"] = len(df)
    return df, stat


def clean_all():
    """
    批量清洗全部基金
    1. 【幂等】先清除processed目录下旧 fund_*.csv
    2. 读取raw/fund_nav下面所有fund_*.csv
    3. 调用clean_fund做行层面清洗，捕获清洗阶段异常
    4. 【批量层】判断时间跨度，不足MIN_DATE_SPAN_DAYS直接整只剔除
    5. 通过的基金写入 data/processed/fund_xxxxxx.csv
    6. 输出clean_report.csv，记录全部基金状态
    """
    # 幂等：删除上一轮残留的基金csv，保留报告文件
    print(f"开始清理 {PROCESSED_DIR} 下旧的 fund_*.csv 文件……")
    for fname in os.listdir(PROCESSED_DIR):
        file_path = os.path.join(PROCESSED_DIR, fname)
        if os.path.isfile(file_path) and fname.startswith("fund_") and fname.endswith(".csv"):
            os.remove(file_path)
    print("旧基金csv清理完毕，开始本轮清洗\n")

    report_list = []
    file_list = [f for f in os.listdir(RAW_FUND_NAV_DIR) if f.startswith("fund_") and f.endswith(".csv")]
    print(f"待清洗基金总数：{len(file_list)}，最小时间门槛 {MIN_DATE_SPAN_DAYS} 自然日")

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
            print(f"[{fund_code}] ⚠️ 清洗后为空，整只剔除")
            continue

        # 批量层：计算清洗后的时间跨度，判断是否达标
        date_start = df_clean["date"].min()
        date_end = df_clean["date"].max()
        clean_span = (date_end - date_start).days

        if clean_span < MIN_DATE_SPAN_DAYS:
            report_list.append({
                "fund_code": fund_code,
                "status": "reject_short_history",
                "reason": f"历史过短:{clean_span}天 < {MIN_DATE_SPAN_DAYS}",
                **stat,
                "clean_date_span_days": clean_span
            })
            print(f"[{fund_code}] 历史过短 {clean_span}天，整只剔除")
            continue

        # 全部校验通过，写入processed
        out_file = f"fund_{fund_code}.csv"
        out_path = os.path.join(PROCESSED_DIR, out_file)
        df_clean.to_csv(out_path, index=False, encoding="utf-8-sig")

        report_list.append({
            "fund_code": fund_code,
            "status": "ok",
            "reason": "success",
            **stat,
            "clean_date_span_days": clean_span
        })
        print(f"[{fund_code}] ✅ ok |原始:{stat['rows_original']} 清洗后:{stat['rows_after_clean']} 跨度:{clean_span}d 可疑跳变:{stat['suspicious_jump_cnt']}")

    # 输出清洗报告（覆盖旧报告）
    df_report = pd.DataFrame(report_list)
    df_report.to_csv(CLEAN_REPORT_PATH, index=False, encoding="utf-8-sig")

    cnt_ok = (df_report["status"] == "ok").sum()
    cnt_err_read = (df_report["status"] == "error_read").sum()
    cnt_err_clean = (df_report["status"] == "error_clean").sum()
    cnt_reject_empty = (df_report["status"] == "reject_empty").sum()
    cnt_reject_short = (df_report["status"] == "reject_short_history").sum()

    print(f"\n=====清洗完成=====")
    print(f"成功保留基金: {cnt_ok} 只")
    print(f"读取异常: {cnt_err_read} 只 | 清洗异常: {cnt_err_clean} 只")
    print(f"清洗后为空剔除: {cnt_reject_empty} 只 | 历史太短剔除: {cnt_reject_short} 只")
    print(f"清洗报告输出至：{CLEAN_REPORT_PATH}")
    return df_report


if __name__ == "__main__":
    clean_all()
