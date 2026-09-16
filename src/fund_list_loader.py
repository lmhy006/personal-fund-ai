import akshare as ak
import pandas as pd
import os

# 路径设置
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
RAW_DATA_DIR = os.path.join(PROJECT_DIR, "data", "raw")
# 获取基金列表
df_stock = ak.fund_open_fund_rank_em(symbol="股票型")
df_mix = ak.fund_open_fund_rank_em(symbol="混合型")
fund_rank_df = pd.concat([df_stock, df_mix], ignore_index=True)
# 保留核心字段
fund_list = fund_rank_df[["基金代码", "基金简称"]].copy()
# 去重
fund_list = fund_list.drop_duplicates(subset=["基金代码"])
# 重置索引
fund_list = fund_list.reset_index(drop=True)
print(f"共获取 {len(fund_list)} 支混合型+股票型基金")
print(fund_list.head())
# 保存到data/raw（全量池：phase2起以该列表为拉取池，含全部股票型+混合型基金）
save_path = os.path.join(RAW_DATA_DIR, "fund_code_list.csv")
fund_list.to_csv(save_path, index=False, encoding="utf-8-sig")

# Phase0调试池留档：取前200只（phase2后仅作历史留档，不再用于拉取）
phase0_pool = fund_list.head(200)
phase0_path = os.path.join(RAW_DATA_DIR, "phase0_fund_pool.csv")
phase0_pool.to_csv(phase0_path, index=False, encoding="utf-8-sig")
print(f"Phase0调试池(200只)留档至 {phase0_path}")
print(f"全量池即 {save_path}（{len(fund_list)}只），data_loader默认从该文件拉取")