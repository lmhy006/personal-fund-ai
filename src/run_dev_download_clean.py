# 开发池流水线：按 fund_dev_pool.csv 拉取净值 → 清洗（范围限定池内）
from data_loader import batch_download_funds
from clean_nav import clean_all

if __name__ == "__main__":
    print("===== 开发池拉取开始 =====")
    batch_download_funds(sleep_sec=2.0, pool_file="fund_dev_pool.csv")
    print("\n===== 拉取完成，开始清洗（仅开发池） =====")
    clean_all(pool_file="fund_dev_pool.csv")
    print("\n===== 开发池拉取+清洗全部完成 =====")
