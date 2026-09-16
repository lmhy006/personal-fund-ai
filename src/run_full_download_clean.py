# phase2全量拉取驱动：全量下载基金净值后自动清洗
# 后台运行：日志输出到 logs/full_pull.log
from data_loader import batch_download_funds
from clean_nav import clean_all

if __name__ == "__main__":
    print("===== 全量拉取开始 =====")
    batch_download_funds(sleep_sec=2.0)
    print("\n===== 拉取完成，开始清洗 =====")
    clean_all()
    print("\n===== 全量拉取+清洗全部完成 =====")
