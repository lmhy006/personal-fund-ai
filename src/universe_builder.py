# phase1.5：开发池构建
# 流程：全量列表 → 簇去重 → 近3年预筛 → 分层随机抽样
import os
import pandas as pd
import akshare as ak
from clean_nav import share_cluster_key

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DEV_POOL_PATH = os.path.join(RAW_DATA_DIR, "fund_dev_pool.csv")

SEED = 42  # 锁定随机种子保证可复现：同一列表+同一seed=同一批开发池


def fetch_fund_master_list() -> pd.DataFrame:
    """
    拉取股票型+混合型全量列表，按基金代码去重，附带近3年业绩列（预筛代理信号）
    :return: 基金代码, 基金简称, 近3年, fund_type
    """
    dfs = []
    for sym in ["股票型", "混合型"]:
        df = ak.fund_open_fund_rank_em(symbol=sym)
        df = df[["基金代码", "基金简称", "近3年"]].copy()
        df["fund_type"] = sym
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    all_df["基金代码"] = all_df["基金代码"].astype(str)
    return all_df.drop_duplicates(subset=["基金代码"]).reset_index(drop=True)


def dedup_share_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    列表层面的份额簇去重（无需净值数据）
    代表代理规则：簇内A类优先，无A取代码升序第一个
    （与clean_nav基于净值跨度的代表规则可能存在差异，差异名单在正式全量时校验）
    """
    df = df.copy()
    df["cluster"] = df["基金简称"].map(share_cluster_key)
    df["is_a"] = df["基金简称"].str.strip().str.endswith("A")
    df = df.sort_values(["cluster", "is_a", "基金代码"], ascending=[True, False, True])
    rep = df.drop_duplicates(subset=["cluster"], keep="first")
    return rep[["基金代码", "基金简称", "近3年", "fund_type"]].reset_index(drop=True)


def build_dev_pool(target: int = 1500, seed: int = SEED) -> pd.DataFrame:
    """
    构建开发池：全量列表 → 簇去重 → 近3年预筛 → 分层随机抽样（seed固定可复现）
    :param target: 抽样目标数（清洗后预计剩~85%，因预筛已排除大概率被剔的新基金）
    :param seed: 随机种子，锁定写入代码与文档，保证实验可复现
    """
    master = fetch_fund_master_list()
    print(f"全量列表：{len(master)} 只（股票型+混合型去重后）")

    rep = dedup_share_classes(master)
    print(f"份额簇去重：{len(master)} -> {len(rep)} 只（保留代表份额）")

    # 预筛：近3年业绩非空 = 成立大概率满3年，能通过清洗的窗口覆盖门槛
    rep = rep.copy()
    rep["has_3y"] = pd.to_numeric(rep["近3年"], errors="coerce").notna()
    eligible = rep[rep["has_3y"]].reset_index(drop=True)
    print(f"近3年业绩预筛：{len(rep)} -> {len(eligible)} 只（剔除大概率过不了窗口门槛的新基金）")

    # 分层随机抽样：股票型/混合型按eligible内比例分配名额
    n_stock_total = len(eligible[eligible["fund_type"] == "股票型"])
    n_mix_total = len(eligible[eligible["fund_type"] == "混合型"])
    n_stock = max(1, min(round(target * n_stock_total / len(eligible)), n_stock_total))
    stock_pool = eligible[eligible["fund_type"] == "股票型"].sample(n=n_stock, random_state=seed)
    mix_pool = eligible[eligible["fund_type"] == "混合型"].sample(
        n=min(target - n_stock, n_mix_total), random_state=seed)
    dev_pool = pd.concat([stock_pool, mix_pool]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    dev_pool = dev_pool[["基金代码", "基金简称", "fund_type"]]
    dev_pool.to_csv(DEV_POOL_PATH, index=False, encoding="utf-8-sig")
    print(f"开发池已保存：{DEV_POOL_PATH}")
    print(f"股票型 {int((dev_pool['fund_type'] == '股票型').sum())} 只 | 混合型 {int((dev_pool['fund_type'] == '混合型').sum())} 只")
    print(f"seed={seed} 已锁定：同一列表+同一seed可重建同一池")
    return dev_pool


if __name__ == "__main__":
    build_dev_pool(target=1500, seed=SEED)
