# fund_ai

A股公募基金（股票型 + 混合型）数据分析与选基 pipeline：数据拉取 → 探查 → 清洗 → 指标分析 → 图表输出。当前完成 phase1（数据工程全链路），phase2（特征工程与模型训练）待启动。

## 环境搭建

- Python 3.12（Windows 实测）
- 安装依赖：

```bash
pip install -r requirements.txt
```

## 项目结构

```
fund_ai/
├── requirements.txt
├── src/
│   ├── fund_list_loader.py   # 拉取股票型+混合型基金列表，切出phase0实验池(前200只)
│   ├── data_loader.py        # 批量下载基金净值 + 沪深300基准（缓存增量更新）
│   ├── eda_nav.py            # 数据探查：逐基金体检报告（只读）
│   ├── clean_nav.py          # 清洗：官方日增长率口径 + 统一窗口 + 份额去重
│   └── analyze_nav.py        # 指标分析 + 三联图 + 单基金CLI
├── data/
│   ├── raw/                  # 原始层：基金池、fund_nav/逐基金净值、benchmark_hs300.csv
│   ├── processed/
│   │   ├── fund_processed/   # 清洗后净值（每基金一个csv + clean_report.csv）
│   │   ├── fund_processed_tmp/      # 清洗临时目录（swap后自动消失）
│   │   └── fund_processed_backup/   # swap失败时残留的备份（下轮自动清理）
│   └── analyze/
│       ├── analysis_summary.csv  # 每基金一行指标
│       └── charts/               # 每基金一张三联图 png
```

## 运行顺序

```bash
# 1. 生成基金池（一次）
python src/fund_list_loader.py

# 2. 批量下载净值 + 基准（每日运行自动增量更新）
python src/data_loader.py
# 全量强刷：python -c "import sys; sys.path.insert(0,'src'); from data_loader import batch_download_funds; batch_download_funds(force_refresh=True)"

# 3. 探查原始数据（可选，重跑覆盖报告）
python src/eda_nav.py

# 4. 清洗（严格共同窗口 + 份额去重 + 原子交换）
python src/clean_nav.py

# 5. 全量分析（指标表 + 102张三联图）
python src/analyze_nav.py

# 单基金查询
python src/analyze_nav.py 340006
```

## 指标口径（重要）

| 口径 | 说明 |
|---|---|
| 日收益 | 官方"日增长率"转小数（真实日收益，已含分红除权调整）。**nav_acc（累计净值）不是复权净值**，分红日对其 pct_change 会被稀释，不作为收益来源 |
| CAGR | 按真实自然年折算 `(end-start)/365.25`；**非** `252/观测数` 折算（后者系统性高估） |
| 波动/夏普/下行偏差/alpha年化 | 252 交易日惯例（与 CAGR 的自然年折算存在口径差异，见 analyze_nav.py docstring） |
| 夏普 | 算术平均超额收益 / 样本标准差 × √252（CFA 口径） |
| 最大回撤 | 复权净值 `(1+daily_ret).cumprod()` 曲线 |
| 回撤修复时间 | 谷底 → 重新站上回撤前峰值（自然日）；未修复的 `drawdown_repaired=False`，`recovery=NaN` |
| alpha/beta | 与沪深300日超额收益 OLS 回归（rf=2%/252）；标准误为 Newey-West HAC（Bartlett 核），p 值用 t 分布 |
| 基准语义 | alpha = 相对沪深300 的超额，**不是**相对同行的超额；`r_squared<0.5` 提示基准错配，解读需谨慎 |
| 分析窗口 | 全体基金严格共同窗口（max首日 ~ min末日，当前 2023-09-14 ~ 2026-09-14），窗口由全局最新净值日 - 3年确定 |
| 基准价格指数 | 沪深300 为**价格指数**（不含成分股分红），基准收益被低估约2%/年，alpha 略微高估 |
| 份额去重 | 同基金不同份额（简称去尾部 A/C/D/E/H/I/M 后缀聚类）只保留代表份额：窗口跨度最长者优先，并列时 A 类优先 |

## 输出物

- `data/analyze/analysis_summary.csv`：每基金一行，含 fund_name、累计收益、年化收益/波动、夏普、最大回撤/修复、VaR95/CVaR95、下行偏差、alpha（含 t/p/R²/样本数）、beta
- `data/analyze/charts/fund_xxxxxx.png`：净值对比 / 回撤 / 日收益分布（VaR、CVaR 竖线）三联图
- `data/processed/fund_processed/clean_report.csv`：每基金清洗状态（ok / reject_short_history / reject_stale_data / reject_duplicate_share / error_*）

## 数据层约定

- `data/raw/` 只增不改；净值缓存在 `data/raw/fund_nav/`，每日运行 `data_loader.py` 时以基准最新交易日为锚点自动增量更新（落后超 3 天重拉）
- 清洗写盘采用 tmp → backup 让位法原子交换，任何失败场景下磁盘上保留完整批次；两次 rename 之间正式路径短暂不存在，**当前流程假设串行执行**（若未来清洗与分析并行运行，需增加任务锁或版本目录+指针方案）
- 网络依赖：基金净值走东财接口，沪深300 走新浪接口（东财 push2his 指数接口在部分环境被拒）；限流防护默认 sleep 1.5s

## 已知局限

- **幸存者偏差**：基金列表来自现存基金（`fund_open_fund_rank_em`），已清盘/转型的不在池内；模型定位应为"现存池内相对排序"
- **基准错配**：行业/小盘基金对沪深300 的 beta/alpha 失真，靠 `r_squared` 门槛辅助过滤
- **无风险利率固定 2%**：beta≈1.4 时 alpha 误差约 ±0.3pp
- EDA 跳变检测、探查报告基于**全历史原始数据**，而清洗报告只覆盖统一窗口内基金，两者计数不可直接对比
