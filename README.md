# fund_ai

A股公募基金（股票型 + 混合型）数据分析与选基 pipeline：数据拉取 → 探查 → 清洗 → 指标分析 → 图表输出 → 研究样本构建 → 面板数据集。当前完成 **Phase 1**（基金评价系统全链路）与 **Phase 1.5**（研究池与数据集建设），Phase 2（walk-forward 验证与模型训练）数据底座已就绪，待实现时序切分器后启动训练。

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
│   ├── fund_list_loader.py    # 全量列表 + phase0调试池留档（phase0时代）
│   ├── universe_builder.py    # Phase1.5：全量列表→份额簇去重→近3年预筛→分层随机抽样(seed=42)→开发池1500只
│   ├── data_loader.py         # 批量下载基金净值 + 沪深300基准（锚点增量更新/断点续传）
│   ├── eda_nav.py             # 数据探查：逐基金体检报告（只读）
│   ├── clean_nav.py           # 清洗：官方日增长率口径 + 双模式(current=三年分析视图 / full=全历史) + 份额去重 + coverage门槛
│   ├── analyze_nav.py         # 指标分析 + 三联图 + 单基金CLI
│   ├── panel_builder.py       # Phase2数据集：fund-month面板（特征≤t信息 + 未来6月标签 + 时变eligibility）
│   ├── run_dev_download_clean.py    # 开发池拉取+清洗驱动（后台挂机用）
│   └── run_full_download_clean.py   # 全量拉取+清洗驱动（phase2正式实验前用）
├── data/
│   ├── raw/                   # 原始层：基金池、fund_nav/全历史净值缓存(1676只)、benchmark_hs300.csv
│   ├── processed/
│   │   ├── fund_processed/    # current模式：三年严格共同窗口(2023-09~2026-09) 1493只 + clean_report.csv(含fund_name/coverage_ratio/zero_ret_cnt)
│   │   ├── fund_history/      # full模式：完整历史 1500只 + clean_history_report.csv
│   │   └── *_tmp / *_backup/  # swap临时与备份目录（正常状态下自动消失）
│   ├── analyze/               # Phase1分析视图：analysis_summary.csv + charts/三联图
│   └── ml/
│       ├── panel.parquet      # fund-month面板（126,124行/1499只/278个月末截面 2003-01~2026-02）
│       └── scores/            # （预留）每月正式评分快照，积累向前验证记录
```

## 运行顺序

```bash
# 1. 生成基金池（一次；universe_builder.py 生成开发池 fund_dev_pool.csv，seed=42 锁定）
python src/universe_builder.py
# 注意：dev_pool 落盘后即为池锚，东财榜单每日漂移，勿重跑覆盖

# 2. 批量下载净值 + 基准（每日运行自动增量更新；pool_file 指定池）
python src/data_loader.py          # 默认读 fund_code_list.csv（全量列表）

# 3. 探查原始数据（可选，重跑覆盖报告）
python src/eda_nav.py

# 4. 双模式清洗
python -c "import sys; sys.path.insert(0,'src'); from clean_nav import clean_all; clean_all(pool_file='fund_dev_pool.csv', window='current')"
python -c "import sys; sys.path.insert(0,'src'); from clean_nav import clean_all; clean_all(pool_file='fund_dev_pool.csv', window='full')"

# 5. Phase1 分析（1493 只指标表 + 三联图）
python src/analyze_nav.py
# 单基金查询：python src/analyze_nav.py 340006

# 6. Phase2 面板构建（全历史 → fund-month 面板 parquet）
python src/panel_builder.py
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
| 披露覆盖率 | QA 门槛：窗口内有效观测数 / 同期基准交易日数 ≥ 95%，过滤周频/月频披露与长期停披露基金（clean_report 含 coverage_ratio 列） |
| 零收益报警 | zero_ret_cnt 仅报警不作剔除标准（低波动基金天然有真 0 收益日）；wealth index 由 daily_ret 现算（`(1+daily_ret).cumprod()`）不落盘 |
| 基准价格指数 | 沪深300 为**价格指数**（不含成分股分红），基准收益被低估约2%/年，alpha 略微高估 |
| 份额去重 | 同基金不同份额（简称去尾部 A/C/D/E/H/I/M 后缀聚类）只保留代表份额：窗口跨度最长者优先，并列时 A 类优先 |

## 面板口径（Phase 2 数据集）

| 口径 | 说明 |
|---|---|
| 面板结构 | fund-month：每行 = (基金 i，月末截面 t)，特征来自过去 12 个月（252 交易日）窗口，仅使用 ≤t 的信息 |
| 标签 | `future_ret_6m = W(t+126)/W(t) − 1`（未来 6 个月复利收益，W 为 wealth index） |
| wealth index | `(1+daily_ret).cumprod()` 现算不落盘；基金净值**预对齐到基准交易日日历**，未披露日 NaN、财富前向保持 |
| 特征集 | ret_1m/3m/6m/12m、vol_12m、sharpe_12m、mdd_12m、beta_12m、alpha_12m（12m 窗口对基准 OLS，普通 SE） |
| 时变 eligibility | `is_eligible(fund, t)`：成立 ≥ 12 个自然月 + 截面及标签期末 15 天内有披露 + 过去 12m coverage ≥ 95%——逐期判定，**不做整基金静态删除** |
| `age_months` | 基金在截面时的年龄（月），分层模型依据：12–36 月短历史段 / ≥36 月长历史段（常量 SHORT_HISTORY_AGE / FULL_HISTORY_AGE） |
| `label_end_date` | 该行标签的结束日；时序切分器据此过滤：**训练行的 label_end_date < 本次预测截面日**（不用 TimeSeriesSplit 的 gap=行数，那会只隔几行基金数据） |
| 当前规模 | 126,124 行 / 1499 只 / 278 截面（2003-01 ~ 2026-02，覆盖多轮牛熊） |

## 输出物

- `data/analyze/analysis_summary.csv`：每基金一行，含 fund_name、累计收益、年化收益/波动、夏普、最大回撤/修复、VaR95/CVaR95、下行偏差、alpha（含 t/p/R²/样本数）、beta
- `data/analyze/charts/fund_xxxxxx.png`：净值对比 / 回撤 / 日收益分布（VaR、CVaR 竖线）三联图
- `data/processed/fund_processed/clean_report.csv`：每基金清洗状态（ok / reject_short_history / reject_stale_data / reject_low_coverage / reject_duplicate_share / error_*），含 fund_name / coverage_ratio / zero_ret_cnt
- `data/processed/fund_history/clean_history_report.csv`：全历史批次报告（full 模式）
- `ml/panel.parquet`：fund-month 面板（Phase2 数据源）

## 数据层约定

- `data/raw/` 只增不改；净值缓存在 `data/raw/fund_nav/`，每日运行 `data_loader.py` 时以基准最新交易日为锚点自动增量更新（落后超 3 天重拉）
- 清洗写盘采用 tmp → backup 让位法原子交换，任何失败场景下磁盘上保留完整批次；两次 rename 之间正式路径短暂不存在，**当前流程假设串行执行**（若未来清洗与分析并行运行，需增加任务锁或版本目录+指针方案）
- **两条 processed 管线**：`current`（三年严格共同窗口，Phase1 分析视图）与 `full`（全历史，Phase2 面板数据源）并存，分别输出 fund_processed/ 与 fund_history/，勿混用
- 评分留存约定：每次正式预测的评分快照落盘 `ml/scores/YYYY-MM.csv`，从第一次正式预测开始积累**向前验证记录**（回测不可替代）
- 网络依赖：基金净值走东财接口，沪深300 走新浪接口（东财 push2his 指数接口在部分环境被拒）；限流防护默认 sleep 1.5s
- pandas 3.0 坑：`datetime64` 默认单位为**微秒**，`.astype("int64")` 取出微秒数；涉及时间戳数值运算必须 `to_numpy(dtype="datetime64[ns]").astype("int64")`（panel_builder 踩过）

## 已知局限

- **幸存者偏差**：基金列表来自现存基金（`fund_open_fund_rank_em`），已清盘/转型的不在池内；模型定位为"现存池内相对排序"；panel 的标签期末 15 天披露检查会把终止基金排除（其最差样本从未进入训练），结论限定为**"现存池的条件性历史研究"**；从第一次正式预测起每月留存评分快照，逐步积累真正的向前验证
- **selection bias 的实证**：头部池（phase0 榜单前200）与随机开发池在完全相同口径下的对比：CAGR 中位 27.8% vs **6.6%**，alpha 显著率 29% vs **0.7%**——任何拿榜单头部池训练的模型都学不到真实分布
- **基准错配**：行业/小盘基金对沪深300 的 beta/alpha 失真，靠 `r_squared` 门槛辅助过滤（开发池中 481/1493 只 R²<0.5）
- **无风险利率固定 2%**：beta≈1.4 时 alpha 误差约 ±0.3pp
- EDA 跳变检测、探查报告基于**全历史原始数据**，而清洗报告只覆盖统一窗口内基金，两者计数不可直接对比
- **新基金（<1年）尚不可预测**：分层模型（Short-history 12–36月 / Full ≥36月）为待验证设计，需按 age 分层分别训练并在各自样本外检验分数可比较性后，才能合并排行
