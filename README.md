# fund_ai

A股公募基金（股票型 + 混合型）数据分析与选基 pipeline：数据拉取 → 探查 → 清洗 → 指标分析 → 图表输出 → 研究样本构建 → 面板数据集。当前完成 **Phase 1**（基金评价系统全链路）、**Phase 1.5**（研究池与数据集建设）与 **Phase 2A**（walk-forward 切分器 + 基线 Rank IC：近一年收益排序 IC=0.093 / 夏普排序 0.085，短历史段与长历史段量级接近）。**Phase 2B**：E1/E2/E1b 线性 + E3 树模型实验完成，经统计审查修正（NW t，n_eff≈42）后所有模型均无增量；runner 对齐评估与逐基金预测留存已完成（对齐后结论不变：单因子 Ridge 与基线完全重合 NW+0.73、树 NW-1.81 不显著）；**战略选择已拍板（2026-09-17）：接受动量基线（近一年收益排序）为第一版候选策略**——特征扩充列独立路线不阻塞；**E4 年龄检验完成**：动量对短/长历史段均适用（组内 IC +0.098/+0.082，NW 均显著）、Top 组年龄构成存在 +4.95pp 系统偏差（NW +2.59，须分段披露）但分层排序无收益增量（NW -0.92）→ **不训练年龄专用模型**；下一步 holdout 终审 → Phase 3 组合优化；不足一年基金评分方案仍待解决。

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
│   ├── data_loader.py         # 批量下载基金净值 + 沪深300基准（逐只全历史；--backfill 回填模式）
│   ├── daily_update.py         # 每日增量：1次请求拿全市场当日净值→按日追加（约30秒，落地后每日运行）
│   ├── eda_nav.py             # 数据探查：逐基金体检报告（只读）
│   ├── clean_nav.py           # 清洗：官方日增长率口径 + 双模式(current=三年分析视图 / full=全历史) + 份额去重 + coverage门槛
│   ├── analyze_nav.py         # 指标分析 + 三联图 + 单基金CLI
│   ├── panel_builder.py       # Phase2数据集：fund-month面板（特征≤t信息 + 未来6月标签 + 时变eligibility）
│   ├── walk_forward_splitter.py    # Phase2A：逐月末walk-forward切分（label_end+lag<T）+ 防泄漏断言 + holdout隔离 + 基线RankIC
│   ├── model_ridge.py              # Phase2B：Ridge实验runner（折内调参纪律 + 特征消融 + 标签模式）
│   ├── model_gbdt.py               # Phase2B：LightGBM实验runner（原生NaN走分支 + 折内早停纪律：rmse/ic两种准则）
│   ├── e4_age_check.py             # E4：主策略年龄适用性/跨段可比性/分层增量空间检验（纯评估无模型）
│   ├── live_score.py               # 上线评分管道：分层（≥12月ret_12m / 6-12月ret_6m低置信度 / <6月不评分）
│   ├── young_fund_check.py         # 不足一年基金研究检验：短窗口动量在各年龄段的 IC（dev 段，NW 判定）
│   ├── holdout_final.py            # 主策略holdout终审（一次性）：末尾12截面验收 + 报告留档
│   ├── run_dev_download_clean.py    # 开发池拉取+清洗驱动（后台挂机用）
│   └── run_full_download_clean.py   # 全量拉取+清洗驱动（phase2正式实验前用）
├── data/
│   ├── raw/                   # 原始层：基金池、fund_nav/全历史净值缓存(1676只)、benchmark_hs300.csv
│   ├── processed/
│   │   ├── fund_processed/    # current模式：三年严格共同窗口(2023-09~2026-09) 1493只 + clean_report.csv(含fund_name/coverage_ratio/zero_ret_cnt)
│   │   ├── fund_history/      # full模式：完整历史 1500只 + clean_history_report.csv
│   │   └── *_tmp / *_backup/  # swap临时与备份目录（正常状态下自动消失）
│   └── analyze/               # Phase1分析视图：analysis_summary.csv + charts/三联图
└── ml/
    ├── panel.parquet          # fund-month面板（126,124行/1499只/278个月末截面 2003-01~2026-02）
    ├── wf_splits/             # Phase2A：wf_manifest.csv（逐折切分审计）+ baseline_rankic.csv（基线RankIC逐月明细）
    ├── experiments/           # Phase2B：ridge_{tag}/gbdt_{tag}_monthly.csv 各实验变体逐月明细
    └── scores/                # （预留）每月正式评分快照，积累向前验证记录
```

## 运行顺序

```bash
# 1. 生成基金池（一次；universe_builder.py 生成开发池 fund_dev_pool.csv，seed=42 锁定）
python src/universe_builder.py
# 注意：dev_pool 落盘后即为池锚，东财榜单每日漂移，勿重跑覆盖

# 2. 批量下载净值 + 基准（两种模式：日常增量 / 长任务回填）
python src/data_loader.py                     # 日常增量：缓存末日期落后锚点>3天才重拉
python src/data_loader.py --backfill          # 回填：只为缺失缓存拉取（全量补拉/断点续传）
#   ⚠️ 长任务（如 9661 只全量补拉，数小时~十几小时）必须用 --backfill：锚点随日期前进，
#      否则前一天拉的进度会在 3 天后被判"过期"而重拉，长任务永远无法收敛。
#      回填完成后若要让缓存跟上最新净值，再用日常增量模式跑一次即可。

# 2b. 日常运行（落地后每天）：**1 次请求**拿全市场当日净值 → 按日追加，约 30 秒
python src/daily_update.py             # 追加最新交易日到各基金缓存（原子写）
python src/daily_update.py --catch-up  # 漏跑多日后补齐：落后≥2日的基金逐只全历史重拉
python src/daily_update.py --dry-run   # 只统计不写盘
#   为什么不用 data_loader：它是逐只整只全历史重拉（9661只≈8小时），只适合回填；
#   东财 fund_open_fund_daily_em 一次返回全市场约 2.4 万只的最近两个交易日净值+日增长率。
#   **不必每天运行**：漏 1~2 个交易日 → daily_update 直接补；漏更多 → --catch-up 逐只补齐
#   （历史净值随时可从逐只接口拉回，数据不会丢，只是补齐更慢）。

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

# 7. Phase2A walk-forward 切分 + 基线 Rank IC（dev folds；末尾12截面holdout默认隔离）
python src/walk_forward_splitter.py                                 # 仅生成逐折切分审计 manifest
python src/walk_forward_splitter.py --baseline ret_12m,sharpe_12m   # 基线：近一年收益/夏普排序的月度 Rank IC
# 最终验收才打开 holdout（单独落盘，只允许跑一次）：--include-holdout

# 8. Phase2B 模型实验（--features/--label-mode/--tag 生成消融变体）
python src/model_ridge.py                                        # E1 Ridge 全特征
python src/model_ridge.py --features ret_12m --tag r12only       # E2 消融示例
python src/model_ridge.py --label-mode rank --tag rank_full      # E1b 秩标签
python src/model_gbdt.py --tag v1                                # E3 LightGBM（rmse早停）
python src/model_gbdt.py --early-stop ic --tag v2_icstop         # E3 v2（IC早停）
python src/e4_age_check.py                                        # E4 主策略年龄检验（纯评估）
# 所有 runner 默认：IC/Top 与基线同在 ret_12m 非缺失行上算（对齐评估），
# 且逐基金预测留存 ml/experiments/preds_{runner}_{tag}_monthly.csv（评估与训练解耦）

# 9. 上线评分（分层：≥12月 ret_12m 主策略 / 6-12月 ret_6m 低置信度 / <6月不评分；无未来标签、不检查未来端点）
python src/live_score.py                                   # 评分日默认=净值最新交易日，落盘 ml/scores/YYYY-MM.csv
python src/live_score.py --as-of 2026-02-27 --verify-panel --no-save   # 与面板同截面逐基金对账（口径自检）
python src/young_fund_check.py                             # 不足一年基金方案的研究依据（dev 段，勿用 holdout）

# 10. holdout 终审（**只允许跑一次**；跑完主策略与评估口径冻结）
python src/holdout_final.py                                # 末尾12截面(2025-03~2026-02)验收，报告留档
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

## 切分口径（Phase 2A walk-forward）

| 口径 | 说明 |
|---|---|
| 训练行规则 | `label_end_date + 公布滞后(默认5自然日) < 预测截面 T`——标签已走完且当时已公布才可入训练；**不是** `t_date < T`（同一行的截面日与标签揭晓日相差约6个月） |
| 切分方式 | 逐月末滚动 expanding（`--train-window-months N` 可改滑动窗）；训练行数 <300 的早期截面跳过（实际 2006-03 起 228 折可用） |
| holdout | 末尾 12 截面（2025-03~2026-02）默认隔离：`folds('dev')` 接口层面拿不到，开发期一切实验不得触碰；最终验收 `--include-holdout` 单独落盘 |
| 防泄漏断言 | 每折强制 assert：训练标签未揭晓、截面日早于 T、测试集为当月横截面——违反即崩溃，口径之争变代码级错误 |
| 预处理纪律 | `FoldPreprocessor`（中位数填补+标准化）fit 仅允许在训练折；基线 RankIC 不需要它（Spearman 免疫单调变换、缺失行剔除） |
| 基线结论(dev) | ret_12m 全样本 mean IC **0.093**（t=6.2，正占比68%）；sharpe_12m **0.085**（t=7.9，71%）；短历史段(12-36月)与长历史段(≥36月)量级接近——Phase2B 模型必须先打过这两条基线 |
| 有效月门槛 | IC 汇总仅计当月横截面 ≥30 行的月份（MIN_TEST_ROWS_FOR_IC）；更小月份照常落盘明细但不入汇总 |
| 年龄组微差 | `age_group` 按 12 月折算（30.4375天/月）vs eligibility 用 365 自然日；面板 age_months 已 round(...,1)（365天→12.0），under12 实际不出现 |

## 模型实验口径与结论（Phase 2B，截至 2026-09-17）

**口径**

| 口径 | 说明 |
|---|---|
| 训练标签 | 默认**月内去均值**（y'=y-当月横截面均值）；它去掉当月全池共同水平，**不是风格调整**，不能据此衡量经理能力。`--label-mode rank` 是月内秩变换（E1b 诊断）。IC 评估与月内平移口径等价；组合收益报绝对/相对300/相对全池三口径 |
| 超参纪律 | Ridge alpha 网格 logspace(-3,3,13)：折内早段拟合（标准化统计只见早段）+ 尾部 24 个月揭晓段按月度 IC 选优 → 全训练折重拟合；只用 ≤T 信息。GBDT：固定保守超参组 + 折内早停（同尾部验证段），`--early-stop rmse/ic` 两种准则 |
| GBDT 特征处理 | **不标准化、不填补**——LightGBM 分裂原生处理 NaN（缺失进默认分支），ret_12m 的 1486 个新基金缺失行直接交给树学；无拟合状态＝无预处理泄漏面 |
| GBDT 成本参数 | v1：lr=0.05/上限2000轮/耐心100 + rmse早停（轮数中位4）；v2 原参数：与 v1 完全相同参数、**只换 IC 早停**（轮数中位9，部分折跑满2000轮上限；唯一差异＝早停准则，干净对照）；v2 受控参数：lr=0.1/上限400轮/耐心50 + IC早停（228折分4片并行后合并，作参数稳健性对照） |
| 评估与判定 | 月度 Rank IC + Top20% 等权三口径收益（相邻月标签重叠 5 个月→组合层统计偏乐观，正式换手/费用回测属 Phase 3）；模型 vs 基线同月配对差 t 检验。**标签重叠修正（审查升级）**：6 月标签逐月重叠 5 个月 → IC 序列 lag1 自相关 0.6~0.8、n_eff≈40~50（naive t 高估约 2.3 倍）——**正式判定一律用 Newey-West HAC t（lag=6）**，naive t 仅存档 |

**首轮实验记录（dev folds 228 折；本表 IC 未做评估行对齐、t 为 naive 值，正式判定见下方对齐评估与 NW(6)）**

| 实验 | 特征集 | 标签 | 旧版 IC | vs ret_12m 基线 naive t |
|---|---|---|---|---|
| E1 | 全 10 特征 | demean | 0.021 | -4.07（正式判定已改为边缘） |
| E2 | 单因子 ret_12m | demean | 0.092 | -0.64 ≈ 基线（管线自证） |
| E2 | 纯动量族(1m/3m/6m/12m) | demean | 0.076 | -1.23 无显著差异 |
| E2 | 动量+风险(sharpe/vol/mdd) | demean | **-0.045** | **-7.53 反号** |
| E2 | 动量+CAPM(alpha/beta) | demean | 0.002 | -4.87 |
| E2 | 动量+age_months | demean | 0.052 | -2.74 |
| E1b | 全 10 特征 | rank | 0.011 | -5.69 |
| E1b | 动量+风险组 | rank | -0.068 | -8.70 |
| E3 | 全 10 特征（LightGBM, rmse早停） | demean | 0.029 | -3.31 |
| E3 | 全 10 特征（LightGBM, IC早停·原参数） | demean | 0.026 | -3.63 |
| E3 | 全 10 特征（LightGBM, IC早停·受控参数分片） | demean | 0.030 | -3.37 |
| E3 | 单因子 ret_12m（树） | demean | -0.009 | -5.56（诊断：早停对单特征树过保守，stump 近乎常数，非公平审判） |

**显著性修正注记（2026-09-17 审查：表中 t 为 naive 值，标签重叠致高估约 2.3 倍）**

| 实验 | naive t | AR1 t | **NW(6) t** | 修正后判定 |
|---|---|---|---|---|
| ridge_v1 全特征 | -4.07 | -1.31 | **-2.02** | 边缘更差 |
| ridge 动量+风险组 | -7.53 | -2.91 | **-4.01** | **显著更差（稳健）** |
| gbdt_v1 / v2_icstop / v2 | -3.3~-3.6 | -1.4~-1.5 | **-1.7~-1.9** | **不显著——"显著更差"收回，改判无增量** |
| gbdt 单因子树 | -5.56 | -2.67 | **-3.06** | 显著更差（诊断性质保留） |
| 基线 ret_12m vs 0 | +6.18 | +2.65 | **+3.37** | **信号仍显著（但 n_eff≈42，置信度打折）** |

**对齐评估复核（2026-09-17，runner 升级：IC/Top 与基线同在 ret_12m 非缺失行上算，并逐基金预测留存 `preds_*.csv` 约 10.8 万行/实验）**：单因子 Ridge IC 0.0947 vs 基线 0.0934（配对差 +0.0013，NW +0.73）——**对齐后自证精确到噪声级**；全特征 Ridge 0.019（NW -2.08 边缘）；LightGBM ic早停 0.029（NW **-1.81 不显著**）。对齐不改变任何结论（每月差中位 2 行的规模），审查点①的量化判断最终确认。

## 主策略年龄检验与战略选择（2026-09-17）

**战略选择（已拍板）**：接受**动量基线（近一年收益排序）为第一版候选策略**。依据：修正后仍显著（NW +3.37）、单因子 Ridge 对齐后与基线完全重合（管线自证精确到噪声级）、线性/树模型在 NW 口径下均无增量、E4 证明其对两个年龄段适用——**不为进入下一阶段而强迫 ML 打赢基线**；特征扩充（规模/换手/经理/持仓，需新数据源）列为独立路线，同时是"经理能力"研究目标的前提，不阻塞 Phase 3。

**E4 年龄检验（`src/e4_age_check.py`，纯评估层，dev folds 对齐行）**

| 问题 | 结果 | NW 判定 |
|---|---|---|
| 适用性：组内动量 IC | short(12-36月) **+0.098** / full(≥36月) **+0.082** | 均显著（+3.21 / +2.90）；组间差不显著（-0.015，-0.66） |
| 跨段可比性：Top20 组年龄构成 | short 占比偏差 **+4.95pp** | **+2.59 显著——动量排序系统性偏向新基金进 Top 组** |
| 分层增量空间：组内排序−全池排序 Top 收益 | **-0.11%** | -0.92 不显著 → **不训练年龄专用模型** |

结论：开发池中两个年龄段的动量 IC 都为正；当前**未检出**按年龄分别取 Top20 的收益增量，故第一版统一排行。Top 组的年龄构成有系统差异，排行榜决定分年龄段披露，方便解读；不足一年基金不在面板 eligibility 内，**全量正式评分前须另行解决**（低置信度或另设输入）。

**补充诊断与边界：**动量基线在四个开发期历史段的平均 IC 均为正（0.066 / 0.091 / 0.126 / 0.091）；按过去 beta 粗分的三组内，动量 IC 也均为正（0.064 / 0.130 / 0.110）。这些结果表明信号不只是跨粗 beta 组排序，**尚不能证明已消除所有风格影响**。单因子树的大量折只训练一轮，产生较多并列预测；这是当前树配置的诊断结果，不能推广成“树模型不能做自证”。评估行对齐、E4 和主策略选择现均已完成，结果以上述较新的对齐评估及 E4 段落为准。

## 上线评分管道（2026-09-17 建成；含不足一年基金分层方案）

**与研究管道的分工**（`src/live_score.py` vs `src/panel_builder.py`）：

| | 研究（panel_builder） | 上线（live_score） |
|---|---|---|
| 信息范围 | 需要未来 126 交易日标签 | **只用过去 252 交易日** |
| eligibility | 成立≥365天 + 当期披露 + coverage≥95% + **标签期末披露**（未来端点） | 分层（下表），**无任何未来检查** |
| 可评截面 | 止于 2026-02（标签完整性所限） | 到净值最新日（2026-09） |

**评分分层（`src/young_fund_check.py` dev 段实定，NW(6) 判定，未使用 holdout）**

| 年龄段 | 信号 | 该段 IC（NW） | 处理 |
|---|---|---|---|
| ≥12月（满 252 交易日） | ret_12m（主策略，与 holdout 验收口径一致） | +0.099（+3.39） | `confidence=main`，进主排行 |
| 6–12月（满 126 交易日） | **ret_6m** | **+0.120（+2.95，91 月）** | `confidence=low`，组内单列，**不与主排行混排**（跨信号分数可比性未验证） |
| <6月 | 唯一可得 ret_1m | +0.073（+1.30 **不显著**，41 月） | **不评分**——明确标注证据不足，不用更短窗口硬凑分数 |

**口径自检（对账）**：面板截面 2026-02-27 → 主策略组 **1498 只交集、最大绝对差 0.000e+00**，分层改造未动摇原口径。

**评分快照**：`ml/scores/YYYY-MM.csv`（列含 rank / rank_lowconf / signal / confidence / age_group）；首次 2026-09 已留存，从本次起积累向前验证记录。

**窗口对比的 holdout 后发现（记录，不改主策略）**：dev 段各窗口 IC = ret_1m +0.050 / ret_3m +0.089 / **ret_6m +0.113** / ret_12m +0.093；差值 ret_6m−ret_12m = +0.0200 但 **NW t=+1.39 不显著** → 不构成更换主策略的依据。主策略已于 holdout 冻结，此项列为“holdout 后变更候选”，需独立向前数据验证。

**全量 universe 扩展（2026-09-18 完成）**：补拉全量 9661 只净值（`--backfill` 断点续传，失败 110 只随后补净）→ `clean_nav(full)` 全量清洗（份额去重后 **5317 只**，异常 0）→ 重跑评分：**5118 只可评分（主策略 4898 + 低置信度 220）**，`ml/scores/2026-09.csv` 已更新为全量版——**6–12 月龄低置信度组首次生效**（220 只），`<6月` 155 只明确不评分。

## holdout 终审（2026-09-17 一次性开启，主策略就此冻结）

`src/holdout_final.py` 用预留的末尾 12 个截面（2025-03~2026-02，dev 期从未触碰）验收主策略，报告留档 `ml/wf_splits/holdout_verdict.txt`。

| 指标 | holdout（12 月） | dev 对照（228 月） |
|---|---|---|
| 主策略 mean IC | **+0.2608** | +0.0934 |
| IC>0 月份占比 | **12/12** | 0.68 |
| NW(6) t | **+5.90** | +3.37 |
| Top20% 绝对收益（6月） | 23.49% | 9.41% |
| Top20% 相对沪深300 | +14.07% | +2.10% |
| Top20% 相对全池 | +9.56% | +1.37% |
| Top−Bottom 价差 | +13.93%（NW +3.62） | — |

**判定**：方向一致、无灾难性失效、显著为正 → 支持进入 Phase 3。

**必须随结论引用的三条限定**：① holdout 段是**单一市场状态**（该段全池 6 月均收益 +13.9%、基准 +9.4% 的强势市），IC 达 dev 期 **2.8 倍**正是**动量收益时变性**（regime 依赖）的表现，**不能假设 0.26 会持续**；② 12 个截面且标签重叠 5/6 → 有效样本约 3~5，本终审用于**排查灾难性失效**，不是确证有效性；③ 现存池条件性研究，组合层未计费用/换手。

**对 Phase 3 的直接要求**：回测必须覆盖多个市场状态（dev 段 2006-2025 含多轮牛熊），不得以 holdout 的高 IC 当作预期收益；主策略与评估口径自本报告起**冻结**，任何后续变更须标注为“holdout 后变更”。

## Phase 3 组合回测（2026-09-18 v2 审查修正）

**口径**（`src/backtest_strategy.py`）：每月按 ret_12m 排序取 Top50 等权（敏感性 20/100）；每月调仓、持 6 月、在持 6 期重叠（月度收益从净值现算，收益先算调仓在后）；费用 申购0.15%+赎回0.5%（费用前后双净值）。**v2 三处修正（2026-09-18 审查）**：① 选股用**无未来端点台账截面**（成立≥365天 + 当期披露 + 过去252日 coverage≥95%，与 live_score 同款 eligibility）——去除研究面板"未来6月末仍披露"的存活条件（持有期内将终止的基金从不入选会美化回测）；② 全池等权基准**与组合同入选时点**（上月台账在座基金在 [上月→本月] 的收益）；③ 市场状态标签**无前瞻**（截至上月末的过去12月沪深300累计，不含当月）。

**dev 段（2005-07 ~ 2025-02，236 个月）**

| N | 年化（费用后） | 费用侵蚀 | 波动 | 夏普 | MDD |
|---|---|---|---|---|---|
| 20 | 11.76% | 1.43pp | 23.9% | 0.50 | -52.3% |
| 50 | **11.84%** | 1.44pp | 23.6% | 0.51 | -55.1% |
| 100 | 11.63% | 1.43pp | 23.1% | 0.51 | -55.6% |

区间基准：沪深300 **7.86%** / 全池等权 **12.65%**。

**结论（修正后依然稳健）**：组合与全池等权**无统计显著差异**（月超额 -0.024%，NW t=-0.24）、与沪深300 亦不显著（NW +0.75）——**IC 正信号（0.093）未转化为显著组合超额**，该结论在去除面板未来披露存活条件、对齐基准时点后不变。**绝对数字对口径高度敏感**（v1 面板口径 3.40% vs v2 台账口径 11.84%，主因起点 2007-09→2005-06 与池子 1498→4476 只），**相对结论稳健**——比超额（费用后 vs 全池），别比绝对收益。

**状态分解（无前瞻标签）**：bull 月均 +2.84% / mix +0.45% / **bear +1.18%**（v2；v1 面板口径的 bear -0.90% 系状态标签含当月收益 + 面板池 + 2007 起点所致）。**不能据此断言"动量只在牛市赚钱"**——绝对收益随状态变化 ≠ 超额随状态变化，需单独检验超额的状态分解（组合构建改进探索的内容）。

**holdout 段（2025-04~2026-02，11 个月，v2）**：年化 45.03%、跑赢全池（36.48%）——**该段已被 holdout 终审与本轮多次使用（已看过），仅作描述性对照；此后任何新设计（特征扩充/状态过滤）的"最终盲测"只能来自向前积累的新数据（ml/scores/ 起），不得再以本段宣称盲测**。组合构建改进列为 holdout 后变更候选，须在 dev 段滚动研究。

## 输出物

- `data/analyze/analysis_summary.csv`：每基金一行，含 fund_name、累计收益、年化收益/波动、夏普、最大回撤/修复、VaR95/CVaR95、下行偏差、alpha（含 t/p/R²/样本数）、beta
- `data/analyze/charts/fund_xxxxxx.png`：净值对比 / 回撤 / 日收益分布（VaR、CVaR 竖线）三联图
- `data/processed/fund_processed/clean_report.csv`：每基金清洗状态（ok / reject_short_history / reject_stale_data / reject_low_coverage / reject_duplicate_share / error_*），含 fund_name / coverage_ratio / zero_ret_cnt
- `data/processed/fund_history/clean_history_report.csv`：全历史批次报告（full 模式）
- `ml/panel.parquet`：fund-month 面板（Phase2 数据源）
- `ml/wf_splits/wf_manifest.csv`：逐月末切分审计（train/test 规模、train_label_end_max＝该轮最晚揭晓的答案日期、ok/skipped/reserved 状态），防泄漏可复核
- `ml/wf_splits/baseline_rankic.csv`：基线 Rank IC 逐月长表（截面 × 特征 × 年龄组），Phase2B 模型的对照底线
- `ml/experiments/ridge_{tag}_monthly.csv`：Phase2B 各实验变体逐折明细（模型IC/双基线IC/Top20三口径收益/折内选定alpha/训练规模与最晚标签日），配对 t 检验打印于控制台
- `ml/experiments/gbdt_{tag}_monthly.csv`：同上（列 `n_rounds`/`valid_ic` 记录折内早停轮数与验证段月度IC）。提交的完整实验汇总为 `gbdt_v1`、`gbdt_v2_icstop`、`gbdt_v2`、`gbdt_v2align` 和 `gbdt_r12only`；分片与冒烟文件是本地中间产物。
- `ml/experiments/preds_{ridge,gbdt}_{tag}_monthly.csv`：逐基金预测留存（fund_code/t_date/age_months/ret_12m/sharpe_12m/y/pred，全体行含缺失）——评估与训练解耦，任何口径（对齐/分年龄/分位组）可离线重算，不必重跑模型
- `ml/experiments/e4_age_check_monthly.csv`：E4 逐月明细（全池/组内 IC、Top 组年龄构成、组内 vs 全池 Top 收益）
- `ml/experiments/young_fund_check_monthly.csv`：年轻基金样本长表（age_bucket × 可得窗口动量 × 未来6月收益，dev 段），不足一年基金方案的研究依据
- `ml/scores/YYYY-MM.csv`：上线评分快照（rank/rank_lowconf/fund_code/fund_name/score/signal/confidence/age_months/age_group/coverage/as_of）——评分日=净值最新交易日；2026-09-18 起为**全量版**（分层：主策略 4898 + 低置信度 220 = 5118 只）
- `ml/wf_splits/holdout_verdict.txt` + `holdout_verdict_monthly.csv`：holdout 终审报告与逐月明细（2026-09-17 一次性开启）

## 数据层约定

- `data/raw/` 只增不改；净值缓存在 `data/raw/fund_nav/`，每日运行 `data_loader.py` 时以基准最新交易日为锚点自动增量更新（落后超 3 天重拉）。**两种模式必须区分**（2026-09-17 增补）：日常增量按锚点判新鲜度（容忍净值 T+1 披露滞后）；`--backfill` 只拉缺失缓存、已有缓存一律跳过——多天长任务不能用日常模式，否则锚点前进会让已完成进度反复作废
- 清洗写盘采用 tmp → backup 让位法原子交换，任何失败场景下磁盘上保留完整批次；两次 rename 之间正式路径短暂不存在，**当前流程假设串行执行**（若未来清洗与分析并行运行，需增加任务锁或版本目录+指针方案）
- **两条 processed 管线**：`current`（三年严格共同窗口，Phase1 分析视图）与 `full`（全历史，Phase2 面板数据源）并存，分别输出 fund_processed/ 与 fund_history/，勿混用
- 评分留存约定：每次正式预测的评分快照落盘 `ml/scores/YYYY-MM.csv`，从第一次正式预测开始积累**向前验证记录**（回测不可替代）。**断档可合法回填**：主策略是无参数规则（动量排序，无拟合），评分只依赖 ≤t 的历史净值 → 事后用全历史重建某历史日评分，数值与当时真跑一致（须标注"事后回填"；缺的是"当时真做了预测"的仪式感，不是信息）
- **落地运行的频率与设备（2026-09-17 结论）**：数据层面**不必每天运行**——漏 1~2 个交易日用 `daily_update.py` 直接补，漏更多用 `--catch-up` 逐只补齐（漏 5 天约 6 小时，可周末后台跑），历史净值随时可拉、不会丢。因此**不需要租服务器**：家用电脑 + Windows 任务计划程序（勾选"错过计划后尽快启动"）即可；只有要 7×24 无人值守（Phase 4 Agent 自动化）时才需常开设备或轻量云主机
- 网络依赖：基金净值走东财接口，沪深300 走新浪接口（东财 push2his 指数接口在部分环境被拒）；限流防护默认 sleep 1.5s
- pandas 3.0 坑：`datetime64` 默认单位为**微秒**，`.astype("int64")` 取出微秒数；涉及时间戳数值运算必须 `to_numpy(dtype="datetime64[ns]").astype("int64")`（panel_builder 踩过）

## 已知局限

- **幸存者偏差**：基金列表来自现存基金（`fund_open_fund_rank_em`），已清盘/转型的不在池内；模型定位为"现存池内相对排序"；panel 的标签期末 15 天披露检查会把终止基金排除（其最差样本从未进入训练），结论限定为**"现存池的条件性历史研究"**；从第一次正式预测起每月留存评分快照，逐步积累真正的向前验证。**升级触发器（2026-09-17 实测，按节奏暂缓；Phase 3 正式回测前必须提醒并启动）**：宣称"可执行历史回测"之前必须补已清盘基金——`fund_open_fund_info_em` 对清盘基金全历史可用（实证：嘉实元和 505888 返回 2014-10~2019-08 完整 258 行，现有 data_loader 零改动可拉），**唯一缺口是清盘代码名单**（AKShare 全部列表接口均不含已清盘基金，funddel.html 已下线；名单需 AMAC 公示或第三方清单）；路径＝拿名单→拉净值→重建历史 universe→重跑 panel/切分/全部实验
- **selection bias 的实证**：头部池（phase0 榜单前200）与随机开发池在完全相同口径下的对比：CAGR 中位 27.8% vs **6.6%**，alpha 显著率 29% vs **0.7%**——任何拿榜单头部池训练的模型都学不到真实分布
- **基准错配**：行业/小盘基金对沪深300 的 beta/alpha 失真，靠 `r_squared` 门槛辅助过滤（开发池中 481/1493 只 R²<0.5）
- **无风险利率固定 2%**：beta≈1.4 时 alpha 误差约 ±0.3pp
- EDA 跳变检测、探查报告基于**全历史原始数据**，而清洗报告只覆盖统一窗口内基金，两者计数不可直接对比
- **不足一年基金已有分层方案（2026-09-17 定，09-18 随全量 universe 生效）**：`src/young_fund_check.py` dev 段实定——6–12 月龄用 **ret_6m**（IC +0.120、NW +2.95）给低置信度评分（`confidence=low`，组内单列不与主排行混排）；**<6 月龄唯一可得 ret_1m 不显著（NW +1.30），明确不评分**，不用更短窗口硬凑分数。已生效于 `ml/scores/2026-09.csv`：主策略 4898 只 + 低置信度组 220 只。
