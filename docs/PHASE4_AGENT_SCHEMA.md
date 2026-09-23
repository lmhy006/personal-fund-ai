# Phase 4 Agent 工具契约（Schema v1.1）

> 状态：**契约定稿 + 薄工具层已完成并验收（v1.1.1，2026-09-22）**。暂不接入 LLM、不执行真实交易、不增加任何研究参数。
> 本文是 **Agent 与生产系统的唯一契约**：任何未在本文登记的工具调用都不被允许。
> 薄工具层实现见 `src/agent_tools.py`（21 项测试通过，`tests/test_agent_tools.py`）；聊天界面不在本阶段范围内（对话/LLM 接入暂缓，待明确启动时再设计）。
>
> **v1.1 变更（相对 v1.0）**：
> 1. `run_risk_scenario(target_vol)` → **`get_risk_scenario_v1()`（无参数）**——只读已冻结的 15% 结果，不提供参数搜索路径；
> 2. `run_production_pipeline()` **对 Agent 不接受任何参数**；`skip_refresh`/`skip_clean` 只留在 CLI（诊断/重试），用它们生成的运行**不写 COMPLETE**（写 test marker、manifest status=test）；
> 3. 状态枚举**加入 `aborted`**：`ok | unavailable | aborted | error | not_executable`；所有示例强制包含全部统一字段（允许 `null`）；
> 4. **`run_id` 是权威快照键**，`month` 仅为便利查询；同月多个完整快照**始终选择 `score_generated_at` 最新者**并返回**全部候选 `run_id`** 与 `multiple_complete_for_month` 警告（不提供"或列表或最新"的二义行为）；显式传入 `run_id` 时**严格读取指定快照**；评分必须读 **snapshot 内 CSV**，不读会被覆盖的 `ml/scores/YYYY-MM.csv`；
> 5. `data_health=FAIL` 阻止**新生产运行与"当前最新"结论**，但**不阻止读取历史完整快照**（系统故障时仍可审计历史）；
> 6. 审计日志允许写入独立路径，但**不得修改 scores / ledger / snapshots**。

## 0. 原则（写死）

1. **只读优先**：Agent 默认只能读，唯一写操作 `run_production_pipeline` 必须由用户明确触发。
2. **冻结参数不可变**：Top N、持有期、信号窗口、费用率、eligibility、显著性阈值、risk target 全部冻结；工具接口**不接收这些参数**（查询展示用的 `top_n` 只影响返回行数，不影响策略）。
3. **停止即停止**：任一"停止条件"命中时，Agent 必须停止作答或明确说明不可用，**不得静默降级**、不得用旧数据当最新、不得用 planned cohort 当已持仓。
4. **可复现**：一切结论必须可追溯到具体 `run_id` + `source_snapshot`。
5. 新研究假设（改参数/转正/训练/搜索）**不在生产工具内**，必须退出 production workflow 重进 research workflow。

## 1. 工具清单

### 1.1 自由调用的只读工具（10 个）

| 工具 | 作用 | 主要输入 | 关键输出 |
|---|---|---|---|
| `get_data_health` | 最近一次数据健康报告 | — | status(PASS/WARN/FAIL)、score_ready、shadow_stale、processed 分布、gap/stale |
| `get_latest_complete_snapshot` | 最近一份带 `COMPLETE` 的正式快照 | — | run_id、manifest 摘要、评分文件哈希 |
| `get_score` | 指定（run_id 权威 / month 便利）的正式评分 | `run_id`（权威）或 `month`（YYYY-MM，可空） | scores 行数、main/low 计数、as_of、**snapshot 内** CSV 路径；同月多完整快照 → 最新 + 全部候选 run_id + 警告 |
| `get_top_funds` | 某次正式评分的 TopN（**仅查询展示**） | `run_id` 或 `month`、`top_n`(≤200) | 主策略 TopN 代码/名称/分数/排名 |
| `get_fund_rank_history` | 某基金在历次正式评分中的排名与分数 | `fund_code` | 各月 rank/score/是否 Top50；无历史月份则明示可用范围 |
| `get_portfolio_state` | 当前 6-cohort ledger 聚合 | — | active cohorts、聚合权重、现金、**planned 列表**、预计费用 |
| `get_shadow_status` | 影子评分状态 | — | 因子截止日、最近影子评分/最近跳过原因、stale 标记 |
| `compare_snapshots` | 两个正式快照的 manifest 字段级对比 | `run_a`、`run_b` | 字段 diff（data_cutoff/scores/cohort/health/…） |
| `get_risk_scenario_v1` | 读**已冻结的 15% 波动目标**情景结果（只读；**无参数**，不提供其他目标值） | — | 冻结口径下的年化/波动/MDD/仓位；返回标注 `scenario=true`；**新目标波动率须回 research workflow** |
| `generate_research_report` | 基于只读数据生成描述性报告 | `topic`（枚举） | Markdown 报告 + 引用的 snapshot/数据源清单 |

### 1.2 必须由用户明确触发的写操作（1 个）

| 工具 | 作用 | 约束 |
|---|---|---|
| `run_production_pipeline` | 运行**冻结策略**的**完整**生产流水线（刷新→清洗→健康→评分→影子→组合→快照） | **对 Agent 不接受任何参数**——包括策略参数（Top N/持有期/信号窗口/费用率/eligibility）与运行模式开关；永远执行完整正式流程 |

- `skip_refresh` / `skip_clean` **只存在于 CLI**（`python src/production_pipeline.py --skip-*`，诊断/重试用），**不注册为 Agent 工具**；
- 任何包含跳过步骤的运行**不得写 `COMPLETE`**（写 `NOT_COMPLETE_TEST` 标记，manifest `status=test`）——它们不是正式 forward 记录；
- 执行后返回新的 `run_id` 与快照引用；任何关键步骤失败返回 `aborted`，不得声称"评分成功"。

### 1.3 不暴露给 Agent 的操作

- `confirm_execution`（成交日确认 → 人工在 `live_portfolio` 执行，Agent 无法访问）
- 修改策略常量（`backtest_strategy.py` 顶部参数等）
- 影子策略转正（`shadow → live` 判定属研究决策）
- 调整显著性门槛 / NW 口径
- 重新训练模型、搜索超参数、新增回测规格
- 修改或删除任何历史快照 / ledger / scores
- 下单、申购、赎回、买卖执行

以上任何"被请求"都视为 **`not_executable`**（拒绝），并在 `errors` 中返回 `code=research_boundary`。

## 2. 统一返回结构（所有工具）

```json
{
  "status": "ok | unavailable | aborted | error | not_executable",
  "run_id": "20260921_104949",
  "data_cutoff": "2026-09-18",
  "strategy_version": "momentum-v1",
  "source_snapshot": "ml/snapshots/20260921_104949",
  "warnings": [],
  "errors": [],
  "provenance": {
    "git_commit_sha": "f8eb4c1...",
    "script_hashes": {"production_pipeline.py": "sha256...", "..." : "..."},
    "score_file_sha256": "229f7062...",
    "score_file": "ml/snapshots/20260921_104949/2026-09.csv",
    "cohort_status": "planned"
  }
}
```

- `status=ok`：可正常作答。
- `status=unavailable`：数据不足/未就绪——Agent **必须停止作答或明示不可用**，不得用旧数据替代。
- `status=aborted`：写流程（`run_production_pipeline`）在中途被中止（任一关键步骤失败）——**绝不声称"评分成功"**，前端以此字段区分"完整"与"中止"。
- `status=error`：工具执行失败（异常）——记录 `errors[].code/message`，不产出结论。
- `status=not_executable`：权限边界拒绝——提示请求属于研究假设，应转 research workflow。
- `warnings` 不阻断作答（如 shadow 因子 stale 但主策略可用），但 Agent 必须在回答中显式带上。
- `provenance` 必须能指向**具体快照**和**当时的代码哈希**——缺任一关键字段即视为不可复现，返回时补 `warnings`。

## 3. 权限矩阵

| 工具 | 读/写 | 需用户触发 | 参数来源 | 命中停止条件时的行为 |
|---|---|---|---|---|
| get_data_health | 只读 | 否 | 系统 | health=FAIL → 其自身仍可返回 FAIL 详情，但其它评分/组合工具须 unavailable |
| get_latest_complete_snapshot | 只读 | 否 | 系统 | 无 COMPLETE 快照 → unavailable |
| get_score | 只读 | 否 | **run_id（权威）或 month（便利）** | 找不到该次/该月正式快照 → unavailable + 引用最近完整快照；同月多完整快照 → 候选列表或最新并附警告 |
| get_top_funds | 只读 | 否 | run_id/month、top_n(≤200) | 同 get_score |
| get_fund_rank_history | 只读 | 否 | fund_code | 无历史月份 → 明示可用范围 |
| get_portfolio_state | 只读 | 否 | 系统 | cohort 全为 planned → 报告"未执行"，不得称已持仓 |
| get_shadow_status | 只读 | 否 | 系统 | 因子 stale → 明示最近跳过原因 |
| compare_snapshots | 只读 | 否 | run_a、run_b | 任一非完整快照 → unavailable |
| get_risk_scenario_v1 | 只读（情景） | 否 | **无参数**（固定 15%） | 返回须带 `scenario=true`；新目标波动率非本工具能力 |
| generate_research_report | 只读 | 否 | topic（枚举） | 依底层数据工具而定 |
| run_production_pipeline | 写 | **是** | **无参数（永远完整正式流程）** | health FAIL / 关键步骤失败 → aborted，不产出"成功评分" |
| （不暴露清单） | — | — | — | not_executable（research_boundary） |

## 4. 错误处理

- **错误分级**：
  - `unavailable`（数据/快照/因子未就绪，暂停作答）
  - `aborted`（写流程中途中止，附中止步骤，绝不声称成功）
  - `error`（执行异常，附 message）
  - `not_executable`（研究边界，拒绝）
- 所有 `errors[]` 元素：`{"code": "<machine_code>", "message": "<human_zh>"}`。
- **审计日志（v1.1）**：工具的每次调用（输入摘要、返回 status、耗时、触发的用户身份）写入**独立审计路径**（`logs/agent_audit/`，与流水线日志同级）；**不得修改 scores / ledger / snapshots** 任何内容——审计与生产产物物理隔离。

## 5. Agent 停止条件（六类，命中即停）

1. **快照缺少 `COMPLETE`** —— `get_latest_complete_snapshot` 找不到带 COMPLETE 标志的快照 → 停止（任何评分/组合结论都不可用）。
2. **`data_health = FAIL`** —— 阻止**新的生产运行**（`run_production_pipeline`）与**"当前最新"结论**（评分/组合类工具返回 `unavailable`）；但**不阻止读取历史完整快照**（`get_score(run_id=...)`、`compare_snapshots`、报告引用历史照常可读——系统故障时仍能审计历史记录）。
3. **shadow 因子过期** —— `get_shadow_status` 报告 stale → 影子相关结论 must 明示"最近影子评分不存在/已跳过（原因）"，不得用旧影子冒充最新。
4. **cohort 仍为 `planned`** —— `get_portfolio_state` 中 planned 列表非空 → 不得宣称"已建仓/已持仓"；只能报告"决策已生成、成交待确认"。
5. **请求涉及冻结参数变更** —— 任何 TopN/持有期/信号窗口/费率/eligibility/显著性/risk 参数的修改意图 → `not_executable`（research_boundary），提示需退出生产工作流重新预登记。
6. **找不到对应月份的正式快照** —— `get_score(month=...)` 无匹配 → `unavailable`，并返回可引用的最近完整快照。

## 6. 工具详细契约（输入/输出/失败样例见下文）

> 每个工具给出单条 JSON 示例（成功 + 失败），实现阶段照此契约写薄工具层。失败场景的**实测验证**见 `docs/PHASE4_FAILURE_SCENARIOS.md`。

### get_data_health
```json
{"status":"ok","run_id":null,"data_cutoff":"2026-09-18","strategy_version":"momentum-v1",
 "source_snapshot":null,"warnings":["shadow 因子 stale：style_index_sz399006.csv"],"errors":[],
 "provenance":{"health_file":"ml/snapshots/20260921_104949/data_health.json"},
 "detail":{"status":"WARN","score_ready":true,"gap_n":0,"stale_n":1,
           "processed":{"median":"2026-09-18","n_at_max":5317,"n_funds":5317}}}
```

### get_latest_complete_snapshot
```json
{"status":"ok","run_id":"20260921_104949",
 "source_snapshot":"ml/snapshots/20260921_104949",
 "data_cutoff":"2026-09-18","strategy_version":"momentum-v1",
 "warnings":[],"errors":[],
 "provenance":{"git_commit_sha":"f8eb4c1...","score_file_sha256":"229f7062...","cohort_status":"planned"},
 "detail":{"manifest_status":"complete","top50":["004320","002910",...], "eligible":5118}}
```
失败（无 COMPLETE）：`{"status":"unavailable","run_id":null,"data_cutoff":null,"strategy_version":"momentum-v1",
"source_snapshot":null,"warnings":[],"errors":[{"code":"no_complete_snapshot","message":"暂无带 COMPLETE 标志的正式快照"}],"provenance":null}`

### get_score(run_id | month)
> **`run_id` 为权威键**；`month` 仅为便利查询。评分永远读取 **snapshot 内 CSV**（`ml/snapshots/{run_id}/{YYYY-MM}.csv`），不读会被覆盖的 `ml/scores/YYYY-MM.csv`。
```json
{"status":"ok","run_id":"20260921_104949","data_cutoff":"2026-09-18","strategy_version":"momentum-v1",
 "source_snapshot":"ml/snapshots/20260921_104949","warnings":[],"errors":[],
 "provenance":{"score_file_sha256":"229f7062...","score_file":"ml/snapshots/20260921_104949/2026-09.csv"},
 "detail":{"month":"2026-09","eligible":5118,"main":4901,"low":217,"as_of":"2026-09-18"}}
```
失败（无该月）：`{"status":"unavailable","run_id":null,"data_cutoff":null,"strategy_version":"momentum-v1",
"source_snapshot":null,"warnings":["最近完整快照：20260921_104949"],
"errors":[{"code":"no_snapshot_for_month","message":"未找到 2026-10 的正式快照"}],"provenance":null}`
同月多个完整快照：**固定规则（v1.1 收口）**——始终选择 `score_generated_at` 最新者，并在 `warnings` 返回**全部候选 `run_id`**；显式传入 `run_id` 时严格读取指定快照（不适用自动选择）：
`"warnings":["multiple_complete_for_month：2026-09 存在 2 个完整快照（20260921_104949, 20260922_100411），已取 score_generated_at 最新（20260922_100411）"]`。

### get_top_funds(run_id | month, top_n)
```json
{"status":"ok","run_id":"20260921_104949","data_cutoff":"2026-09-18","strategy_version":"momentum-v1",
 "source_snapshot":"ml/snapshots/20260921_104949","warnings":["top_n 仅用于展示，不改变冻结策略参数"],"errors":[],
 "provenance":null,
 "detail":{"top":[{"rank":1,"fund_code":"002910","score":1.829}]}}
```
拒绝（top_n>200 或带策略参数）：`{"status":"not_executable","run_id":null,"data_cutoff":null,"strategy_version":"momentum-v1",
"source_snapshot":null,"warnings":[],"errors":[{"code":"research_boundary","message":"不允许通过工具修改策略参数"}],"provenance":null}`

### get_fund_rank_history(fund_code)
```json
{"status":"ok","run_id":null,"data_cutoff":"2026-09-18","strategy_version":"momentum-v1",
 "source_snapshot":"ml/snapshots/20260921_104949",
 "warnings":["history 仅截至当前已完成的正式快照；2027-03 后才能对照 6 个月标签"],"errors":[],
 "provenance":null,
 "detail":{"fund_code":"002910","months":[{"month":"2026-09","rank":1,"in_top50":true}]}}
```

### get_portfolio_state
```json
{"status":"ok","run_id":null,"data_cutoff":"2026-09-18","strategy_version":"momentum-v1",
 "source_snapshot":"ml/snapshots/20260921_104949",
 "warnings":["cohort 2026-09 仍为 planned（成交待确认），不得解读为已持仓"],"errors":[],
 "provenance":{"ledger":"ml/ledger/portfolio_state.json"},
 "detail":{"n_active":0,"cash_weight":1.0,"n_funds":0,"active_cohorts":[],
           "planned":[{"cohort":"2026-09","signal_date":"2026-09-18","execution_date":null}]}}
```

### get_shadow_status
```json
{"status":"unavailable","run_id":null,"data_cutoff":"2026-09-18","strategy_version":"momentum-v1",
 "source_snapshot":null,"warnings":[],
 "errors":[{"code":"shadow_factor_stale","message":"最近影子评分不存在（因子陈旧，已按规则跳过）"}],
 "provenance":{"factors":{"style_index_sz399006.csv":"2026-09-17","sw_industry.min":"2026-09-18"}},
 "detail":{"last_score":null,"skip_reason":"因子 stale：style_index_sz399006.csv 早于打分日"}}
```

### compare_snapshots(run_a, run_b)
```json
{"status":"ok","run_id":null,"data_cutoff":null,"strategy_version":"momentum-v1",
 "source_snapshot":null,"warnings":[],"errors":[],
 "provenance":{"compared":["ml/snapshots/<run_a>","ml/snapshots/<run_b>"]},
 "detail":{"diff":{"data_cutoff":["2026-09-18","<new>"],"eligible":[5118,5118]}}}
```

### get_risk_scenario_v1()（无参数；只读已冻结的 15% 结果）
```json
{"status":"ok","run_id":null,"data_cutoff":"2026-09-18","strategy_version":"momentum-v1",
 "source_snapshot":null,
 "warnings":["scenario=true：本结果仅作情景分析，不构成策略建议；vol_target 不转正"],"errors":[],
 "provenance":{"source":"ml/backtest/vol_target_dev.txt"},
 "detail":{"frozen_target_vol":0.15,"ann_ret":0.0886,"vol":0.1656,"mdd":-0.3804}}
```
> 新目标波动率（如 10%/20%）**不是本工具参数**——属研究假设，须退出生产工作流重新预登记。

### generate_research_report(topic)
```json
{"status":"ok","run_id":null,"data_cutoff":"2026-09-18","strategy_version":"momentum-v1",
 "source_snapshot":"ml/snapshots/20260921_104949","warnings":[],"errors":[],
 "provenance":{"sources":["ml/snapshots/20260921_104949"]},
 "detail":{"topic":"momentum_excess","report":"...(Markdown)"}}
```

### run_production_pipeline（用户触发，**不接受任何参数**）
成功：
```json
{"status":"ok","run_id":"<新run>","data_cutoff":"<新>","strategy_version":"momentum-v1",
 "source_snapshot":"ml/snapshots/<新run>",
 "warnings":["shadow 因子 stale，影子评分已跳过"],"errors":[],
 "provenance":{"git_commit_sha":"<...>","script_hashes":{...}},
 "detail":{"health":"WARN","score":5118,"shadow":"skipped(stale)","portfolio":"planned"}}
```
中止（任一关键步骤失败）：
```json
{"status":"aborted","run_id":"<run>","data_cutoff":null,"strategy_version":"momentum-v1",
 "source_snapshot":null,"warnings":[],
 "errors":[{"code":"run_aborted","message":"关键步骤 step_clean 失败：..."}],
 "provenance":{"aborted_at":"<时间>"}}
```
> 不产出"成功评分"；后续可审计中止位置与原因。

## 7. 实现阶段约定（本轮不实现）

- 薄工具层：单一 Python 模块，每个工具一个函数，严格按本契约返回 JSON；统一审计日志。
- 所有只读工具**只读**：不写 scores/ledger/snapshots；唯一写路径是 `run_production_pipeline`。
- `confirm_execution` 等不暴露项在工具层**不注册**，物理上不可达。
- 契约变更必须走文档更新 + 重新提交 + 重新预登记（等同于研究变更纪律）。

## 8. 边界重申（Phase 4 第一版）

- 只读查询 + 冻结 `run_production_pipeline`；
- 不接 LLM 推理为"自动执行"，不自动下单，不自动改任何参数；
- 任何与冻结参数/影子转正/训练搜索/历史快照修改有关的请求 → `not_executable`（research_boundary）。