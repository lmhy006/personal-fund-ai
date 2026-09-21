# Phase 4 Agent 工具契约（Schema v1）

> 状态：**设计阶段（2026-09-21 验收通过后开始）**。暂不接入 LLM、不执行真实交易、不增加任何研究参数。
> 本文是 **Agent 与生产系统的唯一契约**：任何未在本文登记的工具调用都不被允许。
> 薄工具层代码在本文定稿后再实现；聊天界面不在本阶段范围内。

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
| `get_score` | 指定月份的正式评分 | `as_of`（YYYY-MM，可选） | scores 行数、main/low 计数、as_of、文件路径 |
| `get_top_funds` | 某月正式评分的 TopN（**仅查询展示**） | `as_of`、`top_n`(≤200) | 主策略 TopN 代码/名称/分数/排名 |
| `get_fund_rank_history` | 某基金在历次正式评分中的排名与分数 | `fund_code` | 各月 rank/score/是否 Top50；无历史月份则明示可用范围 |
| `get_portfolio_state` | 当前 6-cohort ledger 聚合 | — | active cohorts、聚合权重、现金、**planned 列表**、预计费用 |
| `get_shadow_status` | 影子评分状态 | — | 因子截止日、最近影子评分/最近跳过原因、stale 标记 |
| `compare_snapshots` | 两个正式快照的 manifest 字段级对比 | `run_a`、`run_b` | 字段 diff（data_cutoff/scores/cohort/health/…） |
| `run_risk_scenario` | 用 `vol_target.py` 做**情景分析**（只读） | `target_vol`（默认 0.15；明确为情景输入） | 情景下的年化/波动/MDD/仓位；返回须标注 `scenario=true` |
| `generate_research_report` | 基于只读数据生成描述性报告 | `topic`（枚举） | Markdown 报告 + 引用的 snapshot/数据源清单 |

### 1.2 必须由用户明确触发的写操作（1 个）

| 工具 | 作用 | 约束 |
|---|---|---|
| `run_production_pipeline` | 运行**冻结策略**的完整生产流水线（刷新→清洗→健康→评分→影子→组合→快照） | **不接受任何策略参数**（Top N、持有期、信号窗口、费率率、eligibility 等一律不接收）；只接收 `skip_refresh`/`skip_clean` 等**运行模式**开关（仍以用户触发为前提） |

执行后返回新的 `run_id` 与快照引用；任何步骤 FAIL 则返回 aborted 状态，不得声称"评分成功"。

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
  "status": "ok | unavailable | error | not_executable",
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
- `status=error`：工具执行失败（异常）——记录 `errors[].code/message`，不产出结论。
- `status=not_executable`：权限边界拒绝——提示请求属于研究假设，应转 research workflow。
- `warnings` 不阻断作答（如 shadow 因子 stale 但主策略可用），但 Agent 必须在回答中显式带上。
- `provenance` 必须能指向**具体快照**和**当时的代码哈希**——缺任一关键字段即视为不可复现，返回时补 `warnings`。

## 3. 权限矩阵

| 工具 | 读/写 | 需用户触发 | 参数来源 | 命中停止条件时的行为 |
|---|---|---|---|---|
| get_data_health | 只读 | 否 | 系统 | health=FAIL → 其自身仍可返回 FAIL 详情，但其它评分/组合工具须 unavailable |
| get_latest_complete_snapshot | 只读 | 否 | 系统 | 无 COMPLETE 快照 → unavailable |
| get_score | 只读 | 否 | as_of（可空=最近） | 找不到该月正式快照 → unavailable + 引用最近完整快照 |
| get_top_funds | 只读 | 否 | as_of、top_n(≤200) | 同 get_score |
| get_fund_rank_history | 只读 | 否 | fund_code | 无历史月份 → 明示可用范围 |
| get_portfolio_state | 只读 | 否 | 系统 | cohort 全为 planned → 报告"未执行"，不得称已持仓 |
| get_shadow_status | 只读 | 否 | 系统 | 因子 stale → 明示最近跳过原因 |
| compare_snapshots | 只读 | 否 | run_a、run_b | 任一非完整快照 → unavailable |
| run_risk_scenario | 只读（情景） | 否 | target_vol（情景输入） | 返回须带 `scenario=true`，不得与策略结论混淆 |
| generate_research_report | 只读 | 否 | topic（枚举） | 依底层数据工具而定 |
| run_production_pipeline | 写 | **是** | 仅运行模式（无策略参数） | health FAIL / 关键步骤失败 → aborted，不产出"成功评分" |
| （不暴露清单） | — | — | — | not_executable（research_boundary） |

## 4. 错误处理

- **错误分级**：
  - `unavailable`（数据/快照/因子未就绪，暂停作答）
  - `error`（执行异常，附 message）
  - `not_executable`（研究边界，拒绝）
- 所有 `errors[]` 元素：`{"code": "<machine_code>", "message": "<human_zh>"}`。
- 审计：工具的每次调用（含输入摘要、返回 status、耗时）在薄工具层统一记录（落地方式在实现阶段定；schema 阶段只约定必有审计日志）。

## 5. Agent 停止条件（六类，命中即停）

1. **快照缺少 `COMPLETE`** —— `get_latest_complete_snapshot` 找不到带 COMPLETE 标志的快照 → 停止（任何评分/组合结论都不可用）。
2. **`data_health = FAIL`** —— 主策略评分/组合类工具一律 `unavailable`（health 本身可返回详情）。
3. **shadow 因子过期** —— `get_shadow_status` 报告 stale → 影子相关结论 must 明示"最近影子评分不存在/已跳过（原因）"，不得用旧影子冒充最新。
4. **cohort 仍为 `planned`** —— `get_portfolio_state` 中 planned 列表非空 → 不得宣称"已建仓/已持仓"；只能报告"决策已生成、成交待确认"。
5. **请求涉及冻结参数变更** —— 任何 TopN/持有期/信号窗口/费率/eligibility/显著性/risk 参数的修改意图 → `not_executable`（research_boundary），提示需退出生产工作流重新预登记。
6. **找不到对应月份的正式快照** —— `get_score(as_of=...)` 无匹配 → `unavailable`，并返回可引用的最近完整快照。

## 6. 工具详细契约（输入/输出/失败样例见下文）

> 每个工具给出单条 JSON 示例（成功 + 失败），实现阶段照此契约写薄工具层。失败场景的**实测验证**见 `docs/PHASE4_FAILURE_SCENARIOS.md`。

### get_data_health
```json
{"status":"ok","data_cutoff":"2026-09-18","run_id":null,
 "source_snapshot":null,"strategy_version":"momentum-v1","warnings":["shadow 因子 stale：style_index_sz399006.csv"],
 "errors":[],"provenance":{"health_file":"ml/snapshots/20260921_104949/data_health.json"},
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
失败（无 COMPLETE）：`{"status":"unavailable","errors":[{"code":"no_complete_snapshot","message":"暂无带 COMPLETE 标志的正式快照"}]}`

### get_score(as_of)
```json
{"status":"ok","run_id":"20260921_104949","data_cutoff":"2026-09-18",
 "source_snapshot":"ml/snapshots/20260921_104949",
 "provenance":{"score_file_sha256":"229f7062..."},
 "detail":{"eligible":5118,"main":4901,"low":217,"as_of":"2026-09-18"}}
```
失败（无该月正式快照）：`{"status":"unavailable","errors":[{"code":"no_snapshot_for_month","message":"未找到 2026-10 的正式快照"}],"warnings":["最近完整快照：20260921_104949"]}`

### get_top_funds(as_of, top_n)
```json
{"status":"ok","source_snapshot":"ml/snapshots/20260921_104949",
 "detail":{"top":[{"rank":1,"fund_code":"002910","score":1.829}, ...]},
 "warnings":["top_n 仅用于展示，不改变冻结策略参数"]}
```
拒绝（top_n>200 或带策略参数）：`{"status":"not_executable","errors":[{"code":"research_boundary","message":"不允许通过工具修改策略参数"}]}`

### get_fund_rank_history(fund_code)
```json
{"status":"ok","detail":{"fund_code":"002910","months":[{"as_of":"2026-09","rank":1,"in_top50":true}]},
 "warnings":["history 仅截至当前已完成的正式快照；2027-03 后才能对照 6 个月标签"]}
```

### get_portfolio_state
```json
{"status":"ok","detail":{"n_active":0,"cash_weight":1.0,"n_funds":0,
 "active_cohorts":[],"planned":[{"cohort":"2026-09","signal_date":"2026-09-18","execution_date":null}]},
 "warnings":["cohort 2026-09 仍为 planned（成交待确认），不得解读为已持仓"]}
```

### get_shadow_status
```json
{"status":"unavailable","detail":{"factors":{"style_index_sz399006.csv":"2026-09-17","sw_industry.min":"2026-09-18"},
 "last_score":null,"skip_reason":"因子 stale：style_index_sz399006.csv 早于打分日"},
 "errors":[{"code":"shadow_factor_stale","message":"最近影子评分不存在（因子陈旧，已按规则跳过）"}]}
```

### compare_snapshots(run_a, run_b)
```json
{"status":"ok","detail":{"diff":{"data_cutoff":["old","new"],"eligible":[5118,5118]}},...}
```

### run_risk_scenario(target_vol=0.15)
```json
{"status":"ok","warnings":["scenario=true：本结果仅作情景分析，不构成策略建议；vol_target 不转正"],
 "detail":{"target_vol":0.15,"ann_ret":0.0886,"vol":0.1656,"mdd":-0.3804}}
```

### generate_research_report(topic)
```json
{"status":"ok","detail":{"topic":"momentum_excess","report":"...(Markdown)","sources":["ml/snapshots/20260921_104949"]}}
```

### run_production_pipeline（用户触发）
```json
{"status":"ok","run_id":"<新run>","source_snapshot":"ml/snapshots/<新run>",
 "detail":{"health":"WARN","score":5118,"shadow":"skipped(stale)","portfolio":"planned"}}
```
中止：`{"status":"error","errors":[{"code":"step_failed","message":"步骤 step_clean 失败：..."}]}`，不产出"成功评分"。

## 7. 实现阶段约定（本轮不实现）

- 薄工具层：单一 Python 模块，每个工具一个函数，严格按本契约返回 JSON；统一审计日志。
- 所有只读工具**只读**：不写 scores/ledger/snapshots；唯一写路径是 `run_production_pipeline`。
- `confirm_execution` 等不暴露项在工具层**不注册**，物理上不可达。
- 契约变更必须走文档更新 + 重新提交 + 重新预登记（等同于研究变更纪律）。

## 8. 边界重申（Phase 4 第一版）

- 只读查询 + 冻结 `run_production_pipeline`；
- 不接 LLM 推理为"自动执行"，不自动下单，不自动改任何参数；
- 任何与冻结参数/影子转正/训练搜索/历史快照修改有关的请求 → `not_executable`（research_boundary）。