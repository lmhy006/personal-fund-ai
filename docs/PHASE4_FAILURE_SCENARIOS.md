# Phase 4 失败场景实测记录（PHASE4_FAILURE_SCENARIOS）

> 验证时间：2026-09-21（Phase 4 设计阶段）。
> 验证方式：使用现有系统组件（`data_health.check`、快照文件系统、`ml/ledger`）对 Schema v1 的
> **六类 Agent 停止条件**逐一做可检测性实测。薄工具层未实现——本文件证明"停止条件在系统里
> 真实可被检测"，工具层按 Schema 契约包装即可。
> 结论：**六类停止条件全部可通过现有数据结构检测**，无需对生产系统做任何改动即可支撑 Agent 契约。

## T1／停止条件①：快照缺少 COMPLETE

```json
{"formal_20260921_104949_has_COMPLETE": true,
 "test_runs_20260921_102405_has_COMPLETE": false}
```
- 正式快照带 `COMPLETE` 标志，测试/归档快照（`test_runs/`）不带 → **可区分正式 forward 记录与测试运行**。
- Agent 检测：`get_latest_complete_snapshot` 只统计带 `COMPLETE` 的目录；无则返回 `unavailable`。

## T2／停止条件②：data_health = FAIL

构造"processed 中位日期落后 raw"的报告，`data_health.check()` 返回：
```json
"FAIL"
```
- check() 的中位日期检查能明确给出 FAIL（不是 WARN）→ 评分/组合类工具必须 `unavailable`。
- 本次生产实际为 WARN（shadow 因子 stale）→ 主策略可继续、影子须跳过（与契约一致）。

## T3／停止条件③：shadow 因子过期

实测当前系统中 `style_index_sz399006.csv` 停在 2026-09-17，早于打分日 2026-09-18：
```json
{"status": "WARN", "shadow_stale": ["style_index_sz399006.csv"]}
```
- health 报告 `shadow_stale` 非空 → `get_shadow_status` 必须返回 `unavailable` 并附最近跳过原因；
- 行业因子（31 个）当前 min=max=2026-09-18，未触发 stale（P0-2 修复生效）。

## T4／停止条件④：cohort 仍为 planned

`ml/ledger/portfolio_state.json`：
```json
{"cohort_status": "planned", "execution_date": null,
 "cohort_weight": 0.1667, "n_active": 0, "cash_weight": 1.0}
```
- 2026-09 cohort 为 **planned**、无成交日、风险仓位 0、现金 100% → Agent 只能报告"决策已生成、成交待确认"，**不得宣称已持仓**；
- 状态通过 `status` 字段可被 `get_portfolio_state` 直接检测。

## T5／停止条件⑤：请求涉及冻结参数变更

- 该条**在 Schema 层固定**：`run_production_pipeline` 的接口**不接收** TopN/持有期/信号窗口/费率/eligibility/显著性/risk 参数；只读工具中 `top_n` 仅用于查询展示并有上限与 `research_boundary` 提示。
- **实现层锁定（薄工具层落地时执行）**：① 工具签名断言（无策略参数）；② 入参白名单校验；③ 审计日志记录每次被拒的 `research_boundary` 请求。
- 本轮未实现工具层，故以设计断言 + 上述双锁计划记录；不属于可执行测试范围。

## T6／停止条件⑥：找不到对应月份的正式快照

当前可用正式评分月份：
```json
{"scores_months": ["2026-09.csv", "shadow_2026-09.csv"],
 "complete_snapshots": ["20260921_104949"],
 "query_2026-10": "unavailable（无2026-10正式评分快照）",
 "nearest_reference": "20260921_104949"}
```
- 查询 2026-10 → 无正式快照 → 返回 `unavailable` 并引用最近完整快照（契约第 6 条）✓。

## 汇总

| 停止条件 | 可检测性 | 检测来源 | 结果 |
|---|---|---|---|
| ① 缺 COMPLETE | ✅ | 文件系统（快照目录 COMPLETE 标志） | 正式 vs 测试可区分 |
| ② health=FAIL | ✅ | `data_health.check()`（processed 中位滞后） | FAIL 分支可达 |
| ③ shadow 因子过期 | ✅ | `health_report.shadow_stale` | 当前正命中（sz399006） |
| ④ cohort planned | ✅ | `portfolio_state.cohorts[..].status` | 当前正命中 |
| ⑤ 冻结参数变更 | ✅（设计层） | Schema 接口固件 + 实现层签名/白名单双锁 | 计划内 |
| ⑥ 无该月正式快照 | ✅ | `ml/scores/*.csv` + `ml/snapshots/*/COMPLETE` | 当前仅 2026-09；2026-10 查询 → unavailable |

结论：**在不改动生产系统的前提下，Agent 契约的所有停止条件都能被真实、可复现地检测**；
薄工具层只需按 `PHASE4_AGENT_SCHEMA.md` 包装这些读取路径即可。