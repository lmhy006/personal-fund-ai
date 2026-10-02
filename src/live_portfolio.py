# live_portfolio.py：Phase 3.5 生产组合——6-cohort ledger（对齐冻结的回测逻辑）
#
# 背景（用户 2026-09-19）：live_score.py 只输出某个月的 Top50，而冻结策略实际是——
#   **每个月新建一个 Top50 cohort、每个 cohort 持有 6 个月、最多同时存在 6 批 cohort**。
#   因此"本月 Top50 ≠ 当前真实目标组合"，必须维护 cohort ledger。
#
# 首五个月建仓规则（本轮**写死**，不基于历史收益比较；README 同步记录）：
#   【方案 A：六个月逐步建仓】
#     第 1 月投资 1/6；第 2 月累计 2/6；……第 6 月达到满仓；
#     未投资部分**明确持有现金**（现金收益按项目既定 2%/年，但本模块只做"目标权重 + 费用"，
#     收益归属由调用方/Agent 按此权重计算）；
#     每个 cohort 的权重恒为 1/HOLD=1/6，与长期 steady state（6 批×1/6）连续。
#   说明：历史回测 dev 段 warm-up 的隐含语义是"active cohort 等权平均（1/n_held）、未建模现金"；
#   方案 A 与回测在 n_held=6 的 steady state 完全一致，仅前 5 个月语义被明确化（不为历史收益差异
#   挑选口径）。若未来要改用方案 B（首月满仓），必须先改本文件顶部常量并重新记录，**不得自行切换**。
#
# 对齐冻结逻辑（与 backtest_strategy.py）：
#   - 选股＝主策略 ret_12m 排名 Top50（组合内等权，1/50）
#   - 每月新建 1 批、持有 HOLD=6 个月后到期移出
#   - 费用＝申购 BUY_FEE=0.15% + 赎回 SELL_FEE=0.5%（组合内部换仓费口径，backtest 同款）
#   - 在持 cohort 权重**不做月度再平衡**（目标权重为建仓时权重）
#
# ⚠️ 收益聚合口径差异（2026-10-02 审计 P1-4，务必先读 docs/PHASE4_前向记账协议.md）：
#   本模块只维护"目标权重 + 执行事实"，**不计算前向收益**；
#   而历史回测（backtest_strategy.py）对当月各基金收益求**简单均值**再对 cohort 求均值，
#   并未按漂移后的份额权重聚合。因此"自然漂移"与回测聚合**不能声称完全一致**；
#   前向纸面账本必须按协议单独实现并与历史口径分列对账。当前**尚未计算可核验的前向绩效**。
#
# 时间字段（生产口径，定义见 README Phase 3.5）：
#   signal_date      ＝评分日（= data_cutoff，用截至该日的净值计算信号）
#   execution_date   ＝信号日后的下一个交易日（**production assumption**：假设信号日次日可提交、
#                      按调仓日净值成交；真实基金申购按哪日净值成交属于 limitation，见 README）
#                      注：新建 cohort 时该字段只是"预计执行日"，**状态恒为 planned**，
#                      只有显式 confirm_execution（写不可变执行事件）才转 active（审计 P1-1）。
#
# 输出：
#   ml/ledger/portfolio_ledger.csv   （每 cohort × 每基金一行，含权重与时间字段）
#   ml/ledger/portfolio_state.json   （当前 active cohorts、聚合目标权重、现金、最近动作）
#   ml/ledger/execution_events.jsonl （append-only 不可变执行事件流）
#
# 审计修复记录（2026-10-02，docs/AUDIT_PRE_PAPER_2026-10-02.md）：
#   P0-1 文本列显式 nullable string dtype（空 execution_date 曾被推断为 float64 → 首次确认 TypeError）
#   P0-2 add_month(persist=False) 纯计算路径；main 的 --no-save 真正不落盘
#   P1-1 新建 cohort 恒为 planned（不再因日历已有次日数据而自动 active）
#   P1-2 确认校验：版本化交易日历 + 来源快照完整性（manifest/评分哈希/Top50/权重）+ 生成时间
#   P1-3 事件流折叠最终执行事实（correction 参与幂等）+ correction 失败回滚 + recover 一致性核对
#        + save 原子写与唯一备份名
import argparse
import copy
import hashlib
import json
import os
import re
import shutil
from datetime import datetime

import pandas as pd

from backtest_strategy import BUY_FEE, HOLD, PROJECT_ROOT, SELL_FEE, TOP_N_DEFAULT
from panel_builder import BENCH_PATH
from trading_calendar import CALENDAR_PATH, UnknownTradingDayError, describe_trading_day

LEDGER_DIR = os.path.join(PROJECT_ROOT, "ml", "ledger")
LEDGER_PATH = os.path.join(LEDGER_DIR, "portfolio_ledger.csv")
STATE_PATH = os.path.join(LEDGER_DIR, "portfolio_state.json")
EXECUTION_EVENTS_PATH = os.path.join(LEDGER_DIR, "execution_events.jsonl")   # 不可变执行事件流（append-only）

COHORT_WEIGHT = 1.0 / HOLD          # 每批权重（方案 A：恒 1/6，含前 5 个月）
TOP_N = TOP_N_DEFAULT               # 50

LEDGER_COLS = ["cohort_id", "fund_code", "weight_in_cohort", "signal_date",
               "execution_date", "created_at", "expire_month", "status"]
# P0-1（2026-10-02 审计）：这些列必须用**可空字符串**读取。历史文件里 execution_date 全空时，
# pandas 默认推断为 float64，随后写入 "YYYY-MM-DD" 会抛
# `TypeError: Invalid value '2026-10-08' for dtype 'float64'`，使首次 paper 确认直接失败。
LEDGER_STR_COLS = ["cohort_id", "fund_code", "signal_date", "execution_date",
                   "created_at", "expire_month", "status"]


def _file_sha256(path: str) -> str | None:
    """文件 sha256（执行事件的状态哈希用）；不存在返回 None。"""
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_ledger(ledger_path: str) -> pd.DataFrame:
    """P0-1：显式可空字符串 dtype 读取 ledger（避免空文本列被推断为 float64）。"""
    if not os.path.exists(ledger_path):
        return pd.DataFrame(columns=LEDGER_COLS)
    df = pd.read_csv(ledger_path, dtype={c: "string" for c in LEDGER_STR_COLS})
    for c in LEDGER_STR_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string")
        else:
            df[c] = pd.Series([pd.NA] * len(df), dtype="string")
    if "weight_in_cohort" in df.columns:
        df["weight_in_cohort"] = pd.to_numeric(df["weight_in_cohort"], errors="coerce")
    else:
        df["weight_in_cohort"] = 0.0
    return df[LEDGER_COLS]


def month_add(key: str, n: int) -> str:
    """'YYYY-MM' + n 个月 → 'YYYY-MM'。"""
    y, m = int(key[:4]), int(key[5:7])
    i = y * 12 + (m - 1) + n
    return f"{i // 12:04d}-{i % 12 + 1:02d}"


def next_trading_day_after(d: pd.Timestamp, calendar_path: str | None = None) -> pd.Timestamp | None:
    """信号日后的下一个交易日（production assumption：可提交日）。

    P1-2（2026-10-02 审计）：优先用**版本化交易日历**（含人工登记的节后交易日），
    日历无法覆盖时退回基准日历；两者都无下一交易日 → None（不得退回 signal_date，
    否则造成 execution_date 早于 score_generated_at 的因果矛盾），由调用方标记 planned/pending。
    """
    try:
        from trading_calendar import next_trading_day as _ntd
        nxt = _ntd(d, calendar_path or CALENDAR_PATH, BENCH_PATH)
        if nxt is not None:
            return nxt
    except Exception:  # noqa: BLE001
        pass
    bc = pd.read_csv(BENCH_PATH, parse_dates=["date"])["date"]
    after = bc[bc > d]
    return pd.Timestamp(after.iloc[0]) if len(after) else None


class LivePortfolio:
    """月度 6-cohort ledger。幂等：某月 cohort 已存在则跳过（重跑安全）。"""

    def __init__(self, ledger_path=LEDGER_PATH, state_path=STATE_PATH,
                 calendar_path=None, bench_path=None, events_path=None):
        self.ledger_path = ledger_path
        self.state_path = state_path
        # 注意：默认参数在**定义时**绑定，故日历/事件路径用 None 哨兵，运行时解析模块级常量，
        # 便于测试与流水线隔离（2026-09-23 教训）。
        self.calendar_path = calendar_path
        self.bench_path = bench_path
        self.events_path = events_path
        self._persist = True            # P0-2：False 时 save() 不落盘（纯计算路径）
        self.ledger = _load_ledger(ledger_path)
        self.state = {}
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                self.state = json.load(f)
        self.state.setdefault("cohorts", {})      # cohort_id -> meta
        self.state.setdefault("last_month", None)

    # ---------------- 路径解析（运行时读取，便于隔离） ----------------
    def _events_path(self) -> str:
        return self.events_path or EXECUTION_EVENTS_PATH

    def _calendar_path(self) -> str:
        return self.calendar_path or CALENDAR_PATH

    def _bench_path(self) -> str:
        return self.bench_path or BENCH_PATH

    # ---------------- 月度动作 ----------------
    def add_month(self, scores_df: pd.DataFrame, top_n: int = TOP_N,
                  cohort_weight: float = COHORT_WEIGHT,
                  score_run: str | None = None, persist: bool = True) -> dict:
        """用本月评分快照推进一个月：识别新增/更新/到期 cohort，写入 ledger 与 state。

        同月重复正式运行（v1.1 边界修正）：
          - cohort **已成交（active/expired）** → 幂等跳过（锁定）；
          - cohort 仍为 **planned**（未成交）→ 用最新正式评分**更新选股**（signal_date / score_run 前进），
            保证"同月数据更新后，未执行的计划反映最新信号"；confirm_execution 后即锁定。

        :param score_run: 来源 run_id（如 ml/snapshots/20260930_233855），供 ledger 溯源。
        :param persist: P0-2（2026-10-02 审计）——False 时为**纯计算路径**：不写盘、不改自身状态
                        （诊断/dry-run 用，防止污染正式 planned 台账）。
        :return: 本月动作摘要（new/updated/skipped + 到期/费用）。
        """
        if persist:
            return self._add_month_impl(scores_df, top_n, cohort_weight, score_run)
        # 纯计算：在副本上跑，结束恢复内存，且 save() 被 _persist 抑制
        prev_state, prev_ledger = copy.deepcopy(self.state), self.ledger.copy()
        self._persist = False
        try:
            return self._add_month_impl(scores_df, top_n, cohort_weight, score_run)
        finally:
            self._persist = True
            self.state, self.ledger = prev_state, prev_ledger

    def _add_month_impl(self, scores_df: pd.DataFrame, top_n: int,
                        cohort_weight: float, score_run: str | None) -> dict:
        if "as_of" not in scores_df.columns or not len(scores_df):
            raise ValueError("scores_df 缺少 as_of 或为空")
        asof = pd.Timestamp(scores_df["as_of"].iloc[0])
        month_key = asof.strftime("%Y-%m")
        cur_status = self.state["cohorts"][month_key].get("status") if month_key in self.state["cohorts"] else None
        if cur_status in ("active", "expired"):
            return {"skipped": True, "month": month_key, "reason": f"cohort {month_key} 已{cur_status}（锁定）"}
        renewed = cur_status == "planned"        # 同月第二次正式评分 → 更新未执行的计划

        main = scores_df[scores_df["confidence"] == "main"] if "confidence" in scores_df else scores_df
        top = main.sort_values("rank").head(top_n) if "rank" in main.columns else \
            main.sort_values("score", ascending=False).head(top_n)
        codes = list(top["fund_code"])
        if len(codes) < 10:
            return {"skipped": True, "month": month_key,
                    "reason": f"主策略可评分基金过少（{len(codes)}）"}

        signal_date = asof.strftime("%Y-%m-%d")
        exec_date = next_trading_day_after(asof, self.calendar_path)
        exec_date_str = exec_date.strftime("%Y-%m-%d") if exec_date is not None else None
        expire_month = month_add(month_key, HOLD)
        now = datetime.now().isoformat(timespec="seconds")

        # 到期：expire_month == 本月的 **active** cohort 标记 expired
        expired_now = []
        for cid in list(self.state["cohorts"]):
            if self.state["cohorts"][cid].get("expire_month") == month_key and \
                    self.state["cohorts"][cid].get("status") == "active":
                self.state["cohorts"][cid]["status"] = "expired"
                self.ledger.loc[self.ledger["cohort_id"] == cid, "status"] = "expired"
                expired_now.append(cid)

        # 同月 planned 更新：替换选股（保留 cohort_id；expire/权重不变）
        if renewed:
            cid = month_key
            meta = self.state["cohorts"][cid]
            meta.update({
                "signal_date": signal_date, "execution_date": exec_date_str,
                "n_codes": len(codes), "updated_at": now,
            })
            if score_run:
                meta["score_run"] = score_run          # 溯源：选股来自哪个正式 run
            self.ledger = self.ledger[self.ledger["cohort_id"] != cid]   # 移除旧选股
            new_rows = pd.DataFrame([{
                "cohort_id": cid, "fund_code": c, "weight_in_cohort": cohort_weight / len(codes),
                "signal_date": signal_date, "execution_date": exec_date_str,
                "created_at": meta.get("created_at", now), "expire_month": expire_month,
                "status": meta["status"]} for c in codes])
            self.ledger = pd.concat([self.ledger, new_rows], ignore_index=True)
            self.save()
            act = self.aggregate()
            return {"status": "updated", "month": month_key, "signal_date": signal_date,
                    "execution_date": exec_date_str, "cohort_status": meta["status"],
                    "new_codes": len(codes), "expired_cohorts": sorted(expired_now),
                    "score_run": score_run, "cash_weight": act["cash_weight"],
                    "n_active_cohorts": act["n_active"]}

        # 新增：**状态恒为 planned**（P1-1，2026-10-02 审计）——execution_date 只是预计执行日，
        # 只有显式 confirm_execution（写不可变执行事件）才把状态转为 active。
        # 旧实现用 `active if exec_date is not None else planned`，只要基准/日历已含次日数据就会
        # 绕过人工确认直接 active（延迟建仓、恢复台账、事后补建月份时都会触发）。
        cid = month_key
        self.state["cohorts"][cid] = {
            "signal_date": signal_date, "execution_date": exec_date_str,
            "expire_month": expire_month, "cohort_weight": cohort_weight,
            "n_codes": len(codes), "created_at": now,
            "status": "planned",
            "execution_date_is_estimate": True,     # 预计执行日，非既成事实
        }
        if score_run:
            self.state["cohorts"][cid]["score_run"] = score_run
        new_rows = pd.DataFrame([{
            "cohort_id": cid, "fund_code": c, "weight_in_cohort": cohort_weight / len(codes),
            "signal_date": signal_date, "execution_date": exec_date_str,
            "created_at": self.state["cohorts"][cid]["created_at"],
            "expire_month": expire_month, "status": "planned"} for c in codes])
        self.ledger = pd.concat([self.ledger, new_rows], ignore_index=True)
        self.state["last_month"] = month_key
        self.save()

        act = self.aggregate()
        sell_w = cohort_weight if expired_now else 0.0
        status = self.state["cohorts"][cid]["status"]
        action = {
            "month": month_key, "signal_date": signal_date, "execution_date": exec_date_str,
            "cohort_status": status,      # planned（待确认）/ active（已确认执行）
            "new_codes": len(codes), "expired_cohorts": sorted(expired_now),
            "expired_codes": int(len(self.ledger[(self.ledger["expire_month"] == month_key)
                                                  & (self.ledger["status"] == "expired")])) if expired_now else 0,
            "buy_weight_risk": cohort_weight if status == "active" else 0.0,  # planned 未投入
            "buy_weight_planned": cohort_weight if status == "planned" else 0.0,
            "sell_weight": sell_w,
            "fee_buy": (cohort_weight * BUY_FEE) if status == "active" else 0.0,
            "fee_sell": sell_w * SELL_FEE,          # 与 backtest 一致：仅到期月收赎回费
            "cash_weight": act["cash_weight"],
            "n_active_cohorts": act["n_active"],
        }
        return action

    # ---------------- 执行确认（planned → active） ----------------
    def confirm_execution(self, cohort_id: str, execution_date: str,
                          exec_type: str = "paper", operator: str = "local_user") -> dict:
        """把 planned cohort 标记为 active（执行确认），并写入**不可变执行事件**。

        前向纸面运行期（2026-09-22 用户）：现阶段明确采用 **paper 纸面组合**（未接交易系统、
        无可靠费用后超额证据），不得把纸面 ledger 描述成真实持仓。

        幂等语义（2026-09-23 审查 P0-2；2026-10-02 审计 P1-3 改为**按事件流折叠的最终事实**判断）：
          - 仅 **planned→active** 允许转换；新建 cohort 不会自动 active（P1-1）；
          - 最终执行事实 == (execution_date, exec_type) 的重复调用返回既有事实（`idempotent=True`），
            **不追加**新事件（含"先修正再按最终事实重试"的情形）；
          - 最终事实不同 → **拒绝**，需显式 `correct_execution` 写 `execution_correction`；
          - 校验（P1-2）：格式、晚于 signal_date、不早于今天（不确认未来执行日）、
            **版本化交易日历**（非交易日拒绝）、**来源快照完整性**
            （run_id/status/signal_mode/signal_date/评分哈希/Top50/台账代码与权重一致）、
            来源生成时间早于所声称的执行时点。
        :param exec_type: "paper"（默认）| "actual"
        :param operator: 操作人（对话适配层/人工提供；默认 local_user）
        :return: 事件记录（首次确认已 append 到 execution_events.jsonl；幂等重试返回既有事实）
        """
        if cohort_id not in self.state["cohorts"]:
            raise KeyError(f"cohort {cohort_id} 不存在")
        if exec_type not in ("paper", "actual"):
            raise ValueError(f"exec_type 仅允许 paper/actual，收到 {exec_type!r}")
        meta = self.state["cohorts"][cohort_id]
        if meta.get("status") == "expired":
            raise ValueError(f"cohort {cohort_id} 已 expired，不能确认执行")

        ctx = self._validate_execution(meta, execution_date, cohort_id)
        fact = self._current_execution_fact(cohort_id)

        if meta.get("status") == "active":
            if fact is None:
                raise ValueError(
                    f"cohort {cohort_id} 状态为 active 但事件流中无确认事实（历史失败窗口）——"
                    f"请先用 recover_execution 恢复，或人工核对后再操作")
            if fact["execution_date"] == execution_date and fact["exec_type"] == exec_type:
                return {**fact["event"], "idempotent": True}      # 最终事实一致：不追加事件
            raise ValueError(
                f"cohort {cohort_id} 已 active，最终事实为 {fact['exec_type']}/{fact['execution_date']}；"
                f"本次请求 {exec_type}/{execution_date} 与之不符——如需修正请用 correct_execution "
                f"写显式 execution_correction 事件")

        # planned → active：先算好待提交状态，再落盘，最后写事件；事件失败则回滚
        new_meta = dict(meta)
        new_meta["execution_date"] = execution_date
        new_meta["status"] = "active"
        new_meta["exec_type"] = exec_type
        new_meta["execution_confirmed_at"] = datetime.now().isoformat(timespec="seconds")
        new_meta["execution_validation"] = ctx
        new_meta.pop("execution_date_is_estimate", None)
        self.state["cohorts"][cohort_id] = new_meta
        self.ledger.loc[self.ledger["cohort_id"] == cohort_id, "execution_date"] = execution_date
        self.ledger.loc[self.ledger["cohort_id"] == cohort_id, "status"] = "active"
        self.save()
        ev = self._build_exec_event(exec_type, cohort_id, execution_date, operator,
                                    validation=ctx, rollback_to=meta)
        try:
            self._append_event(ev)
        except Exception as e:  # noqa: BLE001
            # 事件追加失败 → **回滚为 planned**，避免"active 无事件"的不一致态
            self._restore_cohort(cohort_id, meta)
            raise RuntimeError(f"执行事件写入失败，已回滚为 planned（{e}）") from e
        self._verify_exec_event(ev)
        return ev

    def correct_execution(self, cohort_id: str, new_execution_date: str, new_exec_type: str,
                          reason: str, operator: str = "local_user") -> dict:
        """**显式修正**已确认的执行事实：写 `execution_correction` 事件并更新状态。

        P1-3（2026-10-02 审计）：
          - 修正后的**最终事实**参与幂等：相同修正重试返回既有事件、不追加；
          - 事件追加失败 → 回滚状态并重新保存（与 confirm 同等对待），不留"状态已改、事件未写"。
        仅允许对 active cohort 修正日期/类型；校验同 confirm_execution。
        """
        if cohort_id not in self.state["cohorts"]:
            raise KeyError(f"cohort {cohort_id} 不存在")
        if new_exec_type not in ("paper", "actual"):
            raise ValueError(f"new_exec_type 仅允许 paper/actual，收到 {new_exec_type!r}")
        if not reason or not str(reason).strip():
            raise ValueError("reason 必填（显式修正必须说明原因）")
        meta = self.state["cohorts"][cohort_id]
        if meta.get("status") != "active":
            raise ValueError(f"cohort {cohort_id} 尚未 active，无需修正（先 confirm_execution）")
        ctx = self._validate_execution(meta, new_execution_date, cohort_id)

        fact = self._current_execution_fact(cohort_id)
        if fact and fact["execution_date"] == new_execution_date and fact["exec_type"] == new_exec_type:
            return {**fact["event"], "idempotent": True}       # 相同修正重试：不追加

        old_date, old_type = meta.get("execution_date"), meta.get("exec_type")
        new_meta = dict(meta)
        new_meta["execution_date"] = new_execution_date
        new_meta["exec_type"] = new_exec_type
        new_meta["execution_validation"] = ctx
        self.state["cohorts"][cohort_id] = new_meta
        self.ledger.loc[self.ledger["cohort_id"] == cohort_id, "execution_date"] = new_execution_date
        self.save()
        ev = {
            "event_id": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            "type": "execution_correction",
            "cohort": cohort_id,
            "source_run_id": meta.get("score_run"),
            "confirmed_at": datetime.now().isoformat(timespec="seconds"),
            "execution_date": new_execution_date,
            "exec_type": new_exec_type,
            "operator": operator,
            "previous": {"execution_date": old_date, "exec_type": old_type},
            "reason": str(reason).strip(),
            "validation": ctx,
            "ledger_sha256": _file_sha256(self.ledger_path),
            "state_sha256": _file_sha256(self.state_path),
        }
        try:
            self._append_event(ev)
        except Exception as e:  # noqa: BLE001
            self._restore_cohort(cohort_id, meta)
            raise RuntimeError(f"修正事件写入失败，已回滚状态（{e}）") from e
        self._verify_exec_event(ev)
        return ev

    def recover_execution(self, cohort_id: str, operator: str = "local_user",
                          exec_type: str | None = None,
                          execution_date: str | None = None) -> dict:
        """恢复：核对**状态与事件流是否一致**（P1-3，2026-10-02 审计）。

        - active 且事件流无事实 → 按当前状态补写事件（历史失败窗口）；
        - active 且事实与状态一致 → 无需恢复（返回既有事件）；
        - active 但事实与状态**矛盾** → 拒绝并提示用 correct_execution（不能只判断"事件存在"）；
        - planned 但有事实 → 矛盾，拒绝（状态被人为/异常改回）。
        """
        if cohort_id not in self.state["cohorts"]:
            raise KeyError(f"cohort {cohort_id} 不存在")
        meta = self.state["cohorts"][cohort_id]
        fact = self._current_execution_fact(cohort_id)
        status = meta.get("status")
        if status == "active":
            if fact is None:
                ev = self._build_exec_event(exec_type or meta.get("exec_type") or "paper",
                                            cohort_id,
                                            execution_date or meta.get("execution_date"),
                                            operator, rollback_to=meta)
                ev["recovered_from_missing_event"] = True
                self._append_event(ev)
                self._verify_exec_event(ev)
                return ev
            same = (fact["execution_date"] == meta.get("execution_date")
                    and fact["exec_type"] == (meta.get("exec_type") or fact["exec_type"]))
            if same:
                return {**fact["event"], "recovered": False, "note": "状态与事件一致，无需恢复"}
            raise ValueError(
                f"cohort {cohort_id} 状态（{meta.get('exec_type')}/{meta.get('execution_date')}）与事件流"
                f"最终事实（{fact['exec_type']}/{fact['execution_date']}）矛盾——请用 correct_execution 修正")
        if fact is not None:
            raise ValueError(
                f"cohort {cohort_id} 状态为 {status}，但事件流已有确认事实 "
                f"（{fact['exec_type']}/{fact['execution_date']}）——状态与事件矛盾，请人工核对")
        raise ValueError("只有 active 且缺事件的 cohort 需要恢复（planned 请直接 confirm_execution）")

    # ---- 执行确认辅助 ----
    def _restore_cohort(self, cohort_id: str, meta: dict):
        """把 cohort 回滚到给定 meta（事件写入失败时用），并重新落盘。"""
        self.state["cohorts"][cohort_id] = meta
        self.ledger.loc[self.ledger["cohort_id"] == cohort_id, "execution_date"] = meta.get("execution_date")
        self.ledger.loc[self.ledger["cohort_id"] == cohort_id, "status"] = meta.get("status")
        self.save()

    def _validate_execution(self, meta: dict, execution_date: str, cohort_id: str) -> dict:
        """校验执行日与来源（P1-2，2026-10-02 审计）；返回校验证据（写入事件留痕）。"""
        try:
            d = pd.Timestamp(execution_date)
        except Exception:  # noqa: BLE001
            raise ValueError(f"execution_date 需为合法日期 YYYY-MM-DD，收到 {execution_date!r}") from None
        if d.strftime("%Y-%m-%d") != execution_date:
            raise ValueError(f"execution_date 格式须为 YYYY-MM-DD，收到 {execution_date!r}")
        sig = pd.Timestamp(meta.get("signal_date"))
        if d <= sig:
            raise ValueError(f"execution_date（{execution_date}）必须晚于 signal_date（{meta.get('signal_date')}）")
        if d.date() > datetime.now().date():
            raise ValueError(f"不允许提前确认尚未发生的执行日期（{execution_date} 在未来）")

        # 版本化交易日历（P1-2）：非交易日（含周末/节假日/未登记日期）一律拒绝
        try:
            cal = describe_trading_day(execution_date, self._calendar_path(), self._bench_path())
        except UnknownTradingDayError as e:
            raise ValueError(f"execution_date 无法用版本化交易日历判定：{e}") from e
        if not cal["is_trading_day"]:
            raise ValueError(f"execution_date（{execution_date}）不是交易日（{cal['source']}），不能作为执行日")

        snap = self._validate_source_snapshot(meta, cohort_id)
        gen = snap["manifest"].get("score_generated_at")
        if gen and pd.Timestamp(gen).date() > d.date():
            raise ValueError(
                f"来源快照生成时间（{gen}）晚于执行日（{execution_date}）——因果矛盾，拒绝确认")
        return {"calendar": cal, "source_snapshot": {k: v for k, v in snap.items() if k != "manifest"},
                "score_generated_at": gen,
                "recorded_at": datetime.now().isoformat(timespec="seconds")}

    def _validate_source_snapshot(self, meta: dict, cohort_id: str) -> dict:
        """核对来源快照与台账一致性（P1-2）：run_id/status/signal_mode/signal_date/评分哈希/Top50/权重。"""
        src = meta.get("score_run")
        if not src:
            raise ValueError("cohort 缺少来源 run_id（score_run），无法校验正式快照")
        snap_dir = os.path.join(PROJECT_ROOT, "ml", "snapshots", src)
        complete = os.path.join(snap_dir, "COMPLETE")
        mf_path = os.path.join(snap_dir, "manifest.json")
        if not os.path.exists(complete):
            raise ValueError(f"来源 run_id={src} 不是正式完整快照（缺 COMPLETE），不能确认执行")
        if not os.path.exists(mf_path):
            raise ValueError(f"来源 run_id={src} 缺 manifest.json，不能确认执行")
        with open(mf_path, "r", encoding="utf-8") as f:
            mf = json.load(f)

        if mf.get("run_id") != src:
            raise ValueError(f"来源快照 run_id（{mf.get('run_id')}）与 cohort.score_run（{src}）不一致")
        if mf.get("status") != "complete":
            raise ValueError(f"来源快照 status={mf.get('status')!r} ≠ complete，不能确认执行")
        if mf.get("signal_mode") != "month_end":
            raise ValueError(
                f"来源快照 signal_mode={mf.get('signal_mode')!r} ≠ month_end（非月末正式信号），不能确认执行")
        if mf.get("signal_date") != meta.get("signal_date"):
            raise ValueError(
                f"来源快照 signal_date（{mf.get('signal_date')}）与 cohort.signal_date"
                f"（{meta.get('signal_date')}）不一致——计划与来源漂移，拒绝确认")

        # 评分文件哈希（快照内副本 vs manifest 记录）
        score_hash = mf.get("score_file_sha256")
        score_name = None
        if score_hash:
            for fn in os.listdir(snap_dir):
                if re.fullmatch(r"\d{4}-\d{2}\.csv", fn):
                    score_name = fn
                    break
            if score_name is None:
                raise ValueError(f"来源快照缺评分文件（{snap_dir}），无法核对评分哈希")
            if _file_sha256(os.path.join(snap_dir, score_name)) != score_hash:
                raise ValueError(f"来源快照评分文件哈希与 manifest 记录不符（{score_name}）")

        # Top50 与台账选股（有序）一致
        rows = self.ledger[self.ledger["cohort_id"] == cohort_id]
        if not len(rows):
            raise ValueError(f"台账中找不到 cohort {cohort_id} 的选股行，无法确认执行")
        ledger_codes = [str(c) for c in rows["fund_code"].tolist()]
        mf_top = [str(c) for c in (mf.get("top50") or [])]
        if mf_top and ledger_codes != mf_top:
            diff = set(ledger_codes) ^ set(mf_top)
            raise ValueError(
                f"台账选股与来源快照 Top50 不一致（长度 {len(ledger_codes)} vs {len(mf_top)}；"
                f"差异样例 {sorted(diff)[:5]}）——选股/来源漂移，拒绝确认")
        # 权重合计 ≈ cohort_weight
        wsum = float(rows["weight_in_cohort"].sum())
        cw = meta.get("cohort_weight")
        if cw is not None and abs(wsum - float(cw)) > 1e-9:
            raise ValueError(f"台账权重合计（{wsum}）≠ cohort_weight（{cw}）——权重漂移，拒绝确认")
        return {"snapshot": src, "score_file": score_name, "score_file_sha256": score_hash,
                "top50_size": len(mf_top), "weight_sum": wsum, "manifest": mf}

    def _verify_exec_event(self, ev: dict):
        """确认/修正后自检：事件已落盘、事件哈希与当前 ledger/state 一致。"""
        found = False
        p = self._events_path()
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if json.loads(line).get("event_id") == ev["event_id"]:
                        found = True
                        break
        if not found:
            raise RuntimeError("执行事件自检失败：事件未落盘（状态已 active，请用 recover_execution 恢复）")
        if ev.get("ledger_sha256") != _file_sha256(self.ledger_path) or \
                ev.get("state_sha256") != _file_sha256(self.state_path):
            raise RuntimeError("执行事件自检失败：事件中状态哈希与当前 ledger/state 不一致")

    def _current_execution_fact(self, cohort_id: str) -> dict | None:
        """从事件流**折叠出最终执行事实**（P1-3）：paper/actual 设置事实，correction 覆盖事实。

        :return: {"execution_date","exec_type","event","event_id"} 或 None
        """
        p = self._events_path()
        if not os.path.exists(p):
            return None
        fact = None
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                if ev.get("cohort") != cohort_id:
                    continue
                if ev.get("type") in ("paper", "actual"):
                    fact = {"execution_date": ev.get("execution_date"), "exec_type": ev.get("type"),
                            "event": ev, "event_id": ev.get("event_id")}
                elif ev.get("type") == "execution_correction":
                    fact = {"execution_date": ev.get("execution_date"),
                            "exec_type": ev.get("exec_type") or (fact or {}).get("exec_type"),
                            "event": ev, "event_id": ev.get("event_id")}
        return fact

    def _build_exec_event(self, exec_type: str, cohort_id: str, execution_date: str,
                          operator: str, validation: dict | None = None,
                          rollback_to: dict | None = None) -> dict:
        meta = rollback_to or self.state["cohorts"][cohort_id]
        now = datetime.now()
        ev = {
            "event_id": now.strftime("%Y%m%d_%H%M%S_%f"),
            "type": exec_type,
            "cohort": cohort_id,
            "source_run_id": meta.get("score_run"),
            "confirmed_at": now.isoformat(timespec="seconds"),
            # 模拟成交日（execution_date）与**登记时刻**（confirmed_at）分开记录；
            # 迟于成交日登记时显式给出 lag，便于区分"执行事实"与"登记时间"。
            "recorded_at": now.isoformat(timespec="seconds"),
            "execution_date": execution_date,
            "operator": operator,
            "ledger_sha256": _file_sha256(self.ledger_path),
            "state_sha256": _file_sha256(self.state_path),
        }
        if execution_date:
            try:
                lag = (now.date() - pd.Timestamp(execution_date).date()).days
                ev["execution_lag_days"] = lag
                if lag > 0:
                    ev["note"] = (f"登记晚于所声称的模拟成交日 {lag} 天——成交日按原信号/计划记录，"
                                  f"未因登记延迟而改写")
            except Exception:  # noqa: BLE001
                pass
        if validation:
            ev["validation"] = validation
        return ev

    def _append_event(self, ev: dict):
        os.makedirs(os.path.dirname(self._events_path()) or ".", exist_ok=True)
        with open(self._events_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    @property
    def first_month(self) -> str | None:
        cs = [c for c in self.state["cohorts"]]
        return min(cs) if cs else None

    # ---------------- 聚合 ----------------
    def aggregate(self) -> dict:
        """当前目标组合：各基金聚合权重、现金、**status=active** cohort 摘要。

        planned（成交日未确认）的 cohort 不参与风险权重——现金仍全仓等待。
        """
        active = [c for c, m in self.state["cohorts"].items() if m.get("status") == "active"]
        per_fund = {}
        for cid in active:
            m = self.state["cohorts"][cid]
            rows = self.ledger[(self.ledger["cohort_id"] == cid)
                               & (self.ledger["status"] == "active")]
            for r in rows.itertuples(index=False):
                per_fund[r.fund_code] = per_fund.get(r.fund_code, 0.0) + float(r.weight_in_cohort)
        cash = max(0.0, 1.0 - len(active) * COHORT_WEIGHT)
        return {
            "n_active": len(active), "active_cohorts": sorted(active),
            "n_funds": len(per_fund), "target_weights": dict(sorted(per_fund.items(),
                                                                    key=lambda x: -x[1])),
            "cash_weight": round(cash, 6),
            "sum_weights": round(sum(per_fund.values()) + cash, 6),
        }

    def save(self):
        """落盘 ledger/state（P1-3：临时文件 + 原子替换；备份名含微秒，避免同秒覆盖）。

        `_persist=False`（纯计算路径，P0-2）时**不做任何写盘**。
        写前把上一版 ledger/state 备份到 .bak/（含唯一时间戳，可回滚到上一完整版本）。
        """
        if not getattr(self, "_persist", True):
            return
        os.makedirs(os.path.dirname(self.ledger_path) or ".", exist_ok=True)
        bak_dir = os.path.join(os.path.dirname(self.ledger_path) or ".", ".bak")
        if os.path.exists(self.ledger_path):
            os.makedirs(bak_dir, exist_ok=True)
            tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")     # 微秒：同秒多次 save 不互相覆盖
            shutil.copy2(self.ledger_path, os.path.join(bak_dir, f"ledger_{tag}.csv"))
            if os.path.exists(self.state_path):
                shutil.copy2(self.state_path, os.path.join(bak_dir, f"state_{tag}.json"))
        # 原子写 ledger
        ltmp = self.ledger_path + ".tmp"
        self.ledger.to_csv(ltmp, index=False, encoding="utf-8-sig")
        os.replace(ltmp, self.ledger_path)
        # state：先刷新 portfolio 聚合，再一次原子写（旧实现写两次 JSON，存在半写窗口）
        self.state["portfolio"] = self.aggregate()
        stmp = self.state_path + ".tmp"
        with open(stmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
        os.replace(stmp, self.state_path)


def main():
    ap = argparse.ArgumentParser(description="Phase 3.5 生产组合：6-cohort ledger 维护 / 纸面执行确认")
    ap.add_argument("--scores", help="本月评分快照 CSV（ml/scores/YYYY-MM.csv）；缺省取最近一份")
    ap.add_argument("--confirm-cohort", help="确认执行某个 planned cohort（不填=走 add_month 建仓/更新）")
    ap.add_argument("--execution-date", help="执行确认的实际可交易日 YYYY-MM-DD")
    ap.add_argument("--type", dest="exec_type", default="paper", choices=("paper", "actual"),
                    help="执行类型（默认 paper 纸面组合；actual 仅在未来接真实交易后使用）")
    ap.add_argument("--operator", default="local_user", help="操作人（人工提供）")
    ap.add_argument("--no-save", action="store_true",
                    help="纯计算：不写盘、不改正式 ledger/state（P0-2 审计）")
    args = ap.parse_args()

    pf = LivePortfolio()
    if args.confirm_cohort:
        if not args.execution_date:
            raise SystemExit("--confirm-cohort 必须配合 --execution-date YYYY-MM-DD")
        ev = pf.confirm_execution(args.confirm_cohort, args.execution_date,
                                  args.exec_type, args.operator)
        print("== live_portfolio 执行确认 ==")
        print("  事件:", json.dumps(ev, ensure_ascii=False))
        print(f"  events 追加至: {pf._events_path()}")
        return

    scores_path = args.scores
    if scores_path is None:
        cands = sorted([f for f in os.listdir(os.path.join(PROJECT_ROOT, "ml", "scores"))
                        if re.fullmatch(r"\d{4}-\d{2}\.csv", f)], reverse=True)
        if not cands:
            raise SystemExit("未找到评分快照，请先运行 live_score / production_pipeline")
        scores_path = os.path.join(PROJECT_ROOT, "ml", "scores", cands[0])
    scores = pd.read_csv(scores_path, dtype={"fund_code": str})
    # P0-2：--no-save 现在真正抑制持久化（纯计算路径）
    action = pf.add_month(scores, persist=not args.no_save)
    agg = pf.aggregate()
    print("== live_portfolio ==")
    print("  本月动作:", json.dumps(action, ensure_ascii=False))
    print("  当前:", json.dumps(agg, ensure_ascii=False)[:400])
    print(f"  ledger: {pf.ledger_path}（{len(pf.ledger)} 行）| state: {pf.state_path}")
    if args.no_save:
        print("  （--no-save：纯计算，未写入任何文件）")


if __name__ == "__main__":
    main()
