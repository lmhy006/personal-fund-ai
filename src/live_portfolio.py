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
#   - 在持 cohort 权重**不做月度再平衡**（自然漂移，与回测一致）；本模块报"目标权重"为建仓时权重
#
# 时间字段（生产口径，定义见 README Phase 3.5）：
#   signal_date      ＝评分日（= data_cutoff，用截至该日的净值计算信号）
#   execution_date   ＝信号日后的下一个基准交易日（**production assumption**：假设信号日次日可提交、
#                      按调仓日净值成交；真实基金申购按哪日净值成交属于 limitation，见 README）
#
# 输出：
#   ml/ledger/portfolio_ledger.csv   （每 cohort × 每基金一行，含权重与时间字段）
#   ml/ledger/portfolio_state.json   （当前 active cohorts、聚合目标权重、现金、最近动作、预计费用）
import argparse
import hashlib
import json
import os
from datetime import datetime

import pandas as pd

from backtest_strategy import BUY_FEE, HOLD, PROJECT_ROOT, SELL_FEE, TOP_N_DEFAULT
from panel_builder import BENCH_PATH

LEDGER_DIR = os.path.join(PROJECT_ROOT, "ml", "ledger")
LEDGER_PATH = os.path.join(LEDGER_DIR, "portfolio_ledger.csv")
STATE_PATH = os.path.join(LEDGER_DIR, "portfolio_state.json")
EXECUTION_EVENTS_PATH = os.path.join(LEDGER_DIR, "execution_events.jsonl")   # 不可变执行事件流（append-only）

COHORT_WEIGHT = 1.0 / HOLD          # 每批权重（方案 A：恒 1/6，含前 5 个月）
TOP_N = TOP_N_DEFAULT               # 50


def _file_sha256(path: str) -> str | None:
    """文件 sha256（执行事件的状态哈希用）；不存在返回 None。"""
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

LEDGER_COLS = ["cohort_id", "fund_code", "weight_in_cohort", "signal_date",
               "execution_date", "created_at", "expire_month", "status"]


def month_add(key: str, n: int) -> str:
    """'YYYY-MM' + n 个月 → 'YYYY-MM'。"""
    y, m = int(key[:4]), int(key[5:7])
    i = y * 12 + (m - 1) + n
    return f"{i // 12:04d}-{i % 12 + 1:02d}"


def next_trading_day_after(d: pd.Timestamp) -> pd.Timestamp | None:
    """信号日后的下一个基准交易日（production assumption：可提交日）。

    基准日历止于信号日时返回 None——**不得退回 signal_date**（会造成
    execution_date 早于 score_generated_at 的因果矛盾），由调用方标记为 planned/pending。
    """
    bc = pd.read_csv(BENCH_PATH, parse_dates=["date"])["date"]
    after = bc[bc > d]
    return pd.Timestamp(after.iloc[0]) if len(after) else None


class LivePortfolio:
    """月度 6-cohort ledger。幂等：某月 cohort 已存在则跳过（重跑安全）。"""

    def __init__(self, ledger_path=LEDGER_PATH, state_path=STATE_PATH):
        self.ledger_path = ledger_path
        self.state_path = state_path
        if os.path.exists(ledger_path):
            self.ledger = pd.read_csv(ledger_path, dtype={"fund_code": str})
        else:
            self.ledger = pd.DataFrame(columns=LEDGER_COLS)
        self.state = {}
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                self.state = json.load(f)
        self.state.setdefault("cohorts", {})      # cohort_id -> meta
        self.state.setdefault("last_month", None)

    # ---------------- 月度动作 ----------------
    def add_month(self, scores_df: pd.DataFrame, top_n: int = TOP_N,
                  cohort_weight: float = COHORT_WEIGHT,
                  score_run: str | None = None) -> dict:
        """用本月评分快照推进一个月：识别新增/更新/到期 cohort，写入 ledger 与 state。

        同月重复正式运行（v1.1 边界修正）：
          - cohort **已成交（active/expired）** → 幂等跳过（锁定）；
          - cohort 仍为 **planned**（未成交）→ 用最新正式评分**更新选股**（signal_date / score_run 前进），
            保证"同月数据更新后，未执行的计划反映最新信号"；confirm_execution 后即锁定。

        :param score_run: 来源 run_id（如 ml/snapshots/20260922_100411），供 ledger 溯源。
        :return: 本月动作摘要（new/updated/skipped + 到期/费用）。
        """
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
        exec_date = next_trading_day_after(asof)
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

        # 新增：无已确认成交日 → status="planned"（不参与风险权重，现金仍持有）；
        # 成交日确认后由 confirm_execution 更新为 "active"
        cid = month_key
        self.state["cohorts"][cid] = {
            "signal_date": signal_date, "execution_date": exec_date_str,
            "expire_month": expire_month, "cohort_weight": cohort_weight,
            "n_codes": len(codes), "created_at": now,
            "status": "active" if exec_date is not None else "planned",
        }
        if score_run:
            self.state["cohorts"][cid]["score_run"] = score_run
        new_rows = pd.DataFrame([{
            "cohort_id": cid, "fund_code": c, "weight_in_cohort": cohort_weight / len(codes),
            "signal_date": signal_date, "execution_date": exec_date_str,
            "created_at": self.state["cohorts"][cid]["created_at"],
            "expire_month": expire_month, "status": self.state["cohorts"][cid]["status"]} for c in codes])
        self.ledger = pd.concat([self.ledger, new_rows], ignore_index=True)
        self.state["last_month"] = month_key
        self.save()

        act = self.aggregate()
        sell_w = cohort_weight if expired_now else 0.0
        status = self.state["cohorts"][cid]["status"]
        action = {
            "month": month_key, "signal_date": signal_date, "execution_date": exec_date_str,
            "cohort_status": status,      # planned（成交日未确认）/ active
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

    def confirm_execution(self, cohort_id: str, execution_date: str,
                          exec_type: str = "paper", operator: str = "local_user") -> dict:
        """把 planned cohort 标记为 active（执行确认），并写入**不可变执行事件**。

        前向纸面运行期（2026-09-22 用户）：现阶段明确采用 **paper 纸面组合**（未接交易系统、
        无可靠费用后超额证据），不得把纸面 ledger 描述成真实持仓。
        幂等语义（2026-09-23 用户审查 P0-2）：
          - 仅 **planned→active** 允许转换；
          - **完全相同**的重复调用返回既有事件（`idempotent=True`），**不追加**新事件；
          - active 后不同日期/类型的调用**拒绝**，需显式 `correct_execution` 写
            `execution_correction` 事件修正；
          - 校验：execution_date 格式 YYYY-MM-DD、晚于 signal_date、**不早于今天**
            （不许提前确认尚未发生的执行日期）、来源 run_id 对应正式完整快照（COMPLETE）。
        :param exec_type: "paper"（默认）| "actual"
        :param operator: 操作人（对话适配层/人工提供；默认 local_user）
        :return: 事件记录（首次确认已 append 到 execution_events.jsonl；幂等重试返回既有事件）
        """
        if cohort_id not in self.state["cohorts"]:
            raise KeyError(f"cohort {cohort_id} 不存在")
        if exec_type not in ("paper", "actual"):
            raise ValueError(f"exec_type 仅允许 paper/actual，收到 {exec_type!r}")
        meta = self.state["cohorts"][cohort_id]
        if meta.get("status") == "expired":
            raise ValueError(f"cohort {cohort_id} 已 expired，不能确认执行")
        self._validate_execution(meta, execution_date)

        if meta.get("status") == "active":
            prev = self._latest_exec_event(cohort_id)
            if prev and prev.get("execution_date") == execution_date and prev.get("type") == exec_type:
                return {**prev, "idempotent": True}          # 完全相同重试：不追加事件
            raise ValueError(
                f"cohort {cohort_id} 已 active 且确认内容不同（现有 {prev.get('type') if prev else '?'}/"
                f"{prev.get('execution_date') if prev else '?'}）——如需修正请用 correct_execution "
                f"写显式 execution_correction 事件")

        meta["execution_date"] = execution_date
        meta["status"] = "active"
        meta["exec_type"] = exec_type
        self.state["cohorts"][cohort_id] = meta
        self.ledger.loc[self.ledger["cohort_id"] == cohort_id, "execution_date"] = execution_date
        self.ledger.loc[self.ledger["cohort_id"] == cohort_id, "status"] = "active"
        self.save()
        ev = self._build_exec_event("paper" if exec_type == "paper" else "actual",
                                    cohort_id, execution_date, operator)
        self._append_event(ev)
        return ev

    def correct_execution(self, cohort_id: str, new_execution_date: str, new_exec_type: str,
                          reason: str, operator: str = "local_user") -> dict:
        """**显式修正**已确认的执行事实：写 `execution_correction` 事件并更新状态（幂等修正可重复）。

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
        self._validate_execution(meta, new_execution_date)
        old_date, old_type = meta.get("execution_date"), meta.get("exec_type")
        meta["execution_date"] = new_execution_date
        meta["exec_type"] = new_exec_type
        self.state["cohorts"][cohort_id] = meta
        self.ledger.loc[self.ledger["cohort_id"] == cohort_id, "execution_date"] = new_execution_date
        self.save()
        ev = {
            "event_id": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            "type": "execution_correction",
            "cohort": cohort_id,
            "source_run_id": meta.get("score_run"),
            "confirmed_at": datetime.now().isoformat(timespec="seconds"),
            "execution_date": new_execution_date,
            "operator": operator,
            "previous": {"execution_date": old_date, "exec_type": old_type},
            "reason": str(reason).strip(),
            "ledger_sha256": _file_sha256(self.ledger_path),
            "state_sha256": _file_sha256(self.state_path),
        }
        self._append_event(ev)
        return ev

    # ---- 执行确认辅助 ----
    def _validate_execution(self, meta: dict, execution_date: str):
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
        src = meta.get("score_run")
        if not src:
            raise ValueError("cohort 缺少来源 run_id（score_run），无法校验正式快照")
        snap_dir = os.path.join(PROJECT_ROOT, "ml", "snapshots", src)
        if not os.path.exists(os.path.join(snap_dir, "COMPLETE")):
            raise ValueError(f"来源 run_id={src} 不是正式完整快照（缺 COMPLETE），不能确认执行")

    def _latest_exec_event(self, cohort_id: str) -> dict | None:
        if not os.path.exists(EXECUTION_EVENTS_PATH):
            return None
        last = None
        with open(EXECUTION_EVENTS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                if ev.get("cohort") == cohort_id and ev.get("type") in ("paper", "actual"):
                    last = ev
        return last

    def _build_exec_event(self, exec_type: str, cohort_id: str, execution_date: str,
                          operator: str) -> dict:
        meta = self.state["cohorts"][cohort_id]
        return {
            "event_id": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            "type": exec_type,
            "cohort": cohort_id,
            "source_run_id": meta.get("score_run"),
            "confirmed_at": datetime.now().isoformat(timespec="seconds"),
            "execution_date": execution_date,
            "operator": operator,
            "ledger_sha256": _file_sha256(self.ledger_path),
            "state_sha256": _file_sha256(self.state_path),
        }

    def _append_event(self, ev: dict):
        os.makedirs(LEDGER_DIR, exist_ok=True)
        with open(EXECUTION_EVENTS_PATH, "a", encoding="utf-8") as f:
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
        """写入前先把上一版 ledger/state 备份到 .bak/（回滚副本）——若生产流水线后续步骤失败，
        可用最近一次完整状态回滚，避免半次运行污染正式 ledger。"""
        os.makedirs(LEDGER_DIR, exist_ok=True)
        bak_dir = os.path.join(LEDGER_DIR, ".bak")
        if os.path.exists(self.ledger_path):
            os.makedirs(bak_dir, exist_ok=True)
            import shutil, time as _t
            tag = _t.strftime("%Y%m%d_%H%M%S")
            shutil.copy2(self.ledger_path, os.path.join(bak_dir, f"ledger_{tag}.csv"))
            if os.path.exists(self.state_path):
                shutil.copy2(self.state_path, os.path.join(bak_dir, f"state_{tag}.json"))
        self.ledger.to_csv(self.ledger_path, index=False, encoding="utf-8-sig")
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
        self.state["portfolio"] = self.aggregate()
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Phase 3.5 生产组合：6-cohort ledger 维护 / 纸面执行确认")
    ap.add_argument("--scores", help="本月评分快照 CSV（ml/scores/YYYY-MM.csv）；缺省取最近一份")
    ap.add_argument("--confirm-cohort", help="确认执行某个 planned cohort（不填=走 add_month 建仓/更新）")
    ap.add_argument("--execution-date", help="执行确认的实际可交易日 YYYY-MM-DD")
    ap.add_argument("--type", dest="exec_type", default="paper", choices=("paper", "actual"),
                    help="执行类型（默认 paper 纸面组合；actual 仅在未来接真实交易后使用）")
    ap.add_argument("--operator", default="local_user", help="操作人（人工提供）")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    pf = LivePortfolio()
    if args.confirm_cohort:
        if not args.execution_date:
            raise SystemExit("--confirm-cohort 必须配合 --execution-date YYYY-MM-DD")
        ev = pf.confirm_execution(args.confirm_cohort, args.execution_date,
                                  args.exec_type, args.operator)
        print("== live_portfolio 执行确认 ==")
        print("  事件:", json.dumps(ev, ensure_ascii=False))
        print(f"  events 追加至: {EXECUTION_EVENTS_PATH}")
        return

    scores_path = args.scores
    if scores_path is None:
        import re
        cands = sorted([f for f in os.listdir(os.path.join(PROJECT_ROOT, "ml", "scores"))
                        if re.fullmatch(r"\d{4}-\d{2}\.csv", f)], reverse=True)
        if not cands:
            raise SystemExit("未找到评分快照，请先运行 live_score / production_pipeline")
        scores_path = os.path.join(PROJECT_ROOT, "ml", "scores", cands[0])
    scores = pd.read_csv(scores_path, dtype={"fund_code": str})
    action = pf.add_month(scores)
    agg = pf.aggregate()
    print("== live_portfolio ==")
    print("  本月动作:", json.dumps(action, ensure_ascii=False))
    print("  当前:", json.dumps(agg, ensure_ascii=False)[:400])
    print(f"  ledger: {pf.ledger_path}（{len(pf.ledger)} 行）| state: {pf.state_path}")
    if args.no_save:
        raise SystemExit(0)


if __name__ == "__main__":
    main()