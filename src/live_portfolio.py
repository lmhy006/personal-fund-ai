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
import json
import os
from datetime import datetime

import pandas as pd

from backtest_strategy import BUY_FEE, HOLD, PROJECT_ROOT, SELL_FEE, TOP_N_DEFAULT
from panel_builder import BENCH_PATH

LEDGER_DIR = os.path.join(PROJECT_ROOT, "ml", "ledger")
LEDGER_PATH = os.path.join(LEDGER_DIR, "portfolio_ledger.csv")
STATE_PATH = os.path.join(LEDGER_DIR, "portfolio_state.json")

COHORT_WEIGHT = 1.0 / HOLD          # 每批权重（方案 A：恒 1/6，含前 5 个月）
TOP_N = TOP_N_DEFAULT               # 50

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
                  cohort_weight: float = COHORT_WEIGHT) -> dict:
        """用本月评分快照推进一个月：识别新增/到期 cohort，写入 ledger 与 state。

        :return: 本月动作摘要（新增/到期/费用），若该月已存在则返回 {'skipped': True}。
        """
        if "as_of" not in scores_df.columns or not len(scores_df):
            raise ValueError("scores_df 缺少 as_of 或为空")
        asof = pd.Timestamp(scores_df["as_of"].iloc[0])
        month_key = asof.strftime("%Y-%m")
        if month_key in self.state["cohorts"]:
            return {"skipped": True, "month": month_key}

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

        # 到期：expire_month == 本月的 **active** cohort 标记 expired
        expired_now = []
        for cid in list(self.state["cohorts"]):
            if self.state["cohorts"][cid].get("expire_month") == month_key and \
                    self.state["cohorts"][cid].get("status") == "active":
                self.state["cohorts"][cid]["status"] = "expired"
                self.ledger.loc[self.ledger["cohort_id"] == cid, "status"] = "expired"
                expired_now.append(cid)

        # 新增：无已确认成交日 → status="planned"（不参与风险权重，现金仍持有）；
        # 成交日确认后由 confirm_execution 更新为 "active"
        cid = month_key
        self.state["cohorts"][cid] = {
            "signal_date": signal_date, "execution_date": exec_date_str,
            "expire_month": expire_month, "cohort_weight": cohort_weight,
            "n_codes": len(codes), "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "active" if exec_date is not None else "planned",
        }
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

    def confirm_execution(self, cohort_id: str, execution_date: str) -> dict:
        """把 planned cohort 标记为 active（真实成交日确认后调用）；幂等。"""
        if cohort_id not in self.state["cohorts"]:
            raise KeyError(f"cohort {cohort_id} 不存在")
        meta = self.state["cohorts"][cohort_id]
        meta["execution_date"] = execution_date
        meta["status"] = "active"
        self.state["cohorts"][cohort_id] = meta
        self.ledger.loc[self.ledger["cohort_id"] == cohort_id, "execution_date"] = execution_date
        self.ledger.loc[self.ledger["cohort_id"] == cohort_id, "status"] = "active"
        self.save()
        return {"cohort_id": cohort_id, "execution_date": execution_date, "status": "active"}

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
    ap = argparse.ArgumentParser(description="Phase 3.5 生产组合：6-cohort ledger 维护")
    ap.add_argument("--scores", help="本月评分快照 CSV（ml/scores/YYYY-MM.csv）；缺省取最近一份")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    scores_path = args.scores
    if scores_path is None:
        import re
        cands = sorted([f for f in os.listdir(os.path.join(PROJECT_ROOT, "ml", "scores"))
                        if re.fullmatch(r"\d{4}-\d{2}\.csv", f)], reverse=True)
        if not cands:
            raise SystemExit("未找到评分快照，请先运行 live_score / production_pipeline")
        scores_path = os.path.join(PROJECT_ROOT, "ml", "scores", cands[0])
    scores = pd.read_csv(scores_path, dtype={"fund_code": str})
    pf = LivePortfolio()
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