# paper_ledger.py：最小前向纸面收益账本（2026-10-02 用户第 4 步 / 目标第 ⑥ 项）
#
# 定位：把「封存的记账输入」(ml/paper/paper_inputs_*.json) + 「执行确认事实」
#   (ml/ledger/execution_events.jsonl) 变成**可核验的份额级账本**。
#   本模块只做记账与估值，**不参与任何选股/信号决策**；主策略保持冻结。
#
# 口径（与 docs/PHASE4_前向记账协议.md、封存输入的 protocol 段严格一致，不得在此处改）：
#   - 初始净资产 1.0；cohort 资本 = cohort_weight（方案 A：1/6）
#   - 申购费从申购金额中扣除：净申购额 = 金额 × (1 - buy_fee)，费用计入成本、不摊入份额净值
#   - 收益一律用**官方日增长率**（processed 的 daily_ret，已含分红除权调整）复权累计：
#     建仓日净值为基准，逐日 × (1 + daily_ret)；**不**用 nav 直接比、不用 nav_acc.pct_change
#   - 分红再投资：由上述复权口径自然实现（不另计分红现金）
#   - 现金计息：2%/年，actual/365，按日复利 (1 + rate/365)^days
#   - 净值迟发：建仓成交价取该基金**执行日或之后第一个披露日**净值，并记录 lag_days；
#     估值日则用截至该日最后可用净值（不插值、不猜）
#   - 赎回费 0.5% 仅在到期月计提（到期流程待首批到期前实现）
#
# 账本产物（不可变；dry-run 写 dryrun/ 目录并标 simulated=true，绝不冒充真实执行）：
#   ml/paper/ledgers/paper_ledger_{cohort}.json     建仓明细与份额（含费用、迟发记录）
#   ml/paper/ledgers/valuation_{cohort}_{as_of}.json 指定日估值快照
import argparse
import glob
import hashlib
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from backtest_strategy import PROJECT_ROOT
from clean_nav import HISTORY_DIR

PAPER_DIR = os.path.join(PROJECT_ROOT, "ml", "paper")
LEDGER_DIR = os.path.join(PAPER_DIR, "ledgers")
DRYRUN_DIR = os.path.join(PAPER_DIR, "dryrun")
EXECUTION_EVENTS_PATH = os.path.join(PROJECT_ROOT, "ml", "ledger", "execution_events.jsonl")
LEDGER_VERSION = 1


def _sha256(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def execution_fact(cohort_id: str, events_path: str | None = None) -> dict | None:
    """从事件流折叠该 cohort 的最终执行事实（与 live_portfolio 同口径）。"""
    p = events_path or EXECUTION_EVENTS_PATH
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
                        "event_id": ev.get("event_id"), "operator": ev.get("operator")}
            elif ev.get("type") == "execution_correction":
                fact = {"execution_date": ev.get("execution_date"),
                        "exec_type": ev.get("exec_type") or (fact or {}).get("exec_type"),
                        "event_id": ev.get("event_id"), "operator": ev.get("operator")}
    return fact


def cash_growth(cash: float, days: int, annual_rate: float) -> float:
    """现金计息：2%/年，actual/365，按日复利 (1 + rate/365)^days。

    组合层的未投资现金（方案 A 前五个月的 5/6 等）由上层用同一函数计息，
    保证 cohort 账本与组合层口径一致。
    """
    return float(cash) * (1.0 + float(annual_rate) / 365.0) ** int(days)


class PaperLedger:
    """最小份额级纸面账本：建仓（open_position）+ 估值（value）。"""

    def __init__(self, inputs_path: str | None = None, history_dir: str | None = None,
                 out_dir: str | None = None, dryrun_dir: str | None = None):
        self.history_dir = history_dir or HISTORY_DIR
        self.out_dir = out_dir or LEDGER_DIR
        # 预演目录可注入（测试隔离用，避免 fixture 写进正式 dryrun 目录）
        self.dryrun_dir = dryrun_dir or DRYRUN_DIR
        self.inputs_path = inputs_path or self._latest_inputs()
        if not self.inputs_path or not os.path.exists(self.inputs_path):
            raise FileNotFoundError("找不到封存的记账输入（ml/paper/paper_inputs_*.json）——"
                                    "请先运行 src/freeze_paper_inputs.py")
        with open(self.inputs_path, encoding="utf-8") as f:
            self.inputs = json.load(f)
        self.protocol = self.inputs["protocol"]
        self.cohort = self.inputs["cohort_id"]
        self._series_cache = {}

    @staticmethod
    def _latest_inputs() -> str | None:
        cands = sorted(glob.glob(os.path.join(PAPER_DIR, "paper_inputs_*.json")))
        return cands[-1] if cands else None

    # ---------------- 净值数据 ----------------
    def _series(self, code: str) -> pd.DataFrame:
        if code in self._series_cache:
            return self._series_cache[code]
        path = os.path.join(self.history_dir, f"fund_{code}.csv")
        if not os.path.exists(path):
            raise KeyError(f"缺少净值数据：{path}")
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        missing = [c for c in ("nav", "daily_ret") if c not in df.columns]
        if missing:
            raise ValueError(f"{code} 缺列 {missing}（需 processed fund_history：nav/daily_ret）")
        df = df[["nav", "daily_ret"]].astype(float)
        df["daily_ret"] = df["daily_ret"].fillna(0.0)
        self._series_cache[code] = df
        return df

    def _open_price(self, code: str, execution_date: pd.Timestamp) -> dict:
        """建仓成交价：执行日或之后**第一个披露日**的净值（迟发顺延并记录 lag_days）。"""
        s = self._series(code)
        after = s[s.index >= execution_date]
        if not len(after):
            raise ValueError(f"{code} 在执行日 {execution_date.date()} 及之后没有净值数据"
                             f"（最后可用 {s.index.max().date()}）")
        d = pd.Timestamp(after.index[0])
        return {"nav_date": d.strftime("%Y-%m-%d"), "nav": float(after.iloc[0]["nav"]),
                "lag_days": int((d - execution_date).days)}

    # ---------------- 建仓 ----------------
    def build_position(self, execution_date: str, capital: float | None = None) -> dict:
        """计算建仓明细（不落盘）：金额/费用/净申购额/份额/成交净值/迟发。"""
        ts = pd.Timestamp(execution_date)
        if ts.strftime("%Y-%m-%d") != execution_date:
            raise ValueError(f"execution_date 须为 YYYY-MM-DD，收到 {execution_date!r}")
        top = self.inputs["top50"]
        w_sum = float(sum(float(x["weight"]) for x in top))
        cap = float(capital if capital is not None else self.protocol["cohort_weight"])
        buy_fee = float(self.protocol["buy_fee"])

        positions, invested, fees = [], 0.0, 0.0
        for x in top:
            code = str(x["fund_code"])
            amount = cap * (float(x["weight"]) / w_sum)      # 组内归一化目标金额
            fee = amount * buy_fee
            net = amount - fee
            px = self._open_price(code, ts)
            shares = net / px["nav"]
            positions.append({
                "fund_code": code, "rank": x.get("rank"),
                "amount": amount, "buy_fee": fee, "net_amount": net,
                "nav_at_open": px["nav"], "nav_date": px["nav_date"], "lag_days": px["lag_days"],
                "shares": shares,
            })
            invested += amount
            fees += fee
        cash = cap - invested                                # 方案 A：cohort 内全额投入 → 0
        return {
            "ledger_version": LEDGER_VERSION,
            "cohort": self.cohort,
            "run_id": self.inputs["run_id"],
            "signal_date": self.inputs["signal_date"],
            "execution_date": execution_date,
            "inputs_path": self.inputs_path,
            "inputs_sha256": _sha256(self.inputs_path),
            "protocol": self.protocol,
            "capital": cap,
            "initial_nav": float(self.protocol["initial_nav"]),
            "positions": positions,
            "n_positions": len(positions),
            "invested": invested,
            "total_buy_fee": fees,
            "cash": cash,
            "lag_funds": [p["fund_code"] for p in positions if p["lag_days"] > 0],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def open_position(self, execution_date: str, capital: float | None = None,
                      persist: bool = True, simulate: bool = False,
                      events_path: str | None = None, out_dir: str | None = None) -> dict:
        """建仓并落盘（不可变）。

        - 非 simulate（真实记账）：必须已存在该 cohort 的**执行确认事实**，且 execution_date 与
          事件一致——账本只记录已确认的执行，绝不为未确认的计划建仓；
        - simulate=True（预演，dry-run）：写 dryrun/ 目录并标 `simulated=true`。
        """
        out_dir = out_dir or (self.dryrun_dir if simulate else self.out_dir)
        if not simulate:
            fact = execution_fact(self.cohort, events_path)
            if not fact:
                raise RuntimeError(
                    f"cohort {self.cohort} 尚无执行确认事件——账本只为已确认的执行建仓；"
                    f"如需预演请用 --dry-run（写入 dryrun/ 并标 simulated）")
            if fact["execution_date"] != execution_date:
                raise ValueError(
                    f"execution_date（{execution_date}）与确认事件（{fact['execution_date']}）不一致")
        payload = self.build_position(execution_date, capital)
        payload["simulated"] = bool(simulate)
        if simulate:
            payload["note"] = "预演（dry-run）：未确认执行，不得作为绩效证据"
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"paper_ledger_{self.cohort}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                old = json.load(f)
            a, b = dict(old), dict(payload)
            a.pop("created_at", None)
            b.pop("created_at", None)
            if a == b:
                return {"status": "existing", "path": path, "ledger": payload}
            raise RuntimeError(f"已存在内容不同的账本：{path}（账本不可变，请人工核对）")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return {"status": "opened", "path": path, "ledger": payload}

    # ---------------- 估值 ----------------
    def value(self, as_of: str, ledger: dict | None = None) -> dict:
        """按协议估值：官方日增长率复权累计 + 现金计息（不写盘）。"""
        ts = pd.Timestamp(as_of)
        if ts.strftime("%Y-%m-%d") != as_of:
            raise ValueError(f"as_of 须为 YYYY-MM-DD，收到 {as_of!r}")
        led = ledger or self._load_ledger()
        exec_ts = pd.Timestamp(led["execution_date"])
        if ts < exec_ts:
            raise ValueError(f"估值日（{as_of}）早于建仓日（{led['execution_date']}）")
        rate = float(self.protocol["cash_annual_rate"])

        rows, market_value, lag_rows, last_dates = [], 0.0, [], []
        for p in led["positions"]:
            s = self._series(p["fund_code"])
            open_ts = pd.Timestamp(p["nav_date"])
            win = s[(s.index > open_ts) & (s.index <= ts)]
            factor = float(np.prod(1.0 + win["daily_ret"].to_numpy())) if len(win) else 1.0
            last_date = pd.Timestamp(win.index[-1]) if len(win) else open_ts
            lag = int((ts - last_date).days)
            nav_now = float(p["nav_at_open"]) * factor
            mv = float(p["shares"]) * nav_now
            rows.append({
                "fund_code": p["fund_code"], "shares": p["shares"],
                "nav_at_open": p["nav_at_open"], "nav_date_open": p["nav_date"],
                "growth_factor": factor, "nav_now": nav_now,
                "last_nav_date": last_date.strftime("%Y-%m-%d"),
                "market_value": mv,
                "value_share": None,       # 组装后回填
                "stale_days": lag,
            })
            market_value += mv
            last_dates.append(last_date)
            if lag > 0:
                lag_rows.append({"fund_code": p["fund_code"], "last_nav_date":
                                 last_date.strftime("%Y-%m-%d"), "stale_days": lag})

        days = int((ts - exec_ts).days)
        cash = float(led.get("cash", 0.0))
        cash_grown = cash * (1.0 + rate / 365.0) ** days
        total = market_value + cash_grown
        cap = float(led["capital"])
        for r in rows:
            r["value_share"] = r["market_value"] / total if total else None
        return {
            "cohort": led["cohort"], "run_id": led["run_id"],
            "execution_date": led["execution_date"], "as_of": as_of,
            "days_held": days,
            "capital": cap, "initial_nav": led["initial_nav"],
            "market_value": market_value, "cash": cash, "cash_grown": cash_grown,
            "total_value": total,
            "nav": total / cap if cap else None,          # 组合净值（起点 = initial_nav）
            "return_since_open": (total / cap - 1.0) if cap else None,
            "positions": rows, "n_positions": len(rows),
            "stale_funds": lag_rows,
            "stale_note": "估值日尚未披露净值的基金，按截至该日最后可用净值估值并记 stale_days"
                          "（不插值、不猜）",
            "valuation_basis": "official_daily_growth_rate（复权累计，含分红再投资）",
            "simulated": bool(led.get("simulated")),
            "protocol_sha": _sha256(self.inputs_path),
        }

    def _load_ledger(self, path: str | None = None) -> dict:
        path = path or os.path.join(self.out_dir, f"paper_ledger_{self.cohort}.json")
        if not os.path.exists(path):
            alt = os.path.join(self.dryrun_dir, f"paper_ledger_{self.cohort}.json")
            if os.path.exists(alt):
                path = alt
            else:
                raise FileNotFoundError(f"未找到账本：{path}（请先 --open 建仓）")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def save_valuation(self, val: dict, out_dir: str | None = None) -> str:
        out_dir = out_dir or (self.dryrun_dir if val.get("simulated") else self.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"valuation_{val['cohort']}_{val['as_of']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(val, f, ensure_ascii=False, indent=2)
        return path


def main():
    ap = argparse.ArgumentParser(description="最小前向纸面收益账本（建仓 / 估值）")
    ap.add_argument("--inputs", default=None, help="封存的记账输入 JSON（默认取最近一份）")
    ap.add_argument("--open", action="store_true", help="按封存输入建仓（需已确认执行）")
    ap.add_argument("--value", metavar="AS_OF", help="估值到指定日 YYYY-MM-DD")
    ap.add_argument("--execution-date", default=None, help="建仓执行日 YYYY-MM-DD")
    ap.add_argument("--dry-run", action="store_true",
                    help="预演：不要求已确认执行，写入 ml/paper/dryrun/ 并标 simulated")
    args = ap.parse_args()

    led = PaperLedger(args.inputs)
    print(f"== paper_ledger（cohort={led.cohort}, run_id={led.inputs['run_id']}）==")
    if args.open:
        if not args.execution_date:
            raise SystemExit("--open 需要 --execution-date YYYY-MM-DD")
        res = led.open_position(args.execution_date, simulate=args.dry_run)
        lg = res["ledger"]
        print(f"  状态：{res['status']} | 文件：{res['path']}")
        print(f"  资本 {lg['capital']:.6f} | 持仓 {lg['n_positions']} 只 | 申购费合计 "
              f"{lg['total_buy_fee']:.8f} | 现金 {lg['cash']:.6f}"
              f" | 迟发 {len(lg['lag_funds'])} 只")
        return
    if args.value:
        val = led.value(args.value)
        path = led.save_valuation(val)
        print(f"  估值日 {val['as_of']}（持有 {val['days_held']} 天）")
        print(f"  组合净值 {val['nav']:.6f} | 区间收益 {val['return_since_open']:.4%}"
              f" | 市值 {val['market_value']:.6f} | 现金 {val['cash_grown']:.6f}")
        if val["stale_funds"]:
            print(f"  ⚠️ 迟发未披露 {len(val['stale_funds'])} 只（按最后可用净值估值）")
        print(f"  估值快照：{path}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
