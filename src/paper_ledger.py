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
import shutil
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
                        "event_id": ev.get("event_id"), "operator": ev.get("operator"),
                        "source_run_id": ev.get("source_run_id")}
            elif ev.get("type") == "execution_correction":
                fact = {"execution_date": ev.get("execution_date"),
                        "exec_type": ev.get("exec_type") or (fact or {}).get("exec_type"),
                        "event_id": ev.get("event_id"), "operator": ev.get("operator"),
                        "source_run_id": ev.get("source_run_id")
                        or (fact or {}).get("source_run_id")}
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

    def _data_fingerprint(self, code: str, open_ts: pd.Timestamp, as_of: pd.Timestamp) -> dict:
        """估值所用**数据切片**的指纹（P2，2026-10-02 三次复核）。

        旧实现用整文件 sha256 ⇒ 只要在估值日之后追加了新净值，历史估值就被判"内容不同"而拒绝
        幂等重跑。现在幂等/差异判定以**实际参与估值的数据切片**为准：
        切片 = 该基金 `(建仓成交日, 估值日]` 内参与复权累计的行（date/nav/daily_ret）；
        整文件 sha256 仅作**来源记录**保留（`nav_file_sha256`），不参与幂等判定。
        """
        s = self._series(code)
        win = s[(s.index > open_ts) & (s.index <= as_of)]
        rows = [[d.strftime("%Y-%m-%d"), float(r.nav), float(r.daily_ret)]
                for d, r in zip(win.index, win.itertuples(index=False))]
        payload = json.dumps({"open": open_ts.strftime("%Y-%m-%d"),
                              "as_of": as_of.strftime("%Y-%m-%d"), "rows": rows},
                             ensure_ascii=False, separators=(",", ":"))
        path = os.path.join(self.history_dir, f"fund_{code}.csv")
        return {
            "slice_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "slice_start": open_ts.strftime("%Y-%m-%d"),
            "slice_end": (win.index[-1].strftime("%Y-%m-%d") if len(win)
                          else open_ts.strftime("%Y-%m-%d")),
            "slice_rows": int(len(win)),
            "nav_file": f"fund_{code}.csv",
            "nav_file_sha256": _sha256(path),          # 来源记录（不参与幂等判定）
            "nav_file_last_date": s.index.max().strftime("%Y-%m-%d"),
        }

    def _verify_official_execution(self, cohort: str | None, run_id: str | None,
                                   execution_date: str | None,
                                   events_path: str | None = None) -> dict:
        """正式（非诊断）账本/估值必须能对上执行确认事实（P1，2026-10-02 三次复核）。

        防止"未确认执行的计算结果"绕过建仓校验、直接以正式身份写盘。
        """
        cid = cohort or self.cohort
        fact = execution_fact(cid, events_path)
        if not fact:
            raise RuntimeError(
                f"cohort {cid} 无执行确认事件——正式账本/估值必须绑定已确认的执行，拒绝写盘")
        if execution_date and fact.get("execution_date") != execution_date:
            raise ValueError(
                f"执行事件执行日（{fact.get('execution_date')}）与账本记录（{execution_date}）不一致")
        src_ev = fact.get("source_run_id")
        if not src_ev:
            raise ValueError("执行事件缺少 source_run_id——无法与来源绑定，拒绝正式写盘")
        if run_id and src_ev != run_id:
            raise ValueError(f"执行事件来源（{src_ev}）与账本来源（{run_id}）不一致")
        return fact

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
            # build_position 是**纯计算**（未绑定执行事实）→ 默认诊断身份；
            # open_position 仅在真实记账路径下、核对执行事实后改判为正式（diagnostic=False）。
            "diagnostic": True,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def open_position(self, execution_date: str, capital: float | None = None,
                      persist: bool = True, simulate: bool = False,
                      events_path: str | None = None, out_dir: str | None = None) -> dict:
        """建仓（可落盘，不可变）。

        - `persist=False`：**纯计算**——不写任何文件、不要求已确认执行（测试/诊断用）；P2 修复
          （2026-10-02 用户复核：旧实现忽略了 persist，纯计算路径仍写盘）；
        - `simulate=True`：不要求执行确认，写 `dryrun/` 并标 `simulated=true`（预演留痕）；
        - 真实记账（persist=True 且 simulate=False）：必须存在该 cohort 的**执行确认事实**，且
          ① `execution_date` 与事件一致；② **事件的 `source_run_id` 与封存输入 run_id 一致**
          （P1 修复：防止拿别的 run 的封存输入给本次执行建仓）。
        """
        # P1（2026-10-02 三次复核）：真实记账**先核对执行事实**（含来源绑定），再算价格——
        # 这样"未确认执行"的报错不会被"净值尚未披露"掩盖，语义上也是"先确认、后记账"。
        fact = None
        if persist and not simulate:
            fact = execution_fact(self.cohort, events_path)
            if not fact:
                raise RuntimeError(
                    f"cohort {self.cohort} 尚无执行确认事件——账本只为已确认的执行建仓；"
                    f"如需预演请用 --dry-run（写入 dryrun/ 并标 simulated）")
            if fact["execution_date"] != execution_date:
                raise ValueError(
                    f"execution_date（{execution_date}）与确认事件（{fact['execution_date']}）不一致")
            src_in, src_ev = self.inputs.get("run_id"), fact.get("source_run_id")
            if not src_in:
                raise ValueError("封存输入缺少 run_id——无法与执行事件绑定，拒绝建仓")
            if not src_ev:
                raise ValueError("执行事件缺少 source_run_id——无法与封存输入绑定，拒绝建仓")
            if src_ev != src_in:
                raise ValueError(
                    f"封存输入来源 run_id={src_in} 与执行事件来源 run_id={src_ev} 不一致——"
                    f"不得用其他 run 的封存输入为本次执行建仓")

        payload = self.build_position(execution_date, capital)
        payload["simulated"] = bool(simulate)
        # P1：未确认执行的计算结果必须带**诊断身份**，否则会被当成正式账本、
        # 进而让 value/save_valuation 写进正式目录。
        payload["diagnostic"] = bool(simulate or not persist)
        if simulate:
            payload["note"] = "预演（dry-run）：未确认执行，不得作为绩效证据"
        if not persist:
            payload["note"] = "纯计算（persist=False）：未确认执行、未落盘，不得作为绩效证据"
            return {"status": "computed", "path": None, "ledger": payload}
        if fact is not None:
            payload["execution_event"] = {"event_id": fact.get("event_id"),
                                          "exec_type": fact.get("exec_type"),
                                          "source_run_id": fact.get("source_run_id"),
                                          "operator": fact.get("operator")}
            payload["diagnostic"] = False          # 已绑定确认的执行事实 → 正式身份
            payload.pop("note", None)
        if payload["diagnostic"]:
            self._ensure_diag_dir(out_dir)          # 预演/纯计算产物只能写 dryrun 子树
        out_dir = out_dir or (self.dryrun_dir if payload["diagnostic"] else self.out_dir)
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
    def value(self, as_of: str, ledger: dict | None = None,
              allow_pending: bool = False, events_path: str | None = None) -> dict:
        """按协议估值：官方日增长率复权累计 + 现金计息（不写盘）。

        P1 修复（2026-10-02 用户复核）：
          - 若某只基金的**成交净值日**晚于估值日（建仓尚未实际发生），默认**拒绝生成完整估值**；
            `allow_pending=True` 时标 `status=pending` 且该持仓不计入市值——不会出现
            `stale_days=-1`，也绝不用未来净值倒算历史；
          - 记录逐基金**数据指纹**（净值文件 sha256 + 最后净值日），使"数据变化导致历史估值改变"
            可被 save_valuation 检出。
        返回里的 `cohort_nav` 是**该 cohort 的净值**（以 cohort 资本为分母，起点 1.0）；
        组合层净值由 `portfolio_value()` 汇总（含未投资现金计息）。
        """
        ts = pd.Timestamp(as_of)
        if ts.strftime("%Y-%m-%d") != as_of:
            raise ValueError(f"as_of 须为 YYYY-MM-DD，收到 {as_of!r}")
        led = ledger or self._load_ledger()
        exec_ts = pd.Timestamp(led["execution_date"])
        if ts < exec_ts:
            raise ValueError(f"估值日（{as_of}）早于建仓日（{led['execution_date']}）")
        diag = bool(led.get("simulated") or led.get("diagnostic"))
        if not diag:
            # P1（2026-10-02 三次复核）：正式账本估值前**再次**核对执行事实，
            # 防止"未确认执行的计算结果"绕过建仓校验而以正式身份出估值/写盘
            self._verify_official_execution(led.get("cohort"), led.get("run_id"),
                                           led.get("execution_date"), events_path)

        pending = [p for p in led["positions"] if pd.Timestamp(p["nav_date"]) > ts]
        if pending and not allow_pending:
            first = pending[0]
            raise ValueError(
                f"估值日 {as_of} 早于 {len(pending)} 只基金的成交净值日（如 {first['fund_code']} "
                f"成交于 {first['nav_date']}）——建仓在该日尚未实际发生，拒绝生成完整估值；"
                f"确需诊断可用 allow_pending=True（结果标 pending，不得作为绩效证据）")

        rate = float(self.protocol["cash_annual_rate"])
        rows, market_value, lag_rows, data_hashes, pending_rows = [], 0.0, [], {}, []
        for p in led["positions"]:
            open_ts = pd.Timestamp(p["nav_date"])
            data_hashes[p["fund_code"]] = self._data_fingerprint(p["fund_code"], open_ts, ts)
            if open_ts > ts:                       # 仅 allow_pending 路径可达
                pending_rows.append({"fund_code": p["fund_code"], "nav_date": p["nav_date"],
                                     "why": "成交净值日晚于估值日，建仓尚未发生"})
                rows.append({"fund_code": p["fund_code"], "shares": p["shares"],
                             "nav_at_open": p["nav_at_open"], "nav_date_open": p["nav_date"],
                             "growth_factor": None, "nav_now": None, "last_nav_date": None,
                             "market_value": 0.0, "value_share": None, "stale_days": None,
                             "pending": True})
                continue
            s = self._series(p["fund_code"])
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
                "stale_days": lag, "pending": False,
            })
            market_value += mv
            if lag > 0:
                lag_rows.append({"fund_code": p["fund_code"], "last_nav_date":
                                 last_date.strftime("%Y-%m-%d"), "stale_days": lag})

        days = int((ts - exec_ts).days)
        cash = float(led.get("cash", 0.0))
        cash_grown = cash_growth(cash, days, rate)
        total = market_value + cash_grown
        cap = float(led["capital"])
        for r in rows:
            if r["value_share"] is None and total:
                r["value_share"] = r["market_value"] / total
        cohort_nav = (total / cap) if cap else None
        return {
            "cohort": led["cohort"], "run_id": led["run_id"],
            "execution_date": led["execution_date"], "as_of": as_of,
            "days_held": days,
            "capital": cap, "initial_nav": led["initial_nav"],
            "market_value": market_value, "cash": cash, "cash_grown": cash_grown,
            "total_value": total,
            # **该 cohort 的净值**（分母 = cohort 资本，起点 initial_nav）；组合层另见 portfolio_value()
            "cohort_nav": cohort_nav,
            "nav": cohort_nav,                       # 历史别名（等价 cohort_nav，勿再称"组合净值"）
            "cohort_return_since_open": (cohort_nav - 1.0) if cohort_nav is not None else None,
            "return_since_open": (cohort_nav - 1.0) if cohort_nav is not None else None,
            "positions": rows, "n_positions": len(rows),
            "stale_funds": lag_rows,
            "pending_funds": pending_rows,
            "status": "pending" if pending_rows else "complete",
            "stale_note": "估值日尚未披露净值的基金，按截至该日最后可用净值估值并记 stale_days"
                          "（不插值、不猜）",
            "valuation_basis": "official_daily_growth_rate（复权累计，含分红再投资）",
            "data_hashes": data_hashes,
            "diagnostic": diag,                    # 与账本一致：诊断结果不得写正式目录
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

    @staticmethod
    def _is_within(child: str, parent: str) -> bool:
        """child 是否等于 parent 或位于其子树内（大小写与分隔符归一）。"""
        c = os.path.normcase(os.path.normpath(os.path.abspath(child)))
        p = os.path.normcase(os.path.normpath(os.path.abspath(parent)))
        return c == p or c.startswith(p.rstrip("\\/") + os.sep)

    def _ensure_diag_dir(self, out_dir: str | None):
        """诊断/预演产物**只允许**写入配置的 dryrun 目录及其子目录（P1，2026-10-02 四次复核）。

        旧守卫只在"目标路径恰好等于正式目录"时拒绝，于是 `out_dir=正式目录/子目录` 或
        `open_position(simulate=True, out_dir=正式目录)` 都能把预演产物写进正式区，
        随后正式建仓会被"已有不同内容账本"挡住。现在两个入口共用同一判定。
        """
        if out_dir is None:
            return
        if not self._is_within(out_dir, self.dryrun_dir):
            raise RuntimeError(
                f"诊断/预演产物只能写入 dryrun 目录及其子目录（{os.path.abspath(self.dryrun_dir)}），"
                f"收到 {os.path.abspath(out_dir)}")

    @staticmethod
    def _comparable(val: dict) -> dict:
        """幂等比较视图：剔除易变字段与**纯来源记录**（P2 2026-10-02 三次复核）。

        剔除项：`generated_at`、`revision`（每次写入不同），以及每只基金的整文件哈希
        `nav_file_sha256` / `nav_file_last_date`——它们只作来源记录，**不参与差异判定**；
        差异判定以 `slice_sha256`（实际参与估值的数据切片）为准，因此"估值日之后追加净值"
        不会误判为内容变化，而"估值区间内数据被改动"仍会被检出。
        """
        v = json.loads(json.dumps(val, ensure_ascii=False, default=str))
        v.pop("generated_at", None)
        v.pop("revision", None)
        for fp in (v.get("data_hashes") or {}).values():
            if isinstance(fp, dict):
                fp.pop("nav_file_sha256", None)
                fp.pop("nav_file_last_date", None)
        return v

    def save_valuation(self, val: dict, out_dir: str | None = None, revision: bool = False,
                       revision_reason: str | None = None) -> dict:
        """写估值快照（不可变）。

        P1 修复（2026-10-02 用户复核）：旧实现无条件覆盖同名文件——同日重跑会覆盖旧文件，
        净值数据被改动后历史估值也会被"重写成新值"。现在：
          - 相同内容（忽略 generated_at）重跑 → 幂等返回 `existing`；
          - 内容不同 → **拒绝覆盖**（含"数据指纹变化"引起的差异，可据此发现数据被改）；
          - 确需修订 → 显式 `revision=True`：备份旧版为 `.rev_{n}` 并在新文件记录修订原因；
          - `status=pending` 的估值不得落盘（建仓尚未发生）。
        """
        if val.get("status") == "pending":
            raise RuntimeError("估值处于 pending（成交净值日尚未到来）——不得落盘为完成估值")
        diag = bool(val.get("simulated") or val.get("diagnostic"))
        if diag:
            self._ensure_diag_dir(out_dir)          # 只能写 dryrun 目录及其子目录
        if not diag:
            # P1（2026-10-02 三次复核）：正式估值落盘前**再次**核对执行事实
            self._verify_official_execution(val.get("cohort"), val.get("run_id"),
                                           val.get("execution_date"))
        out_dir = out_dir or (self.dryrun_dir if diag else self.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"valuation_{val['cohort']}_{val['as_of']}.json")
        payload = dict(val)
        payload.setdefault("generated_at", datetime.now().isoformat(timespec="seconds"))
        exists = os.path.exists(path)
        if exists:
            with open(path, encoding="utf-8") as f:
                old = json.load(f)
            if self._comparable(old) == self._comparable(payload):
                return {"status": "existing", "path": path}
            if not revision:
                raise RuntimeError(
                    f"已存在内容不同的估值：{path}——估值不可变（差异可能来自净值数据被改动）；"
                    f"确需修订请显式 revision=True（会备份旧版并记录原因）")
            n = 1 + len(glob.glob(path + ".rev_*"))
            bak = f"{path}.rev_{n}"
            shutil.copy2(path, bak)
            payload["revision"] = {"rev": n + 1, "backup": os.path.basename(bak),
                                   "reason": revision_reason or "未注明",
                                   "revised_at": datetime.now().isoformat(timespec="seconds")}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return {"status": "revised" if exists else "written", "path": path}

    # ---------------- 组合层（2026-10-02 用户：cohort 净值 ≠ 组合净值） ----------------
    def portfolio_value(self, as_of: str, ledger_paths: list | None = None,
                        total_capital: float = 1.0, include_simulated: bool = False) -> dict:
        """组合层净值汇总：`Σ(cohort_nav × cohort_weight) + 未投资现金计息`。

        建仓日无价格变化、尚无利息时（单 cohort）应得 `1/6×(1−buy_fee) + 5/6 = 0.99975`。
        估值日晚于其 execution_date 才计入的 cohort 才算已建仓；未建仓部分权重留在现金。

        现金计息按**现金流逐段**计算（P1，2026-10-02 三次复核）：旧实现把"最终剩余现金"整段
        从最早建仓日起计息，遗漏了"后来要投入的现金在此前赚到的利息"。正确做法是在时间轴上
        逐段推进——先计息到该 cohort 的建仓日，再扣除其投入资本：两批相隔 32 天、基金收益为零时
        组合净值应为 `1.000962429`（旧实现给 `1.000669943`，少计约 2.93bp）。
        """
        ts = pd.Timestamp(as_of)
        if ledger_paths is None:
            ledger_paths = sorted(glob.glob(os.path.join(self.out_dir, "paper_ledger_*.json")))
            if include_simulated:      # 预演场景：把 dryrun 账本也纳入汇总
                ledger_paths += sorted(glob.glob(
                    os.path.join(self.dryrun_dir, "paper_ledger_*.json")))
        rate = float(self.protocol["cash_annual_rate"])
        entries = []
        for p in ledger_paths:
            with open(p, encoding="utf-8") as f:
                led = json.load(f)
            if led.get("simulated") and not include_simulated:
                continue
            exec_ts = pd.Timestamp(led["execution_date"])
            if exec_ts > ts:
                continue                                   # 估值日尚未建仓
            entries.append({"led": led, "path": p, "exec_ts": exec_ts,
                            "weight": float(led["capital"]) / float(total_capital)})
        entries.sort(key=lambda e: e["exec_ts"])

        cohorts, segments = [], []
        cash = float(total_capital)
        prev = None
        for e in entries:
            if prev is not None:
                d = int((e["exec_ts"] - prev).days)
                if d > 0:
                    before = cash
                    cash = cash_growth(cash, d, rate)
                    segments.append({"kind": "interest", "from": prev.strftime("%Y-%m-%d"),
                                     "to": e["exec_ts"].strftime("%Y-%m-%d"), "days": d,
                                     "cash_before": before, "cash_after": cash})
            cash -= e["weight"] * float(total_capital)
            segments.append({"kind": "invest", "date": e["exec_ts"].strftime("%Y-%m-%d"),
                             "cohort": e["led"]["cohort"],
                             "amount": -e["weight"] * float(total_capital), "cash_after": cash})
            v = self.value(as_of, ledger=e["led"])
            cohorts.append({"cohort": e["led"]["cohort"], "weight": e["weight"],
                            "cohort_nav": v["cohort_nav"], "market_value": v["market_value"],
                            "cash_grown": v["cash_grown"], "status": v["status"],
                            "ledger": os.path.basename(e["path"])})
            prev = e["exec_ts"]
        days_total = int((ts - entries[0]["exec_ts"]).days) if entries else 0
        if prev is not None:
            d = int((ts - prev).days)
            if d > 0:
                before = cash
                cash = cash_growth(cash, d, rate)
                segments.append({"kind": "interest", "from": prev.strftime("%Y-%m-%d"),
                                 "to": as_of, "days": d,
                                 "cash_before": before, "cash_after": cash})
        cash_weight = cash / float(total_capital)
        invested = sum(c["cohort_nav"] * c["weight"] for c in cohorts)
        return {
            "as_of": as_of, "n_cohorts": len(cohorts), "cohorts": cohorts,
            "cash_weight": cash_weight, "cash_grown": cash,
            "cash_segments": segments, "cash_days_total": days_total,
            "invested_value": invested, "portfolio_nav": invested + cash,
            "portfolio_return_since_start": invested + cash - 1.0,
            "basis": "组合净值 = Σ(cohort_nav × cohort_weight) + 未投资现金计息"
                     "（cohort_nav 是单个 cohort 的净值，起点 1.0）",
            "cash_note": "现金按 2%/年 actual-365 日复利，**按现金流逐段计息**"
                         "（先计息到各 cohort 建仓日、再扣除其投入资本；明细见 cash_segments）",
        }


def main():
    ap = argparse.ArgumentParser(description="最小前向纸面收益账本（建仓 / 估值 / 组合汇总）")
    ap.add_argument("--inputs", default=None, help="封存的记账输入 JSON（默认取最近一份）")
    ap.add_argument("--open", action="store_true", help="按封存输入建仓（需已确认执行）")
    ap.add_argument("--value", metavar="AS_OF", help="估值到指定日 YYYY-MM-DD（写 cohort 估值快照）")
    ap.add_argument("--portfolio", metavar="AS_OF", help="组合层净值汇总（含未投资现金计息，不写盘）")
    ap.add_argument("--execution-date", default=None, help="建仓执行日 YYYY-MM-DD")
    ap.add_argument("--dry-run", action="store_true",
                    help="预演：不要求已确认执行，写入 ml/paper/dryrun/ 并标 simulated")
    ap.add_argument("--no-save", action="store_true", help="纯计算，不写任何文件（建仓）")
    ap.add_argument("--revision", action="store_true",
                    help="估值修订：备份旧版并记录原因（默认拒绝覆盖不同内容）")
    ap.add_argument("--revision-reason", default=None, help="估值修订原因（配合 --revision）")
    args = ap.parse_args()

    led = PaperLedger(args.inputs)
    print(f"== paper_ledger（cohort={led.cohort}, run_id={led.inputs['run_id']}）==")
    if args.open:
        if not args.execution_date:
            raise SystemExit("--open 需要 --execution-date YYYY-MM-DD")
        res = led.open_position(args.execution_date, simulate=args.dry_run,
                                persist=not args.no_save)
        lg = res["ledger"]
        print(f"  状态：{res['status']} | 文件：{res['path'] or '（--no-save 纯计算，未写盘）'}")
        print(f"  封存输入：{led.inputs_path}")
        print(f"  cohort 资本 {lg['capital']:.6f} | 持仓 {lg['n_positions']} 只 | 申购费合计 "
              f"{lg['total_buy_fee']:.8f} | 现金 {lg['cash']:.6f}"
              f" | 迟发 {len(lg['lag_funds'])} 只")
        return
    if args.value:
        val = led.value(args.value)
        res = led.save_valuation(val, revision=args.revision,
                                 revision_reason=args.revision_reason)
        pv = led.portfolio_value(args.value, include_simulated=bool(val.get("simulated")))
        print(f"  估值日 {val['as_of']}（持有 {val['days_held']} 天）")
        print(f"  **cohort 净值 {val['cohort_nav']:.6f}** | cohort 区间收益 "
              f"{val['cohort_return_since_open']:.4%} | 市值 {val['market_value']:.6f}"
              f" | cohort 现金 {val['cash_grown']:.6f}")
        print(f"  组合净值 {pv['portfolio_nav']:.6f}（{pv['n_cohorts']} 个已建仓 cohort；"
              f"未投资现金权重 {pv['cash_weight']:.6f}，逐段计息 {pv['cash_days_total']} 天）")
        if val["stale_funds"]:
            print(f"  ⚠️ 迟发未披露 {len(val['stale_funds'])} 只（按最后可用净值估值）")
        print(f"  估值快照：{res['path']}（{res['status']}）")
        return
    if args.portfolio:
        pv = led.portfolio_value(args.portfolio, include_simulated=args.dry_run)
        print(f"  组合净值 {pv['portfolio_nav']:.6f} | 区间收益 "
              f"{pv['portfolio_return_since_start']:.4%} | 已建仓 cohort {pv['n_cohorts']} 个"
              f" | 现金权重 {pv['cash_weight']:.6f}（逐段计息 {pv['cash_days_total']} 天 → "
              f"{pv['cash_grown']:.6f}）")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
