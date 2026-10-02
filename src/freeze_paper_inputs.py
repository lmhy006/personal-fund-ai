# freeze_paper_inputs.py：封存首次前向纸面记账的**输入**（2026-10-02 用户第 2 步）
#
# 为什么需要：`docs/PHASE4_前向记账协议.md` 已定协议但账本尚未实现。在 10-8 首次 paper 确认前，
#   必须把"记账要用的输入"**一次性固定并封存**，否则事后无法证明"当时用的是哪套参数、哪份评分、
#   哪个可执行全池"。封存物不可变：内容不同的重复封存会被拒绝（需 --force 并留备份）。
#
# 封存内容：
#   ① 协议参数（规范化初始净资产、申购费扣法、分红/复权口径、现金计息、净值迟发处理、
#      再平衡与到期规则）——**与冻结策略一致，不得在此处改口径**；
#   ② 来源快照身份与哈希（run_id/signal_date/manifest/评分文件/COMPLETE）；
#   ③ Top50 选股与逐只目标权重；
#   ④ main 全池对照集合（可执行全池的当日子集，单独 CSV + 哈希）——供未来"相对全池"对照；
#   ⑤ 全部输入文件的 sha256。
#
# 输出（默认 ml/paper/）：
#   paper_inputs_{run_id}.json      封存主文件（含协议与哈希）
#   main_universe_{signal_date}.csv main 全池对照集合（fund_code/rank/score）
import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime

import pandas as pd

from backtest_strategy import BUY_FEE, HOLD, PROJECT_ROOT, SELL_FEE
from live_portfolio import COHORT_WEIGHT, TOP_N

SNAPSHOTS_DIR = os.path.join(PROJECT_ROOT, "ml", "snapshots")
PAPER_DIR = os.path.join(PROJECT_ROOT, "ml", "paper")
PAPER_INPUTS_VERSION = 1


def _sha256(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _forward_snapshots(snapshots_dir: str) -> list:
    """可作前向证据的正式快照（COMPLETE + status=complete + month_end + 非重建），按生成时间升序。

    与 agent_tools._complete_snapshots 同口径：事后重建（is_reconstruction/forward_eligible=false）
    与非正式运行（run_mode=test/observation）不得作为前向输入来源。
    """
    out = []
    if not os.path.isdir(snapshots_dir):
        return out
    for d in sorted(os.listdir(snapshots_dir)):
        snap = os.path.join(snapshots_dir, d)
        mf_path = os.path.join(snap, "manifest.json")
        if not (os.path.exists(os.path.join(snap, "COMPLETE")) and os.path.exists(mf_path)):
            continue
        try:
            with open(mf_path, encoding="utf-8") as f:
                man = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        if man.get("status") != "complete" or man.get("signal_mode") != "month_end":
            continue
        if man.get("is_reconstruction") or man.get("forward_eligible") is False:
            continue
        if man.get("run_mode") in ("test", "observation"):
            continue
        out.append({"run_id": d, "dir": snap, "manifest": man})
    out.sort(key=lambda x: str(x["manifest"].get("score_generated_at", "")))
    return out


def _protocol() -> dict:
    """记账协议参数（单一真源；改这里等于改口径，必须重新封存并记录）。"""
    return {
        "initial_nav": 1.0,                     # 规范化初始净资产（每 cohort 1/6，起步净值 1.0）
        "cohort_weight": COHORT_WEIGHT,         # 1/6（方案 A 逐月建仓，未投部分持现金）
        "hold_months": HOLD,                    # 6
        "buy_fee": BUY_FEE,                     # 申购费 0.15%（0.0015），成交金额按净值扣费口径
        "sell_fee": SELL_FEE,                   # 赎回费 0.5%（0.005），仅到期月计提
        "buy_fee_deduction": "费用从申购金额中扣除（净申购额 = 金额 × (1 - buy_fee)），"
                             "费用额计入成本，不摊入份额净值",
        "return_basis": "official_daily_growth_rate",
        "return_basis_note": "一律用净值表官方『日增长率』累计；不用累计净值 pct_change、不用净值比",
        "dividend_policy": "reinvest_via_daily_growth",
        "dividend_note": "分红再投资：官方日增长率本身为复权口径，等于分红再投且不计税费；"
                         "不做分红现金另计，也不对分红日做特殊处理",
        "nav_used": "单位净值(nav)",
        "nav_lag_policy": "若执行日（或估值日）该基金净值尚未披露：顺延取该基金**下一披露日**净值，"
                          "并在账本记录 lag_days；不插值、不用上一日净值冒充、不猜",
        "cash_annual_rate": 0.02,
        "cash_day_count": "actual/365",
        "rebalance_policy": "cohort 内不做月度再平衡（目标权重 = 建仓权重）；"
                            "与历史回测的等权算术聚合口径分列对账，见 docs/PHASE4_前向记账协议.md",
        "expire_rule": "持有满 HOLD 个月后于到期月移出（expire_month），赎回费仅在到期月计提",
        "weighting": "cohort 内 50 只等权（cohort_weight / n_codes）",
        "performance_not_computed": True,
        "performance_note": "本次封存只固定输入；账本尚未实现，**不得把 active 状态当成绩效验证**",
    }


def build_inputs(run_id: str | None = None, snapshots_dir: str | None = None,
                 out_dir: str | None = None, write_main: bool = True) -> dict:
    """构造封存内容（`write_main=False` 时纯内存计算，不落盘，供 dry-run）。"""
    snapshots_dir = snapshots_dir or SNAPSHOTS_DIR
    out_dir = out_dir or PAPER_DIR
    snaps = _forward_snapshots(snapshots_dir)
    if not snaps:
        raise RuntimeError(f"未找到可作前向证据的正式快照（{snapshots_dir}）")
    if run_id:
        match = [s for s in snaps if s["run_id"] == run_id]
        if not match:
            raise RuntimeError(f"run_id={run_id} 不是可作前向证据的正式快照（检查 COMPLETE/status/"
                               f"signal_mode/是否重建）")
        snap = match[0]
    else:
        snap = snaps[-1]

    man = snap["manifest"]
    signal_date = str(man.get("signal_date"))
    cohort_id = signal_date[:7]
    score_file = None
    for fn in sorted(os.listdir(snap["dir"])):
        if fn.endswith(".csv") and fn[:7] == cohort_id:
            score_file = fn
            break
    if score_file is None:
        raise RuntimeError(f"快照 {snap['run_id']} 内找不到评分文件（{cohort_id}*.csv）")
    score_path = os.path.join(snap["dir"], score_file)
    scores = pd.read_csv(score_path, dtype={"fund_code": str})

    top = man.get("top50") or []
    if not top:
        raise RuntimeError(f"快照 {snap['run_id']} 的 manifest 缺 top50")
    rank_map = {}
    if "rank" in scores.columns:
        # 低置信度基金没有排名（NaN）——只对 main 记排名
        rank_map = {str(r.fund_code): (None if pd.isna(r.rank) else int(r.rank))
                    for r in scores.itertuples(index=False)}
    weight_per_fund = COHORT_WEIGHT / len(top)
    top50 = [{"fund_code": str(c), "rank": rank_map.get(str(c)), "weight": weight_per_fund}
             for c in top]

    main = scores[scores["confidence"] == "main"] if "confidence" in scores.columns else scores
    main = main.sort_values("rank") if "rank" in main.columns else main
    main_cols = [c for c in ("fund_code", "rank", "score", "as_of", "confidence") if c in main.columns]
    main_df = main[main_cols].copy()

    # main 全池对照集合：统一 \n 行尾，保证"内存哈希 == 落盘文件哈希"
    main_csv = main_df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    main_name = f"main_universe_{signal_date}.csv"
    main_path = os.path.join(out_dir, main_name)
    main_hash = hashlib.sha256(main_csv).hexdigest()
    if write_main:
        os.makedirs(out_dir, exist_ok=True)
        with open(main_path, "wb") as f:
            f.write(main_csv)

    payload = {
        "paper_inputs_version": PAPER_INPUTS_VERSION,
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": snap["run_id"],
        "signal_date": signal_date,
        "cohort_id": cohort_id,
        "source_snapshot": {
            "dir": snap["dir"],
            "manifest_sha256": _sha256(os.path.join(snap["dir"], "manifest.json")),
            "score_file": score_file,
            "score_file_sha256": _sha256(score_path),
            "complete_flag": True,
            "signal_mode": man.get("signal_mode"),
            "forward_eligible": man.get("forward_eligible", True),
            "data_cutoff": man.get("data_cutoff"),
            "factor_cutoff": man.get("factor_cutoff"),
            "stale_n": man.get("stale_n"), "gap_n": man.get("gap_n"),
        },
        "protocol": _protocol(),
        "n_codes": len(top50),
        "weight_per_fund": weight_per_fund,
        "top50": top50,
        "main_universe": {"count": int(len(main_df)), "file": main_name, "sha256": main_hash,
                          "note": "可执行全池对照集合（评分日 main 全集）；未来『相对全池』对照以它为准"},
        "hashes": {
            "manifest.json": _sha256(os.path.join(snap["dir"], "manifest.json")),
            score_file: _sha256(score_path),
            main_name: main_hash,
        },
        "frozen_by": "freeze_paper_inputs.py",
    }
    return payload


def freeze(run_id: str | None = None, snapshots_dir: str | None = None,
           out_dir: str | None = None, force: bool = False) -> dict:
    """封存（不可变）：已存在且内容一致 → 返回既有；内容不同 → 拒绝（--force 备份后重建）。"""
    out_dir = out_dir or PAPER_DIR
    os.makedirs(out_dir, exist_ok=True)
    payload = build_inputs(run_id, snapshots_dir, out_dir, write_main=True)
    path = os.path.join(out_dir, f"paper_inputs_{payload['run_id']}.json")
    if os.path.exists(path) and not force:
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        a, b = dict(old), dict(payload)
        a.pop("frozen_at", None)
        b.pop("frozen_at", None)
        if a == b:
            return {"status": "existing", "path": path, "run_id": payload["run_id"],
                    "note": "封存物已存在且内容一致（不可变，未改动）"}
        raise RuntimeError(
            f"已存在内容不同的封存文件：{path}——封存物不可变，请人工核对；"
            f"确需重建请显式 --force（会先备份旧文件）")
    if os.path.exists(path) and force:
        bak = path + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(path, bak)
        payload["rebuilt_from"] = os.path.basename(bak)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return {"status": "frozen", "path": path, "run_id": payload["run_id"],
            "signal_date": payload["signal_date"], "n_codes": payload["n_codes"],
            "main_universe": payload["main_universe"]}


def main():
    ap = argparse.ArgumentParser(description="封存首次前向纸面记账输入（不可变）")
    ap.add_argument("--run-id", default=None, help="来源正式快照 run_id（缺省=最近一个可作前向证据的月末快照）")
    ap.add_argument("--force", action="store_true", help="已存在不同内容时备份后重建（默认拒绝）")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要封存的内容，不落盘")
    args = ap.parse_args()

    if args.dry_run:
        payload = build_inputs(args.run_id, write_main=False)
        print(json.dumps(payload, ensure_ascii=False, indent=2)[:3000])
        return
    res = freeze(args.run_id, force=args.force)
    print("== freeze_paper_inputs ==")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
