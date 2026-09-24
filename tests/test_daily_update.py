# test_daily_update.py：daily_update 增量分支测试（P0：latest 单列增长率 / prev 无专属增长率不补）
# mock fetch_daily + 临时 NAV_DIR，不触碰真实缓存。
import os
import shutil
import sys
import unittest
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import daily_update  # noqa: E402


def _write_cache(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _daily_df(latest_nav, prev_nav, growth, latest="2026-09-24", prev="2026-09-23"):
    df = pd.DataFrame({
        f"{prev}-单位净值": [prev_nav], f"{latest}-单位净值": [latest_nav],
        "日增长率": [growth], f"{latest}-累计净值": [latest_nav + 0.1],
        f"{prev}-累计净值": [prev_nav + 0.1],
    }, index=pd.Index(["005555"], name="基金代码"))
    return df, [prev, latest]


class TestDailyUpdateBranches(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml",
                                f"_du_test_{os.getpid()}_{id(self)}")
        os.makedirs(self.tmp, exist_ok=True)
        self._orig = {k: getattr(daily_update, k) for k in ("NAV_DIR", "DEFAULT_POOL")}
        daily_update.NAV_DIR = self.tmp
        self.pool = os.path.join(self.tmp, "fund_code_list.csv")
        pd.DataFrame({"基金代码": ["005555"]}).to_csv(self.pool, index=False, encoding="utf-8-sig")

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(daily_update, k, v)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_latest_appends_official_single_growth(self):
        """P0：追加 latest 时增长率取官方单列"日增长率"（非空），不会被清洗删日。"""
        # 缓存到 9-23（前一交易日）
        _write_cache(os.path.join(self.tmp, "fund_005555.csv"), [
            {"date": "2026-09-22", "nav": 1.4000, "日增长率": 0.2, "nav_acc": 1.50},
            {"date": "2026-09-23", "nav": 1.5000, "日增长率": 0.5, "nav_acc": 1.60},
        ])
        with mock.patch.object(daily_update, "fetch_daily",
                               return_value=_daily_df(1.51, 1.5, 0.6)):
            stat = daily_update.run_daily_update(pool=self.pool, auto_catchup_max=0)
        d = pd.read_csv(os.path.join(self.tmp, "fund_005555.csv"), parse_dates=["date"])
        row = d[d["date"] == pd.Timestamp("2026-09-24")].iloc[0]
        self.assertEqual(stat["updated"], 1)
        self.assertEqual(stat["gap"], 0)
        self.assertAlmostEqual(row["nav"], 1.51)
        self.assertAlmostEqual(row["日增长率"], 0.6)          # 官方单列增长率，非空
        self.assertEqual(d["日增长率"].isna().sum(), 0)       # 无空增长率行（不会被清洗删日）

    def test_prev_without_daily_growth_not_appended(self):
        """P0/P0-1：补 prev 时无日期专属增长率 → 不追加（转 catch-up），避免污染/空值。"""
        _write_cache(os.path.join(self.tmp, "fund_005555.csv"), [
            {"date": "2026-09-22", "nav": 1.4000, "日增长率": 0.2, "nav_acc": 1.50},
        ])
        # 接口只给单列增长率（属于 latest=9-24），prev=9-23 无专属增长率列
        with mock.patch.object(daily_update, "fetch_daily",
                               return_value=_daily_df(1.51, 1.5, 0.6)):
            stat = daily_update.run_daily_update(pool=self.pool, auto_catchup_max=0)
        d = pd.read_csv(os.path.join(self.tmp, "fund_005555.csv"), parse_dates=["date"])
        self.assertNotIn(pd.Timestamp("2026-09-23"), set(d["date"]))   # 不追加 prev
        self.assertEqual(stat["no_growth_prev"], 1)                    # 转入 catch-up 清单
        self.assertEqual(stat["updated"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)