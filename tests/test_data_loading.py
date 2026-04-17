from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from d3qn_stock.envs.make_env import load_price_data


class TestLoadPriceData(unittest.TestCase):
    def test_custom_stock_columns_are_mapped(self) -> None:
        csv_content = "\n".join(
            [
                "trade_date,open_px,high_px,low_px,close_px,vol",
                "2024-01-03,100,101,99,100.5,1000",
                "2024-01-02,99,100,98,99.8,1200",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample.csv"
            csv_path.write_text(csv_content)
            df = load_price_data(
                csv_path,
                close_column="close_px",
                date_column="trade_date",
                open_column="open_px",
                high_column="high_px",
                low_column="low_px",
                volume_column="vol",
            )

        self.assertEqual(list(df.columns), ["Date", "Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(df.iloc[0]["Date"].strftime("%Y-%m-%d"), "2024-01-02")
        self.assertAlmostEqual(float(df.iloc[1]["Close"]), 100.5)


if __name__ == "__main__":
    unittest.main()
