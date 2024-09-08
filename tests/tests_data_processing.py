import unittest
import pandas as pd
import numpy as np
from datetime import datetime, date
from src.data_processing import (
    excel_to_dataframe,
    merge_static_and_monthly,
    calculate_seasoning,
    calculate_time_to_reversion,
    calculate_is_post_seller_purchase_date,
    calculate_n_missed_payments,
    calculate_default_in_month,
    calculate_recovery_in_month,
)


class TestDataProcessing(unittest.TestCase):

    def test_excel_to_dataframe(self):
        # This test would require a sample Excel file
        # For now, we'll just check if the function exists
        self.assertTrue(callable(excel_to_dataframe))

    def test_merge_static_and_monthly(self):
        static_df = pd.DataFrame({"loan_id": [1, 2], "static_data": ["A", "B"]})
        monthly_df1 = pd.DataFrame(
            {"loan_id": [1, 2], "2022-01-01": [100, 200], "2022-02-01": [101, 201]}
        )
        monthly_df2 = pd.DataFrame(
            {"loan_id": [1, 2], "2022-01-01": [10, 20], "2022-02-01": [11, 21]}
        )

        result = merge_static_and_monthly(
            static_df, [monthly_df1, monthly_df2], ["Value1", "Value2"]
        )

        self.assertEqual(result.shape, (4, 5))
        self.assertTrue("month" in result.columns)
        self.assertTrue("Value1" in result.columns)
        self.assertTrue("Value2" in result.columns)

    def test_calculate_seasoning(self):
        df = pd.DataFrame(
            {
                "month": pd.to_datetime(["2022-03-01", "2022-04-01", "2022-05-01"]),
                "origination_date": pd.to_datetime(
                    ["2022-01-01", "2022-01-01", "2022-06-01"]
                ),
            }
        )
        result = calculate_seasoning(df)
        expected = pd.Series([2, 3, np.nan])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_calculate_time_to_reversion(self):
        df = pd.DataFrame(
            {
                "month": pd.to_datetime(["2022-03-01"]),
                "reversion_date": pd.to_datetime(["2022-06-01"]),
            }
        )
        result = calculate_time_to_reversion(df)
        expected = pd.Series([-3], dtype="int32")
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_calculate_is_post_seller_purchase_date(self):
        row = {
            "month": pd.to_datetime("2022-03-01"),
            "investor_1_acquisition_date": pd.to_datetime("2022-01-01"),
        }
        self.assertTrue(calculate_is_post_seller_purchase_date(row))

    def test_calculate_n_missed_payments(self):
        df = pd.DataFrame(
            {
                "loan_id": [1, 1, 1, 2, 2, 2],
                "month": [
                    "2023-01-01",
                    "2023-02-01",
                    "2023-03-01",
                    "2023-01-01",
                    "2023-02-01",
                    "2023-03-01",
                ],
                "payment_due": [100, 100, 100, 200, 200, 200],
                "payment_made": [0, 100, 0, 200, 0, 200],
            }
        )
        result = calculate_n_missed_payments(df)
        self.assertEqual(list(result), [1, 0, 1, 0, 1, 0])

    def test_calculate_default_in_month(self):
        df = pd.DataFrame(
            {
                "loan_id": [1, 1, 1, 1, 2, 2, 2, 2],
                "month": [
                    "2023-01-01",
                    "2023-02-01",
                    "2023-03-01",
                    "2023-04-01",
                    "2023-01-01",
                    "2023-02-01",
                    "2023-03-01",
                    "2023-04-01",
                ],
                "payment_due": [100, 100, 100, 100, 200, 200, 200, 200],
                "payment_made": [100, 0, 0, 0, 200, 0, 0, 200],
            }
        )
        df["n_missed_payments"] = calculate_n_missed_payments(df)
        result = calculate_default_in_month(df)
        self.assertEqual(
            list(result), [False, False, False, True, False, False, False, False]
        )

    def test_calculate_recovery_in_month(self):
        df = pd.DataFrame(
            {
                "loan_id": [1, 1, 1, 1, 2, 2, 2, 2],
                "month": [
                    "2023-01-01",
                    "2023-02-01",
                    "2023-03-01",
                    "2023-04-01",
                    "2023-01-01",
                    "2023-02-01",
                    "2023-03-01",
                    "2023-04-01",
                ],
                "default_in_month": [False, True, True, True, False, True, True, True],
                "payment_made": [100, 0, 50, 0, 200, 0, 0, 100],
            }
        )
        result = calculate_recovery_in_month(df)
        self.assertEqual(
            list(result), [False, False, True, False, False, False, False, True]
        )


if __name__ == "__main__":
    unittest.main()
