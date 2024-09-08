import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings


MAX_TERM = 180  # Assuming a maximum loan term of 15 years


# @numba.jit(nopython=True)
def calculate_monthly_payment(principal, annual_rate, term_months):
    monthly_rate = annual_rate / 12
    return (
        principal
        * (monthly_rate * (1 + monthly_rate) ** term_months)
        / ((1 + monthly_rate) ** term_months - 1)
    )


# @numba.jit(nopython=True)
def calculate_cashflows(
    balance, rate, term_months, cpr, cdr, recovery_rate, recovery_lag
):
    monthly_payment = calculate_monthly_payment(balance, rate, term_months)
    remaining_balance = balance
    cashflows = np.zeros(term_months)

    for month in range(term_months):
        if remaining_balance <= 0:
            break

        interest_payment = remaining_balance * (rate / 12)
        principal_payment = monthly_payment - interest_payment

        # Apply CPR
        prepayment = remaining_balance * (1 - (1 - cpr[month]) ** (1 / 12))

        # Apply CDR
        default = (remaining_balance - prepayment) * (1 - (1 - cdr[month]) ** (1 / 12))

        # Calculate recovery
        if month + recovery_lag < term_months:
            cashflows[month + recovery_lag] += default * recovery_rate[month]

        cashflows[month] += interest_payment + principal_payment + prepayment
        remaining_balance -= principal_payment + prepayment + default

    return cashflows


def results_to_dataframe(results, loans_df):
    """
    Convert the results to a DataFrame, accounting for different origination dates.
    """
    # Find the earliest origination date and the latest possible end date
    start_date = pd.to_datetime(loans_df["origination_date"].min())
    max_term = 360  # Assuming a maximum loan term of 30 years
    end_date = start_date + pd.DateOffset(months=max_term)

    full_date_range = pd.date_range(start=start_date, end=end_date, freq="ME")

    cashflows_dict = {
        f"Loan_{loan_id}": pd.Series(index=full_date_range, dtype=float)
        for loan_id, _ in results
    }

    for loan_id, cashflows in results:
        # Get the origination date for this loan
        loan_start_date = pd.to_datetime(
            loans_df.loc[loans_df["loan_id"] == loan_id, "origination_date"].iloc[0]
        )
        # Create a date range for this specific loan
        loan_date_range = pd.date_range(
            start=loan_start_date, periods=len(cashflows), freq="ME"
        )
        cashflows_dict[f"Loan_{loan_id}"].loc[loan_date_range] = cashflows

    df = pd.DataFrame(cashflows_dict, index=full_date_range)

    df = df.fillna(0)

    df["Total"] = df.sum(axis=1)
    return df


def run_model(
    loans_df,
    cpr_vector,
    cdr_vector,
    recovery_rate,
    recovery_lag,
    cpr_multiplier=1,
    cdr_multiplier=1,
    scenario=None,
):
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    results = []

    if isinstance(recovery_rate, float):
        recovery_rate = np.full(len(cpr_vector), recovery_rate)

    for _, loan in loans_df.iterrows():
        origination_date = pd.to_datetime(loan["origination_date"])
        reversion_date = pd.to_datetime(loan["reversion_date"])

        pre_reversion_months = (reversion_date.year - origination_date.year) * 12 + (
            reversion_date.month - origination_date.month
        )
        total_months = MAX_TERM  # Assuming 15-year loans
        total_months = min(MAX_TERM, len(cpr_vector))
        if scenario is not None:
            cpr_vector = scenario(cpr_vector, loan, pre_reversion_months)

        pre_reversion_rate = loan["pre_reversion_fixed_rate"]
        post_reversion_rate = (
            loan["post_reversion_boe_margin"] + 0.05
        )  # Assuming BOE base rate is 5%

        pre_reversion_cashflows = calculate_cashflows(
            loan["original_balance"],
            pre_reversion_rate,
            pre_reversion_months,
            cpr_vector[:pre_reversion_months],
            cdr_vector[:pre_reversion_months],
            recovery_rate,
            recovery_lag,
        )

        post_reversion_balance = loan["original_balance"] - np.sum(
            pre_reversion_cashflows
        )
        post_reversion_cashflows = calculate_cashflows(
            post_reversion_balance,
            post_reversion_rate,
            total_months - pre_reversion_months,
            cpr_vector[pre_reversion_months:] * cpr_multiplier,
            cdr_vector[pre_reversion_months:] * cdr_multiplier,
            recovery_rate,
            recovery_lag,
        )

        loan_cashflows = np.concatenate(
            [pre_reversion_cashflows, post_reversion_cashflows]
        )
        results.append((loan["loan_id"], loan_cashflows))

    return results


def aggregate_cashflows(results):
    max_length = max(len(cf) for _, cf in results)
    aggregated = np.zeros(max_length)

    for _, cashflows in results:
        aggregated[: len(cashflows)] += cashflows

    return aggregated


def get_cdr_vector(rate):
    return np.full(MAX_TERM, rate)


def get_cpr_vector(rate):
    return np.full(MAX_TERM, rate)


def get_recovery_rate_vector(rate):
    return np.full(MAX_TERM, rate)


def display_positive_cashflows(results_df, num_loans=5, min_cashflow=0):
    """
    Display cashflow information for multiple loans where the cashflow is greater than a specified minimum.
    """
    # Get columns that start with 'Loan_'
    loan_columns = [col for col in results_df.columns if col.startswith("Loan_")]

    # Select a subset of loans if there are more than num_loans
    if len(loan_columns) > num_loans:
        loan_columns = loan_columns[:num_loans]

    # Create a mask for rows where at least one loan has a cashflow > min_cashflow
    mask = results_df[loan_columns].gt(min_cashflow).any(axis=1)

    # Apply the mask and select only the loan columns
    filtered_df = results_df[mask][loan_columns]

    # Display the results
    print(f"Displaying cashflows > {min_cashflow} for {len(loan_columns)} loans:")
    print(filtered_df)
