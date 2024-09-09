import pandas as pd
from typing import Tuple, List
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import csv


import matplotlib.pyplot as plt


def excel_to_dataframe(
    file_path: str, sheet_name: str, header_start: Tuple[int, int] = (0, 0)
) -> pd.DataFrame:
    """
    Read an Excel file into a pandas DataFrame, specifying the starting row and column for the header.
    """
    # Read the Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    header_row, header_col = header_start

    # Set the header using the specified row
    headers = df.iloc[header_row, header_col:].tolist()

    headers = [h.date() if isinstance(h, datetime) else h.lower() for h in headers]

    new_df = pd.DataFrame(
        df.iloc[header_row + 1 :, header_col:].values, columns=headers
    )

    new_df = new_df.replace({pd.NaT: np.nan})
    datetime_columns = new_df.select_dtypes(include=["datetime64"]).columns

    for col in datetime_columns:
        new_df[col] = new_df[col].dt.normalize()

    return new_df


def summary(df, name="DataFrame"):
    """
    Display summary statistics for a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze
    name (str): A name for the DataFrame, used in print statements

    Returns:
    None: This function prints information but does not return any value
    """
    print(f"Summary Statistics for {name}")
    print("-" * 40)

    print(f"\nShape of {name}: {df.shape}")

    print(f"\nFirst few rows of {name}:")
    print(df.head().to_string())

    print(f"\nColumn names in {name}:")
    print(df.columns.tolist())

    print(f"\nData types of columns in {name}:")
    print(df.dtypes)

    print(f"\nSummary statistics for numeric columns in {name}:")
    print(df.describe().to_string())

    print(f"\nNon-null count and data types in {name}:")
    print(df.info())

    print(f"\nMissing values in each column of {name}:")
    print(df.isnull().sum())

    print("\nEnd of Summary Statistics")
    print("-" * 40)


def plot_distribution(
    df,
    product_column="product",
    title="Distribution of Products",
    xlabel="Product",
    ylabel="Product Count",
):
    """
    Plot the distribution of products in a DataFrame. Returns: matplotlib.figure.Figure:
    The figure object containing the plot
    """
    # Set the style
    sns.set_style("whitegrid", {"axes.facecolor": ".9"})

    # Create the plot
    plt.figure(figsize=(12, 6))
    g = sns.catplot(
        x=product_column,
        data=df,
        kind="count",
        hue=product_column,
        legend=False,
        aspect=1.5,
    )

    # Customize the plot
    g.ax.set(xlabel=xlabel, ylabel=ylabel)
    g.figure.suptitle(title, fontsize=16)

    # Color the bars
    n_colors = len(df[product_column].unique())
    palette = sns.color_palette("RdBu_r", n_colors)
    for i, bar in enumerate(g.ax.patches):
        bar.set_facecolor(palette[i % n_colors])

    # Adjust layout and return the figure
    plt.tight_layout()
    return g.figure


def plot_distribution_by_category(
    df, category_column, target_column=None, rotation=90, figsize=(12, 6)
):
    """
    Create distribution plots of a categorical variable, optionally split by a target variable.
    """
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Overall distribution
    sns.countplot(x=category_column, data=df, palette="GnBu_r", ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=rotation)
    ax1.set_title(f"Overall Distribution of {category_column}")
    ax1.set_ylabel("Count")

    # Distribution by target (if provided)
    if target_column:
        if df[target_column].dtype.kind in "bifc":  # Check if target is numeric
            # For numeric targets, calculate mean and plot as bar chart
            mean_by_category = (
                df.groupby(category_column)[target_column]
                .mean()
                .sort_values(ascending=False)
            )
            sns.barplot(
                x=mean_by_category.index,
                y=mean_by_category.values,
                palette="RdBu_r",
                ax=ax2,
            )
            ax2.set_title(f"Average {target_column} by {category_column}")
            ax2.set_ylabel(f"Average {target_column}")
        else:
            # For categorical targets, use countplot
            sns.countplot(
                x=category_column, hue=target_column, data=df, palette="RdBu_r", ax=ax2
            )
            ax2.set_title(f"Distribution of {category_column} by {target_column}")
            ax2.set_ylabel("Count")

        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=rotation)
    else:
        fig.delaxes(ax2)
        fig.set_size_inches(figsize[0] / 2, figsize[1])

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_loan_characteristics(df, x_col, y_col, hue_col, facet_col, title):
    """
    Create a FacetGrid scatter plot of loan characteristics.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the loan data
    x_col (str): Column name for x-axis
    y_col (str): Column name for y-axis
    hue_col (str): Column name for color differentiation
    facet_col (str): Column name for facet differentiation
    title (str): Title for the plot

    Returns:
    matplotlib.figure.Figure: The figure object containing the plot
    """
    g = sns.FacetGrid(df, col=facet_col, hue=hue_col, col_wrap=2, height=6, aspect=1.5)
    g.map(plt.scatter, x_col, y_col, alpha=0.5)
    g.add_legend()
    g.figure.suptitle(title, fontsize=16)
    plt.tight_layout()
    return g.figure


def merge_static_and_monthly(
    static_df: pd.DataFrame,
    monthly_dfs: List[pd.DataFrame],
    column_names: List[str],
) -> pd.DataFrame:
    """
    Merge static dataframe with multiple monthly dataframes.
    """
    # Ensure loan_id and month columns are consistent in types
    static_df["loan_id"] = static_df["loan_id"].astype(str)

    # Melt the first monthly dataframe
    melted = monthly_dfs[0].melt(
        id_vars=["loan_id"], var_name="month", value_name=column_names[0]
    )
    melted["loan_id"] = melted["loan_id"].astype(str)
    melted["month"] = pd.to_datetime(melted["month"], errors="coerce")

    # Merge subsequent monthly dataframes
    for df, col_name in zip(monthly_dfs[1:], column_names[1:]):
        temp = df.melt(id_vars=["loan_id"], var_name="month", value_name=col_name)
        temp["loan_id"] = temp["loan_id"].astype(str)
        temp["month"] = pd.to_datetime(temp["month"], errors="coerce")
        melted = pd.merge(melted, temp, on=["loan_id", "month"], how="outer")

    # Merge with static data
    result = pd.merge(melted, static_df, on="loan_id", how="left")

    # Convert payment_due and payment_made to float, handling non-numeric values
    if "payment_due" in result.columns:
        result["payment_due"] = pd.to_numeric(result["payment_due"], errors="coerce")
    if "payment_made" in result.columns:
        result["payment_made"] = pd.to_numeric(result["payment_made"], errors="coerce")

    # Sort and reset index
    result = result.sort_values(["loan_id", "month"]).reset_index(drop=True)

    return result


def calculate_current_balance(df):
    """
    Calculate the current_balance for each row in the dataframe.
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()

    # Sort the dataframe copy by loan_id and month
    df_copy = df_copy.sort_values(["loan_id", "month"])

    # Create a new column for current_balance
    df["current_balance"] = np.nan

    # Group by loan_id
    for _, loan_df in df_copy.groupby("loan_id"):
        # Find the first non-NaN Month_end_balance
        start_idx = loan_df["me_balances"].first_valid_index()
        if start_idx is not None:
            # Set the first current_balance
            df_copy.loc[start_idx, "current_balance"] = df_copy.loc[
                start_idx, "me_balances"
            ]
            # Calculate current_balance for subsequent months
            for idx in loan_df.index[loan_df.index > start_idx]:
                prev_balance = df_copy.loc[
                    df_copy.index[df_copy.index < idx].max(), "current_balance"
                ]
                payment_due = df_copy.loc[idx, "payment_due"]
                payment_made = df_copy.loc[idx, "payment_made"]
                df_copy.loc[idx, "current_balance"] = round(
                    prev_balance + payment_due - payment_made, 5
                )

    # Return the current_balance series, aligned with the original dataframe's index
    return df_copy["current_balance"].reindex(df.index)


def print_sample_loans(df, n_loans=3):
    """
    Print a sample of the dataframe for a specified number of loans,
    showing only rows where Month_end_balance is not NaN.
    """
    # Get a list of unique loan IDs
    unique_loans = df["loan_id"].unique()

    # Ensure we don't try to print more loans than exist
    n_loans = min(n_loans, len(unique_loans))

    # Select the first n_loans
    selected_loans = unique_loans[:n_loans]
    print(f"Printing sample data for {n_loans} loans:")
    print("selected_loans:", selected_loans)
    for loan in selected_loans:
        loan_df = df[(df["loan_id"] == loan) & (df["Month End Balances"].notna())]

        print(f"\nLoan ID: {loan}")
        print(loan_df[["month", "Month End Balances", "current_balance"]].head())
        print("-" * 80)  # Print a separator line


# Dynamic calculations (vary by loan and calendar month)
def calculate_seasoning(df):
    """
    Calculate seasoning for each row in the DataFrame.
    Only calculates for months on or after the origination date.
    Returns NaN for months before the origination date.
    """
    # Calculate the difference in months
    months_diff = (
        df["month"].dt.year * 12
        + df["month"].dt.month
        - df["origination_date"].dt.year * 12
        - df["origination_date"].dt.month
    )

    # Only keep positive differences (months on or after origination date)
    seasoning = months_diff.where(months_diff >= 0)

    return seasoning


def calculate_n_missed_payments(df):
    # Sort the DataFrame by loan_id and month
    df = df.sort_values(["loan_id", "month"])

    # Create a boolean column indicating missed payments
    df["missed_payment"] = (df["payment_due"] > 0) & (df["payment_made"] == 0)

    # Initialize a counter for missed payments
    df["n_missed"] = 0

    # Iterate over each loan_id group
    for _, group in df.groupby("loan_id"):
        missed_count = 0
        for idx, row in group.iterrows():
            if row["missed_payment"]:
                missed_count += (
                    1  # Increment the missed count for consecutive missed payments
                )
            else:
                missed_count = 0  # Reset the counter when payment is made
            df.at[idx, "n_missed"] = (
                missed_count  # Update the DataFrame with the counter
            )

    # Remove the temporary 'missed_payment' column
    df = df.drop(columns=["missed_payment"])
    return df["n_missed"]


def calculate_prepaid_in_month(df):
    """Flag indicating if the borrower prepaid in a given month."""
    return np.where(
        (df["payment_made"].notna()) & (df["payment_due"].notna()),
        (df["payment_made"] > df["payment_due"]) & (df["current_balance"] == 0),
        False,
    )


def calculate_default_in_month(df):
    """
    Flag indicating if the borrower defaulted in a given month.
    Default occurs when a borrower misses three payments in a row.
    Once default occurs, all subsequent months are also marked as default.
    """
    df = df.sort_values(["loan_id", "month"])

    df["default_in_month"] = df["n_missed_payments"] >= 3
    df["default_in_month"] = df.groupby("loan_id")["default_in_month"].cummax()

    return df["default_in_month"]


def calculate_recovery_in_month(df):
    """Flag indicating if a recovery has been made post-default in a given month."""
    df = df.sort_values(["loan_id", "month"])

    if "default_in_month" not in df.columns:
        df["default_in_month"] = calculate_default_in_month(df)

    return df["default_in_month"] & (df["payment_made"] > 0)


def calculate_is_recovery_payment(df):
    """
    Flag indicating if the associated payment has been made post-default.
    """
    return (
        df["default_in_month"] & df["payment_made"].notna() & (df["payment_made"] > 0)
    )


def calculate_time_to_reversion(df):
    """Calculate the integer number of months until the loan reverts."""
    d1, d2 = df["month"], df["reversion_date"]

    return d1.dt.year * 12 + d1.dt.month - d2.dt.year * 12 - d2.dt.month


def calculate_is_post_seller_purchase_date(df, seller_purchase_date="31/12/2020"):
    """Check if this time period is after the seller purchased this loan."""
    return pd.to_datetime(df["month"]) > pd.to_datetime(seller_purchase_date)


# Static calculations (vary by loan but are the same for each calendar month)
def calculate_date_of_default(df, column_name="date_of_default"):
    """Calculate the date of default for each loan and merge it back into the original dataframe."""
    default_dates = (
        df.loc[df["default_in_month"] == 1, ["loan_id", "month"]]
        .groupby("loan_id")
        .first()
    )
    default_dates.columns = [column_name]

    # Merge default dates back into the original dataframe
    df = df.merge(default_dates, on="loan_id", how="left")

    return df


def calculate_months_since_default(df, column_name="months_since_default"):
    """Calculate the number of months since default for each loan and merge it back into the original dataframe."""
    df[column_name] = np.nan

    mask = df["month"] >= df["date_of_default"]
    d1, d2 = df["month"], df["date_of_default"]
    df.loc[mask, column_name] = (
        d1.dt.year * 12 + d1.dt.month - d2.dt.year * 12 - d2.dt.month
    )

    return df[column_name]


def calculate_postdefault_recoveries(df, column_name="postdefault_recoveries"):
    """
    Calculate the cumulative recoveries post-default for each loan and merge it back into the original dataframe.
    """
    # Create a boolean mask for payments made after the default date
    post_default_mask = df["month"] > df["date_of_default"]

    # Calculate cumulative sum of payments made post-default for each loan
    post_default_payments = (
        df[(df["date_of_default"].notna()) & (post_default_mask)]
        .groupby("loan_id", as_index=False)
        .agg({"payment_made": "sum"})
    )

    post_default_payments.columns = ["loan_id", column_name]
    post_default_payments["loan_id"] = post_default_payments["loan_id"].astype("str")

    # Merge the post-default payments back into the original dataframe
    df = df.merge(post_default_payments, on=["loan_id"], how="left")

    # Fill NaN values with 0 for loans that haven't defaulted or for periods before default
    df[column_name] = df[column_name].fillna(0)

    return df


def calculate_prepayment_date(df, column_name="prepayment_date"):
    """
    Calculate the prepayment date for each loan and merge it back into the original dataframe.
    A prepayment is identified when payment_made > payment_due and current_balance == 0.
    """
    prepayment_dates = (
        df.loc[
            (df["payment_made"] > df["payment_due"]) & (df["current_balance"] == 0),
            ["loan_id", "month"],
        ]
        .groupby("loan_id")
        .first()
    )
    prepayment_dates.columns = [column_name]

    # Merge prepayment dates back into the original dataframe
    df = df.merge(prepayment_dates, on="loan_id", how="left")

    return df


def calculate_date_of_recovery(df, column_name="date_of_recovery"):
    """Calculate the date of recovery as the month of the last non-zero payment
    when the date_of_default is not NaN, and merge it back into the original dataframe.
    """

    # Initialize the 'date_of_recovery' column with NaN
    df[column_name] = np.nan

    # Filter loans with non-null date_of_default
    defaulted_loans = df[df["date_of_default"].notna()]

    # Group by loan_id to find the last non-zero payment for each loan
    for loan_id, group in defaulted_loans.groupby("loan_id"):
        # Filter rows with non-zero payments
        non_zero_payments = group[group["payment_made"] > 0]

        if not non_zero_payments.empty:
            # Find the last non-zero payment's month
            last_payment_month = non_zero_payments["month"].max()

            # Update the 'date_of_recovery' for all rows of that loan_id
            df.loc[df["loan_id"] == loan_id, column_name] = last_payment_month.date()

    return df


def calculate_exposure_at_default(df, column_name="exposure_at_default"):
    """
    Calculate the exposure at default (EAD) as the current balance on the date of default.
    If date_of_default is NaN, exposure_at_default is also NaN.
    """
    # Initialize the 'exposure_at_default' column with NaN
    df[column_name] = np.nan

    # Filter loans with non-null date_of_default
    defaulted_loans = df[df["date_of_default"].notna()]

    # Iterate over each loan_id group
    for loan_id, group in defaulted_loans.groupby("loan_id"):
        # Get the row where the month matches the date_of_default
        default_row = group[group["month"] == group["date_of_default"]]

        if not default_row.empty:
            # Get the current balance at the date_of_default
            exposure_at_default = default_row["current_balance"].values[0]

            # Update the 'exposure_at_default' for all rows of that loan_id
            df.loc[df["loan_id"] == loan_id, column_name] = exposure_at_default

    return df


def calculate_recovery_percent(df, column_name="recovery_percent"):
    """
    Calculate the recovery percentage as the ratio of postdefault_recoveries
    to the exposure at default (EAD) for each loan. If either is NaN, recovery_percent is also NaN.
    """
    # Initialize the 'recovery_percent' column with NaN
    df[column_name] = np.nan

    # Filter loans with non-null date_of_default and exposure_at_default
    valid_loans = df[
        (df["date_of_default"].notna()) & (df["exposure_at_default"].notna())
    ]

    # Calculate recovery_percent as (postdefault_recoveries / exposure_at_default) * 100
    df.loc[valid_loans.index, column_name] = (
        valid_loans["postdefault_recoveries"] / valid_loans["exposure_at_default"]
    ) * 100

    return df


def save_to_csv(df, loan_id="10"):
    temp_df = df[(df["loan_id"] == "10") & (df["Month End Balances"].notnull())][
        [
            "loan_id",
            "Month End Balances",
            "month",
            "current_balance",
            "payment_due",
            "payment_made",
            "prepayment_date",
        ]
    ]
    temp_df.to_csv(f"loan_{loan_id}.csv", index=False)


def create_pivot_table(aggregated_df, pivots, curve_vars):
    """
    Create a pivot table from the aggregated data.
    """

    pivot_cumulative = aggregated_df.pivot(
        index="seasoning",
        columns=pivots[1:],
        values=curve_vars["cumulative_rate_column"],
    )
    pivot_rate = aggregated_df.pivot(
        index="seasoning", columns=pivots[1:], values=curve_vars["rate_column"]
    )

    if len(pivots) > 2:
        pivot_cumulative.columns = [
            f"{curve_vars['cumulative_rate_column'].lower()}_{'_'.join(map(str, col))}"
            for col in pivot_cumulative.columns.values
        ]
        pivot_rate.columns = [
            f"{curve_vars['rate_column'].lower()}_{'_'.join(map(str, col))}"
            for col in pivot_rate.columns.values
        ]
    else:
        pivot_cumulative.columns = [
            f"{curve_vars['cumulative_rate_column'].lower()}_{col}"
            for col in pivot_cumulative.columns
        ]
        pivot_rate.columns = [
            f"{curve_vars['rate_column'].lower()}_{col}" for col in pivot_rate.columns
        ]

    pivot_table = pd.concat([pivot_cumulative, pivot_rate], axis=1)
    pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)
    pivot_table.reset_index(inplace=True)

    return pivot_table


def calculate_recovery_curve(df, pivots=None, include_default_vintage=False):
    """
    Calculate recovery curve with flexible pivot options and optional default vintage inclusion.

    :param df: Input DataFrame
    :param pivots: List of column names to pivot on (default: ["months_since_default"])
    :param include_default_vintage: Boolean, whether to include default vintage in calculations
    :return: DataFrame with calculated recovery curve
    """
    if pivots is None:
        pivots = ["months_since_default"]

    if include_default_vintage:
        df["year_of_default"] = df["date_of_default"].dt.year
        group_columns = ["year_of_default"] + pivots
    else:
        group_columns = pivots

    # Define aggregation functions
    agg_functions = {
        "postdefault_recoveries": lambda x: round(
            x[
                (df.loc[x.index, "months_since_default"] >= 0)
                & (df.loc[x.index, "is_recovery_payment"])
            ].sum(),
            5,
        ),
        "exposure_at_default": lambda x: round(
            x[df.loc[x.index, "months_since_default"] >= 0].sum(),
            5,
        ),
    }

    # Group by all pivots
    aggregated_df = df.groupby(group_columns).agg(agg_functions).reset_index()

    if include_default_vintage:
        # Calculate cumulative sum and recovery percentage within each group
        aggregated_df = (
            aggregated_df.groupby(["year_of_default"])
            .apply(
                lambda x: x.assign(
                    postdefault_recoveries_cumsum=x["postdefault_recoveries"].cumsum(),
                    recovery_percent=lambda df: (
                        df["postdefault_recoveries_cumsum"]
                        / df["exposure_at_default"].iloc[0]
                        * 100
                    ),
                )
            )
            .reset_index(drop=True)
        )
    else:
        aggregated_df["postdefault_recoveries_cumsum"] = aggregated_df[
            "postdefault_recoveries"
        ].cumsum()
        aggregated_df["recovery_percent"] = (
            aggregated_df["postdefault_recoveries_cumsum"]
            / aggregated_df.iloc[0]["exposure_at_default"]
            * 100
        )

    return aggregated_df


def validate_pivots(df, pivots, primary_pivot):
    """
    Validate and ensure the primary pivot is in the pivots list.
    """
    if pivots is None:
        pivots = [primary_pivot]
    elif primary_pivot not in pivots:
        pivots = [primary_pivot] + pivots

    valid_pivots = [pivot for pivot in pivots if pivot in df.columns]
    if len(valid_pivots) != len(pivots):
        invalid_pivots = set(pivots) - set(valid_pivots)
        print(f"Warning: The following pivots are not valid columns: {invalid_pivots}")
    return valid_pivots


def set_curve_variables(curve_types):
    """
    Set up curve-specific variables based on the curve types.
    """
    curve_vars = {}
    for curve_type in curve_types:
        if curve_type == "prepayment":
            curve_vars[curve_type] = {
                "event_column": "prepaid_in_month",
                "amount_column": "prepayment",
                "rate_column": "SMM",
                "cumulative_rate_column": "CPR",
            }
        elif curve_type == "default":
            curve_vars[curve_type] = {
                "event_column": "default_in_month",
                "amount_column": "amount_default",
                "rate_column": "MDR",
                "cumulative_rate_column": "CDR",
            }
        else:
            raise ValueError(f"Unsupported curve type: {curve_type}")
    return curve_vars


def calculate_amounts(df, curve_vars):
    """
    Calculate the amounts for the given curve types.
    """
    for curve_type, vars in curve_vars.items():
        if curve_type == "prepayment":
            df.loc[df[vars["event_column"]], vars["amount_column"]] = (
                df["payment_made"] - df["payment_due"]
            )
        elif curve_type == "default":
            df.loc[df[vars["event_column"]], vars["amount_column"]] = df[
                "current_balance"
            ]
    return df


def aggregate_data(df, pivots, curve_vars):
    """
    Aggregate data based on pivots and calculate rates.
    """
    agg_dict = {
        "current_balance": lambda x: round(
            x[
                (~df.loc[x.index, "default_in_month"])
                & (~df.loc[x.index, "prepaid_in_month"])
            ].sum(),
            5,
        )
    }

    for vars in curve_vars.values():
        agg_dict[vars["amount_column"]] = lambda x: round(x.sum(), 5)

    aggregated_df = df.groupby(pivots).agg(agg_dict).reset_index()

    # Drop rows where current_balance is 0
    aggregated_df = aggregated_df[aggregated_df["current_balance"] > 0]

    for vars in curve_vars.values():
        aggregated_df[vars["rate_column"]] = (
            aggregated_df[vars["amount_column"]] / aggregated_df["current_balance"]
        )
        aggregated_df[vars["cumulative_rate_column"]] = (
            1 - (1 - aggregated_df[vars["rate_column"]]) ** 12
        ) * 100

    return aggregated_df


def create_pivot_table(aggregated_df, pivots, curve_vars, primary_pivot):
    """
    Create a pivot table from the aggregated data.
    """
    if len(pivots) > 1:
        pivot_tables = []
        for curve_type, vars in curve_vars.items():
            pivot = aggregated_df.pivot(
                index=primary_pivot,
                columns=pivots[1:],
                values=vars["cumulative_rate_column"],
            )

            if len(pivots) > 2:
                pivot.columns = [
                    f"{vars['cumulative_rate_column'].lower()}_{'_'.join(map(str, col))}"
                    for col in pivot.columns.values
                ]
            else:
                pivot.columns = [
                    f"{vars['cumulative_rate_column'].lower()}_{col}"
                    for col in pivot.columns
                ]

            pivot_tables.append(pivot)

        pivot_table = pd.concat(pivot_tables, axis=1)
        pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)
        pivot_table.reset_index(inplace=True)
    else:
        pivot_table = aggregated_df

    return pivot_table


def calculate_curves(df, curve_types, pivots=None, primary_pivot="seasoning"):
    """
    Calculate financial curves based on input data.

    :param df: Input DataFrame
    :param curve_types: List of curve types to calculate (e.g., ['prepayment', 'default'])
    :param pivots: List of column names to pivot on (default: [primary_pivot])
    :param primary_pivot: Primary pivot column (default: 'seasoning')
    :return: DataFrame with calculated curves
    """
    pivots = validate_pivots(df, pivots, primary_pivot)
    curve_vars = set_curve_variables(curve_types)
    df = calculate_amounts(df, curve_vars)
    aggregated_df = aggregate_data(df, pivots, curve_vars)
    pivot_table = create_pivot_table(aggregated_df, pivots, curve_vars, primary_pivot)
    return pivot_table


def plot_curves(
    data,
    title,
    xlabel,
    ylabel,
    figsize=(10, 6),
    markers=None,
    colors=None,
    legend_loc="best",
):
    """
    Plot one or multiple curves on a single graph.

    :param data: List of dictionaries, each containing 'x', 'y', and 'label' for a curve
    :param title: String, title of the plot
    :param xlabel: String, label for x-axis
    :param ylabel: String, label for y-axis
    :param figsize: Tuple, figure size (width, height) in inches
    :param markers: List of marker styles for each curve (optional)
    :param colors: List of colors for each curve (optional)
    :param legend_loc: String or int, location of the legend (default: 'best')
    """
    plt.figure(figsize=figsize)

    for i, curve in enumerate(data):
        marker = markers[i] if markers and i < len(markers) else "o"
        color = colors[i] if colors and i < len(colors) else None
        plt.plot(
            curve["x"], curve["y"], marker=marker, color=color, label=curve["label"]
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    if any("label" in curve for curve in data):
        plt.legend(loc=legend_loc)

    plt.tight_layout()
    plt.show()


def read_csv_to_list(filename, column_index=1, skip_header=True, convert_to_float=True):
    """
    Read a CSV file and return its contents as a list of lists.
    """
    try:
        with open(filename, "r", newline="") as csvfile:
            csvreader = csv.reader(csvfile)

            if skip_header:
                next(csvreader)  # Skip the header row

            if convert_to_float:
                data = [float(row[column_index]) for row in csvreader]
            else:
                data = [int(row[column_index]) for row in csvreader]

        print(
            f"Successfully read {len(data)} values from column {column_index} in {filename}"
        )
        return data

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []
    except (csv.Error, ValueError, IndexError) as e:
        print(f"Error reading CSV file: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []
