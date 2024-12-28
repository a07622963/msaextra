import streamlit as st
import pandas as pd
import numpy as np

# --------------------------------------------------
# Configuration: Define Metric Types and Display Names
# --------------------------------------------------
METRICS_INFO = {
    "Installations": "numeric",
    "Total_trails": "numeric",
    "Total_Paid": "numeric",
    "installTotrail": "percentage",
    "trialToPaid": "percentage",
    "installTopaid": "percentage",
    "Sum of USD_Amount": "currency",
    "installation_LTV": "currency"
}

DISPLAY_METRIC_NAMES = {
    "Installations": "Installations",
    "Total_trails": "Total Trials",
    "Total_Paid": "Total Paid",
    "installTotrail": "Install Trial",
    "trialToPaid": "Trial to Paid",
    "installTopaid": "Install to Paid",
    "Sum of USD_Amount": "Sum of USD Amount",
    "installation_LTV": "Installation LTV"
}

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def parse_numeric(series: pd.Series) -> pd.Series:
    """
    Convert numeric-like strings (e.g., "1,405") to floats.
    Non-convertible values become NaN.

    Args:
        series (pd.Series): Series containing numeric-like strings.

    Returns:
        pd.Series: Series with numeric values.
    """
    return pd.to_numeric(series.str.replace(",", "", regex=True), errors="coerce")

def parse_percentage(series: pd.Series) -> pd.Series:
    """
    Convert percentage strings (e.g., "21.31%") to floats.
    Handles missing '%' signs or invalid values gracefully.

    Args:
        series (pd.Series): Series containing percentage strings.

    Returns:
        pd.Series: Series with percentage values as floats.
    """
    return pd.to_numeric(series.astype(str).str.replace("%", "", regex=True), errors="coerce")

def parse_currency(series: pd.Series) -> pd.Series:
    """
    Convert currency strings (e.g., "$1,142.94") to floats.
    Removes non-numeric characters gracefully.

    Args:
        series (pd.Series): Series containing currency strings.

    Returns:
        pd.Series: Series with currency values as floats.
    """
    return pd.to_numeric(series.str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce")

def parse_and_aggregate(df: pd.DataFrame, column: str, col_type: str) -> float:
    """
    Parse a specific column based on its type and return an aggregated value.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column name to parse and aggregate.
        col_type (str): The type of the column ('numeric', 'currency', 'percentage').

    Returns:
        float: The aggregated value (sum or mean) or np.nan if not applicable.
    """
    if df.empty or column not in df.columns:
        return np.nan

    col_data = df[column].astype(str)

    if col_type == "numeric":
        return parse_numeric(col_data).sum(skipna=True)
    elif col_type == "currency":
        return parse_currency(col_data).sum(skipna=True)
    elif col_type == "percentage":
        return parse_percentage(col_data).mean(skipna=True)

    return np.nan

def color_change(val):
    """
    Returns a CSS style string based on the value.
    Positive values are green, negative values are red.

    Args:
        val (float): The value to evaluate.

    Returns:
        str: CSS style string.
    """
    if isinstance(val, (int, float)):
        if val > 0:
            return 'color: green'
        elif val < 0:
            return 'color: red'
    return ''

def float_formatter(x):
    """
    Formats float values to two decimal places with comma as thousand separator.
    Returns "N/A" for NaN values.

    Args:
        x (float): The float value to format.

    Returns:
        str: Formatted string.
    """
    return f"{x:,.2f}" if pd.notna(x) else "N/A"

@st.cache_data
def load_data(file) -> pd.DataFrame:
    """
    Loads CSV data and caches it for performance optimization.

    Args:
        file: The uploaded CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(file)

def get_experiment_labels(df: pd.DataFrame) -> list:
    """
    Retrieves unique experiment labels from the 'First Confirmed Experiment' column.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        list: A list of unique experiment labels.
    """
    if "First Confirmed Experiment" in df.columns:
        unique_experiments = df["First Confirmed Experiment"].dropna().unique()
        return list(unique_experiments)
    return []

# --------------------------------------------------
# Main Application
# --------------------------------------------------

def main():
    """
    The main function that orchestrates the Streamlit application.
    """
    # Set page configuration
    st.set_page_config(page_title="Experiment Comparison Dashboard", layout="wide")

    st.title("Experiment Comparison Dashboard")

    # Sidebar for file upload and filters
    with st.sidebar:
        st.header("File Upload and Filters")
        uploaded_file = st.file_uploader("Upload Experiment Data", type="csv")

    if uploaded_file:
        try:
            # Load and preprocess data
            df = load_data(uploaded_file)

            # Normalize 'first_lang' if present
            if "first_lang" in df.columns:
                df["first_lang"] = df["first_lang"].astype(str).str.lower().str.strip()

            # Retrieve experiment labels
            experiment_labels = get_experiment_labels(df)
            if len(experiment_labels) < 2:
                st.error("The uploaded file must contain at least two unique experiments in the 'First Confirmed Experiment' column.")
                return

            # Select Control and Test experiments
            with st.sidebar:
                exp1 = st.selectbox("Select Control Experiment", options=experiment_labels, index=0)
                exp2 = st.selectbox(
                    "Select Test Experiment", 
                    options=[exp for exp in experiment_labels if exp != exp1], 
                    index=0
                )

            # Split data into Control and Test
            df_a = df[df["First Confirmed Experiment"] == exp1].copy()
            df_b = df[df["First Confirmed Experiment"] == exp2].copy()

        except Exception as e:
            st.error(f"Error processing the CSV file: {e}")
            return

        # Apply Country Code filter if available
        df_a, df_b = apply_country_filter(df_a, df_b)

        # Apply First Language filter if available
        df_a, df_b = apply_language_filter(df_a, df_b)

        # Display DataFrames
        display_dataframes(exp1, df_a, exp2, df_b)

        # Metrics Comparison
        results_df = display_metrics_comparison(df_a, df_b, exp1, exp2)

        # Option to Download Results
        download_results(results_df)

        # Language Metrics Comparison
        display_language_metrics_comparison(df_a, df_b, exp1, exp2)

        # Country Metrics Comparison
        display_country_metrics_comparison(df_a, df_b, exp1, exp2)

        # Winner Determination Table based on Installation LTV
        display_winner_determination(df_a, df_b, exp1, exp2)

    else:
        st.info("Please upload a CSV file in the sidebar to begin.")

def apply_country_filter(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple:
    """
    Applies country code filters to both Control and Test DataFrames.

    Args:
        df_a (pd.DataFrame): Control experiment DataFrame.
        df_b (pd.DataFrame): Test experiment DataFrame.

    Returns:
        tuple: Filtered Control and Test DataFrames.
    """
    required_columns = {"Country Code"}
    if required_columns.issubset(df_a.columns) and required_columns.issubset(df_b.columns):
        all_country_codes = sorted(set(df_a["Country Code"].dropna()).union(df_b["Country Code"].dropna()))
        with st.sidebar:
            selected_country_codes = st.multiselect(
                "Filter by Country Code:",
                options=all_country_codes,
                default=all_country_codes
            )
        if selected_country_codes:
            df_a = df_a[df_a["Country Code"].isin(selected_country_codes)]
            df_b = df_b[df_b["Country Code"].isin(selected_country_codes)]
    return df_a, df_b

def apply_language_filter(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple:
    """
    Applies first language filters to both Control and Test DataFrames.

    Args:
        df_a (pd.DataFrame): Control experiment DataFrame.
        df_b (pd.DataFrame): Test experiment DataFrame.

    Returns:
        tuple: Filtered Control and Test DataFrames.
    """
    required_columns = {"first_lang"}
    if required_columns.issubset(df_a.columns) and required_columns.issubset(df_b.columns):
        all_first_langs = sorted(set(df_a["first_lang"].dropna()).union(df_b["first_lang"].dropna()))
        with st.sidebar:
            selected_first_langs = st.multiselect(
                "Filter by First Language:",
                options=all_first_langs,
                default=all_first_langs
            )
        if selected_first_langs:
            df_a = df_a[df_a["first_lang"].isin(selected_first_langs)]
            df_b = df_b[df_b["first_lang"].isin(selected_first_langs)]
    return df_a, df_b

def display_dataframes(exp1: str, df_a: pd.DataFrame, exp2: str, df_b: pd.DataFrame):
    """
    Displays the Control and Test DataFrames in expandable sections.

    Args:
        exp1 (str): Control experiment label.
        df_a (pd.DataFrame): Control experiment DataFrame.
        exp2 (str): Test experiment label.
        df_b (pd.DataFrame): Test experiment DataFrame.
    """
    with st.expander(exp1, expanded=False):
        st.dataframe(df_a, use_container_width=True)
    with st.expander(exp2, expanded=False):
        st.dataframe(df_b, use_container_width=True)

def display_metrics_comparison(df_a: pd.DataFrame, df_b: pd.DataFrame, exp1: str, exp2: str) -> pd.DataFrame:
    """
    Computes and displays the metrics comparison between Control and Test experiments.

    Args:
        df_a (pd.DataFrame): Control experiment DataFrame.
        df_b (pd.DataFrame): Test experiment DataFrame.
        exp1 (str): Control experiment label.
        exp2 (str): Test experiment label.

    Returns:
        pd.DataFrame: The comparison results DataFrame.
    """
    st.subheader("Metrics Comparison")
    results = []

    for metric, col_type in METRICS_INFO.items():
        if metric in df_a.columns or metric in df_b.columns:
            val_a = parse_and_aggregate(df_a, metric, col_type)
            val_b = parse_and_aggregate(df_b, metric, col_type)

            if pd.isna(val_a) and pd.isna(val_b):
                difference = pct_change = np.nan
            else:
                difference = val_b - val_a if not (pd.isna(val_b) or pd.isna(val_a)) else np.nan
                pct_change = (difference / val_a) * 100 if val_a != 0 and not pd.isna(val_a) else np.nan

            display_name = DISPLAY_METRIC_NAMES.get(metric, metric)
            results.append([display_name, val_a, val_b, difference, pct_change])
        else:
            display_name = DISPLAY_METRIC_NAMES.get(metric, metric)
            results.append([display_name, np.nan, np.nan, np.nan, np.nan])

    results_df = pd.DataFrame(
        results,
        columns=["Metric", f"{exp1}", f"{exp2}", f"Difference vs {exp1}", f"% Change vs {exp1}"]
    )

    styled_df = (
        results_df.style
        .applymap(color_change, subset=[f"Difference vs {exp1}", f"% Change vs {exp1}"])
        .format({
            f"{exp1}": float_formatter,
            f"{exp2}": float_formatter,
            f"Difference vs {exp1}": float_formatter,
            f"% Change vs {exp1}": lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
        })
    )
    st.dataframe(styled_df, use_container_width=True)

    return results_df

def download_results(results_df: pd.DataFrame):
    """
    Provides a download button for the metrics comparison results as a CSV file.

    Args:
        results_df (pd.DataFrame): The DataFrame containing the comparison results.
    """
    st.download_button(
        label="Download Comparison Results",
        data=results_df.to_csv(index=False).encode('utf-8'),
        file_name='metrics_comparison.csv',
        mime='text/csv',
    )

def display_language_metrics_comparison(df_a: pd.DataFrame, df_b: pd.DataFrame, exp1: str, exp2: str):
    """
    Computes and displays the metrics comparison segmented by language.

    Args:
        df_a (pd.DataFrame): Control experiment DataFrame.
        df_b (pd.DataFrame): Test experiment DataFrame.
        exp1 (str): Control experiment label.
        exp2 (str): Test experiment label.
    """
    st.subheader("Language Metrics Comparison")
    required_columns = {"first_lang"}
    if required_columns.issubset(df_a.columns) and required_columns.issubset(df_b.columns):
        unique_languages = sorted(set(df_a["first_lang"].dropna()).intersection(df_b["first_lang"].dropna()))
        languages_to_compare = ["ar", "en"]

        if all(lang in unique_languages for lang in languages_to_compare):
            merged_metrics = []

            for metric, col_type in METRICS_INFO.items():
                if metric in df_a.columns and metric in df_b.columns:
                    # AR Language Metrics
                    ar_a = parse_and_aggregate(df_a[df_a["first_lang"] == "ar"], metric, col_type)
                    ar_b = parse_and_aggregate(df_b[df_b["first_lang"] == "ar"], metric, col_type)
                    pct_change_ar = ((ar_b - ar_a) / ar_a) * 100 if ar_a not in [0, np.nan] else np.nan

                    # EN Language Metrics
                    en_a = parse_and_aggregate(df_a[df_a["first_lang"] == "en"], metric, col_type)
                    en_b = parse_and_aggregate(df_b[df_b["first_lang"] == "en"], metric, col_type)
                    pct_change_en = ((en_b - en_a) / en_a) * 100 if en_a not in [0, np.nan] else np.nan

                    display_name = DISPLAY_METRIC_NAMES.get(metric, metric)
                    merged_metrics.append([display_name, pct_change_ar, pct_change_en])

            merged_metrics_df = pd.DataFrame(
                merged_metrics,
                columns=["Metric", "AR - % Change", "EN - % Change"]
            )

            styled_merged_df = (
                merged_metrics_df.style
                .applymap(color_change, subset=["AR - % Change", "EN - % Change"])
                .format({
                    "AR - % Change": lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A",
                    "EN - % Change": lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
                })
            )
            st.dataframe(styled_merged_df, use_container_width=True)
        else:
            missing_langs = [lang.upper() for lang in languages_to_compare if lang not in unique_languages]
            st.warning(f"The required languages ({', '.join(missing_langs)}) are not available in the datasets.")
    else:
        st.warning("Missing 'first_lang' column in one of the datasets.")

def display_country_metrics_comparison(df_a: pd.DataFrame, df_b: pd.DataFrame, exp1: str, exp2: str):
    """
    Computes and displays the metrics comparison segmented by country.

    Args:
        df_a (pd.DataFrame): Control experiment DataFrame.
        df_b (pd.DataFrame): Test experiment DataFrame.
        exp1 (str): Control experiment label.
        exp2 (str): Test experiment label.
    """
    st.subheader("Country Metrics Comparison")
    required_columns = {"Country Code"}
    if required_columns.issubset(df_a.columns) and required_columns.issubset(df_b.columns):
        unique_countries = sorted(set(df_a["Country Code"].dropna()).intersection(df_b["Country Code"].dropna()))

        if unique_countries:
            metrics_list = [
                "Installations",
                "Total Trials",
                "Total Paid",
                "Install Trial",
                "Trial to Paid",
                "Install to Paid",
                "Sum of USD Amount",
                "Installation LTV"
            ]
            merged_country_metrics = []

            for country in unique_countries:
                df_a_country = df_a[df_a["Country Code"] == country]
                df_b_country = df_b[df_b["Country Code"] == country]
                total_paid_a = parse_and_aggregate(df_a_country, "Total_Paid", METRICS_INFO["Total_Paid"])

                if pd.isna(total_paid_a) or total_paid_a == 0:
                    continue  # Exclude countries with Total_Paid=0 in Control

                country_metrics = [country]

                for metric in metrics_list:
                    metric_key = next((k for k, v in DISPLAY_METRIC_NAMES.items() if v == metric), None)
                    if metric_key and metric_key in METRICS_INFO:
                        col_type = METRICS_INFO[metric_key]
                        a_val = parse_and_aggregate(df_a_country, metric_key, col_type)
                        b_val = parse_and_aggregate(df_b_country, metric_key, col_type)

                        if pd.isna(a_val) or a_val == 0:
                            pct_change = np.nan
                        else:
                            pct_change = ((b_val - a_val) / a_val) * 100 if not pd.isna(b_val) else np.nan

                        country_metrics.append(pct_change)
                    else:
                        country_metrics.append(np.nan)

                merged_country_metrics.append(country_metrics)

            if merged_country_metrics:
                merged_country_metrics_df = pd.DataFrame(
                    merged_country_metrics,
                    columns=[
                        "Country",
                        "Installations",
                        "Total Trials",
                        "Total Paid",
                        "Install Trial",
                        "Trial to Paid",
                        "Install to Paid",
                        "Sum of USD Amount",
                        "Installation LTV"
                    ]
                )

                styled_country_metrics_df = (
                    merged_country_metrics_df.style
                    .applymap(color_change, subset=merged_country_metrics_df.columns[1:])
                    .format({
                        metric: lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A" for metric in merged_country_metrics_df.columns[1:]
                    })
                )
                st.dataframe(styled_country_metrics_df, use_container_width=True)
            else:
                st.warning("No countries to display after excluding those with Total Paid = 0.")
        else:
            st.warning("No common countries found in both datasets after applying filters.")
    else:
        st.warning("Missing 'Country Code' column in one of the datasets.")

def display_winner_determination(df_a: pd.DataFrame, df_b: pd.DataFrame, exp1: str, exp2: str):
    """
    Computes and displays the Winner Determination table based on 'Installation LTV',
    segmented by Country and Language.

    Args:
        df_a (pd.DataFrame): Control experiment DataFrame.
        df_b (pd.DataFrame): Test experiment DataFrame.
        exp1 (str): Control experiment label.
        exp2 (str): Test experiment label.
    """
    st.subheader("Winner Determination by Country and Language")

    # Ensure necessary columns exist
    required_columns = {"Country Code", "first_lang", "installation_LTV"}
    if not required_columns.issubset(df_a.columns) or not required_columns.issubset(df_b.columns):
        st.warning("Missing one of the required columns ('Country Code', 'first_lang', 'installation_LTV') in the datasets.")
        return

    # Merge unique combinations of Country and Language from both datasets
    unique_combinations = pd.concat([
        df_a[['Country Code', 'first_lang']].drop_duplicates(),
        df_b[['Country Code', 'first_lang']].drop_duplicates()
    ]).drop_duplicates()

    winner_data = []

    for _, row in unique_combinations.iterrows():
        country = row['Country Code']
        language = row['first_lang']

        # Filter data for the current Country and Language
        df_a_subset = df_a[(df_a['Country Code'] == country) & (df_a['first_lang'] == language)]
        df_b_subset = df_b[(df_b['Country Code'] == country) & (df_b['first_lang'] == language)]

        # Aggregate Installation LTV
        ltv_a = parse_and_aggregate(df_a_subset, "installation_LTV", METRICS_INFO["installation_LTV"])
        ltv_b = parse_and_aggregate(df_b_subset, "installation_LTV", METRICS_INFO["installation_LTV"])

        # Determine Winner
        if pd.isna(ltv_a) and pd.isna(ltv_b):
            winner = "N/A"
            difference = np.nan
            pct_change = np.nan
        elif pd.isna(ltv_a):
            winner = exp2
            difference = ltv_b
            pct_change = np.nan
        elif pd.isna(ltv_b):
            winner = exp1
            difference = -ltv_a
            pct_change = np.nan
        else:
            difference = ltv_b - ltv_a
            pct_change = (difference / ltv_a) * 100 if ltv_a != 0 else np.nan
            if ltv_b > ltv_a:
                winner = exp2
            elif ltv_b < ltv_a:
                winner = exp1
            else:
                winner = "Tie"

        winner_data.append({
            "Country": country,
            "Language": language.upper(),
            f"{exp1} Installation LTV": ltv_a,
            f"{exp2} Installation LTV": ltv_b,
            "Difference": difference,
            "% Change": pct_change,
            "Winner": winner
        })

    winner_df = pd.DataFrame(winner_data)

    # Define a function to determine the color based on the winner
    def winner_color(row):
        if row["Winner"] == exp2:
            return ['color: green'] * len(row)
        elif row["Winner"] == exp1:
            return ['color: red'] * len(row)
        else:
            return [''] * len(row)

    # Apply styling
    styled_winner_df = (
        winner_df.style
        .apply(winner_color, axis=1)
        .format({
            f"{exp1} Installation LTV": float_formatter,
            f"{exp2} Installation LTV": float_formatter,
            "Difference": float_formatter,
            "% Change": lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A",
            "Winner": lambda x: f"🟢 {x}" if x == exp2 else (f"🔴 {x}" if x == exp1 else x)
        })
    )

    st.dataframe(styled_winner_df, use_container_width=True)

if __name__ == "__main__":
    main()
