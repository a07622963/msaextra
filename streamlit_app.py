import streamlit as st
import pandas as pd
import numpy as np

# --------------------------------------------------
# Configuration: Define metric types and display names
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
    Convert a series of numeric-like strings (e.g., "582", "1,405") to floats.
    Non-convertible values become NaN.
    """
    return pd.to_numeric(series.str.replace(",", "", regex=True), errors="coerce")

def parse_percentage(series: pd.Series) -> pd.Series:
    """
    Convert a series of percentage strings (e.g. "21.31%") to floats in 0–100 range.
    Handles missing % signs or invalid values gracefully.
    """
    return pd.to_numeric(series.astype(str).str.replace("%", "", regex=True), errors="coerce")

def parse_currency(series: pd.Series) -> pd.Series:
    """
    Convert currency strings (e.g. "$1,142.94") to float (1142.94).
    Handles any non-numeric characters gracefully.
    """
    return pd.to_numeric(series.str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce")

def parse_and_aggregate(df: pd.DataFrame, column: str, col_type: str) -> float:
    """
    Parse a specific column from df according to col_type,
    then return an aggregated value:
      - numeric/currency => SUM of that column
      - percentage       => AVERAGE of that column
    If df is empty or column doesn't exist, return np.nan.
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
    """
    return f"{x:,.2f}" if pd.notna(x) else "N/A"

@st.cache_data
def load_data(file):
    """
    Caches the CSV loading to optimize performance.
    """
    return pd.read_csv(file)

def get_experiment_labels(df):
    """
    Retrieves unique experiment labels from the 'First Confirmed Experiment' column.
    Returns a list of unique experiments.
    """
    if "First Confirmed Experiment" in df.columns:
        unique_experiments = df["First Confirmed Experiment"].dropna().unique()
        return list(unique_experiments)
    return []

# --------------------------------------------------
# Main Application
# --------------------------------------------------
def main():
    # Set page layout to wide
    st.set_page_config(layout="wide")
    
    st.title("Experiment Comparison Dashboard")

    # Sidebar for file upload and experiment selection
    with st.sidebar:
        st.header("File Upload and Filters")
        uploaded_file = st.file_uploader("Upload Experiment Data", type="csv")

    if uploaded_file:
        try:
            # Load data
            df = load_data(uploaded_file)

            # Normalize 'first_lang' to lowercase and strip whitespace
            if "first_lang" in df.columns:
                df["first_lang"] = df["first_lang"].astype(str).str.lower().str.strip()

            # Retrieve experiment labels
            experiment_labels = get_experiment_labels(df)
            if len(experiment_labels) < 2:
                st.error("The uploaded file must contain at least two unique experiments in the 'First Confirmed Experiment' column.")
                return

            # Allow user to select Control and Test experiments
            with st.sidebar:
                exp1 = st.selectbox("Select Control Experiment", options=experiment_labels, index=0)
                exp2 = st.selectbox("Select Test Experiment", options=[exp for exp in experiment_labels if exp != exp1], index=0)

            # Split the DataFrame into Experiment A (Control) and Experiment B (Test)
            df_a = df[df["First Confirmed Experiment"] == exp1].copy()
            df_b = df[df["First Confirmed Experiment"] == exp2].copy()

        except Exception as e:
            st.error(f"Error processing the CSV file: {e}")
            return

        # Filter by Country Code
        if {"Country Code"}.issubset(df_a.columns) and {"Country Code"}.issubset(df_b.columns):
            all_country_codes = sorted(set(df_a["Country Code"].dropna()).union(df_b["Country Code"].dropna()))
            with st.sidebar:
                selected_country_codes = st.multiselect(
                    "Filter by Country Code:",
                    all_country_codes,
                    default=all_country_codes
                )
            if selected_country_codes:
                df_a = df_a[df_a["Country Code"].isin(selected_country_codes)]
                df_b = df_b[df_b["Country Code"].isin(selected_country_codes)]

        # Filter by First Language
        if {"first_lang"}.issubset(df_a.columns) and {"first_lang"}.issubset(df_b.columns):
            all_first_langs = sorted(set(df_a["first_lang"].dropna()).union(df_b["first_lang"].dropna()))
            with st.sidebar:
                selected_first_langs = st.multiselect(
                    "Filter by First Language:",
                    all_first_langs,
                    default=all_first_langs
                )
            if selected_first_langs:
                df_a = df_a[df_a["first_lang"].isin(selected_first_langs)]
                df_b = df_b[df_b["first_lang"].isin(selected_first_langs)]

        # Display dataframes with enhanced labels
        with st.expander(exp1, expanded=False):
            st.dataframe(df_a, use_container_width=True)
        with st.expander(exp2, expanded=False):
            st.dataframe(df_b, use_container_width=True)

        # Metrics Comparison
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

        # Option to download results
        st.download_button(
            label="Download Comparison Results",
            data=results_df.to_csv(index=False).encode('utf-8'),
            file_name='metrics_comparison.csv',
            mime='text/csv',
        )

        # Language Metrics Comparison
        st.subheader("Language Metrics Comparison")
        if {"first_lang"}.issubset(df_a.columns) and {"first_lang"}.issubset(df_b.columns):
            unique_languages = sorted(set(df_a["first_lang"].dropna()).intersection(df_b["first_lang"].dropna()))
            languages_to_compare = ["ar", "en"]
            if all(lang in unique_languages for lang in languages_to_compare):
                merged_metrics = []
                for metric, col_type in METRICS_INFO.items():
                    if metric in df_a.columns and metric in df_b.columns:
                        # Aggregate metrics for AR
                        df_a_ar = df_a[df_a["first_lang"] == "ar"]
                        df_b_ar = df_b[df_b["first_lang"] == "ar"]
                        ar_a = parse_and_aggregate(df_a_ar, metric, col_type)
                        ar_b = parse_and_aggregate(df_b_ar, metric, col_type)
                        pct_change_ar = ((ar_b - ar_a) / ar_a) * 100 if ar_a != 0 and not pd.isna(ar_a) else np.nan

                        # Aggregate metrics for EN
                        df_a_en = df_a[df_a["first_lang"] == "en"]
                        df_b_en = df_b[df_b["first_lang"] == "en"]
                        en_a = parse_and_aggregate(df_a_en, metric, col_type)
                        en_b = parse_and_aggregate(df_b_en, metric, col_type)
                        pct_change_en = ((en_b - en_a) / en_a) * 100 if en_a != 0 and not pd.isna(en_a) else np.nan

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

        # Country Metrics Comparison
        st.subheader("Country Metrics Comparison")
        if {"Country Code"}.issubset(df_a.columns) and {"Country Code"}.issubset(df_b.columns):
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
                        continue  # Exclude country with Total_Paid=0 in Experiment A

                    country_metrics = [country]
                    for metric in metrics_list:
                        metric_key = next((k for k, v in DISPLAY_METRIC_NAMES.items() if v == metric), None)
                        if metric_key and metric_key in METRICS_INFO:
                            col_type = METRICS_INFO[metric_key]
                            a_val = parse_and_aggregate(df_a_country, metric_key, col_type)
                            b_val = parse_and_aggregate(df_b_country, metric_key, col_type)
                            if pd.isna(a_val) and pd.isna(b_val):
                                pct_change = np.nan
                            elif a_val == 0 or pd.isna(a_val):
                                pct_change = np.nan
                            else:
                                pct_change = ((b_val - a_val) / a_val) * 100 if not pd.isna(b_val - a_val) else np.nan
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
                        .applymap(color_change, subset=metrics_list)
                        .format({
                            metric: lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A" for metric in metrics_list
                        })
                    )
                    st.dataframe(styled_country_metrics_df, use_container_width=True)
                else:
                    st.warning("No countries to display after excluding those with Total Paid = 0.")
            else:
                st.warning("No common countries found in both datasets after applying filters.")
        else:
            st.warning("Missing 'Country Code' column in one of the datasets.")
    else:
        st.info("Please upload a CSV file in the sidebar to begin.")

if __name__ == "__main__":
    main()
