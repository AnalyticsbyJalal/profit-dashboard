import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================================================
# BASIC APP CONFIG
# =========================================================
st.set_page_config(
    page_title="Business Profit Dashboard",
    page_icon="📊",
    layout="wide",
)

# Light / corporate default colors are controlled mainly by Streamlit theme
# in .streamlit/config.toml on your repo, not here. This app assumes your
# config.toml uses a light theme.


# =========================================================
# HELPERS
# =========================================================
@st.cache_data
def load_file(uploaded_file: io.BytesIO) -> pd.DataFrame:
    """Load CSV or Excel into a DataFrame."""
    if uploaded_file is None:
        return pd.DataFrame()

    name = uploaded_file.name.lower()

    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            # Requires openpyxl in requirements.txt
            try:
                return pd.read_excel(uploaded_file, engine="openpyxl")
            except ImportError:
                st.error(
                    "Missing optional dependency `openpyxl` for Excel files.\n\n"
                    "Add this line to your `requirements.txt` on GitHub:\n\n"
                    "`openpyxl`"
                )
                return pd.DataFrame()
        else:
            st.error("Unsupported file type. Please upload CSV or XLSX.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading file **{uploaded_file.name}**: {e}")
        return pd.DataFrame()


def prepare_data(df: pd.DataFrame,
                 revenue_col: str,
                 cost_col: str,
                 date_col: str,
                 product_col: str) -> pd.DataFrame:
    """
    Standardize columns and compute revenue / cost / profit.

    Validates that each mapping (revenue, cost, date, product) is a different
    column to avoid duplicate names like '__revenue__'.
    """

    # --- Validate mapping: all roles must be different columns ---
    mapping = {
        "Revenue": revenue_col,
        "Cost": cost_col,
        "Date": date_col,
        "Product": product_col,
    }
    if len(set(mapping.values())) < 4:
        msg = (
            "Each mapping (**Revenue**, **Cost**, **Date**, **Product**) "
            "must point to a **different column**.\n\n"
            "Current selection:\n"
            f"- Revenue → `{revenue_col}`\n"
            f"- Cost → `{cost_col}`\n"
            f"- Date → `{date_col}`\n"
            f"- Product → `{product_col}`\n\n"
            "Please choose four different columns in the mapping section above."
        )
        st.error(msg)
        st.stop()

    df = df.copy()

    # Rename to internal names used everywhere else
    df.rename(
        columns={
            revenue_col: "__revenue__",
            cost_col: "__cost__",
            date_col: "__date__",
            product_col: "product",
        },
        inplace=True,
    )

    # Ensure numeric
    df["__revenue__"] = pd.to_numeric(df["__revenue__"], errors="coerce").fillna(0.0)
    df["__cost__"] = pd.to_numeric(df["__cost__"], errors="coerce").fillna(0.0)

    # Parse dates
    df["__date__"] = pd.to_datetime(df["__date__"], errors="coerce")
    df = df[~df["__date__"].isna()]

    # Profit
    df["__profit__"] = df["__revenue__"] - df["__cost__"]

    # Month-year helper
    df["month"] = df["__date__"].dt.to_period("M").dt.to_timestamp()

    return df


def build_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    ms = (
        df.groupby("month")[["__revenue__", "__cost__", "__profit__"]]
        .sum()
        .reset_index()
        .rename(
            columns={
                "month": "Month",
                "__revenue__": "Revenue",
                "__cost__": "Cost",
                "__profit__": "Profit",
            }
        )
    )
    ms["Margin %"] = np.where(
        ms["Revenue"] != 0,
        ms["Profit"] / ms["Revenue"] * 100,
        0.0,
    )
    return ms


def build_product_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    ps = (
        df.groupby("product")[["__revenue__", "__cost__", "__profit__"]]
        .sum()
        .reset_index()
        .rename(
            columns={
                "product": "Product",
                "__revenue__": "Revenue",
                "__cost__": "Cost",
                "__profit__": "Profit",
            }
        )
    )
    ps["Margin %"] = np.where(
        ps["Revenue"] != 0,
        ps["Profit"] / ps["Revenue"] * 100,
        0.0,
    )
    ps = ps.sort_values("Profit", ascending=False)
    return ps


def describe_trend(series: pd.Series) -> str:
    if len(series) < 3:
        return "flat"

    last3 = series.tail(3)
    x = np.arange(len(last3))
    coef = np.polyfit(x, last3.values, 1)[0]

    if coef > 0:
        return "up"
    elif coef < 0:
        return "down"
    return "flat"


# =========================================================
# SIDEBAR: LOGO + FILE UPLOAD + PAGE NAV
# =========================================================
with st.sidebar:
    # Logo if logo.png exists
    try:
        st.image("logo.png", use_column_width=True)
    except Exception:
        st.markdown("### Analytics by Jalal")

    st.markdown("### Navigation")
    page = st.radio(
        "",
        ["Dashboard", "Raw Data", "Settings / Info"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### Upload your data")
    uploaded_files = st.file_uploader(
        "Upload one or more CSV/XLSX files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

# =========================================================
# LOAD + COMBINE RAW DATA
# =========================================================
if uploaded_files:
    dfs = []
    for f in uploaded_files:
        df_part = load_file(f)
        if not df_part.empty:
            df_part["__source_file__"] = f.name
            dfs.append(df_part)

    if dfs:
        df_raw = pd.concat(dfs, ignore_index=True)
    else:
        df_raw = pd.DataFrame()
else:
    df_raw = pd.DataFrame()

# =========================================================
# DASHBOARD PAGE
# =========================================================
if page == "Dashboard":
    st.title("Business Profit Dashboard")
    st.write(
        "Upload one or more CSV/XLSX files in the sidebar, then map your "
        "revenue, cost, date, and product columns."
    )

    if df_raw.empty:
        st.info("Upload at least one file to get started.")
        st.stop()

    st.markdown("### Column Mapping")

    columns = list(df_raw.columns)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        revenue_col = st.selectbox("Revenue column", options=columns, index=0)
    with col2:
        cost_col = st.selectbox("Cost column", options=columns, index=min(1, len(columns) - 1))
    with col3:
        date_col = st.selectbox("Date column", options=columns, index=min(2, len(columns) - 1))
    with col4:
        product_col = st.selectbox("Product column", options=columns, index=min(3, len(columns) - 1))

    df = prepare_data(df_raw, revenue_col, cost_col, date_col, product_col)
    if df.empty:
        st.warning("No valid rows after cleaning.")
        st.stop()

    # =====================================================
    # FILTERS
    # =====================================================
    st.markdown("### Filters")

    product_options = sorted(df["product"].astype(str).unique())
    min_date = df["__date__"].min()
    max_date = df["__date__"].max()

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_products = st.multiselect(
            "Products",
            options=product_options,
            default=product_options,
        )
    with col_f2:
        date_range = st.date_input(
            "Date range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )

    if selected_products:
        df = df[df["product"].isin(selected_products)]

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(
            seconds=1
        )
        df = df[(df["__date__"] >= start_date) & (df["__date__"] <= end_date)]

    if df.empty:
        st.warning("No data after applying filters.")
        st.stop()

    # =====================================================
    # SUMMARIES
    # =====================================================
    monthly_summary = build_monthly_summary(df)
    product_summary = build_product_summary(df)

    total_revenue = df["__revenue__"].sum()
    total_cost = df["__cost__"].sum()
    total_profit = df["__profit__"].sum()
    margin_pct = total_profit / total_revenue * 100 if total_revenue != 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"${total_revenue:,.0f}")
    c2.metric("Total Cost", f"${total_cost:,.0f}")
    c3.metric("Total Profit", f"${total_profit:,.0f}")
    c4.metric("Profit Margin", f"{margin_pct:,.1f}%")

    # =====================================================
    # CHARTS
    # =====================================================
    st.markdown("### Revenue & Profit Over Time")

    if not monthly_summary.empty:
        line_df = monthly_summary.sort_values("Month")
        fig = px.line(
            line_df,
            x="Month",
            y=["Revenue", "Profit"],
            labels={"value": "Amount", "variable": "Metric"},
        )
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    # Product table
    st.markdown("### Top Products by Profit")
    if not product_summary.empty:
        st.dataframe(
            product_summary.style.format(
                {
                    "Revenue": "${:,.0f}",
                    "Cost": "${:,.0f}",
                    "Profit": "${:,.0f}",
                    "Margin %": "{:,.1f}%",
                }
            ),
            use_container_width=True,
        )

    # =====================================================
    # INSIGHTS
    # =====================================================
    st.markdown("---")
    st.subheader("💡 Insights")

    insights = []

    # 1) Most profitable product
    if not product_summary.empty:
        top_prod = product_summary.iloc[0]
        insights.append(
            f"{top_prod['Product']} is your most profitable product with "
            f"profit of **${top_prod['Profit']:,.0f}** and a margin of "
            f"**{top_prod['Margin %']:,.1f}%**."
        )

    # 2) Latest month performance
    if not monthly_summary.empty:
        ms_sorted = monthly_summary.sort_values("Month")
        latest = ms_sorted.iloc[-1]

        latest_month_label = latest["Month"].strftime("%B %Y")
        latest_revenue = latest["Revenue"]
        latest_profit = latest["Profit"]

        latest_line = (
            f"In the latest month (**{latest_month_label}**), you generated "
            f"**${latest_revenue:,.0f} in revenue** and "
            f"**${latest_profit:,.0f} in profit**."
        )
        insights.append(latest_line)

        # Month-over-month
        if len(ms_sorted) > 1:
            prev = ms_sorted.iloc[-2]
            prev_rev = prev["Revenue"]
            if prev_rev != 0:
                mom_change = (latest_revenue - prev_rev) / prev_rev * 100
                direction = "up" if mom_change > 0 else "down"
                insights.append(
                    f"Month-over-month, revenue is **{mom_change:,.1f}%** ({direction} vs. the prior month)."
                )

        # Year-over-year
        this_month = latest["Month"].month
        this_year = latest["Month"].year
        same_month_last_year = ms_sorted[
            (ms_sorted["Month"].dt.month == this_month)
            & (ms_sorted["Month"].dt.year == this_year - 1)
        ]
        if not same_month_last_year.empty:
            last_year_rev = same_month_last_year.iloc[0]["Revenue"]
            if last_year_rev != 0:
                yoy_change = (latest_revenue - last_year_rev) / last_year_rev * 100
                direction = "up" if yoy_change > 0 else "down"
                insights.append(
                    f"Year-over-year, revenue for {latest_month_label} is **{yoy_change:,.1f}%** ({direction})."
                )

        # Trend last 3 months
        if len(ms_sorted) >= 3:
            trend = describe_trend(ms_sorted["Revenue"])
            insights.append(
                f"Revenue has been trending **{trend}** over the last three months."
            )

    if not insights:
        st.write("Insights will appear here once your data is mapped.")
    else:
        for bullet in insights:
            st.markdown(f"- {bullet}")

    # =====================================================
    # EXECUTIVE SUMMARY
    # =====================================================
    st.markdown("---")
    st.subheader("📝 Executive Summary")

    summary_parts = []

    if not product_summary.empty:
        top_prod = product_summary.iloc[0]
        summary_parts.append(
            f"{top_prod['Product']} is currently your top performer, "
            f"delivering **${top_prod['Profit']:,.0f} in profit** at a "
            f"**{top_prod['Margin %']:,.1f}% margin**."
        )

    if not monthly_summary.empty:
        ms_sorted = monthly_summary.sort_values("Month")
        latest = ms_sorted.iloc[-1]
        latest_month_label = latest["Month"].strftime("%B %Y")
        summary_parts.append(
            f"In {latest_month_label}, the business generated "
            f"**${latest['Revenue']:,.0f} in revenue** and "
            f"**${latest['Profit']:,.0f} in profit**."
        )

        if len(ms_sorted) >= 3:
            trend = describe_trend(ms_sorted["Revenue"])
            summary_parts.append(
                f"Over the last three months, revenue has been trending **{trend}**."
            )

    if summary_parts:
        st.write(" ".join(summary_parts))
    else:
        st.write("An executive summary will appear here once your data is mapped.")


# =========================================================
# RAW DATA PAGE
# =========================================================
elif page == "Raw Data":
    st.title("Raw Data")

    if df_raw.empty:
        st.info("Upload at least one file to view raw data.")
        st.stop()

    st.write("Below is the combined raw data from all uploaded files.")
    st.dataframe(df_raw, use_container_width=True)


# =========================================================
# SETTINGS / INFO PAGE
# =========================================================
else:  # page == "Settings / Info"
    st.title("Settings / Info")

    st.markdown("### What this app does")
    st.markdown(
        """
- Uploads CSV or Excel sales/profit data.
- Lets you map revenue, cost, date, and product columns.
- Applies filters (date & product).
- Shows KPIs, charts, tables, insights, and an executive summary.
- Supports basic forecasting upgrades, AI insights, and PDF export in future versions.
        """
    )

    st.markdown("### How to turn advanced features on later")
    st.markdown(
        """
In future upgrades we can add:

1. **User authentication** (Streamlit-Authenticator / Supabase).
2. **Forecasting** (linear regression, Prophet, ARIMA) with 3/6/12-month predictions.
3. **AI insights** powered by ChatGPT inside the app.
4. **More filters** (region, customer, etc.).
5. **Downloadable PDF reports** with logo, charts, tables, and commentary.
6. **Multi-page navigation** for Overview, Forecasting, Product Performance, etc.
7. **Custom domain** like `dashboard.analyticsbyjalal.com`.
        """
    )

    st.markdown(
        "Right now you have a clean, working dashboard. When you're ready, "
        "we can layer in any of the upgrades above."
    )
