# app.py – Clean, stable version (Settings page simplified)
# ---------------------------------------------------------
# Features:
# - CSV/XLSX upload
# - Column mapping (date / product / revenue / cost)
# - Robust numeric + date handling (no TypeError)
# - Sidebar filters (product + date range)
# - KPIs, charts, tables
# - Insights + Executive Summary (no markdown glitches)
# - Multi-page navigation (Dashboard / Raw Data / Settings)
#
# Upgrade flags are kept for future auth/AI/forecasting/PDF work.

from __future__ import annotations

from datetime import datetime
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -------------------------------------------------------------------
# Upgrade flags (for future features)
# -------------------------------------------------------------------
ENABLE_AUTH = False          # Future: login / multi-user
ENABLE_AI_INSIGHTS = False   # Future: ChatGPT narrative
ENABLE_FORECASTING = False   # Future: Prophet / ARIMA etc.
ENABLE_PDF_EXPORT = False    # Future: PDF export


# -------------------------------------------------------------------
# Page config – light look
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Business Profit Dashboard",
    layout="wide",
    page_icon="📊",
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _get_series(df: pd.DataFrame, col_name: Optional[str]) -> pd.Series:
    """
    Safely return a 1D Series for a given column name.
    Fixes the case where df[col] might be a DataFrame and
    caused the pd.to_numeric TypeError.
    """
    if not col_name:
        return pd.Series(dtype="float64")

    obj = df[col_name]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]

    return obj


def load_files(uploaded_files) -> pd.DataFrame:
    """Read all uploaded CSV/XLSX files into a single DataFrame."""
    if not uploaded_files:
        return pd.DataFrame()

    dfs = []
    for f in uploaded_files:
        name = f.name.lower()
        try:
            if name.endswith(".csv"):
                df = pd.read_csv(f)
            elif name.endswith(".xlsx") or name.endswith(".xls"):
                df = pd.read_excel(f)  # needs openpyxl in requirements.txt
            else:
                st.warning(f"Unsupported file type: {f.name}")
                continue
            dfs.append(df)
        except Exception as e:
            st.error(f"Error reading file {f.name}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def prepare_data(
    df_raw: pd.DataFrame,
    revenue_col: Optional[str],
    cost_col: Optional[str],
    date_col: Optional[str],
    product_col: Optional[str],
) -> pd.DataFrame:
    """
    Normalize the raw uploaded data into standard columns:
      __revenue__, __cost__, __profit__, __margin_pct__,
      __date__, Month, Year, __product__
    """
    df = df_raw.copy()

    # Revenue
    if revenue_col:
        rev = _get_series(df, revenue_col)
        df["__revenue__"] = pd.to_numeric(
            rev.astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        ).fillna(0.0)
    else:
        df["__revenue__"] = 0.0

    # Cost
    if cost_col:
        cost = _get_series(df, cost_col)
        df["__cost__"] = pd.to_numeric(
            cost.astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        ).fillna(0.0)
    else:
        df["__cost__"] = 0.0

    # Profit and margin
    df["__profit__"] = df["__revenue__"] - df["__cost__"]
    df["__margin_pct__"] = np.where(
        df["__revenue__"] != 0,
        df["__profit__"] / df["__revenue__"] * 100.0,
        0.0,
    )

    # Date
    if date_col:
        date_series = _get_series(df, date_col)
        df["__date__"] = pd.to_datetime(date_series, errors="coerce")
    else:
        df["__date__"] = pd.NaT

    df["Month"] = df["__date__"].dt.to_period("M").dt.to_timestamp()
    df["Year"] = df["__date__"].dt.year

    # Product
    if product_col:
        prod = _get_series(df, product_col)
        df["__product__"] = prod.astype(str).fillna("Unknown")
    else:
        df["__product__"] = "Unknown"

    return df


def build_summaries(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return monthly and product-level summaries."""
    if df.empty:
        monthly_summary = pd.DataFrame(
            columns=["Month", "Revenue", "Cost", "Profit", "Margin %"]
        )
        product_summary = pd.DataFrame(
            columns=["Product", "Revenue", "Cost", "Profit", "Margin %"]
        )
        return monthly_summary, product_summary

    monthly_summary = (
        df.groupby("Month", dropna=True)[["__revenue__", "__cost__", "__profit__"]]
        .sum()
        .reset_index()
        .rename(
            columns={
                "__revenue__": "Revenue",
                "__cost__": "Cost",
                "__profit__": "Profit",
            }
        )
    )
    monthly_summary["Margin %"] = np.where(
        monthly_summary["Revenue"] != 0,
        monthly_summary["Profit"] / monthly_summary["Revenue"] * 100.0,
        0.0,
    )

    product_summary = (
        df.groupby("__product__", dropna=False)[["__revenue__", "__cost__", "__profit__"]]
        .sum()
        .reset_index()
        .rename(
            columns={
                "__product__": "Product",
                "__revenue__": "Revenue",
                "__cost__": "Cost",
                "__profit__": "Profit",
            }
        )
    )
    product_summary["Margin %"] = np.where(
        product_summary["Revenue"] != 0,
        product_summary["Profit"] / product_summary["Revenue"] * 100.0,
        0.0,
    )

    return monthly_summary, product_summary


def build_insights(
    monthly_summary: pd.DataFrame, product_summary: pd.DataFrame
) -> Tuple[List[str], str]:
    """Generate list of insight bullets + an Executive Summary paragraph."""
    insights: List[str] = []
    exec_parts: List[str] = []

    # Most profitable product
    if not product_summary.empty:
        top_row = product_summary.sort_values("Profit", ascending=False).iloc[0]
        top_prod = top_row["Product"]
        top_profit = top_row["Profit"]
        top_margin = top_row["Margin %"]

        insights.append(
            f"{top_prod} is your most profitable product with profit of "
            f"${top_profit:,.0f} and a margin of {top_margin:,.1f}%."
        )
        exec_parts.append(
            f"{top_prod} is currently your top performer, delivering "
            f"${top_profit:,.0f} in profit at a {top_margin:,.1f}% margin."
        )

    # Monthly stats
    mom_pct = None
    yoy_pct = None
    trend_word = None
    latest_month = None
    latest_rev = None
    latest_prof = None

    if not monthly_summary.empty:
        ms = monthly_summary.sort_values("Month")
        latest = ms.iloc[-1]
        latest_month = latest["Month"]
        latest_rev = latest["Revenue"]
        latest_prof = latest["Profit"]

        insights.append(
            f"In the latest month ({latest_month:%b %Y}), you generated "
            f"${latest_rev:,.0f} in revenue and ${latest_prof:,.0f} in profit."
        )
        exec_parts.append(
            f"In {latest_month:%b %Y}, revenue was ${latest_rev:,.0f} "
            f"with ${latest_prof:,.0f} in profit."
        )

        # Month-over-month
        if len(ms) >= 2:
            prev = ms.iloc[-2]
            prev_rev = prev["Revenue"]
            if prev_rev != 0:
                mom_pct = (latest_rev - prev_rev) / prev_rev * 100.0
                direction = "up" if mom_pct >= 0 else "down"
                insights.append(
                    f"Month-over-month, revenue changed by {mom_pct:,.1f}% "
                    f"({direction} vs. the prior month)."
                )

        # Year-over-year (same month last year)
        if len(ms) >= 13:
            target_month = latest_month.month
            target_year = latest_month.year - 1
            mask_last_year = (ms["Month"].dt.month == target_month) & (
                ms["Month"].dt.year == target_year
            )
            if mask_last_year.any():
                last_year_rev = ms.loc[mask_last_year, "Revenue"].iloc[0]
                if last_year_rev != 0:
                    yoy_pct = (latest_rev - last_year_rev) / last_year_rev * 100.0
                    direction = "up" if yoy_pct >= 0 else "down"
                    insights.append(
                        f"Year-over-year, revenue for {latest_month:%b} is "
                        f"{yoy_pct:,.1f}% ({direction} vs. {target_year})."
                    )

        # Trend over last 3 months
        if len(ms) >= 3:
            last3 = ms.tail(3)["Revenue"].values
            if last3[0] < last3[-1]:
                trend_word = "up"
            elif last3[0] > last3[-1]:
                trend_word = "down"
            else:
                trend_word = "flat"

            insights.append(
                f"Revenue has been trending {trend_word} over the last three months."
            )
            exec_parts.append(
                f"Revenue has been trending {trend_word} over the last three months."
            )

    if yoy_pct is not None and latest_month is not None:
        direction = "higher" if yoy_pct >= 0 else "lower"
        exec_parts.append(
            f"Compared with the same month last year, revenue is "
            f"{abs(yoy_pct):,.1f}% {direction}."
        )

    executive_summary = " ".join(exec_parts) if exec_parts else ""
    return insights, executive_summary


# -------------------------------------------------------------------
# Pages
# -------------------------------------------------------------------
def render_dashboard(df: pd.DataFrame, monthly_summary: pd.DataFrame, product_summary: pd.DataFrame):
    st.markdown("## Business Profit Dashboard")

    if df.empty:
        st.info("Upload data, map your columns above, and the dashboard will appear here.")
        return

    # KPIs
    total_revenue = df["__revenue__"].sum()
    total_cost = df["__cost__"].sum()
    total_profit = df["__profit__"].sum()
    margin_pct = (total_profit / total_revenue * 100.0) if total_revenue != 0 else 0.0

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total Revenue", f"${total_revenue:,.0f}")
    kpi_cols[1].metric("Total Cost", f"${total_cost:,.0f}")
    kpi_cols[2].metric("Total Profit", f"${total_profit:,.0f}")
    kpi_cols[3].metric("Overall Margin", f"{margin_pct:,.1f}%")

    st.markdown("---")

    # Charts
    chart_col1, chart_col2 = st.columns(2)

    if not monthly_summary.empty:
        fig_rev = px.line(
            monthly_summary,
            x="Month",
            y=["Revenue", "Profit"],
            title="Revenue & Profit Over Time",
            markers=True,
        )
        fig_rev.update_layout(legend_title_text="Metric")
        chart_col1.plotly_chart(fig_rev, use_container_width=True)

    if not product_summary.empty:
        top_products = product_summary.sort_values("Profit", ascending=False).head(10)
        fig_prod = px.bar(
            top_products,
            x="Product",
            y="Profit",
            title="Top Products by Profit",
        )
        chart_col2.plotly_chart(fig_prod, use_container_width=True)

    st.markdown("---")

    # Insights
    st.subheader("💡 Insights")
    insights, exec_summary = build_insights(monthly_summary, product_summary)

    if not insights:
        st.write("Insights will appear here once your data is processed.")
    else:
        for bullet in insights:
            st.markdown(f"- {bullet}")

    st.markdown("---")
    st.subheader("📝 Executive Summary")

    if exec_summary:
        st.write(exec_summary)
    else:
        st.write("A narrative summary will appear here after your data is processed.")


def render_raw_data(df: pd.DataFrame, monthly_summary: pd.DataFrame, product_summary: pd.DataFrame):
    st.markdown("## Raw Data & Tables")

    if df.empty:
        st.info("No data loaded yet.")
        return

    st.markdown("### Sample of raw data")
    st.dataframe(df.head(100))

    st.markdown("---")
    st.markdown("### Monthly summary")
    st.dataframe(monthly_summary)

    st.markdown("---")
    st.markdown("### Product summary")
    st.dataframe(product_summary)


def render_settings():
    """
    Super-simple settings/info page – only one markdown call to avoid
    any possible AttributeError inside Streamlit helpers.
    """
    st.markdown(
        """
        ## Settings / Info

        ### What this app does

        - Uploads CSV or Excel sales/profit data.  
        - Lets you map revenue, cost, date, and product columns.  
        - Computes total revenue, cost, profit, and margins.  
        - Shows KPIs, charts, tables, insights, and an executive summary.  
        - Has hooks for future authentication, AI insights, forecasting, and PDF export.  

        ### Upgrade roadmap

        1. **User authentication** (Streamlit Authenticator / Supabase).  
        2. **Forecasting** (linear regression, Prophet, ARIMA – 3/6/12-month projections).  
        3. **AI insights** (ChatGPT-powered commentary and strategy).  
        4. **Downloadable PDF reports** (logo, KPIs, charts, tables, insights).  
        5. **Multi-page analytics suite** (Overview, Forecasting, Products, Profitability, Operations, Region).  
        """
    )


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    st.title("Business Profit Dashboard")

    # Sidebar: logo + upload
    try:
        st.sidebar.image("logo.png", use_column_width=True)
    except Exception:
        st.sidebar.markdown("### AnalyticsByJalal")

    st.sidebar.markdown("### Upload your data")
    uploaded_files = st.sidebar.file_uploader(
        "CSV or Excel", type=["csv", "xlsx", "xls"], accept_multiple_files=True
    )

    df_raw = load_files(uploaded_files)

    # Column mapping (main area)
    df = pd.DataFrame()
    monthly_summary = pd.DataFrame()
    product_summary = pd.DataFrame()

    if df_raw.empty:
        st.info("Upload at least one CSV or Excel file using the sidebar to get started.")
    else:
        st.markdown("## Column Mapping")

        cols = list(df_raw.columns)

        def guess_idx(keyword: str) -> int:
            for i, c in enumerate(cols):
                if keyword.lower() in str(c).lower():
                    return i
            return 0

        map_col1, map_col2, map_col3, map_col4 = st.columns(4)

        with map_col1:
            revenue_col = st.selectbox(
                "Revenue column", cols, index=guess_idx("rev")
            )
        with map_col2:
            cost_col = st.selectbox(
                "Cost column", cols, index=guess_idx("cost")
            )
        with map_col3:
            date_col = st.selectbox(
                "Date column", cols, index=guess_idx("date")
            )
        with map_col4:
            product_col = st.selectbox(
                "Product column", cols, index=guess_idx("prod")
            )

        # Only prepare data if we have all mappings
        if all([revenue_col, cost_col, date_col, product_col]):
            df = prepare_data(df_raw, revenue_col, cost_col, date_col, product_col)

    # If we have a prepared df, build filters and summaries
    if not df.empty:
        # Sidebar filters
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Filters")

        # Product filter
        product_options = sorted(df["__product__"].unique().tolist())
        selected_products = st.sidebar.multiselect(
            "Products",
            options=product_options,
            default=product_options,
        )

        # Date range filter
        min_date = df["__date__"].min()
        max_date = df["__date__"].max()

        if pd.notnull(min_date) and pd.notnull(max_date):
            date_range = st.sidebar.date_input(
                "Date range",
                value=(min_date.date(), max_date.date()),
            )
            # date_input can return a single date or tuple
            if isinstance(date_range, (list, tuple)):
                start_date, end_date = date_range
            else:
                start_date = end_date = date_range

            # Apply filters
            df_filtered = df.copy()
            if selected_products:
                df_filtered = df_filtered[df_filtered["__product__"].isin(selected_products)]

            df_filtered = df_filtered[
                (df_filtered["__date__"] >= pd.to_datetime(start_date))
                & (df_filtered["__date__"] <= pd.to_datetime(end_date))
            ]
        else:
            df_filtered = df.copy()
            if selected_products:
                df_filtered = df_filtered[df_filtered["__product__"].isin(selected_products)]

        df = df_filtered
        monthly_summary, product_summary = build_summaries(df)

    # Navigation
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigation", ["Dashboard", "Raw Data", "Settings / Info"])

    if page == "Dashboard":
        render_dashboard(df, monthly_summary, product_summary)
    elif page == "Raw Data":
        render_raw_data(df, monthly_summary, product_summary)
    else:
        render_settings()


if __name__ == "__main__":
    main()
