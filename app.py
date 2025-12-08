import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import streamlit as st

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Business Profit Dashboard",
    page_icon="📊",
    layout="wide",
)

# Feature flags (optional, future use)
ENABLE_AUTH = False
ENABLE_AI_INSIGHTS = False
ENABLE_PDF_EXPORT = False

# Session defaults
if "filter_presets" not in st.session_state:
    st.session_state.filter_presets = {}  # name -> dict of filters

if "product_filter" not in st.session_state:
    st.session_state.product_filter = None

if "date_range" not in st.session_state:
    st.session_state.date_range = None


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def load_data(uploaded_file, sheet_name=None):
    if uploaded_file is None:
        return None

    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        # Excel
        xls = pd.ExcelFile(uploaded_file)
        if sheet_name is None:
            # default: first sheet
            sheet_name = xls.sheet_names[0]
        df = xls.parse(sheet_name)

    return df


def map_columns(df, revenue_col, cost_col, date_col, product_col):
    """Create normalized internal columns."""
    df = df.copy()

    # Coerce numeric
    if revenue_col:
        df["__revenue__"] = pd.to_numeric(df[revenue_col], errors="coerce")
    else:
        df["__revenue__"] = 0.0

    if cost_col:
        df["__cost__"] = pd.to_numeric(df[cost_col], errors="coerce")
    else:
        df["__cost__"] = 0.0

    df["__profit__"] = df["__revenue__"] - df["__cost__"]

    # Date
    if date_col:
        df["__date__"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df["__date__"] = pd.NaT

    # Product
    if product_col:
        df["product"] = df[product_col].astype(str)
    else:
        df["product"] = "Unknown"

    return df


def summarize_monthly(df):
    if df["__date__"].isna().all():
        return None

    df = df.dropna(subset=["__date__"]).copy()
    df["Month"] = df["__date__"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby("Month", as_index=False)[["__revenue__", "__cost__", "__profit__"]]
        .sum()
        .sort_values("Month")
    )

    monthly["Margin %"] = np.where(
        monthly["__revenue__"] != 0,
        monthly["__profit__"] / monthly["__revenue__"] * 100,
        0.0,
    )

    return monthly


def summarize_products(df):
    product_summary = (
        df.groupby("product", as_index=False)[["__revenue__", "__cost__", "__profit__"]]
        .sum()
        .sort_values("__profit__", ascending=False)
    )
    product_summary["Margin %"] = np.where(
        product_summary["__revenue__"] != 0,
        product_summary["__profit__"] / product_summary["__revenue__"] * 100,
        0.0,
    )
    return product_summary


def build_insights_and_exec(df, monthly_summary, product_summary):
    insights = []
    exec_parts = []

    # 1) Most profitable product
    if product_summary is not None and not product_summary.empty:
        top = product_summary.iloc[0]
        top_product = top["product"]
        top_profit = top["__profit__"]
        top_margin = top["Margin %"]

        insights.append(
            f"**{top_product}** is your most profitable product "
            f"with profit of **${top_profit:,.0f}** and a margin of **{top_margin:.1f}%**."
        )

        exec_parts.append(
            f"{top_product} is currently your top performer, delivering "
            f"**${top_profit:,.0f} in profit** at a **{top_margin:.1f}% margin**."
        )
    else:
        top_product = None

    # 2) Monthly performance / latest month
    if monthly_summary is not None and not monthly_summary.empty:
        ms = monthly_summary.sort_values("Month")
        latest = ms.iloc[-1]
        latest_month = latest["Month"]
        latest_rev = latest["__revenue__"]
        latest_prof = latest["__profit__"]

        if len(ms) >= 2:
            prev = ms.iloc[-2]
            mom_rev = (
                (latest_rev - prev["__revenue__"]) / prev["__revenue__"] * 100
                if prev["__revenue__"] != 0
                else np.nan
            )
        else:
            mom_rev = np.nan

        # YoY: compare with same month last year if exists
        yoy_rev = np.nan
        if len(ms) >= 13:
            target_month = (latest_month - pd.DateOffset(years=1)).to_period("M")
            ms["MonthPeriod"] = ms["Month"].dt.to_period("M")
            match = ms[ms["MonthPeriod"] == target_month]
            if not match.empty:
                last_year_rev = match.iloc[-1]["__revenue__"]
                yoy_rev = (
                    (latest_rev - last_year_rev) / last_year_rev * 100
                    if last_year_rev != 0
                    else np.nan
                )

        # Trend: last 3 months
        trend_phrase = ""
        if len(ms) >= 3:
            last3 = ms.iloc[-3:]
            revs = last3["__revenue__"].values
            if all(np.diff(revs) > 0):
                trend_phrase = "up"
            elif all(np.diff(revs) < 0):
                trend_phrase = "down"
            else:
                trend_phrase = "mixed"

        # Insights text
        insights.append(
            f"In the latest month (**{latest_month:%B %Y}**), you generated "
            f"**${latest_rev:,.0f}** in revenue and **${latest_prof:,.0f}** in profit."
        )

        if not np.isnan(mom_rev):
            direction = "up" if mom_rev >= 0 else "down"
            insights.append(
                f"Month-over-month, revenue is **{mom_rev:+.1f}%** "
                f"({direction} vs. the prior month)."
            )

        if not np.isnan(yoy_rev):
            direction = "up" if yoy_rev >= 0 else "down"
            insights.append(
                f"Year-over-year, revenue is **{yoy_rev:+.1f}%** for "
                f"{latest_month:%B}."
            )

        if trend_phrase:
            insights.append(
                f"Revenue has been trending **{trend_phrase}** over the last three months."
            )

        # Exec summary parts
        exec_parts.append(
            f"In {latest_month:%B %Y}, you generated **${latest_rev:,.0f} in revenue** "
            f"and **${latest_prof:,.0f} in profit**."
        )

        if not np.isnan(yoy_rev):
            exec_parts.append(
                f"Compared with the same month last year, revenue is "
                f"**{yoy_rev:+.1f}%**."
            )

        if trend_phrase:
            exec_parts.append(
                f"Revenue has been trending **{trend_phrase}** over the last three months."
            )

    exec_summary = " ".join(exec_parts) if exec_parts else "No data available yet."

    return insights, exec_summary


def build_excel_download(df, monthly_summary, product_summary):
    """Create an in-memory Excel file with multiple sheets."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Raw Data")
        if monthly_summary is not None:
            monthly_summary.to_excel(writer, index=False, sheet_name="Monthly Summary")
        if product_summary is not None:
            product_summary.to_excel(writer, index=False, sheet_name="Product Summary")
    output.seek(0)
    return output


# =========================================================
# LAYOUT – SIDEBAR: FILE & FILTERS
# =========================================================

if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", use_column_width=True)

st.sidebar.title("Business Profit Dashboard")

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Product Details", "Export Reports", "Settings / Info"],
)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file", type=["csv", "xlsx", "xls"]
)

sheet_name = None
df_raw = None

if uploaded_file is not None:
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(uploaded_file)
        if len(xls.sheet_names) > 1:
            sheet_name = st.sidebar.selectbox(
                "Select sheet", options=xls.sheet_names
            )
        else:
            sheet_name = xls.sheet_names[0]
        df_raw = xls.parse(sheet_name)
    else:
        df_raw = pd.read_csv(uploaded_file)

# Column mapping
revenue_col = cost_col = date_col = product_col = None
if df_raw is not None:
    cols = list(df_raw.columns)
    st.sidebar.markdown("### Column Mapping")
    revenue_col = st.sidebar.selectbox(
        "Revenue column", ["(none)"] + cols, index=0 if "revenue" not in ",".join(cols).lower() else cols.index("revenue")
    )
    cost_col = st.sidebar.selectbox("Cost column (optional)", ["(none)"] + cols)
    date_col = st.sidebar.selectbox("Date column (optional)", ["(none)"] + cols)
    product_col = st.sidebar.selectbox("Product column (optional)", ["(none)"] + cols)

    if revenue_col == "(none)":
        revenue_col = None
    if cost_col == "(none)":
        cost_col = None
    if date_col == "(none)":
        date_col = None
    if product_col == "(none)":
        product_col = None

# Prepare main df
df = None
monthly_summary = None
product_summary = None

if df_raw is not None:
    df = map_columns(df_raw, revenue_col, cost_col, date_col, product_col)

    # =======================
    # FILTERS + PRESETS
    # =======================
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filters")

    # Product filter
    product_options = sorted(df["product"].dropna().unique().tolist())
    default_products = product_options

    if st.session_state.product_filter is None:
        st.session_state.product_filter = default_products

    product_filter = st.sidebar.multiselect(
        "Products",
        options=product_options,
        default=st.session_state.product_filter,
        key="product_filter",
    )

    # Date range filter
    if not df["__date__"].isna().all():
        min_date = df["__date__"].min().date()
        max_date = df["__date__"].max().date()

        if st.session_state.date_range is None:
            st.session_state.date_range = (min_date, max_date)

        date_range = st.sidebar.slider(
            "Date range",
            min_value=min_date,
            max_value=max_date,
            value=st.session_state.date_range,
            key="date_range",
        )
    else:
        date_range = None

    # Apply the filters
    if product_filter:
        df = df[df["product"].isin(product_filter)]

    if date_range and not df["__date__"].isna().all():
        start, end = date_range
        mask = (df["__date__"].dt.date >= start) & (df["__date__"].dt.date <= end)
        df = df[mask]

    # =======================
    # PRESET MANAGEMENT
    # =======================
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filter Presets")

    preset_name = st.sidebar.text_input("Preset name")
    existing_presets = list(st.session_state.filter_presets.keys())
    preset_to_load = st.sidebar.selectbox(
        "Existing presets", ["(none)"] + existing_presets
    )

    col_save, col_load, col_delete = st.sidebar.columns(3)

    with col_save:
        if st.button("Save", use_container_width=True):
            if preset_name.strip():
                st.session_state.filter_presets[preset_name.strip()] = {
                    "products": list(st.session_state.product_filter),
                    "date_range": st.session_state.date_range,
                }
                st.sidebar.success(f"Saved preset: {preset_name.strip()}")
            else:
                st.sidebar.warning("Enter a preset name to save.")

    with col_load:
        if st.button("Load", use_container_width=True):
            if preset_to_load != "(none)":
                preset = st.session_state.filter_presets[preset_to_load]
                st.session_state.product_filter = preset.get("products", product_options)
                st.session_state.date_range = preset.get(
                    "date_range",
                    (min_date, max_date)
                    if not df["__date__"].isna().all()
                    else st.session_state.date_range,
                )
                st.sidebar.success(f"Loaded preset: {preset_to_load}")
            else:
                st.sidebar.warning("Select a preset to load.")

    with col_delete:
        if st.button("Delete", use_container_width=True):
            if preset_to_load != "(none)" and preset_to_load in st.session_state.filter_presets:
                del st.session_state.filter_presets[preset_to_load]
                st.sidebar.success(f"Deleted preset: {preset_to_load}")
            else:
                st.sidebar.warning("Select a preset to delete.")

    # Recompute summaries after filters
    monthly_summary = summarize_monthly(df)
    product_summary = summarize_products(df)


# =========================================================
# MAIN PAGES
# =========================================================

st.title("Business Profit Dashboard")

if df is None:
    st.info("Upload a CSV or Excel file and map your columns in the sidebar to begin.")
else:
    # Build insights + executive summary once
    insights, exec_summary = build_insights_and_exec(df, monthly_summary, product_summary)

    # ----------------- PAGE: DASHBOARD -----------------
    if page == "Dashboard":
        st.markdown("### Key Metrics")

        total_revenue = df["__revenue__"].sum()
        total_cost = df["__cost__"].sum()
        total_profit = df["__profit__"].sum()
        margin_pct = (total_profit / total_revenue * 100) if total_revenue != 0 else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Revenue", f"${total_revenue:,.0f}")
        c2.metric("Total Cost", f"${total_cost:,.0f}")
        c3.metric("Total Profit", f"${total_profit:,.0f}")
        c4.metric("Overall Margin", f"{margin_pct:.1f}%")

        st.markdown("---")

        # Charts
        if monthly_summary is not None and not monthly_summary.empty:
            st.subheader("Revenue & Profit Over Time")
            fig = px.line(
                monthly_summary,
                x="Month",
                y=["__revenue__", "__profit__"],
                labels={"value": "Amount", "variable": "Metric"},
            )
            fig.update_layout(legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)

        if product_summary is not None and not product_summary.empty:
            st.subheader("Top Products by Profit")
            fig2 = px.bar(
                product_summary.head(15),
                x="product",
                y="__profit__",
                labels={"__profit__": "Profit", "product": "Product"},
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.subheader("💡 Insights")
        if insights:
            for bullet in insights:
                st.markdown(f"- {bullet}")
        else:
            st.write("Insights will appear here once you have valid data.")

        st.markdown("---")
        st.subheader("📝 Executive Summary")
        st.markdown(exec_summary)

    # ----------------- PAGE: PRODUCT DETAILS -----------------
    elif page == "Product Details":
        st.subheader("Product Summary Table")

        if product_summary is None or product_summary.empty:
            st.write("No product data available.")
        else:
            # Pretty column names
            display_df = product_summary.rename(
                columns={
                    "product": "Product",
                    "__revenue__": "Revenue",
                    "__cost__": "Cost",
                    "__profit__": "Profit",
                    "Margin %": "Margin %",
                }
            )
            st.dataframe(display_df, use_container_width=True)

    # ----------------- PAGE: EXPORT REPORTS -----------------
    elif page == "Export Reports":
        st.subheader("Export Reports")

        if df is None or df.empty:
            st.write("Upload and map data first.")
        else:
            # Product CSV
            if product_summary is not None:
                export_df = product_summary.rename(
                    columns={
                        "product": "Product",
                        "__revenue__": "Revenue",
                        "__cost__": "Cost",
                        "__profit__": "Profit",
                        "Margin %": "Margin %",
                    }
                )
                csv_bytes = export_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download product summary (CSV)",
                    data=csv_bytes,
                    file_name="product_summary.csv",
                    mime="text/csv",
                )

            # Full Excel
            excel_bytes = build_excel_download(df, monthly_summary, product_summary)
            st.download_button(
                "Download full Excel report",
                data=excel_bytes,
                file_name="profit_dashboard_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    # ----------------- PAGE: SETTINGS / INFO -----------------
    elif page == "Settings / Info":
        st.markdown("---")
        st.subheader("Settings & Info")

        st.write(
            """
### What this app does

- Uploads CSV or Excel sales/profit data.
- Lets you map revenue, cost, date, and product columns.
- Applies filters (date & product) with **saved presets**.
- Shows KPIs, charts, tables, insights, and an executive summary.
- Builds a simple 6-month forecast based on trend (approximate).
- Supports optional authentication, AI Insights, and PDF export via flags.

### How to turn features on/off

Edit the top of `app.py` and set these values:
"""
        )

        st.code(
            '''
ENABLE_AUTH = False        # True to enable login
ENABLE_AI_INSIGHTS = False # True to enable OpenAI-based insights
ENABLE_PDF_EXPORT = False  # True to enable PDF download
            ''',
            language="python",
        )

        st.markdown("---")
        st.markdown(
            "Built by **AnalyticsbyJalal** · Use this as a demo app or a starting point "
            "for a client-facing analytics product."
        )

