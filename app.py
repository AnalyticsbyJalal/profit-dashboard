import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import streamlit as st

# -------------------------------------------------
# Page config & basic styling
# -------------------------------------------------
st.set_page_config(
    page_title="Business Profit Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Simple corporate-style CSS
st.markdown(
    """
    <style>
        .main { background-color: #f7f9fc; }
        h1, h2, h3, h4 { color: #12355b; }
        .stMetric { background-color: white; border-radius: 10px; padding: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Header with optional logo
# -------------------------------------------------
cols_header = st.columns([0.15, 0.85])

with cols_header[0]:
    try:
        logo = Image.open("logo.png")
        st.image(logo, use_column_width=True)
    except Exception:
        st.write("")

with cols_header[1]:
    st.title("Business Profit Dashboard")
    st.caption(
        "Corporate-style analytics for revenue, cost, profit, and margin performance."
    )

st.markdown("---")

# -------------------------------------------------
# File upload
# -------------------------------------------------
st.subheader("Upload data")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file", type=["csv", "xlsx", "xls"], accept_multiple_files=False
)

if not uploaded_file:
    st.info("Upload a CSV or Excel file to get started.")
    st.stop()

filename = uploaded_file.name.lower()
ext = filename.split(".")[-1]

# Read CSV or Excel
if ext == "csv":
    df_raw = pd.read_csv(uploaded_file)
else:
    # Excel with possible multiple sheets
    xls = pd.ExcelFile(uploaded_file)
    if len(xls.sheet_names) == 1:
        sheet_name = xls.sheet_names[0]
    else:
        sheet_name = st.selectbox("Select sheet to analyze", xls.sheet_names)
    df_raw = xls.parse(sheet_name)

if df_raw.empty:
    st.warning("The uploaded file appears to be empty.")
    st.stop()

st.write("Preview of uploaded data:")
st.dataframe(df_raw.head())

# -------------------------------------------------
# Column mapping
# -------------------------------------------------
st.markdown("---")
st.subheader("Column Mapping")

cols = list(df_raw.columns)

with st.container():
    c1, c2 = st.columns(2)
    with c1:
        revenue_col = st.selectbox("Revenue column", options=cols)
        cost_col = st.selectbox("Cost column (optional)", options=["(none)"] + cols)
        if cost_col == "(none)":
            cost_col = None

    with c2:
        date_col = st.selectbox("Date column (optional)", options=["(none)"] + cols)
        if date_col == "(none)":
            date_col = None

        product_col = st.selectbox("Product column (optional)", options=["(none)"] + cols)
        if product_col == "(none)":
            product_col = None

# -------------------------------------------------
# Prepare working dataframe with numeric and date fields
# -------------------------------------------------
df = df_raw.copy()

# Revenue
df["__revenue__"] = pd.to_numeric(df[revenue_col], errors="coerce")

# Cost
if cost_col:
    df["__cost__"] = pd.to_numeric(df[cost_col], errors="coerce")
else:
    df["__cost__"] = 0.0

# Profit & Margin
df["__profit__"] = df["__revenue__"] - df["__cost__"]
df["__margin_pct__"] = np.where(
    df["__revenue__"] != 0, df["__profit__"] / df["__revenue__"] * 100, np.nan
)

# Date & Month
if date_col:
    df["__date__"] = pd.to_datetime(df[date_col], errors="coerce")
    df["Month"] = df["__date__"].dt.to_period("M").dt.to_timestamp()
else:
    df["__date__"] = pd.NaT
    df["Month"] = pd.NaT

# -------------------------------------------------
# Aggregations
# -------------------------------------------------
# Key totals
total_revenue = df["__revenue__"].sum(skipna=True)
total_cost = df["__cost__"].sum(skipna=True)
total_profit = df["__profit__"].sum(skipna=True)
overall_margin = (
    (total_profit / total_revenue * 100) if total_revenue not in (0, np.nan) else np.nan
)

# Monthly summary
monthly_summary = None
if df["Month"].notna().any():
    monthly_summary = (
        df.dropna(subset=["Month"])
        .groupby("Month", as_index=False)
        .agg(
            {
                "__revenue__": "sum",
                "__cost__": "sum",
                "__profit__": "sum",
            }
        )
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
        monthly_summary["Profit"] / monthly_summary["Revenue"] * 100,
        np.nan,
    )

# Product summary
product_summary = None
if product_col:
    product_summary = (
        df.groupby(product_col, as_index=False)
        .agg(
            {
                "__revenue__": "sum",
                "__cost__": "sum",
                "__profit__": "sum",
            }
        )
        .rename(
            columns={
                "__revenue__": "Revenue",
                "__cost__": "Cost",
                "__profit__": "Profit",
            }
        )
    )
    product_summary["Margin %"] = np.where(
        product_summary["Revenue"] != 0,
        product_summary["Profit"] / product_summary["Revenue"] * 100,
        np.nan,
    )
    product_summary = product_summary.sort_values("Profit", ascending=False)

# -------------------------------------------------
# Key Metrics
# -------------------------------------------------
st.markdown("---")
st.subheader("Key Metrics")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Revenue", f"${total_revenue:,.0f}")
m2.metric("Total Cost", f"${total_cost:,.0f}")
m3.metric("Total Profit", f"${total_profit:,.0f}")
m4.metric(
    "Profit Margin",
    f"{overall_margin:,.1f}%" if not np.isnan(overall_margin) else "N/A",
)

# -------------------------------------------------
# Charts
# -------------------------------------------------
st.markdown("---")
st.subheader("Visuals")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    if monthly_summary is not None and not monthly_summary.empty:
        fig_line = px.line(
            monthly_summary,
            x="Month",
            y=["Revenue", "Profit"],
            title="Revenue and Profit Over Time",
            markers=True,
        )
        fig_line.update_layout(legend_title_text="")
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Add a date column to view revenue and profit over time.")

with chart_col2:
    if product_summary is not None and not product_summary.empty:
        fig_bar = px.bar(
            product_summary.head(15),
            x=product_col,
            y="Profit",
            title="Top Products by Profit",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Add a product column to view top products by profit.")

# Detailed tables
with st.expander("View detailed tables"):
    if monthly_summary is not None:
        st.write("Monthly summary")
        st.dataframe(monthly_summary)
    if product_summary is not None:
        st.write("Product summary")
        st.dataframe(product_summary)

# -------------------------------------------------
# Export reports
# -------------------------------------------------
st.markdown("---")
st.subheader("Export Reports")

# Product summary CSV
if product_summary is not None and not product_summary.empty:
    csv_buf = product_summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download product summary (CSV)",
        data=csv_buf,
        file_name="product_summary.csv",
        mime="text/csv",
    )
else:
    st.caption("Product summary CSV is unavailable (no product column mapped).")

# Full Excel report
excel_buf = io.BytesIO()
with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
    df_raw.to_excel(writer, index=False, sheet_name="Raw Data")

    if monthly_summary is not None:
        monthly_summary.to_excel(writer, index=False, sheet_name="Monthly Summary")

    if product_summary is not None:
        product_summary.to_excel(writer, index=False, sheet_name="Product Summary")

st.download_button(
    label="Download full Excel report",
    data=excel_buf.getvalue(),
    file_name="business_profit_report.xlsx",
    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    ),
)

# -------------------------------------------------
# Insights & Executive Summary
# -------------------------------------------------
st.markdown("---")
st.subheader("Insights")

insights: list[str] = []
exec_parts: list[str] = []

# 1) Most profitable product
if product_summary is not None and not product_summary.empty:
    top_prod = product_summary.iloc[0]
    top_name = str(top_prod[product_col])
    top_profit = float(top_prod["Profit"])
    top_margin = float(top_prod["Margin %"])

    insights.append(
        f"{top_name} is your most profitable product with profit of "
        f"${top_profit:,.0f} and a margin of {top_margin:,.1f}%."
    )

    exec_parts.append(
        f"{top_name} is currently your top performer, delivering "
        f"${top_profit:,.0f} in profit at a {top_margin:,.1f}% margin."
    )

# 2) Monthly performance + MoM + YoY + Trend
mom_phrase = ""
yoy_phrase = ""
trend_phrase = ""

if monthly_summary is not None and not monthly_summary.empty:
    ms = monthly_summary.sort_values("Month").copy()
    ms_indexed = ms.set_index("Month")

    latest_row = ms.iloc[-1]
    latest_month = latest_row["Month"]
    latest_rev = float(latest_row["Revenue"])
    latest_prof = float(latest_row["Profit"])

    insights.append(
        f"In the latest month ({latest_month.strftime('%B %Y')}), "
        f"you generated ${latest_rev:,.0f} in revenue and "
        f"${latest_prof:,.0f} in profit."
    )

    # Month-over-month growth
    prev_month = latest_month - pd.DateOffset(months=1)
    if prev_month in ms_indexed.index:
        prev_rev = float(ms_indexed.loc[prev_month, "Revenue"])
        if prev_rev != 0:
            mom_change = (latest_rev - prev_rev) / abs(prev_rev) * 100
            insights.append(
                f"Month-over-month, revenue changed by "
                f"{mom_change:+.1f}% compared to {prev_month.strftime('%b %Y')}."
            )
            mom_phrase = (
                f"Revenue is {mom_change:+.1f}% versus {prev_month.strftime('%b %Y')}."
            )

    # Year-over-year growth
    prev_year_month = latest_month - pd.DateOffset(years=1)
    if prev_year_month in ms_indexed.index:
        prev_year_rev = float(ms_indexed.loc[prev_year_month, "Revenue"])
        if prev_year_rev != 0:
            yoy_change = (latest_rev - prev_year_rev) / abs(prev_year_rev) * 100
            insights.append(
                f"Year-over-year, revenue for {latest_month.strftime('%B')} "
                f"is {yoy_change:+.1f}% versus {prev_year_month.strftime('%B %Y')}."
            )
            yoy_phrase = (
                f"Compared with {prev_year_month.strftime('%B %Y')}, "
                f"revenue is {yoy_change:+.1f}%."
            )

    # Simple trend over last 3 months
    if len(ms) >= 3:
        last3 = ms.tail(3)
        revs = last3["Revenue"].values
        if np.all(np.diff(revs) > 0):
            trend_phrase = "Revenue has been trending up over the last three months."
            insights.append(trend_phrase)
        elif np.all(np.diff(revs) < 0):
            trend_phrase = "Revenue has been trending down over the last three months."
            insights.append(trend_phrase)

    exec_parts.append(
        f"In {latest_month.strftime('%B %Y')}, the business generated "
        f"${latest_rev:,.0f} in revenue and ${latest_prof:,.0f} in profit. "
        f"{mom_phrase} {yoy_phrase} {trend_phrase}".strip()
    )

# 3) Loss warnings
loss_warnings = []

if product_summary is not None and not product_summary.empty:
    loss_products = product_summary[product_summary["Profit"] < 0]
    if not loss_products.empty:
        names = ", ".join(loss_products[product_col].astype(str).head(5))
        loss_warnings.append(f"Loss-making products include: {names}.")

if monthly_summary is not None and not monthly_summary.empty:
    loss_months = monthly_summary[monthly_summary["Profit"] < 0]
    if not loss_months.empty:
        months_txt = ", ".join(loss_months["Month"].dt.strftime("%b %Y"))
        loss_warnings.append(
            f"Some months show negative profit, including: {months_txt}."
        )

for w in loss_warnings:
    insights.append("Warning: " + w)

if loss_warnings:
    exec_parts.append("Additionally, " + " ".join(w.rstrip(".") + "." for w in loss_warnings))

# Display insights bullets
if not insights:
    st.write("Insights will appear here once more data is available.")
else:
    for text in insights:
        st.markdown(f"- {text}")

# -------------------------------------------------
# Executive Summary
# -------------------------------------------------
st.markdown("---")
st.subheader("Executive Summary")

executive_summary = " ".join(exec_parts).strip()
if not executive_summary:
    executive_summary = (
        "A concise summary could not be generated yet. "
        "Upload data with mapped revenue, date, and product columns."
    )

st.write(executive_summary)
