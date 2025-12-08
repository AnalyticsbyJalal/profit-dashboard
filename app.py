import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import streamlit as st

# -------------------------------------------------
# Feature flags (flip to True when ready)
# -------------------------------------------------
ENABLE_AUTH = False          # requires streamlit-authenticator
ENABLE_AI_INSIGHTS = False   # requires openai + API key
ENABLE_PDF_EXPORT = False    # requires fpdf

# -------------------------------------------------
# Optional authentication
# -------------------------------------------------
if ENABLE_AUTH:
    try:
        import streamlit_authenticator as stauth

        # Example credentials (REPLACE with your own!)
        names = ["Jalal Reslan"]
        usernames = ["jalal"]
        passwords = ["password123"]  # use hashed in real life

        hashed_passwords = stauth.Hasher(passwords).generate()
        credentials = {
            "usernames": {
                usernames[0]: {
                    "name": names[0],
                    "password": hashed_passwords[0],
                }
            }
        }

        authenticator = stauth.Authenticate(
            credentials,
            "profit_dashboard_cookie",
            "profit_dashboard_key",
            cookie_expiry_days=7,
        )

        name, auth_status, username = authenticator.login("Login", "main")

        if auth_status is False:
            st.error("Username/password is incorrect")
            st.stop()
        elif auth_status is None:
            st.warning("Please enter your username and password")
            st.stop()

        with st.sidebar:
            st.write(f"Logged in as **{name}**")
            authenticator.logout("Logout", "sidebar")

    except ImportError:
        st.warning(
            "Authentication is enabled but `streamlit-authenticator` is not installed.\n"
            "Install with `pip install streamlit-authenticator` or set ENABLE_AUTH = False."
        )
        st.stop()

# -------------------------------------------------
# Page config & CSS
# -------------------------------------------------
st.set_page_config(
    page_title="Business Profit Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
# Header
# -------------------------------------------------
header_cols = st.columns([0.15, 0.85])

with header_cols[0]:
    try:
        logo = Image.open("logo.png")
        st.image(logo, use_column_width=True)
    except Exception:
        st.write("")

with header_cols[1]:
    st.title("Business Profit Dashboard")
    st.caption(
        "Corporate-style analytics for revenue, cost, profit, margin and forecasts."
    )

st.markdown("---")

# -------------------------------------------------
# Data upload
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

# Read data
if ext == "csv":
    df_raw = pd.read_csv(uploaded_file)
else:
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
# Build working dataframe
# -------------------------------------------------
df = df_raw.copy()

df["__revenue__"] = pd.to_numeric(df[revenue_col], errors="coerce")

if cost_col:
    df["__cost__"] = pd.to_numeric(df[cost_col], errors="coerce")
else:
    df["__cost__"] = 0.0

df["__profit__"] = df["__revenue__"] - df["__cost__"]
df["__margin_pct__"] = np.where(
    df["__revenue__"] != 0, df["__profit__"] / df["__revenue__"] * 100, np.nan
)

if date_col:
    df["__date__"] = pd.to_datetime(df[date_col], errors="coerce")
    df["Month"] = df["__date__"].dt.to_period("M").dt.to_timestamp()
else:
    df["__date__"] = pd.NaT
    df["Month"] = pd.NaT

# -------------------------------------------------
# Sidebar: navigation & filters
# -------------------------------------------------
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        [
            "Overview",
            "Forecasting",
            "AI Insights",
            "Settings / Info",
        ],
    )

    st.markdown("---")
    st.header("Filters")

    # Date filter
    if df["__date__"].notna().any():
        min_date = df["__date__"].min()
        max_date = df["__date__"].max()
        date_range = st.date_input(
            "Date range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            mask = (df["__date__"] >= pd.Timestamp(start_date)) & (
                df["__date__"] <= pd.Timestamp(end_date)
            )
            df = df[mask]
    else:
        st.caption("No valid date column mapped; date filter disabled.")

    # Product filter
    if product_col:
        prods = df[product_col].dropna().unique().tolist()
        selected_prods = st.multiselect("Filter products", prods, default=prods)
        if selected_prods:
            df = df[df[product_col].isin(selected_prods)]
    else:
        st.caption("No product column mapped; product filter disabled.")

# -------------------------------------------------
# Recompute aggregations AFTER filters
# -------------------------------------------------
total_revenue = df["__revenue__"].sum(skipna=True)
total_cost = df["__cost__"].sum(skipna=True)
total_profit = df["__profit__"].sum(skipna=True)
overall_margin = (
    (total_profit / total_revenue * 100) if total_revenue not in (0, np.nan) else np.nan
)

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

# =================================================
# PAGE 1: OVERVIEW
# =================================================
if page == "Overview":
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
            st.info("Map a valid date column to see revenue and profit over time.")

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
            st.info("Map a product column to see top products by profit.")

    with st.expander("View detailed tables"):
        if monthly_summary is not None:
            st.write("Monthly summary")
            st.dataframe(monthly_summary)
        if product_summary is not None:
            st.write("Product summary")
            st.dataframe(product_summary)

    # ---------- Export section ----------
    st.markdown("---")
    st.subheader("Export Reports")

    if product_summary is not None and not product_summary.empty:
        csv_buf = product_summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download product summary (CSV)",
            data=csv_buf,
            file_name="product_summary.csv",
            mime="text/csv",
        )
    else:
        st.caption("Product summary CSV unavailable (no product column mapped).")

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

    # Optional PDF export
    if ENABLE_PDF_EXPORT:
        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Business Profit Dashboard Summary", ln=True)

            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 8, f"Total Revenue: ${total_revenue:,.0f}", ln=True)
            pdf.cell(0, 8, f"Total Profit: ${total_profit:,.0f}", ln=True)
            if not np.isnan(overall_margin):
                pdf.cell(0, 8, f"Profit Margin: {overall_margin:,.1f}%", ln=True)

            pdf_bytes = pdf.output(dest="S").encode("latin-1")
            st.download_button(
                "Download summary PDF",
                data=pdf_bytes,
                file_name="summary.pdf",
                mime="application/pdf",
            )
        except ImportError:
            st.warning(
                "PDF export enabled but `fpdf` is not installed. "
                "Install with `pip install fpdf` or set ENABLE_PDF_EXPORT = False."
            )

    # ---------- Insights & Executive Summary ----------
    st.markdown("---")
    st.subheader("Insights")

    insights = []
    exec_parts = []

    # Most profitable product
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

    # Monthly performance
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

        if len(ms) >= 3:
            last3 = ms.tail(3)
            revs = last3["Revenue"].values
            if np.all(np.diff(revs) > 0):
                trend_phrase = "Revenue has been trending up over the last three months."
                insights.append(trend_phrase)
            elif np.all(np.diff(revs) < 0):
                trend_phrase = (
                    "Revenue has been trending down over the last three months."
                )
                insights.append(trend_phrase)

        exec_parts.append(
            f"In {latest_month.strftime('%B %Y')}, the business generated "
            f"${latest_rev:,.0f} in revenue and ${latest_prof:,.0f} in profit. "
            f"{mom_phrase} {yoy_phrase} {trend_phrase}".strip()
        )

    # Loss warnings
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
        exec_parts.append(
            "Additionally, " + " ".join(w.rstrip(".") + "." for w in loss_warnings)
        )

    if not insights:
        st.write("Insights will appear here once more data is available.")
    else:
        for text in insights:
            st.markdown(f"- {text}")

    st.markdown("---")
    st.subheader("Executive Summary")

    executive_summary = " ".join(exec_parts).strip()
    if not executive_summary:
        executive_summary = (
            "A concise summary could not be generated yet. "
            "Upload data with mapped revenue, date, and product columns."
        )

    st.write(executive_summary)

# =================================================
# PAGE 2: FORECASTING
# =================================================
elif page == "Forecasting":
    st.markdown("---")
    st.subheader("Forecasts (simple trend)")

    if monthly_summary is None or monthly_summary.empty:
        st.info("Forecasting requires a valid date column and monthly data.")
        st.stop()

    ms = monthly_summary.sort_values("Month").copy()
    ms["t"] = np.arange(len(ms))

    # Simple linear regression with numpy.polyfit for revenue & profit
    def forecast_series(series, periods=6):
        t = np.arange(len(series))
        # fall back if all NaN or constant
        if len(series.dropna()) < 2:
            return None, None
        coef = np.polyfit(t, series, 1)
        trend = np.poly1d(coef)
        future_t = np.arange(len(series), len(series) + periods)
        return trend, future_t

    rev_trend, rev_future_t = forecast_series(ms["Revenue"])
    prof_trend, prof_future_t = forecast_series(ms["Profit"])

    future_months = pd.date_range(
        ms["Month"].max() + pd.offsets.MonthBegin(1),
        periods=6,
        freq="MS",
    )

    forecast_df = pd.DataFrame({"Month": ms["Month"], "Revenue": ms["Revenue"], "Profit": ms["Profit"]})

    if rev_trend is not None:
        rev_forecast_vals = rev_trend(rev_future_t)
        prof_forecast_vals = prof_trend(prof_future_t) if prof_trend is not None else np.zeros_like(
            rev_forecast_vals
        )
        future_df = pd.DataFrame(
            {
                "Month": future_months,
                "Revenue": rev_forecast_vals,
                "Profit": prof_forecast_vals,
                "Type": "Forecast",
            }
        )
        hist_df = forecast_df.copy()
        hist_df["Type"] = "Actual"
        combined = pd.concat([hist_df, future_df], ignore_index=True)

        st.write("Next 6-month forecast (simple trend):")
        fig_f = px.line(
            combined,
            x="Month",
            y=["Revenue", "Profit"],
            color="Type",
            title="Actual vs Forecast (Revenue & Profit)",
            markers=True,
        )
        st.plotly_chart(fig_f, use_container_width=True)
    else:
        st.info("Not enough non-NaN data to build a forecast.")

    with st.expander("Forecast data"):
        st.dataframe(forecast_df)

# =================================================
# PAGE 3: AI INSIGHTS
# =================================================
elif page == "AI Insights":
    st.markdown("---")
    st.subheader("AI Insights (optional)")

    st.write(
        "This page is a scaffold for AI-generated commentary using the OpenAI API.\n\n"
        "To enable it:\n"
        "1. Install the OpenAI Python library: `pip install openai`\n"
        "2. Set `ENABLE_AI_INSIGHTS = True` at the top of `app.py`.\n"
        "3. Set the environment variable `OPENAI_API_KEY` with your API key.\n"
    )

    if not ENABLE_AI_INSIGHTS:
        st.info("AI insights are disabled. Set ENABLE_AI_INSIGHTS = True to use.")
        st.stop()

    try:
        import openai

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OPENAI_API_KEY environment variable is not set.")
            st.stop()
        openai.api_key = api_key

        # Build a short text description of the data we have
        summary_prompt = f"""
You are a financial analyst. Provide a concise but insightful commentary (4-6 bullet points)
on the following business performance:

Total revenue: {total_revenue:,.0f}
Total profit: {total_profit:,.0f}
Overall margin: {overall_margin:,.1f}%

We also have monthly and product level breakdowns.
Highlight trends, risks, and opportunities in plain English.
"""
        st.write("Click the button below to generate AI commentary.")
        if st.button("Generate AI insights"):
            with st.spinner("Contacting OpenAI and generating insights..."):
                resp = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful senior FP&A analyst.",
                        },
                        {"role": "user", "content": summary_prompt},
                    ],
                    temperature=0.3,
                )
                text = resp["choices"][0]["message"]["content"]
                st.markdown(text)

    except ImportError:
        st.warning(
            "OpenAI library not installed. Run `pip install openai` or set "
            "ENABLE_AI_INSIGHTS = False."
        )

# =================================================
# PAGE 4: SETTINGS / INFO
# =================================================
elif page == "Settings / Info":
    st.markdown("---")
    st.subheader("Settings & Info")

    st.write(
        """
### What this app does

- Uploads CSV or Excel sales/profit data.
- Lets you map revenue, cost, date, and product columns.
- Applies filters (date & product).
- Shows KPIs, charts, tables, insights, and an executive summary.
- Builds a simple 6-month forecast based on trend.
- Supports optional authentication, AI Insights, and PDF export via flags.

### How to turn features on/off

Edit the top of `app.py`:

```python
ENABLE_AUTH = False          # True to enable login
ENABLE_AI_INSIGHTS = False   # True to enable OpenAI-based insights
ENABLE_PDF_EXPORT = False    # True to enable PDF download
