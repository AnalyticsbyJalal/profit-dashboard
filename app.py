import os
import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# Optional: only used if you have the OpenAI Python package installed
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


# -----------------------------------------------------------------------------
# CONFIG FLAGS
# -----------------------------------------------------------------------------
ENABLE_AUTH = True           # Step 1 - simple password login
ENABLE_AI_INSIGHTS = True    # Step 3 - AI narrative (requires OPENAI_API_KEY)


# -----------------------------------------------------------------------------
# AUTHENTICATION HELPERS
# -----------------------------------------------------------------------------
def check_password() -> bool:
    """
    Simple password gate using Streamlit session_state.
    Password is read from Streamlit secrets if available:
        APP_PASSWORD
    Fallback (if not set) = "Jarvis1997$"
    """
    if not ENABLE_AUTH:
        return True

    # Already logged in?
    if st.session_state.get("authenticated", False):
        return True

    # Get password from secrets or default
    secrets_pwd = st.secrets.get("APP_PASSWORD", None)
    app_password = secrets_pwd if secrets_pwd else "Jarvis1997$"

    st.title("🔐 Business Profit Dashboard")
    st.subheader("Login")

    pwd = st.text_input("Password", type="password")
    if st.button("Unlock"):
        if pwd == app_password:
            st.session_state["authenticated"] = True
            st.experimental_rerun()
        else:
            st.error("Incorrect password. Please try again.")

    return False


# -----------------------------------------------------------------------------
# DATA PREPARATION
# -----------------------------------------------------------------------------
def load_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    """Read CSV or Excel into a DataFrame."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            # openpyxl must be listed in requirements.txt
            return pd.read_excel(uploaded_file)
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading file {uploaded_file.name}: {e}")
        return pd.DataFrame()


def prepare_data(
    df_raw: pd.DataFrame,
    revenue_col: str | None,
    cost_col: str | None,
    date_col: str | None,
    product_col: str | None,
):
    """
    Clean and enrich the raw data:
      - coerce revenue / cost to numeric
      - parse dates
      - fill product if missing
      - compute profit and margin
      - build product_summary and monthly_summary
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = df_raw.copy()

    # --- Revenue ---
    if revenue_col and revenue_col in df.columns:
        df["__revenue__"] = pd.to_numeric(df[revenue_col], errors="coerce").fillna(0.0)
    else:
        df["__revenue__"] = 0.0

    # --- Cost ---
    if cost_col and cost_col in df.columns:
        df["__cost__"] = pd.to_numeric(df[cost_col], errors="coerce").fillna(0.0)
    else:
        df["__cost__"] = 0.0

    # --- Date ---
    if date_col and date_col in df.columns:
        df["__date__"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df["__date__"] = pd.NaT

    # --- Product ---
    if product_col and product_col in df.columns:
        df["__product__"] = df[product_col].fillna("Unknown").astype(str)
    else:
        df["__product__"] = "Unknown"

    # --- Profit & Margin ---
    df["__profit__"] = df["__revenue__"] - df["__cost__"]
    df["__margin_pct__"] = np.where(
        df["__revenue__"] != 0,
        df["__profit__"] / df["__revenue__"] * 100.0,
        0.0,
    )

    # --- Product Summary ---
    product_summary = (
        df.groupby("__product__", dropna=False)
        .agg(
            Revenue=("__revenue__", "sum"),
            Cost=("__cost__", "sum"),
            Profit=("__profit__", "sum"),
        )
        .reset_index()
        .rename(columns={"__product__": "Product"})
    )
    if not product_summary.empty:
        product_summary["Margin %"] = np.where(
            product_summary["Revenue"] != 0,
            product_summary["Profit"] / product_summary["Revenue"] * 100.0,
            0.0,
        )

    # --- Monthly Summary ---
    # Use Year-Month for aggregation
    if df["__date__"].notna().any():
        df["__month__"] = df["__date__"].dt.to_period("M").dt.to_timestamp()
    else:
        df["__month__"] = pd.NaT

    monthly_summary = (
        df.dropna(subset=["__month__"])
        .groupby("__month__")
        .agg(
            Revenue=("__revenue__", "sum"),
            Cost=("__cost__", "sum"),
            Profit=("__profit__", "sum"),
        )
        .reset_index()
        .rename(columns={"__month__": "Month"})
    )

    return df, product_summary, monthly_summary


# -----------------------------------------------------------------------------
# INSIGHTS GENERATION (non-AI)
# -----------------------------------------------------------------------------
def generate_text_insights(product_summary: pd.DataFrame, monthly_summary: pd.DataFrame):
    insights = []

    # 1) Most profitable product
    if not product_summary.empty:
        top_row = product_summary.sort_values("Profit", ascending=False).iloc[0]
        prod = top_row["Product"]
        profit = top_row["Profit"]
        margin = top_row["Margin %"]
        insights.append(
            f"{prod} is your most profitable product with profit of ${profit:,.0f} "
            f"and a margin of {margin:.1f}%."
        )

    # 2) Latest month snapshot + MoM
    if not monthly_summary.empty:
        ms = monthly_summary.sort_values("Month")
        latest = ms.iloc[-1]
        latest_month = latest["Month"]
        latest_rev = latest["Revenue"]
        latest_prof = latest["Profit"]

        # Month-over-month
        mom_text = ""
        if len(ms) >= 2:
            prev = ms.iloc[-2]
            if prev["Revenue"] != 0:
                mom_change = (latest_rev - prev["Revenue"]) / prev["Revenue"] * 100.0
                mom_text = f" Month-over-month, revenue changed by {mom_change:+.1f}% versus the prior month."
            else:
                mom_text = " Month-over-month comparison is not available (previous revenue was 0)."

        insights.append(
            f"In the latest month ({latest_month:%B %Y}), you generated "
            f"${latest_rev:,.0f} in revenue and ${latest_prof:,.0f} in profit."
            + mom_text
        )

        # 3) Year-over-year if we have at least 13 months
        yoy_text = ""
        if len(ms) >= 13:
            # Compare last month to same month last year
            this_month = latest_month.month
            this_year = latest_month.year
            last_year_mask = (ms["Month"].dt.month == this_month) & (ms["Month"].dt.year == this_year - 1)
            if last_year_mask.any():
                last_year_row = ms[last_year_mask].iloc[-1]
                if last_year_row["Revenue"] != 0:
                    yoy_change = (latest_rev - last_year_row["Revenue"]) / last_year_row["Revenue"] * 100.0
                    yoy_text = f" Year-over-year, revenue is {yoy_change:+.1f}% versus the same month last year."
        if yoy_text:
            insights.append(yoy_text)

        # 4) Trend over last 3 months
        if len(ms) >= 3:
            last3 = ms.tail(3)
            if last3["Revenue"].is_monotonic_increasing:
                trend = "up"
            elif last3["Revenue"].is_monotonic_decreasing:
                trend = "down"
            else:
                trend = "mixed"
            insights.append(f"Revenue has been trending {trend} over the last three months.")

    return insights


def generate_exec_summary(product_summary: pd.DataFrame, monthly_summary: pd.DataFrame) -> str:
    """Short, human-readable executive summary paragraph."""
    if product_summary.empty or monthly_summary.empty:
        return "Data is loaded, but there is not enough information yet to build an executive summary."

    # Top product
    top_row = product_summary.sort_values("Profit", ascending=False).iloc[0]
    prod = top_row["Product"]
    prod_profit = top_row["Profit"]
    prod_margin = top_row["Margin %"]

    ms = monthly_summary.sort_values("Month")
    latest = ms.iloc[-1]
    latest_month = latest["Month"]
    latest_rev = latest["Revenue"]
    latest_prof = latest["Profit"]

    # YoY
    yoy_part = ""
    if len(ms) >= 13:
        this_month = latest_month.month
        this_year = latest_month.year
        last_year_mask = (ms["Month"].dt.month == this_month) & (ms["Month"].dt.year == this_year - 1)
        if last_year_mask.any():
            last_year_row = ms[last_year_mask].iloc[-1]
            if last_year_row["Revenue"] != 0:
                yoy_change = (latest_rev - last_year_row["Revenue"]) / last_year_row["Revenue"] * 100.0
                yoy_part = f" Compared with the same month last year, revenue is {yoy_change:+.1f}%."

    # Trend last 3 months
    trend_part = ""
    if len(ms) >= 3:
        last3 = ms.tail(3)
        if last3["Revenue"].is_monotonic_increasing:
            trend_word = "up"
        elif last3["Revenue"].is_monotonic_decreasing:
            trend_word = "down"
        else:
            trend_word = "mixed"
        trend_part = f" Over the last three months, revenue has been trending {trend_word}."

    summary = (
        f"{prod} is currently your top performer, delivering ${prod_profit:,.0f} in profit "
        f"at a {prod_margin:.1f}% margin. In the latest month ({latest_month:%B %Y}), "
        f"the business generated ${latest_rev:,.0f} in revenue and ${latest_prof:,.0f} in profit."
        f"{yoy_part}{trend_part}"
    )

    return summary


# -----------------------------------------------------------------------------
# AI INSIGHTS (Step 3 – optional)
# -----------------------------------------------------------------------------
def generate_ai_narrative(
    product_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    total_revenue: float,
    total_profit: float,
) -> str:
    """Call OpenAI (if configured) to generate a narrative."""
    if not ENABLE_AI_INSIGHTS:
        return "AI Insights are disabled."

    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OpenAI is not configured. Add OPENAI_API_KEY to Streamlit secrets to enable AI insights."

    if OpenAI is None:
        return "OpenAI Python package is not installed. Add `openai` to requirements.txt."

    if monthly_summary.empty:
        return "Not enough data to generate AI insights yet."

    client = OpenAI(api_key=api_key)

    # Prepare compact text summary for the model
    prod_csv = product_summary.to_csv(index=False)
    monthly_csv = monthly_summary.to_csv(index=False)

    prompt = f"""
You are an FP&A and analytics expert.

Here is product performance data (CSV):
{prod_csv}

Here is monthly performance data (CSV):
{monthly_csv}

Total revenue: {total_revenue:,.2f}
Total profit: {total_profit:,.2f}

Write a concise narrative (3–5 short paragraphs) that covers:
- key revenue and profit drivers,
- product performance highlights,
- month-over-month and high-level trend commentary,
- 2–3 concrete business recommendations.

Write it in a professional but friendly tone, suitable for an executive audience.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert financial analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=700,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:  # pragma: no cover
        return f"Error calling OpenAI: {e}"


# -----------------------------------------------------------------------------
# MAIN APP LAYOUT
# -----------------------------------------------------------------------------
def main():
    # --- Auth gate ---
    if not check_password():
        return

    # --- Page config ---
    st.set_page_config(
        page_title="Business Profit Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📊 Business Profit Dashboard")

    # --- Sidebar: file upload ---
    st.sidebar.header("Upload your data")
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more CSV/XLSX files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    df_raw = pd.DataFrame()
    if uploaded_files:
        dfs = [load_file(f) for f in uploaded_files]
        dfs = [d for d in dfs if not d.empty]
        if dfs:
            df_raw = pd.concat(dfs, ignore_index=True)
        else:
            st.warning("No valid data loaded yet.")

    if df_raw.empty:
        st.info("Upload one or more CSV/XLSX files in the sidebar to get started.")
        return

    # --- Data preview ---
    with st.expander("🔍 Data preview", expanded=True):
        st.dataframe(df_raw.head(100))

    # --- Column mapping ---
    st.markdown("---")
    st.subheader("🧩 Column Mapping")

    cols = list(df_raw.columns)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        revenue_col = st.selectbox("Revenue column", cols, index=0 if cols else None)
    with col2:
        cost_col = st.selectbox("Cost column (optional)", ["(none)"] + cols, index=1 if len(cols) > 1 else 0)
        if cost_col == "(none)":
            cost_col = None
    with col3:
        date_col = st.selectbox("Date column (optional)", ["(none)"] + cols, index=1 if len(cols) > 1 else 0)
        if date_col == "(none)":
            date_col = None
    with col4:
        product_col = st.selectbox("Product column (optional)", ["(none)"] + cols, index=1 if len(cols) > 1 else 0)
        if product_col == "(none)":
            product_col = None

    # --- Prepare data ---
    df, product_summary, monthly_summary = prepare_data(
        df_raw,
        revenue_col=revenue_col,
        cost_col=cost_col,
        date_col=date_col,
        product_col=product_col,
    )

    if df.empty:
        st.warning("Unable to prepare data. Check your column mapping and try again.")
        return

    total_revenue = float(df["__revenue__"].sum())
    total_cost = float(df["__cost__"].sum())
    total_profit = float(df["__profit__"].sum())
    overall_margin = (total_profit / total_revenue * 100.0) if total_revenue != 0 else 0.0

    # -----------------------------------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📌 Key Metrics")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Revenue", f"${total_revenue:,.0f}")
    kpi2.metric("Total Cost", f"${total_cost:,.0f}")
    kpi3.metric("Total Profit", f"${total_profit:,.0f}")
    kpi4.metric("Profit Margin", f"{overall_margin:.1f}%")

    # -----------------------------------------------------------------------------
    # CHARTS
    # -----------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📈 Revenue & Profit Over Time")

    if not monthly_summary.empty:
        chart_df = monthly_summary.set_index("Month")[["Revenue", "Profit"]]
        st.line_chart(chart_df)
    else:
        st.info("No valid date column selected. Add a date column to see time-series charts.")

    # -----------------------------------------------------------------------------
    # PRODUCT TABLE
    # -----------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🏷️ Product Performance")
    if not product_summary.empty:
        st.dataframe(
            product_summary.sort_values("Profit", ascending=False),
            use_container_width=True,
        )
    else:
        st.info("No product column selected; all rows are treated as 'Unknown'.")

    # -----------------------------------------------------------------------------
    # TEXTUAL INSIGHTS
    # -----------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("💡 Insights")

    text_insights = generate_text_insights(product_summary, monthly_summary)
    if not text_insights:
        st.write("Insights will appear here once enough data is available.")
    else:
        for bullet in text_insights:
            st.markdown(f"- {bullet}")

    # -----------------------------------------------------------------------------
    # EXECUTIVE SUMMARY
    # -----------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📋 Executive Summary")

    exec_summary = generate_exec_summary(product_summary, monthly_summary)
    st.write(exec_summary)

    # -----------------------------------------------------------------------------
    # AI INSIGHTS (Step 3)
    # -----------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🤖 AI Insights (beta)")

    if st.button("Generate AI Narrative"):
        with st.spinner("Calling OpenAI and generating narrative..."):
            ai_text = generate_ai_narrative(
                product_summary=product_summary,
                monthly_summary=monthly_summary,
                total_revenue=total_revenue,
                total_profit=total_profit,
            )
        st.markdown(ai_text)


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
