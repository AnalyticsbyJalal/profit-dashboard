import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ================================================================
# FEATURE FLAGS – turn future upgrades on/off here
# ================================================================
ENABLE_AUTH = True           # 1) Simple password gate
ENABLE_FORECASTING = True    # 2) Forecasting page
ENABLE_AI_INSIGHTS = True    # 3) ChatGPT-style AI insights

# ================================================================
# OPTIONAL: OpenAI for AI Insights
# ================================================================
OPENAI_AVAILABLE = False
try:
    if ENABLE_AI_INSIGHTS:
        import openai

        # Expect key in Streamlit secrets for security
        openai.api_key = st.secrets.get("OPENAI_API_KEY", None)
        if openai.api_key:
            OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# ================================================================
# BASIC AUTHENTICATION
# ================================================================
def require_auth():
    """Very simple, optional auth gate. Controlled by ENABLE_AUTH."""
    if not ENABLE_AUTH:
        return

    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False

    if st.session_state.auth_ok:
        return

    st.sidebar.markdown("### 🔐 Login")
    password = st.sidebar.text_input("Password", type="password")

    expected_pwd = st.secrets.get("APP_PASSWORD", "Jarvis1997$")

    if st.sidebar.button("Unlock"):
        if password == expected_pwd:
            st.session_state.auth_ok = True
            st.sidebar.success("Access granted.")
        else:
            st.sidebar.error("Incorrect password.")

    if not st.session_state.auth_ok:
        st.stop()


# ================================================================
# DATA PREP HELPERS
# ================================================================
def load_file(f) -> pd.DataFrame:
    """Load CSV or Excel, return DataFrame."""
    name = f.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(f)
    else:
        # Requires openpyxl – make sure it's in requirements.txt
        return pd.read_excel(f)


def prepare_data(
    df_raw: pd.DataFrame,
    revenue_col: str,
    cost_col: Optional[str],
    date_col: Optional[str],
    product_col: Optional[str],
    region_col: Optional[str],
    customer_col: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df        – cleaned row-level data
      monthly   – monthly revenue/profit summary
      products  – product-level profit summary
    """
    df = df_raw.copy()

    # Normalise column names for indexing
    cols_lower = {c.lower(): c for c in df.columns}

    def pick(col_name: Optional[str]) -> Optional[str]:
        if not col_name:
            return None
        return cols_lower.get(col_name.lower(), col_name)

    rev_src = pick(revenue_col)
    cost_src = pick(cost_col)
    date_src = pick(date_col)
    prod_src = pick(product_col)
    region_src = pick(region_col)
    customer_src = pick(customer_col)

    # Revenue & cost numeric
    df["__revenue__"] = pd.to_numeric(df[rev_src], errors="coerce").fillna(0.0)

    if cost_src and cost_src in df.columns:
        df["__cost__"] = pd.to_numeric(df[cost_src], errors="coerce").fillna(0.0)
    else:
        df["__cost__"] = 0.0

    df["__profit__"] = df["__revenue__"] - df["__cost__"]

    # Date
    if date_src and date_src in df.columns:
        df["__date__"] = pd.to_datetime(df[date_src], errors="coerce")
    else:
        df["__date__"] = pd.NaT

    # Product
    if prod_src and prod_src in df.columns:
        df["__product__"] = df[prod_src].astype(str)
    else:
        df["__product__"] = "Unknown"

    # Region
    if region_src and region_src in df.columns:
        df["__region__"] = df[region_src].astype(str)
    else:
        df["__region__"] = "Unspecified"

    # Customer
    if customer_src and customer_src in df.columns:
        df["__customer__"] = df[customer_src].astype(str)
    else:
        df["__customer__"] = "Unspecified"

    # Monthly summary
    if df["__date__"].notna().any():
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
    else:
        monthly = pd.DataFrame(
            columns=["Month", "__revenue__", "__cost__", "__profit__", "Margin %"]
        )

    # Product summary
    products = (
        df.groupby("__product__", as_index=False)[["__revenue__", "__cost__", "__profit__"]]
        .sum()
        .rename(columns={"__product__": "Product"})
    )
    products["Margin %"] = np.where(
        products["__revenue__"] != 0,
        products["__profit__"] / products["__revenue__"] * 100,
        0.0,
    )
    products = products.sort_values("__profit__", ascending=False)

    return df, monthly, products


# ================================================================
# KPI / INSIGHTS UTILITIES
# ================================================================
def calc_key_metrics(df: pd.DataFrame) -> Tuple[float, float, float]:
    total_rev = df["__revenue__"].sum()
    total_cost = df["__cost__"].sum()
    total_profit = df["__profit__"].sum()
    return total_rev, total_cost, total_profit


def format_currency(x: float) -> str:
    return f"${x:,.0f}"


def render_insights(monthly: pd.DataFrame, products: pd.DataFrame):
    st.markdown("### 💡 Insights")

    if monthly.empty:
        st.write("Insights will appear here once you map your columns and include a date column.")
        return

    total_rev = monthly["__revenue__"].sum()
    total_profit = monthly["__profit__"].sum()

    # Top product
    if not products.empty:
        top_prod = products.iloc[0]
        st.write(
            f"- **{top_prod['Product']}** is your most profitable product with "
            f"profit of {format_currency(top_prod['__profit__'])} and a margin of "
            f"{top_prod['Margin %']:.1f}%."
        )

    # Latest month
    latest = monthly.iloc[-1]
    latest_month = latest["Month"].strftime("%B %Y")
    st.write(
        f"- In the latest month (**{latest_month}**), you generated "
        f"{format_currency(latest['__revenue__'])} in revenue and "
        f"{format_currency(latest['__profit__'])} in profit."
    )

    # MoM change
    if len(monthly) >= 2:
        prev = monthly.iloc[-2]
        if prev["__revenue__"] != 0:
            mom = (latest["__revenue__"] - prev["__revenue__"]) / prev["__revenue__"] * 100
            direction = "up" if mom >= 0 else "down"
            st.write(
                f"- Month-over-month, revenue is **{mom:+.1f}%** ({direction} vs. the previous month)."
            )

    # Trend – last 3 months
    if len(monthly) >= 3:
        last3 = monthly.tail(3)
        rev_trend = np.polyfit(range(3), last3["__revenue__"], 1)[0]
        trend_word = "upward" if rev_trend > 0 else "downward"
        st.write(
            f"- Revenue has been trending **{trend_word}** over the last three months."
        )

    # Executive summary
    st.markdown("---")
    st.markdown("### 📝 Executive Summary")

    if products.empty:
        st.write("Not enough data yet for an executive summary.")
        return

    best = products.iloc[0]
    summary = (
        f"{best['Product']} is currently your top performer, delivering "
        f"{format_currency(best['__profit__'])} in profit at a "
        f"{best['Margin %']:.1f}% margin. "
        f"Total revenue in the dataset is {format_currency(total_rev)}, "
        f"with total profit of {format_currency(total_profit)}."
    )

    st.write(summary)


# ================================================================
# FORECASTING
# ================================================================
def build_forecast(monthly: pd.DataFrame, horizon_months: int = 12) -> pd.DataFrame:
    """
    Very simple linear trend forecast over revenue.
    Returns a DataFrame with historical + forecasted months.
    """
    if monthly.empty or len(monthly) < 3:
        return pd.DataFrame()

    monthly = monthly.sort_values("Month").reset_index(drop=True)

    x = np.arange(len(monthly))
    y = monthly["__revenue__"].values

    slope, intercept = np.polyfit(x, y, 1)

    last_idx = x[-1]
    future_idx = np.arange(last_idx + 1, last_idx + 1 + horizon_months)

    last_month = monthly["Month"].iloc[-1]
    future_months = [last_month + pd.DateOffset(months=i) for i in range(1, horizon_months + 1)]

    future_rev = slope * future_idx + intercept
    future_df = pd.DataFrame(
        {"Month": future_months, "__revenue__": future_rev, "is_forecast": True}
    )

    hist_df = monthly[["Month", "__revenue__"]].copy()
    hist_df["is_forecast"] = False

    combined = pd.concat([hist_df, future_df], ignore_index=True)
    return combined


# ================================================================
# AI INSIGHTS
# ================================================================
def generate_ai_insights(monthly: pd.DataFrame, products: pd.DataFrame) -> str:
    if not OPENAI_AVAILABLE:
        return "OpenAI is not configured. Add OPENAI_API_KEY to Streamlit secrets to enable AI insights."

    if monthly.empty:
        return "Please upload data and map your columns (including a date column) first."

    latest = monthly.iloc[-1]
    latest_month = latest["Month"].strftime("%B %Y")
    total_rev = monthly["__revenue__"].sum()
    total_profit = monthly["__profit__"].sum()

    top_products = (
        products[["Product", "__revenue__", "__profit__", "Margin %"]]
        .head(5)
        .to_dict(orient="records")
    )

    prompt = f"""
You are a financial analytics assistant.
Here is a summary of business performance:

Latest month: {latest_month}
Latest month revenue: {latest['__revenue__']:.2f}
Latest month profit: {latest['__profit__']:.2f}
Total revenue (all data): {total_rev:.2f}
Total profit (all data): {total_profit:.2f}

Top products (up to 5, with revenue, profit, and margin):
{top_products}

Please provide:
1) A short narrative of how the business is performing.
2) 3 concrete recommendations to improve profit.
3) 2 potential risks or red flags you see.
Keep it concise and easy to understand.
"""

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert FP&A business analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message["content"].strip()
    except Exception as e:
        return f"Error while calling OpenAI: {e}"


# ================================================================
# MAIN APP
# ================================================================
def main():
    st.set_page_config(
        page_title="Business Profit Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    require_auth()

    st.title("Business Profit Dashboard")

    # ------------------------------------------------------------
    # Sidebar – navigation
    # ------------------------------------------------------------
    page = st.sidebar.radio("Navigation", ["Dashboard", "Forecasting", "Raw Data", "Settings / Info"])

    # File upload
    st.sidebar.markdown("### 📂 Upload your data")
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more CSV/XLSX files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload one or more CSV/XLSX files in the sidebar to get started.")
        if page == "Settings / Info":
            render_settings()
        st.stop()

    # Load and concatenate
    try:
        dfs = [load_file(f) for f in uploaded_files]
        df_raw = pd.concat(dfs, ignore_index=True)
    except Exception as e:
        st.error(f"Error reading file(s): {e}")
        st.stop()

    # ------------------------------------------------------------
    # Column mapping
    # ------------------------------------------------------------
    st.markdown("### Column Mapping")

    cols = list(df_raw.columns)
    if not cols:
        st.error("Uploaded file(s) seem to be empty.")
        st.stop()

    # Best-guess defaults
    def guess(name: str, fallback: str) -> str:
        matches = [c for c in cols if name.lower() in c.lower()]
        return matches[0] if matches else fallback

    revenue_col = st.selectbox("Revenue column", cols, index=cols.index(guess("revenue", cols[0])))
    cost_col = st.selectbox("Cost column", cols, index=cols.index(guess("cost", cols[0]))) if len(cols) > 1 else None
    date_col = st.selectbox("Date column", cols, index=cols.index(guess("date", cols[0]))) if len(cols) > 1 else None
    product_col = st.selectbox("Product column", cols, index=cols.index(guess("product", cols[0]))) if len(cols) > 1 else None

    # NEW: optional region & customer mapping
    region_col = st.selectbox(
        "Region column (optional)",
        ["<None>"] + cols,
        index=0,
    )
    region_col = None if region_col == "<None>" else region_col

    customer_col = st.selectbox(
        "Customer column (optional)",
        ["<None>"] + cols,
        index=0,
    )
    customer_col = None if customer_col == "<None>" else customer_col

    df, monthly, products = prepare_data(
        df_raw,
        revenue_col,
        cost_col,
        date_col,
        product_col,
        region_col,
        customer_col,
    )

    # ------------------------------------------------------------
    # Filters (Product, Date range, Region, Customer)
    # ------------------------------------------------------------
    st.sidebar.markdown("### 🔍 Filters")

    # Product filter
    product_options = sorted(df["__product__"].unique())
    selected_products = st.sidebar.multiselect(
        "Products", product_options, default=product_options
    )

    # Region filter
    region_options = sorted(df["__region__"].unique())
    if len(region_options) > 1 or region_options[0] != "Unspecified":
        selected_regions = st.sidebar.multiselect(
            "Regions", region_options, default=region_options
        )
    else:
        selected_regions = region_options  # no real regions – treat as all

    # Customer filter
    customer_options = sorted(df["__customer__"].unique())
    if len(customer_options) > 1 or customer_options[0] != "Unspecified":
        selected_customers = st.sidebar.multiselect(
            "Customers", customer_options, default=customer_options
        )
    else:
        selected_customers = customer_options

    # Date filter
    if df["__date__"].notna().any():
        min_date = df["__date__"].min()
        max_date = df["__date__"].max()
        date_range = st.sidebar.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            mask = (df["__date__"] >= pd.to_datetime(start_date)) & (
                df["__date__"] <= pd.to_datetime(end_date)
            )
            df = df[mask].copy()
    else:
        st.sidebar.info("No valid date column mapped; date filter disabled.")

    # Apply categorical filters
    if selected_products:
        df = df[df["__product__"].isin(selected_products)].copy()
    if selected_regions:
        df = df[df["__region__"].isin(selected_regions)].copy()
    if selected_customers:
        df = df[df["__customer__"].isin(selected_customers)].copy()

    # Recalculate summaries on filtered data
    if df.empty:
        st.warning("No data after filters – adjust filters or column mapping.")
        st.stop()

    _, monthly_filtered, products_filtered = prepare_data(
        df,
        revenue_col=revenue_col,
        cost_col=cost_col,
        date_col=date_col,
        product_col=product_col,
        region_col=region_col,
        customer_col=customer_col,
    )

    # Page routing
    if page == "Dashboard":
        render_dashboard(df, monthly_filtered, products_filtered)
    elif page == "Forecasting":
        render_forecasting(monthly_filtered)
    elif page == "Raw Data":
        render_raw(df)
    elif page == "Settings / Info":
        render_settings(monthly_filtered, products_filtered)


# ================================================================
# PAGE RENDERERS
# ================================================================
def render_dashboard(df: pd.DataFrame, monthly: pd.DataFrame, products: pd.DataFrame):
    st.markdown("## 📊 Overview")

    total_rev, total_cost, total_profit = calc_key_metrics(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", format_currency(total_rev))
    col2.metric("Total Cost", format_currency(total_cost))
    col3.metric("Total Profit", format_currency(total_profit))
    margin = (total_profit / total_rev * 100) if total_rev != 0 else 0.0
    col4.metric("Overall Margin", f"{margin:.1f}%")

    # Charts
    st.markdown("### Revenue & Profit Over Time")
    if monthly.empty:
        st.write("No valid date column mapped; can't plot monthly trends.")
    else:
        monthly_sorted = monthly.sort_values("Month")
        chart_df = monthly_sorted[["Month", "__revenue__", "__profit__"]].rename(
            columns={"__revenue__": "Revenue", "__profit__": "Profit"}
        )
        chart_df = chart_df.set_index("Month")
        st.line_chart(chart_df)

    st.markdown("### Top Products by Profit")
    st.dataframe(
        products[["Product", "__revenue__", "__profit__", "Margin %"]]
        .rename(
            columns={
                "__revenue__": "Revenue",
                "__profit__": "Profit",
            }
        )
        .style.format({"Revenue": "${:,.0f}", "Profit": "${:,.0f}", "Margin %": "{:,.1f}%"}),
        use_container_width=True,
    )

    st.markdown("---")
    render_insights(monthly, products)

    if ENABLE_AI_INSIGHTS:
        st.markdown("---")
        st.markdown("### 🤖 AI Insights (beta)")
        if st.button("Generate AI Narrative"):
            with st.spinner("Asking the AI analyst..."):
                msg = generate_ai_insights(monthly, products)
            st.write(msg)
        else:
            st.caption("Click the button to generate AI-driven commentary (if OpenAI is configured).")


def render_forecasting(monthly: pd.DataFrame):
    st.markdown("## 📈 Forecasting")

    if not ENABLE_FORECASTING:
        st.info("Forecasting is currently disabled via feature flag.")
        return

    if monthly.empty or len(monthly) < 3:
        st.warning("Need at least 3 months of data to build a trend-based forecast.")
        return

    horizon = st.slider("Forecast horizon (months)", 3, 24, 12)
    fc = build_forecast(monthly, horizon_months=horizon)

    if fc.empty:
        st.warning("Could not build a forecast from the current data.")
        return

    chart_df = fc[["Month", "__revenue__", "is_forecast"]].copy()
    chart_df["Type"] = np.where(chart_df["is_forecast"], "Forecast", "Actual")

    st.markdown("### Revenue: Actual vs Forecast")
    pivot = chart_df.pivot(index="Month", columns="Type", values="__revenue__")
    st.line_chart(pivot)

    st.markdown("### Forecast Details")
    fut = fc[fc["is_forecast"]].copy()
    fut_display = fut[["Month", "__revenue__"]].rename(
        columns={"__revenue__": "Forecast Revenue"}
    )
    st.dataframe(
        fut_display.style.format({"Forecast Revenue": "${:,.0f}"}),
        use_container_width=True,
    )


def render_raw(df: pd.DataFrame):
    st.markdown("## 📄 Raw Data")
    st.dataframe(df, use_container_width=True)


def render_settings(
    monthly: Optional[pd.DataFrame] = None,
    products: Optional[pd.DataFrame] = None,
):
    st.markdown("## ⚙️ Settings / Info")

    st.markdown(
        """
This app is built for **Business Profit Analytics**:

- Upload CSV or Excel sales/profit data.
- Map revenue, cost, date, product, region, and customer columns.
- Filter by product, region, customer, and date range.
- View KPIs, charts, insights, and an executive summary.
- (Optional) Use basic authentication, forecasting, and AI insights.
"""
    )

    st.markdown("### Feature Flags")
    st.code(
        f"""
ENABLE_AUTH = {ENABLE_AUTH}
ENABLE_FORECASTING = {ENABLE_FORECASTING}
ENABLE_AI_INSIGHTS = {ENABLE_AI_INSIGHTS}
""",
        language="python",
    )

    if ENABLE_AI_INSIGHTS:
        st.markdown("### OpenAI Configuration")
        if OPENAI_AVAILABLE:
            st.success("OpenAI key found in Streamlit secrets. AI Insights is enabled.")
        else:
            st.warning(
                "AI Insights is enabled, but OpenAI is not fully configured.\n\n"
                "- Add `openai` to `requirements.txt`\n"
                "- Add `OPENAI_API_KEY` to Streamlit secrets"
            )

    st.markdown("### Roadmap")
    st.write(
        """
Planned upgrades:

1. Stronger authentication (per-user, roles).
2. More advanced forecasting models (Prophet / ARIMA).
3. Richer AI narratives and scenario analysis.
4. Additional pages for product, region, and customer analysis.
5. PDF exports with your logo and executive summary.
"""
    )


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    main()

