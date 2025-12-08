import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================================================
# FEATURE FLAGS
# =========================================================
ENABLE_AUTH = False            # Password gate (set password in st.secrets["APP_PASSWORD"])
ENABLE_FORECASTING = True      # Show Forecasting page (simple regression)
ENABLE_AI_INSIGHTS = False     # Use OpenAI for extra narrative (requires openai + API key)
ENABLE_PDF_EXPORT = False      # Placeholder for PDF export (requires fpdf2 or similar)
ENABLE_ADVANCED_FILTERS = True # Product/date + region/customer filters


# =========================================================
# PAGE CONFIG + THEME
# =========================================================
st.set_page_config(
    page_title="Business Profit Dashboard",
    page_icon="📊",
    layout="wide",
)

LIGHT_CORPORATE_CSS = """
<style>
/* Main background */
.reportview-container .main {
    background-color: #f5f7fb;
}

/* Light sidebar theme */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    color: #111827;
    border-right: 1px solid #e5e7eb;
}

/* Sidebar headings */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #111827;
}

/* Metric cards */
.metric-card {
    padding: 18px 20px;
    border-radius: 12px;
    background-color: #ffffff;
    box-shadow: 0 2px 6px rgba(15, 28, 63, 0.08);
    border: 1px solid #e1e4f0;
}
.metric-label {
    font-size: 0.85rem;
    color: #6b7280;
    font-weight: 500;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #111827;
}
.metric-sub {
    font-size: 0.8rem;
    color: #9ca3af;
}

/* Section cards */
.section-card {
    padding: 18px 20px;
    border-radius: 12px;
    background-color: #ffffff;
    box-shadow: 0 2px 6px rgba(15, 28, 63, 0.05);
    border: 1px solid #e1e4f0;
}
</style>
"""
st.markdown(LIGHT_CORPORATE_CSS, unsafe_allow_html=True)


# =========================================================
# AUTHENTICATION (optional)
# =========================================================
def check_auth():
    """Very simple password gate using st.secrets['APP_PASSWORD']."""
    correct_password = st.secrets.get("APP_PASSWORD", "").strip()
    if not correct_password:
        st.warning(
            "Authentication is enabled, but no `APP_PASSWORD` is set in Streamlit secrets. "
            "Set it under **Settings → Secrets** in Streamlit Cloud."
        )
        return

    st.sidebar.markdown("### Login")
    pwd = st.sidebar.text_input("Password", type="password")
    if pwd == "":
        st.stop()
    if pwd != correct_password:
        st.sidebar.error("Incorrect password.")
        st.stop()


# =========================================================
# HELPERS
# =========================================================
def load_uploaded_data(uploaded_files):
    """Load CSV/XLSX files into a single DataFrame."""
    if not uploaded_files:
        return pd.DataFrame()

    frames = []
    for f in uploaded_files:
        name = f.name.lower()
        try:
            if name.endswith(".csv"):
                df = pd.read_csv(f)
                frames.append(df)
            elif name.endswith(".xlsx") or name.endswith(".xls"):
                try:
                    # requires openpyxl; make sure it's in requirements.txt
                    xls = pd.ExcelFile(f, engine="openpyxl")
                except Exception as e:
                    st.error(
                        f"Error reading file {f.name}: {e}. "
                        "Make sure `openpyxl` is listed in requirements.txt."
                    )
                    continue

                for sheet in xls.sheet_names:
                    df = xls.parse(sheet)
                    frames.append(df)
        except Exception as e:
            st.error(f"Error reading file {f.name}: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def prepare_data(df, revenue_col, cost_col, date_col, product_col):
    """Standardize columns and compute revenue / cost / profit."""
    df = df.copy()

    df.rename(
        columns={
            revenue_col: "__revenue__",
            cost_col: "__cost__",
            date_col: "__date__",
            product_col: "product",
        },
        inplace=True,
    )

    df["__revenue__"] = pd.to_numeric(df["__revenue__"], errors="coerce").fillna(0.0)
    df["__cost__"] = pd.to_numeric(df["__cost__"], errors="coerce").fillna(0.0)

    df["__date__"] = pd.to_datetime(df["__date__"], errors="coerce")
    df = df[~df["__date__"].isna()]

    df["__profit__"] = df["__revenue__"] - df["__cost__"]

    return df


def build_product_summary(df):
    if "product" not in df.columns:
        return pd.DataFrame()

    grp = (
        df.groupby("product", as_index=False)[["__revenue__", "__cost__", "__profit__"]]
        .sum()
    )
    grp["Margin %"] = np.where(
        grp["__revenue__"] != 0,
        grp["__profit__"] / grp["__revenue__"] * 100,
        np.nan,
    )
    return grp.sort_values("__profit__", ascending=False)


def build_monthly_summary(df):
    temp = df.copy()
    temp["Month"] = temp["__date__"].dt.to_period("M").dt.to_timestamp()
    grp = (
        temp.groupby("Month", as_index=False)[["__revenue__", "__cost__", "__profit__"]]
        .sum()
    )
    grp["Margin %"] = np.where(
        grp["__revenue__"] != 0,
        grp["__profit__"] / grp["__revenue__"] * 100,
        np.nan,
    )
    return grp.sort_values("Month")


def build_insights_and_exec(df, monthly_summary, product_summary):
    """Generate Insights bullets + Executive Summary (clean formatting!)."""
    insights = []
    exec_parts = []

    # 1) Most profitable product
    if product_summary is not None and not product_summary.empty:
        top = product_summary.iloc[0]
        top_product = top["product"]
        top_profit = top["__profit__"]
        top_margin = top["Margin %"]

        insights.append(
            f"**{top_product}** is your most profitable product with profit of "
            f"**${top_profit:,.0f}** and a margin of **{top_margin:.1f}%**."
        )

        exec_parts.append(
            f"{top_product} is currently your top performer, delivering "
            f"**${top_profit:,.0f}** in profit with a margin of **{top_margin:.1f}%**."
        )

    # 2) Monthly performance / MoM / YoY / trend
    if monthly_summary is not None and not monthly_summary.empty:
        ms = monthly_summary.sort_values("Month").copy()
        latest = ms.iloc[-1]
        latest_month = latest["Month"]
        latest_rev = latest["__revenue__"]
        latest_prof = latest["__profit__"]

        insights.append(
            f"In the latest month (**{latest_month:%B %Y}**), you generated "
            f"**${latest_rev:,.0f}** in revenue and **${latest_prof:,.0f}** in profit."
        )

        # Month-over-month
        mom_rev = np.nan
        if len(ms) >= 2:
            prev = ms.iloc[-2]
            base = prev["__revenue__"]
            if base != 0:
                mom_rev = (latest_rev - base) / base * 100

        if not np.isnan(mom_rev):
            direction = "up" if mom_rev >= 0 else "down"
            insights.append(
                f"Month-over-month, revenue is **{mom_rev:+.1f}%** "
                f"({direction} vs. the prior month)."
            )

        # Year-over-year (if we have at least 13 months)
        yoy_rev = np.nan
        if len(ms) >= 13:
            target_period = (latest_month - pd.DateOffset(years=1)).to_period("M")
            ms["Period"] = ms["Month"].dt.to_period("M")
            match = ms[ms["Period"] == target_period]
            if not match.empty:
                last_year_rev = match.iloc[-1]["__revenue__"]
                if last_year_rev != 0:
                    yoy_rev = (latest_rev - last_year_rev) / last_year_rev * 100

        if not np.isnan(yoy_rev):
            direction = "up" if yoy_rev >= 0 else "down"
            insights.append(
                f"Year-over-year, revenue for {latest_month:%B} is "
                f"**{yoy_rev:+.1f}%** ({direction})."
            )

        # Trend last 3 months
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

        if trend_phrase:
            insights.append(
                f"Revenue has been trending **{trend_phrase}** over the last three months."
            )

        exec_parts.append(
            f"In {latest_month:%B %Y}, you generated "
            f"**${latest_rev:,.0f}** in revenue and **${latest_prof:,.0f}** in profit."
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


def build_simple_forecast(monthly_summary, periods=6):
    """Simple linear regression forecast over monthly revenue."""
    if monthly_summary is None or monthly_summary.empty:
        return pd.DataFrame()

    ms = monthly_summary.sort_values("Month").copy()
    ms = ms.dropna(subset=["__revenue__"])
    if len(ms) < 3:
        return pd.DataFrame()

    x = np.arange(len(ms))
    y = ms["__revenue__"].values
    m, b = np.polyfit(x, y, 1)

    future_idx = np.arange(len(ms), len(ms) + periods)
    future_y = m * future_idx + b

    last_month = ms["Month"].iloc[-1]
    future_months = [last_month + pd.DateOffset(months=i + 1) for i in range(periods)]

    df_actual = pd.DataFrame(
        {"Month": ms["Month"], "Revenue": y, "Type": "Actual"}
    )
    df_forecast = pd.DataFrame(
        {"Month": future_months, "Revenue": future_y, "Type": "Forecast"}
    )

    return pd.concat([df_actual, df_forecast], ignore_index=True)


def generate_ai_narrative(df, monthly_summary, product_summary):
    """Optional AI narrative using OpenAI (only if ENABLE_AI_INSIGHTS)."""
    try:
        from openai import OpenAI
    except Exception:
        return (
            "AI insights are enabled, but the `openai` package is not installed. "
            "Add `openai` to requirements.txt to use this feature."
        )

    api_key = st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return (
            "AI insights are enabled, but no OpenAI API key is configured. "
            "Add `OPENAI_API_KEY` to your Streamlit secrets or environment."
        )

    client = OpenAI(api_key=api_key)

    # Build a compact data summary for the prompt
    basic_metrics = {
        "total_revenue": float(df['__revenue__'].sum()),
        "total_profit": float(df['__profit__'].sum()),
        "num_rows": int(len(df)),
    }

    prompt = (
        "You are a senior FP&A analyst. Based on the following metrics, "
        "produce a concise narrative with strategy recommendations, risks and opportunities.\n\n"
        f"Basic metrics: {basic_metrics}\n\n"
        "Write 2–3 short paragraphs, plain text, no bullet points."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful financial analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI: {e}"


def download_product_summary_csv(product_summary):
    csv_buf = io.StringIO()
    product_summary.to_csv(csv_buf, index=False)
    return csv_buf.getvalue().encode("utf-8")


def download_full_excel_report(df, monthly_summary, product_summary):
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Raw Data")
        monthly_summary.to_excel(writer, index=False, sheet_name="Monthly Summary")
        product_summary.to_excel(writer, index=False, sheet_name="Product Summary")
    excel_buf.seek(0)
    return excel_buf


# =========================================================
# SIDEBAR + AUTH
# =========================================================
if ENABLE_AUTH:
    check_auth()

# Logo
try:
    st.sidebar.image("logo.png", use_column_width=True)
except Exception:
    st.sidebar.markdown("### Analytics By Jalal")

st.sidebar.markdown("---")

nav_pages = [
    "Overview",
    "Forecasting",
    "Product Performance",
    "Profitability",
    "Region / Customer",
    "Raw Data",
    "Settings / Info",
]
page = st.sidebar.radio("Navigation", nav_pages)

st.sidebar.markdown("### Upload your data")
uploaded_files = st.sidebar.file_uploader(
    "CSV or Excel",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
)

# =========================================================
# LOAD DATA
# =========================================================
df_raw = load_uploaded_data(uploaded_files)

if df_raw.empty:
    st.title("Business Profit Dashboard")
    st.write(
        "Upload one or more **CSV/XLSX** files in the sidebar to get started. "
        "You’ll then map your revenue, cost, date, and product columns."
    )
    st.stop()

# =========================================================
# COLUMN MAPPING
# =========================================================
st.markdown("### Column Mapping")

cols = df_raw.columns.tolist()

c1, c2, c3, c4 = st.columns(4)
with c1:
    revenue_col = st.selectbox("Revenue column", cols, index=0)
with c2:
    cost_col = st.selectbox("Cost column", cols, index=min(1, len(cols) - 1))
with c3:
    date_col = st.selectbox("Date column", cols, index=min(2, len(cols) - 1))
with c4:
    product_col = st.selectbox("Product column", cols, index=min(3, len(cols) - 1))

df = prepare_data(df_raw, revenue_col, cost_col, date_col, product_col)

# Optional region/customer detection
region_candidates = ["region", "Region", "REGION", "area", "Area"]
customer_candidates = ["customer", "Customer", "CUSTOMER", "client", "Client"]

region_col = next((c for c in df_raw.columns if c in region_candidates), None)
customer_col = next((c for c in df_raw.columns if c in customer_candidates), None)

if region_col and region_col not in df.columns:
    df[region_col] = df_raw[region_col]
if customer_col and customer_col not in df.columns:
    df[customer_col] = df_raw[customer_col]

# =========================================================
# FILTERS
# =========================================================
product_options = sorted(df["product"].dropna().unique()) if "product" in df.columns else []
date_min = df["__date__"].min()
date_max = df["__date__"].max()

selected_region_values = None
selected_customer_values = None

if ENABLE_ADVANCED_FILTERS:
    st.markdown("### Filters")
    fc1, fc2, fc3 = st.columns([2, 2, 2])
    with fc1:
        selected_products = st.multiselect(
            "Products",
            options=product_options,
            default=product_options,
        )
    with fc2:
        selected_range = st.date_input(
            "Date range",
            value=(date_min.date(), date_max.date()),
        )
    with fc3:
        if region_col:
            selected_region_values = st.multiselect(
                "Region",
                options=sorted(df[region_col].dropna().unique()),
                default=sorted(df[region_col].dropna().unique()),
            )

    if selected_products:
        df = df[df["product"].isin(selected_products)]
    if isinstance(selected_range, (list, tuple)) and len(selected_range) == 2:
        start, end = selected_range
        df = df[(df["__date__"] >= pd.to_datetime(start)) & (df["__date__"] <= pd.to_datetime(end))]
    if region_col and selected_region_values:
        df = df[df[region_col].isin(selected_region_values)]

    if customer_col:
        selected_customer_values = st.multiselect(
            "Customer",
            options=sorted(df[customer_col].dropna().unique()),
            default=sorted(df[customer_col].dropna().unique()),
        )
        if selected_customer_values:
            df = df[df[customer_col].isin(selected_customer_values)]

if df.empty:
    st.warning("Your current filter selection returns no rows. Adjust filters to see data.")
    st.stop()

# =========================================================
# SUMMARIES
# =========================================================
monthly_summary = build_monthly_summary(df)
product_summary = build_product_summary(df)

total_revenue = df["__revenue__"].sum()
total_cost = df["__cost__"].sum()
total_profit = df["__profit__"].sum()
overall_margin = (total_profit / total_revenue * 100) if total_revenue != 0 else np.nan

insights, exec_summary = build_insights_and_exec(df, monthly_summary, product_summary)

# =========================================================
# PAGE CONTENT
# =========================================================

# --------------- OVERVIEW (main dashboard) ----------------
if page == "Overview":
    st.title("Business Profit Dashboard – Overview")

    # Key metrics
    st.markdown("#### Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Total Revenue</div>
              <div class="metric-value">${total_revenue:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Total Cost</div>
              <div class="metric-value">${total_cost:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Total Profit</div>
              <div class="metric-value">${total_profit:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        margin_text = "N/A" if np.isnan(overall_margin) else f"{overall_margin:.1f}%"
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Overall Margin</div>
              <div class="metric-value">{margin_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Charts
    st.markdown("#### Revenue & Profit Over Time")
    if not monthly_summary.empty:
        fig_time = px.line(
            monthly_summary,
            x="Month",
            y=["__revenue__", "__profit__"],
            labels={"value": "Amount", "Month": "Month", "variable": "Metric"},
            title="Monthly Revenue & Profit",
        )
        fig_time.update_layout(legend_title_text="")
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("Not enough data to plot monthly trend yet.")

    st.markdown("#### Top Products by Profit")
    if not product_summary.empty:
        fig_prod = px.bar(
            product_summary.head(15),
            x="product",
            y="__profit__",
            title="Top Products by Profit",
            labels={"__profit__": "Profit", "product": "Product"},
        )
        st.plotly_chart(fig_prod, use_container_width=True)
    else:
        st.info("No product breakdown available yet.")

    # Insights & Executive summary
    st.markdown("---")
    c_left, c_right = st.columns([1.2, 1])
    with c_left:
        st.subheader("💡 Insights")
        if insights:
            for bullet in insights:
                st.markdown(f"- {bullet}")
        else:
            st.write("Insights will appear here once there is enough data.")
    with c_right:
        st.subheader("📝 Executive Summary")
        st.write(exec_summary)

        if ENABLE_AI_INSIGHTS:
            if st.button("Generate AI Insights (Experimental)"):
                ai_text = generate_ai_narrative(df, monthly_summary, product_summary)
                st.markdown("---")
                st.markdown(ai_text)

    # Export section
    st.markdown("---")
    st.subheader("⬇️ Export Reports")

    if not product_summary.empty:
        csv_data = download_product_summary_csv(product_summary)
        st.download_button(
            "Download product summary (CSV)",
            data=csv_data,
            file_name="product_summary.csv",
            mime="text/csv",
        )
    else:
        st.caption("Product summary export will be available once there is product data.")

    if not df.empty and not monthly_summary.empty and not product_summary.empty:
        excel_buf = download_full_excel_report(df, monthly_summary, product_summary)
        st.download_button(
            "Download full Excel report",
            data=excel_buf,
            file_name="profit_dashboard_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.caption("Full Excel report will be available once there is enough data.")

    if ENABLE_PDF_EXPORT:
        st.caption(
            "PDF export placeholder: when you're ready, we can plug in `fpdf2` or a similar "
            "library to generate branded PDF reports with logo, tables, and charts."
        )

# --------------- FORECASTING ----------------
elif page == "Forecasting":
    st.title("📈 Forecasting")

    if not ENABLE_FORECASTING:
        st.info("Forecasting is currently disabled. Set ENABLE_FORECASTING = True.")
    else:
        if monthly_summary.empty:
            st.warning("Not enough data to build a forecast yet.")
        else:
            st.markdown(
                "This is a simple linear regression forecast over monthly **revenue**. "
                "We can later upgrade this to Prophet / ARIMA for richer seasonality."
            )
            forecast_df = build_simple_forecast(monthly_summary, periods=6)
            if forecast_df.empty:
                st.warning("Need at least 3 months of data for a forecast.")
            else:
                fig_fc = px.line(
                    forecast_df,
                    x="Month",
                    y="Revenue",
                    color="Type",
                    title="6-Month Revenue Forecast (Simple Regression)",
                    labels={"Revenue": "Revenue", "Month": "Month"},
                )
                st.plotly_chart(fig_fc, use_container_width=True)

                st.markdown("#### Forecast Data")
                st.dataframe(forecast_df)

# --------------- PRODUCT PERFORMANCE ----------------
elif page == "Product Performance":
    st.title("🏷️ Product Performance")

    if product_summary.empty:
        st.info("No product summary yet.")
    else:
        st.markdown("### Profit by Product")
        st.dataframe(product_summary)

        fig_prod = px.bar(
            product_summary,
            x="product",
            y="__profit__",
            title="Profit by Product",
            labels={"__profit__": "Profit", "product": "Product"},
        )
        st.plotly_chart(fig_prod, use_container_width=True)

# --------------- PROFITABILITY ----------------
elif page == "Profitability":
    st.title("💵 Profitability")

    if monthly_summary.empty:
        st.info("No monthly summary yet.")
    else:
        st.markdown("### Monthly Margin %")
        ms = monthly_summary.copy()
        fig_margin = px.line(
            ms,
            x="Month",
            y="Margin %",
            title="Monthly Profit Margin %",
            labels={"Margin %": "Margin %", "Month": "Month"},
        )
        st.plotly_chart(fig_margin, use_container_width=True)

        st.markdown("### Monthly Profit Table")
        st.dataframe(ms[["Month", "__revenue__", "__cost__", "__profit__", "Margin %"]])

# --------------- REGION / CUSTOMER ----------------
elif page == "Region / Customer":
    st.title("🌍 Region / Customer Insights")

    if not region_col and not customer_col:
        st.info(
            "No region or customer columns were detected. "
            "If your data has them, name the columns something like "
            "`Region`/`region` or `Customer`/`customer`."
        )
    else:
        if region_col:
            st.markdown(f"### Revenue by {region_col}")
            reg_grp = (
                df.groupby(region_col, as_index=False)[["__revenue__", "__profit__"]]
                .sum()
            )
            fig_reg = px.bar(
                reg_grp,
                x=region_col,
                y="__revenue__",
                title=f"Revenue by {region_col}",
                labels={"__revenue__": "Revenue"},
            )
            st.plotly_chart(fig_reg, use_container_width=True)
            st.dataframe(reg_grp)

        if customer_col:
            st.markdown(f"### Top Customers by Revenue ({customer_col})")
            cust_grp = (
                df.groupby(customer_col, as_index=False)[["__revenue__", "__profit__"]]
                .sum()
                .sort_values("__revenue__", ascending=False)
                .head(50)
            )
            fig_cust = px.bar(
                cust_grp,
                x=customer_col,
                y="__revenue__",
                title="Top Customers by Revenue",
                labels={"__revenue__": "Revenue"},
            )
            st.plotly_chart(fig_cust, use_container_width=True)
            st.dataframe(cust_grp)

# --------------- RAW DATA ----------------
elif page == "Raw Data":
    st.title("Raw Data Preview")
    st.markdown("### Filtered Dataset")
    st.dataframe(df.head(1000))

    st.markdown("### Monthly Summary")
    if not monthly_summary.empty:
        st.dataframe(monthly_summary)
    else:
        st.info("No monthly summary yet.")

    st.markdown("### Product Summary")
    if not product_summary.empty:
        st.dataframe(product_summary)
    else:
        st.info("No product summary yet.")

# --------------- SETTINGS / INFO ----------------
else:
    st.title("Settings & Info")

    st.write(
        """
        ### What this app does

        - Uploads CSV or Excel sales/profit data (supports multiple files & sheets).
        - Lets you map revenue, cost, date, and product columns.
        - Applies filters on product, date, and (optionally) region and customer.
        - Shows KPIs, charts, and an Insights + Executive Summary section.
        - Builds monthly and product-level summaries.
        - Exports product summary (CSV) and a full Excel report.
        """
    )

    st.write(
        """
        ### Advanced features we wired in

        - **Authentication** (ENABLE_AUTH): password gate using `st.secrets["APP_PASSWORD"]`.
        - **Forecasting** (ENABLE_FORECASTING): simple 6-month regression forecast now,
          with room to upgrade to Prophet / ARIMA later.
        - **AI Insights** (ENABLE_AI_INSIGHTS): uses OpenAI when `OPENAI_API_KEY` is set.
        - **PDF Export** (ENABLE_PDF_EXPORT): placeholder for branded PDF reports.
        - **Multi-page navigation**: Overview, Forecasting, Product Performance,
          Profitability, Region/Customer, Raw Data, Settings/Info.
        """
    )

    st.write(
        """
        ### Next steps

        - Add a custom domain like `dashboard.analyticsbyjalal.com` via Streamlit Cloud.
        - When you're ready, we can refine:
          - Mobile responsiveness
          - Dark mode toggle
          - More advanced forecasting models
          - Fully branded PDF templates
        """
    )
