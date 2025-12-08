import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime

# =========================================================
# FEATURE FLAGS (simple, safe ones only)
# =========================================================
ENABLE_AI_INSIGHTS = False
ENABLE_PDF_EXPORT = False  # reserved for later
ENABLE_ADVANCED_FILTERS = True

# =========================================================
# PAGE CONFIG + THEME
# =========================================================
st.set_page_config(
    page_title="Business Profit Dashboard",
    page_icon="📊",
    layout="wide",
)

CUSTOM_CSS = """
<style>
/* Corporate blue header + clean cards */
.reportview-container .main {
    background-color: #f5f7fb;
}
[data-testid="stSidebar"] {
    background-color: #0f1c3f;
    color: white;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: white;
}
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
.section-card {
    padding: 18px 20px;
    border-radius: 12px;
    background-color: #ffffff;
    box-shadow: 0 2px 6px rgba(15, 28, 63, 0.05);
    border: 1px solid #e1e4f0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================
# HELPERS
# =========================================================
def load_uploaded_data(uploaded_files):
    """Load one or more uploaded CSV/XLSX files into a single DataFrame."""
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
                xls = pd.ExcelFile(f)
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

    # Normalize names
    df.rename(
        columns={
            revenue_col: "__revenue__",
            cost_col: "__cost__",
            date_col: "__date__",
            product_col: "product",
        },
        inplace=True,
    )

    # Clean types
    df["__revenue__"] = pd.to_numeric(df["__revenue__"], errors="coerce").fillna(0.0)
    df["__cost__"] = pd.to_numeric(df["__cost__"], errors="coerce").fillna(0.0)

    # Date
    df["__date__"] = pd.to_datetime(df["__date__"], errors="coerce")
    df = df[~df["__date__"].isna()]

    # Profit
    df["__profit__"] = df["__revenue__"] - df["__cost__"]

    return df


def build_product_summary(df):
    """Group by product -> revenue / cost / profit / margin."""
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
    """Aggregate by month."""
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
    """Generate Insights bullets + Executive Summary text."""
    insights = []
    exec_parts = []

    # 1) Most profitable product
    if product_summary is not None and not product_summary.empty:
        top = product_summary.iloc[0]
        top_product = top["product"]
        top_profit = top["__profit__"]
        top_margin = top["Margin %"]

        # Insights bullet
        insights.append(
            f"**{top_product}** is your most profitable product with profit of "
            f"**${top_profit:,.0f}** and a margin of **{top_margin:.1f}%**."
        )

        # Exec summary pieces
        exec_parts.append(
            f"{top_product} is currently your top performer, delivering "
            f"**${top_profit:,.0f}** in profit."
        )
        exec_parts.append(
            f"That corresponds to a margin of **{top_margin:.1f}%**."
        )

    # 2) Monthly performance / MoM / YoY / trend
    if monthly_summary is not None and not monthly_summary.empty:
        ms = monthly_summary.sort_values("Month").copy()
        latest = ms.iloc[-1]
        latest_month = latest["Month"]
        latest_rev = latest["__revenue__"]
        latest_prof = latest["__profit__"]

        # Latest month insight
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

        # Trend over last 3 months
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

        # Exec summary for month performance
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
# SIDEBAR
# =========================================================
# Logo (optional)
try:
    st.sidebar.image("logo.png", use_column_width=True)
except Exception:
    st.sidebar.markdown("### Business Profit Dashboard")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Raw Data", "Settings / Info"],
)

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

# Filter options
product_options = sorted(df["product"].dropna().unique()) if "product" in df.columns else []
date_min = df["__date__"].min()
date_max = df["__date__"].max()

if ENABLE_ADVANCED_FILTERS:
    st.markdown("### Filters")
    fc1, fc2 = st.columns([2, 2])
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

    if selected_products:
        df = df[df["product"].isin(selected_products)]
    if isinstance(selected_range, (list, tuple)) and len(selected_range) == 2:
        start, end = selected_range
        df = df[(df["__date__"] >= pd.to_datetime(start)) & (df["__date__"] <= pd.to_datetime(end))]

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
# PAGE: DASHBOARD
# =========================================================
if page == "Dashboard":
    st.title("Business Profit Dashboard")

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
        if np.isnan(overall_margin):
            margin_text = "N/A"
        else:
            margin_text = f"{overall_margin:.1f}%"
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

    # Insights & Executive Summary
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

# =========================================================
# PAGE: RAW DATA
# =========================================================
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

# =========================================================
# PAGE: SETTINGS / INFO
# =========================================================
else:
    st.markdown("---")
    st.subheader("Settings & Info")

    st.write(
        """
        ### What this app does

        - Uploads CSV or Excel sales/profit data (supports multiple files & sheets).
        - Lets you map revenue, cost, date, and product columns.
        - Applies filters on product and date.
        - Shows KPIs, charts, and an Insights + Executive Summary section.
        - Builds monthly and product-level summaries.
        - Lets you export product summary (CSV) and a full Excel report.
        """
    )

    st.write(
        """
        ### Feature flags (top of `app.py`)

        ```python
        ENABLE_AI_INSIGHTS = False
        ENABLE_PDF_EXPORT = False
        ENABLE_ADVANCED_FILTERS = True
        ```

        You can flip these to `True` later as we add more advanced features.
        """
    )
