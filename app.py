import os
from datetime import datetime, date
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

# Optional: only used if you have the OpenAI Python package installed
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

# Optional: PDF generation for reports
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# Optional: Prophet for advanced forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False


# -----------------------------------------------------------------------------
# CONFIG FLAGS
# -----------------------------------------------------------------------------
ENABLE_AUTH = True           # simple password login
ENABLE_AI_INSIGHTS = True    # AI narrative (requires OPENAI_API_KEY)


# -----------------------------------------------------------------------------
# PAGE CONFIG (must be first Streamlit call)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AnalyticsByJalal — Profit Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# AUTHENTICATION
# -----------------------------------------------------------------------------
def check_password() -> bool:
    """
    Simple password gate using Streamlit session_state.
    Password is read from Streamlit secrets if available:
        APP_PASSWORD
    Fallback (if not set) = "jalal2025"
    """
    if not ENABLE_AUTH:
        return True

    if st.session_state.get("authenticated", False):
        return True

    secrets_pwd = st.secrets.get("APP_PASSWORD", None)
    app_password = secrets_pwd if secrets_pwd else "jalal2025"

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
# DATA PREP + SUMMARY
# -----------------------------------------------------------------------------
def load_file(uploaded_file) -> pd.DataFrame:
    """Read CSV or Excel into a DataFrame."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(uploaded_file)
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading file {uploaded_file.name}: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    """
    Built-in demo dataset with:
    - date, product, region, country, category, customer, revenue, cost
    """
    np.random.seed(42)

    n_rows = 600
    dates = pd.date_range(start="2023-01-01", periods=18, freq="MS")
    products = ["Alpha", "Bravo", "Charlie", "Delta"]
    regions = ["North", "South", "East", "West"]
    countries = ["USA", "Canada", "Mexico"]
    categories = ["Frozen", "Fresh", "Ambient"]
    customers = [f"Customer_{i}" for i in range(1, 21)]

    rows = []
    for _ in range(n_rows):
        d = np.random.choice(dates)
        p = np.random.choice(products)
        r = np.random.choice(regions)
        ctry = np.random.choice(countries)
        cat = np.random.choice(categories)
        cust = np.random.choice(customers)
        base_rev = np.random.uniform(500, 5000)
        noise = np.random.normal(scale=300)
        revenue = max(100, base_rev + noise)
        cost = revenue * np.random.uniform(0.5, 0.85)
        rows.append([d, p, r, ctry, cat, cust, revenue, cost])

    df = pd.DataFrame(
        rows,
        columns=[
            "date",
            "product",
            "region",
            "country",
            "category",
            "customer",
            "revenue",
            "cost",
        ],
    )
    return df


@st.cache_data(show_spinner=False)
def summarize_prepared(df: pd.DataFrame):
    """Build product_summary and monthly_summary from df that already has internal columns."""
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Product summary
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

    # Monthly summary
    if df["__date__"].notna().any():
        df_month = df.copy()
        df_month["__month__"] = df_month["__date__"].dt.to_period("M").dt.to_timestamp()
        monthly_summary = (
            df_month.dropna(subset=["__month__"])
            .groupby("__month__")
            .agg(
                Revenue=("__revenue__", "sum"),
                Cost=("__cost__", "sum"),
                Profit=("__profit__", "sum"),
            )
            .reset_index()
            .rename(columns={"__month__": "Month"})
        )
    else:
        monthly_summary = pd.DataFrame()

    return product_summary, monthly_summary


def prepare_data(
    df_raw: pd.DataFrame,
    revenue_col: str | None,
    cost_col: str | None,
    date_col: str | None,
    product_col: str | None,
    region_col: str | None,
    customer_col: str | None,
    country_col: str | None,
    category_col: str | None,
):
    """
    Clean and enrich the raw data:
      - coerce revenue / cost to numeric
      - parse dates
      - fill product if missing
      - compute profit and margin
      - map optional region / customer / country / category
    Then build product_summary and monthly_summary.
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = df_raw.copy()

    # Revenue
    if revenue_col and revenue_col in df.columns:
        df["__revenue__"] = pd.to_numeric(df[revenue_col], errors="coerce").fillna(0.0)
    else:
        df["__revenue__"] = 0.0

    # Cost
    if cost_col and cost_col in df.columns:
        df["__cost__"] = pd.to_numeric(df[cost_col], errors="coerce").fillna(0.0)
    else:
        df["__cost__"] = 0.0

    # Date
    if date_col and date_col in df.columns:
        df["__date__"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df["__date__"] = pd.NaT

    # Product
    if product_col and product_col in df.columns:
        df["__product__"] = df[product_col].fillna("Unknown").astype(str)
    else:
        df["__product__"] = "Unknown"

    # Region
    if region_col and region_col in df.columns:
        df["__region__"] = df[region_col].fillna("Unspecified Region").astype(str)
    else:
        df["__region__"] = "Unspecified Region"

    # Customer
    if customer_col and customer_col in df.columns:
        df["__customer__"] = df[customer_col].fillna("Unspecified Customer").astype(str)
    else:
        df["__customer__"] = "Unspecified Customer"

    # Country
    if country_col and country_col in df.columns:
        df["__country__"] = df[country_col].fillna("Unspecified Country").astype(str)
    else:
        df["__country__"] = "Unspecified Country"

    # Category
    if category_col and category_col in df.columns:
        df["__category__"] = df[category_col].fillna("Unspecified Category").astype(str)
    else:
        df["__category__"] = "Unspecified Category"

    # Profit & margin
    df["__profit__"] = df["__revenue__"] - df["__cost__"]
    df["__margin_pct__"] = np.where(
        df["__revenue__"] != 0,
        df["__profit__"] / df["__revenue__"] * 100.0,
        0.0,
    )

    product_summary, monthly_summary = summarize_prepared(df)
    return df, product_summary, monthly_summary


# -----------------------------------------------------------------------------
# INSIGHTS (TEXT)
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

    # 2) Latest month + MoM
    if not monthly_summary.empty:
        ms = monthly_summary.sort_values("Month")
        latest = ms.iloc[-1]
        latest_month = latest["Month"]
        latest_rev = latest["Revenue"]
        latest_prof = latest["Profit"]

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

        # 3) YoY
        if len(ms) >= 13:
            this_month = latest_month.month
            this_year = latest_month.year
            last_year_mask = (ms["Month"].dt.month == this_month) & (ms["Month"].dt.year == this_year - 1)
            if last_year_mask.any():
                last_year_row = ms[last_year_mask].iloc[-1]
                if last_year_row["Revenue"] != 0:
                    yoy_change = (latest_rev - last_year_row["Revenue"]) / last_year_row["Revenue"] * 100.0
                    insights.append(
                        f"Year-over-year, revenue for {latest_month:%B} is {yoy_change:+.1f}% versus the same month last year."
                    )

        # 4) Trend last 3 months
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

    top_row = product_summary.sort_values("Profit", ascending=False).iloc[0]
    prod = top_row["Product"]
    prod_profit = top_row["Profit"]
    prod_margin = top_row["Margin %"]

    ms = monthly_summary.sort_values("Month")
    latest = ms.iloc[-1]
    latest_month = latest["Month"]
    latest_rev = latest["Revenue"]
    latest_prof = latest["Profit"]

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
# FORECASTING
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_linear_forecast(monthly_summary: pd.DataFrame, periods: int = 12) -> pd.DataFrame | None:
    """
    Simple linear-trend forecast on monthly revenue.
    Returns DataFrame with Month, Revenue, Type (Actual/Forecast).
    """
    if monthly_summary is None or monthly_summary.empty or len(monthly_summary) < 3:
        return None

    ms = monthly_summary.sort_values("Month").copy()
    x = np.arange(len(ms))
    y = ms["Revenue"].values

    # Linear trend
    slope, intercept = np.polyfit(x, y, 1)
    future_x = np.arange(len(ms), len(ms) + periods)
    future_y = intercept + slope * future_x

    last_month = ms["Month"].iloc[-1]
    future_months = [last_month + pd.offsets.MonthBegin(i + 1) for i in range(periods)]

    actual_df = ms.copy()
    actual_df["Type"] = "Actual"

    forecast_df = pd.DataFrame(
        {
            "Month": future_months,
            "Revenue": future_y,
            "Type": ["Forecast"] * periods,
        }
    )

    combined = pd.concat([actual_df, forecast_df], ignore_index=True)
    return combined


@st.cache_data(show_spinner=False)
def build_prophet_forecast(monthly_summary: pd.DataFrame, periods: int = 12) -> pd.DataFrame | None:
    """
    Advanced forecast using Prophet (if available).
    Returns DataFrame with Month, yhat, yhat_lower, yhat_upper, Type.
    """
    if not PROPHET_AVAILABLE:
        return None
    if monthly_summary is None or monthly_summary.empty or len(monthly_summary) < 3:
        return None

    ms = monthly_summary.sort_values("Month").copy()
    df_p = ms[["Month", "Revenue"]].rename(columns={"Month": "ds", "Revenue": "y"})

    m = Prophet()
    m.fit(df_p)

    future = m.make_future_dataframe(periods=periods, freq="MS")  # Month Start
    forecast = m.predict(future)

    # Merge back actual vs forecast
    merged = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    merged["Type"] = np.where(merged["ds"].isin(df_p["ds"]), "Actual", "Forecast")
    merged.rename(columns={"ds": "Month"}, inplace=True)

    return merged


# -----------------------------------------------------------------------------
# AI INSIGHTS
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
- product, region, and customer performance highlights (if present),
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
# UTIL: GUESS COLUMN INDICES
# -----------------------------------------------------------------------------
def guess_column_index(columns: list[str], keywords: list[str]) -> int | None:
    """
    Try to guess which column index matches any of the provided keywords.
    """
    lowered = [c.lower() for c in columns]
    for kw in keywords:
        for i, col in enumerate(lowered):
            if kw in col:
                return i
    return None


# -----------------------------------------------------------------------------
# PDF REPORT GENERATION
# -----------------------------------------------------------------------------
def build_pdf_report(
    total_revenue: float,
    total_cost: float,
    total_profit: float,
    overall_margin: float,
    product_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    exec_summary: str,
    insights: list[str],
) -> bytes:
    """
    Build a simple but professional PDF report with:
    - Logo (if logo.png exists)
    - Title + date
    - KPIs
    - Executive Summary
    - Insights bullets
    - Top products (limited)
    - Monthly summary (limited)
    Returns PDF as bytes.
    """
    from io import BytesIO

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50

    # Logo
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        try:
            c.drawImage(logo_path, 40, y - 40, width=100, preserveAspectRatio=True, mask="auto")
        except Exception:
            pass

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(160, y - 10, "Business Performance Report")

    c.setFont("Helvetica", 10)
    c.drawString(160, y - 26, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    y -= 70

    # KPIs
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Key Metrics")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Total Revenue: ${total_revenue:,.0f}")
    y -= 14
    c.drawString(50, y, f"Total Cost:    ${total_cost:,.0f}")
    y -= 14
    c.drawString(50, y, f"Total Profit:  ${total_profit:,.0f}")
    y -= 14
    c.drawString(50, y, f"Profit Margin: {overall_margin:.1f}%")

    y -= 24

    # Executive Summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Executive Summary")
    y -= 16
    c.setFont("Helvetica", 10)

    from textwrap import wrap

    for line in wrap(exec_summary, width=100):
        if y < 80:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 10)
        c.drawString(50, y, line)
        y -= 12

    y -= 18

    # Insights
    if insights:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "Key Insights")
        y -= 16
        c.setFont("Helvetica", 10)
        for bullet in insights[:5]:
            for line in wrap(bullet, width=95):
                if y < 80:
                    c.showPage()
                    y = height - 50
                    c.setFont("Helvetica", 10)
                c.drawString(50, y, f"- {line}")
                y -= 12
            y -= 4

    # New page for tables
    c.showPage()
    y = height - 50

    # Top Products
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Top Products by Profit")
    y -= 18
    c.setFont("Helvetica", 9)

    if not product_summary.empty:
        top_products = product_summary.sort_values("Profit", ascending=False).head(10)
        c.drawString(50, y, "Product")
        c.drawString(220, y, "Revenue")
        c.drawString(320, y, "Profit")
        c.drawString(410, y, "Margin %")
        y -= 12
        c.line(50, y, 500, y)
        y -= 12

        for _, row in top_products.iterrows():
            if y < 80:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica-Bold", 12)
                c.drawString(40, y, "Top Products by Profit (cont.)")
                y -= 18
                c.setFont("Helvetica", 9)

            c.drawString(50, y, str(row["Product"])[:22])
            c.drawRightString(290, y, f"${row['Revenue']:,.0f}")
            c.drawRightString(380, y, f"${row['Profit']:,.0f}")
            c.drawRightString(470, y, f"{row['Margin %']:,.1f}%")
            y -= 12
    else:
        c.drawString(50, y, "No product data available for this period.")
        y -= 14

    # Monthly summary
    if not monthly_summary.empty:
        y -= 24
        if y < 120:
            c.showPage()
            y = height - 50

        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "Monthly Performance")
        y -= 18
        c.setFont("Helvetica", 9)

        c.drawString(50, y, "Month")
        c.drawRightString(220, y, "Revenue")
        c.drawRightString(320, y, "Profit")
        y -= 12
        c.line(50, y, 500, y)
        y -= 12

        for _, row in monthly_summary.sort_values("Month").tail(18).iterrows():
            if y < 80:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica-Bold", 12)
                c.drawString(40, y, "Monthly Performance (cont.)")
                y -= 18
                c.setFont("Helvetica", 9)

            month_str = row["Month"].strftime("%Y-%m")
            c.drawString(50, y, month_str)
            c.drawRightString(220, y, f"${row['Revenue']:,.0f}")
            c.drawRightString(320, y, f"${row['Profit']:,.0f}")
            y -= 12

    c.showPage()
    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# -----------------------------------------------------------------------------
# DATA EXPORT HELPERS
# -----------------------------------------------------------------------------
def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to Excel bytes for download."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buffer.getvalue()


# -----------------------------------------------------------------------------
# MAIN APP (MULTI-PAGE)
# -----------------------------------------------------------------------------
def main():
    if not check_password():
        return

    # -------------------------------------------------------------------------
    # THEME TOGGLE (Light / Dark)
    # -------------------------------------------------------------------------
    theme = st.sidebar.radio("🎨 Theme", ["Light", "Dark"], index=0)

    if theme == "Dark":
        st.markdown(
            """
            <style>
            .block-container {
                padding-top: 1rem;
                background-color: #0e1117;
                color: #f9fafb;
            }
            body {
                background-color: #0e1117;
            }
            .stApp {
                background-color: #0e1117;
            }
            .stMetric-value, .stMetric-label {
                color: #f9fafb !important;
            }
            .css-1d391kg, .css-18e3th9 {
                background-color: #111827 !important;
                color: #f9fafb !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .block-container { padding-top: 1rem; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # Branded header
    # -------------------------------------------------------------------------
    header_col1, header_col2 = st.columns([1, 5])
    with header_col1:
        if os.path.exists("logo.png"):
            st.image("logo.png", use_column_width=True)
        else:
            st.empty()
    with header_col2:
        st.markdown(
            """
            <div style="padding-left: 10px;">
                <h1 style="margin-bottom: 0;">AnalyticsByJalal Dashboard</h1>
                <p style="color: #9ca3af; margin-top: 4px;">
                    Profitability • Performance • Forecasts • AI Insights
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # SIDEBAR: data source + upload
    # -------------------------------------------------------------------------
    st.sidebar.header("Data Source")
    data_source = st.sidebar.radio(
        "Choose data source",
        ["Upload your own file(s)", "Use demo sample data"],
        index=0,
    )

    df_raw = pd.DataFrame()

    if data_source == "Use demo sample data":
        df_raw = load_sample_data()
        st.sidebar.success("Using built-in demo dataset.")
    else:
        st.sidebar.markdown("### Upload your data")
        uploaded_files = st.sidebar.file_uploader(
            "Upload one or more CSV/XLSX files",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            dfs = [load_file(f) for f in uploaded_files]
            dfs = [d for d in dfs if not d.empty]
            if dfs:
                df_raw = pd.concat(dfs, ignore_index=True)

    if df_raw.empty:
        st.info("Upload one or more CSV/XLSX files in the sidebar or select 'Use demo sample data' to get started.")
        return

    with st.expander("🔍 Data preview", expanded=True):
        st.dataframe(df_raw.head(100), use_container_width=True)

    # -------------------------------------------------------------------------
    # COLUMN MAPPING with smart defaults
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🧩 Column Mapping")

    cols = list(df_raw.columns)

    rev_guess = guess_column_index(cols, ["revenue", "sales", "amount"])
    cost_guess = guess_column_index(cols, ["cost", "cogs", "expense"])
    date_guess = guess_column_index(cols, ["date", "month", "period"])
    prod_guess = guess_column_index(cols, ["product", "sku", "item", "name"])
    region_guess = guess_column_index(cols, ["region", "area", "zone"])
    customer_guess = guess_column_index(cols, ["customer", "client", "account", "cust"])
    country_guess = guess_column_index(cols, ["country", "nation"])
    category_guess = guess_column_index(cols, ["category", "segment", "type", "class"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        revenue_col = st.selectbox(
            "Revenue column",
            cols,
            index=rev_guess if rev_guess is not None else 0,
        )
    with col2:
        cost_options = ["(none)"] + cols
        cost_index = (cost_guess + 1) if cost_guess is not None else 0
        cost_col = st.selectbox(
            "Cost column (optional)",
            cost_options,
            index=cost_index if cost_index < len(cost_options) else 0,
        )
        if cost_col == "(none)":
            cost_col = None
    with col3:
        date_options = ["(none)"] + cols
        date_index = (date_guess + 1) if date_guess is not None else 0
        date_col = st.selectbox(
            "Date column (optional)",
            date_options,
            index=date_index if date_index < len(date_options) else 0,
        )
        if date_col == "(none)":
            date_col = None
    with col4:
        prod_options = ["(none)"] + cols
        prod_index = (prod_guess + 1) if prod_guess is not None else 0
        product_col = st.selectbox(
            "Product column (optional)",
            prod_options,
            index=prod_index if prod_index < len(prod_options) else 0,
        )
        if product_col == "(none)":
            product_col = None

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        region_options = ["(none)"] + cols
        region_index = (region_guess + 1) if region_guess is not None else 0
        region_col = st.selectbox(
            "Region column (optional)",
            region_options,
            index=region_index if region_index < len(region_options) else 0,
        )
        if region_col == "(none)":
            region_col = None
    with col6:
        customer_options = ["(none)"] + cols
        customer_index = (customer_guess + 1) if customer_guess is not None else 0
        customer_col = st.selectbox(
            "Customer column (optional)",
            customer_options,
            index=customer_index if customer_index < len(customer_options) else 0,
        )
        if customer_col == "(none)":
            customer_col = None
    with col7:
        country_options = ["(none)"] + cols
        country_index = (country_guess + 1) if country_guess is not None else 0
        country_col = st.selectbox(
            "Country column (optional)",
            country_options,
            index=country_index if country_index < len(country_options) else 0,
        )
        if country_col == "(none)":
            country_col = None
    with col8:
        category_options = ["(none)"] + cols
        category_index = (category_guess + 1) if category_guess is not None else 0
        category_col = st.selectbox(
            "Category column (optional)",
            category_options,
            index=category_index if category_index < len(category_options) else 0,
        )
        if category_col == "(none)":
            category_col = None

    # -------------------------------------------------------------------------
    # PREPARE DATA
    # -------------------------------------------------------------------------
    df, product_summary_full, monthly_summary_full = prepare_data(
        df_raw,
        revenue_col=revenue_col,
        cost_col=cost_col,
        date_col=date_col,
        product_col=product_col,
        region_col=region_col,
        customer_col=customer_col,
        country_col=country_col,
        category_col=category_col,
    )

    if df.empty:
        st.warning("Unable to prepare data. Check your column mapping and try again.")
        return

    # -------------------------------------------------------------------------
    # FILTERS (Product / Region / Customer / Country / Category / Date)
    # -------------------------------------------------------------------------
    st.sidebar.header("Filters")

    mask = pd.Series(True, index=df.index)

    # Product filter
    if "__product__" in df.columns:
        product_options = sorted(df["__product__"].unique())
        selected_products = st.sidebar.multiselect(
            "Products",
            product_options,
            default=product_options,
        )
        if selected_products:
            mask &= df["__product__"].isin(selected_products)

    # Region filter
    if "__region__" in df.columns:
        region_opts = sorted(df["__region__"].unique())
        if not (len(region_opts) == 1 and region_opts[0] == "Unspecified Region"):
            selected_regions = st.sidebar.multiselect(
                "Regions",
                region_opts,
                default=region_opts,
            )
            if selected_regions:
                mask &= df["__region__"].isin(selected_regions)

    # Customer filter
    if "__customer__" in df.columns:
        customer_opts = sorted(df["__customer__"].unique())
        if not (len(customer_opts) == 1 and customer_opts[0] == "Unspecified Customer"):
            selected_customers = st.sidebar.multiselect(
                "Customers",
                customer_opts,
                default=customer_opts,
            )
            if selected_customers:
                mask &= df["__customer__"].isin(selected_customers)

    # Country filter
    if "__country__" in df.columns:
        country_opts = sorted(df["__country__"].unique())
        if not (len(country_opts) == 1 and country_opts[0] == "Unspecified Country"):
            selected_countries = st.sidebar.multiselect(
                "Countries",
                country_opts,
                default=country_opts,
            )
            if selected_countries:
                mask &= df["__country__"].isin(selected_countries)

    # Category filter
    if "__category__" in df.columns:
        category_opts = sorted(df["__category__"].unique())
        if not (len(category_opts) == 1 and category_opts[0] == "Unspecified Category"):
            selected_categories = st.sidebar.multiselect(
                "Categories",
                category_opts,
                default=category_opts,
            )
            if selected_categories:
                mask &= df["__category__"].isin(selected_categories)

    # Date range filter
    has_dates = df["__date__"].notna().any()
    if has_dates:
        min_date = df["__date__"].min().date()
        max_date = df["__date__"].max().date()
        date_range = st.sidebar.date_input(
            "Date range",
            value=(min_date, max_date),
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date

        mask &= df["__date__"].between(
            pd.to_datetime(start_date),
            pd.to_datetime(end_date),
        )
    else:
        start_date = end_date = None
        st.sidebar.info("No valid date column mapped; date filter disabled.")

    df_filtered = df[mask].copy()
    if df_filtered.empty:
        st.warning("No data after filters. Try relaxing some filters.")
        return

    product_summary, monthly_summary = summarize_prepared(df_filtered)

    total_revenue = float(df_filtered["__revenue__"].sum())
    total_cost = float(df_filtered["__cost__"].sum())
    total_profit = float(df_filtered["__profit__"].sum())
    overall_margin = (total_profit / total_revenue * 100.0) if total_revenue != 0 else 0.0

    # Precompute global things used in multiple pages
    text_insights = generate_text_insights(product_summary, monthly_summary)
    exec_summary = generate_exec_summary(product_summary, monthly_summary)

    # -------------------------------------------------------------------------
    # MULTI-PAGE NAV
    # -------------------------------------------------------------------------
    page = st.sidebar.radio(
        "📂 View",
        ["Overview", "Product Performance", "Customer & Region", "Forecasting", "AI & Reports"],
        index=0,
    )

    # -------------------------------------------------------------------------
    # PAGE: OVERVIEW
    # -------------------------------------------------------------------------
    if page == "Overview":
        st.markdown("---")
        st.subheader("📌 Key Metrics (filtered)")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Revenue", f"${total_revenue:,.0f}")
        kpi2.metric("Total Cost", f"${total_cost:,.0f}")
        kpi3.metric("Total Profit", f"${total_profit:,.0f}")
        kpi4.metric("Profit Margin", f"{overall_margin:.1f}%")

        st.markdown("---")
        st.subheader("📈 Revenue & Profit Over Time")

        if not monthly_summary.empty:
            chart_df = monthly_summary.set_index("Month")[["Revenue", "Profit"]]
            st.line_chart(chart_df, use_container_width=True)
        else:
            st.info("No valid date column selected or no data in chosen date range.")

        st.markdown("---")
        st.subheader("📋 Executive Summary")
        st.write(exec_summary)

        # NEW: Data exports from Overview (filtered)
        st.markdown("---")
        st.subheader("📥 Download Data (filtered view)")

        col_dl1, col_dl2, col_dl3 = st.columns(3)
        with col_dl1:
            if st.button("Prepare filtered data (Excel)"):
                st.session_state["filtered_data_ready"] = True
        with col_dl2:
            if st.button("Prepare product summary (Excel)"):
                st.session_state["product_summary_ready"] = True
        with col_dl3:
            if st.button("Prepare monthly summary (Excel)"):
                st.session_state["monthly_summary_ready"] = True

        # Show download buttons if prepared
        if st.session_state.get("filtered_data_ready", False):
            try:
                excel_bytes = df_to_excel_bytes(df_filtered)
                st.download_button(
                    label="⬇️ Download filtered data (Excel)",
                    data=excel_bytes,
                    file_name="filtered_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                st.error(f"Error exporting filtered data: {e}")

        if st.session_state.get("product_summary_ready", False):
            if product_summary.empty:
                st.info("Product summary is empty for the current filters.")
            else:
                try:
                    excel_bytes_ps = df_to_excel_bytes(product_summary)
                    st.download_button(
                        label="⬇️ Download product summary (Excel)",
                        data=excel_bytes_ps,
                        file_name="product_summary.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                except Exception as e:
                    st.error(f"Error exporting product summary: {e}")

        if st.session_state.get("monthly_summary_ready", False):
            if monthly_summary.empty:
                st.info("Monthly summary is empty for the current filters.")
            else:
                try:
                    excel_bytes_ms = df_to_excel_bytes(monthly_summary)
                    st.download_button(
                        label="⬇️ Download monthly summary (Excel)",
                        data=excel_bytes_ms,
                        file_name="monthly_summary.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                except Exception as e:
                    st.error(f"Error exporting monthly summary: {e}")

    # -------------------------------------------------------------------------
    # PAGE: PRODUCT PERFORMANCE
    # -------------------------------------------------------------------------
    elif page == "Product Performance":
        st.markdown("---")
        st.subheader("🏷️ Top Products by Profit")

        if not product_summary.empty:
            top_n = (
                product_summary.sort_values("Profit", ascending=False)
                .head(10)
                .set_index("Product")
            )
            st.bar_chart(top_n["Profit"], use_container_width=True)
        else:
            st.info("No product data available for the current filters.")

        st.markdown("### 📋 Product Performance Table")
        if not product_summary.empty:
            st.dataframe(
                product_summary.sort_values("Profit", ascending=False),
                use_container_width=True,
            )

    # -------------------------------------------------------------------------
    # PAGE: CUSTOMER & REGION
    # -------------------------------------------------------------------------
    elif page == "Customer & Region":
        st.markdown("---")
        st.subheader("🌍 Region & Country Breakdown")

        # Region chart
        if "__region__" in df_filtered.columns and df_filtered["__region__"].nunique() > 1:
            region_perf = (
                df_filtered.groupby("__region__")
                .agg(Revenue=("__revenue__", "sum"), Profit=("__profit__", "sum"))
                .sort_values("Profit", ascending=False)
            )
            st.markdown("#### Profit by Region")
            st.bar_chart(region_perf["Profit"], use_container_width=True)
        else:
            st.info("No detailed region data available after filters.")

        # Country chart
        if "__country__" in df_filtered.columns and df_filtered["__country__"].nunique() > 1:
            country_perf = (
                df_filtered.groupby("__country__")
                .agg(Revenue=("__revenue__", "sum"), Profit=("__profit__", "sum"))
                .sort_values("Profit", ascending=False)
            )
            st.markdown("#### Profit by Country")
            st.bar_chart(country_perf["Profit"], use_container_width=True)

        st.markdown("---")
        st.subheader("👤 Customer Performance")

        if "__customer__" in df_filtered.columns and df_filtered["__customer__"].nunique() > 1:
            customer_perf = (
                df_filtered.groupby("__customer__")
                .agg(Revenue=("__revenue__", "sum"), Profit=("__profit__", "sum"))
                .sort_values("Profit", ascending=False)
            )
            st.markdown("#### Top 15 Customers by Profit")
            st.dataframe(customer_perf.head(15), use_container_width=True)
        else:
            st.info("No detailed customer data available after filters.")

    # -------------------------------------------------------------------------
    # PAGE: FORECASTING
    # -------------------------------------------------------------------------
    elif page == "Forecasting":
        st.markdown("---")
        st.subheader("📈 Forecasting")

        if monthly_summary.empty or len(monthly_summary) < 3:
            st.info("Need at least 3 months of data (after filters) to generate a forecast.")
        else:
            model_choice = st.radio(
                "Forecast model",
                ["Linear trend", "Prophet (advanced)"],
                index=0,
                help="Prophet requires the `prophet` package in requirements.txt",
            )

            if model_choice == "Prophet (advanced)" and not PROPHET_AVAILABLE:
                st.warning(
                    "Prophet is not installed. Add `prophet` to your `requirements.txt` "
                    "on GitHub to enable advanced forecasting. Falling back to linear trend."
                )
                model_choice = "Linear trend"

            if model_choice == "Linear trend":
                forecast_df = build_linear_forecast(monthly_summary, periods=12)
                if forecast_df is None:
                    st.info("Could not build linear forecast (not enough data).")
                else:
                    future_part = forecast_df[forecast_df["Type"] == "Forecast"].copy()
                    if not future_part.empty:
                        rev_3 = future_part["Revenue"].head(3).sum()
                        rev_6 = future_part["Revenue"].head(6).sum()
                        rev_12 = future_part["Revenue"].head(12).sum()
                        st.write(
                            f"Projected revenue over the next **3 months**: `${rev_3:,.0f}`  |  "
                            f"**6 months**: `${rev_6:,.0f}`  |  **12 months**: `${rev_12:,.0f}`"
                        )

                    chart_data = forecast_df.set_index("Month")[["Revenue", "Type"]]
                    actual = chart_data[chart_data["Type"] == "Actual"][["Revenue"]].rename(
                        columns={"Revenue": "Actual Revenue"}
                    )
                    fc = chart_data[chart_data["Type"] == "Forecast"][["Revenue"]].rename(
                        columns={"Revenue": "Forecast Revenue"}
                    )
                    combined_chart = actual.join(fc, how="outer")
                    st.line_chart(combined_chart, use_container_width=True)

                    st.markdown(
                        "💬 _Forecast is based on a simple linear trend. Prophet is available as an upgrade when installed._"
                    )

            else:  # Prophet advanced
                prophet_df = build_prophet_forecast(monthly_summary, periods=12)
                if prophet_df is None:
                    st.info("Could not build Prophet forecast (check data or installation).")
                else:
                    # Split actual vs forecast
                    actual = prophet_df[prophet_df["Type"] == "Actual"].copy()
                    fc = prophet_df[prophet_df["Type"] == "Forecast"].copy()

                    # High-level numbers for forecast horizon
                    rev_3 = fc["yhat"].head(3).sum()
                    rev_6 = fc["yhat"].head(6).sum()
                    rev_12 = fc["yhat"].head(12).sum()
                    st.write(
                        f"Projected revenue over the next **3 months** (Prophet): `${rev_3:,.0f}`  |  "
                        f"**6 months**: `${rev_6:,.0f}`  |  **12 months**: `${rev_12:,.0f}`"
                    )

                    # Build chart
                    chart_df = prophet_df.set_index("Month")[["yhat", "yhat_lower", "yhat_upper", "Type"]]
                    actual_series = chart_df[chart_df["Type"] == "Actual"][["yhat"]].rename(
                        columns={"yhat": "Actual Revenue"}
                    )
                    fc_series = chart_df[chart_df["Type"] == "Forecast"][["yhat"]].rename(
                        columns={"yhat": "Forecast Revenue"}
                    )

                    combined = actual_series.join(fc_series, how="outer")
                    st.line_chart(combined, use_container_width=True)

                    st.caption(
                        "Lower/upper forecast interval is available in the data table below (Prophet output)."
                    )

                    with st.expander("📊 Prophet raw forecast data"):
                        st.dataframe(
                            prophet_df.tail(24)[["Month", "yhat", "yhat_lower", "yhat_upper", "Type"]],
                            use_container_width=True,
                        )

    # -------------------------------------------------------------------------
    # PAGE: AI & REPORTS
    # -------------------------------------------------------------------------
    elif page == "AI & Reports":
        st.markdown("---")
        st.subheader("📋 Executive Summary")
        st.write(exec_summary)

        st.markdown("---")
        st.subheader("💡 Insights")
        if not text_insights:
            st.write("Insights will appear here once enough data is available.")
        else:
            for bullet in text_insights:
                st.markdown(f"- {bullet}")

        st.markdown("---")
        st.subheader("🧾 Download Report")

        if not REPORTLAB_AVAILABLE:
            st.info(
                "To enable PDF export, add `reportlab` to your `requirements.txt`.\n\n"
                "Example:\n\n"
                "`reportlab`\n"
            )
        else:
            pdf_bytes = build_pdf_report(
                total_revenue=total_revenue,
                total_cost=total_cost,
                total_profit=total_profit,
                overall_margin=overall_margin,
                product_summary=product_summary,
                monthly_summary=monthly_summary,
                exec_summary=exec_summary,
                insights=text_insights,
            )
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name="business_performance_report.pdf",
                mime="application/pdf",
            )

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
