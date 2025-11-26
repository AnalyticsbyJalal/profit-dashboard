import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from io import BytesIO
from datetime import datetime
from PIL import Image, UnidentifiedImageError

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader


# -------------------------------------------------
# Page config + logo loading
# -------------------------------------------------
try:
    logo = Image.open("logo.png")
except (FileNotFoundError, UnidentifiedImageError):
    logo = None

st.set_page_config(
    page_title="Business Profit Dashboard - AnalyticsByJalal",
    page_icon=logo if logo is not None else "📊",
    layout="wide",
)


# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def clean_number(series: pd.Series) -> pd.Series:
    """Convert strings like '1,234.56' or '$1,234' to float, keep NaN on failure."""
    def _clean(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        x = str(x).replace(",", "").replace("$", "").strip()
        try:
            return float(x)
        except ValueError:
            return np.nan

    return series.map(_clean)


def build_monthly_summary(df, date_col, rev_col, cost_col):
    """Aggregate by month for revenue / cost / profit."""
    if date_col is None:
        return None

    temp = df.copy()
    temp["__date__"] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=["__date__"])

    temp["Month"] = temp["__date__"].dt.to_period("M").dt.to_timestamp()

    agg = {"__revenue__": "sum"}
    if cost_col is not None:
        agg["__cost__"] = "sum"

    g = temp.groupby("Month", dropna=False).agg(agg).reset_index()
    g["Revenue"] = g["__revenue__"]
    if cost_col is not None:
        g["Cost"] = g["__cost__"]
        g["Profit"] = g["Revenue"] - g["Cost"]
    else:
        g["Cost"] = 0.0
        g["Profit"] = g["Revenue"]

    return g[["Month", "Revenue", "Cost", "Profit"]]


def build_product_summary(df, product_col, rev_col, cost_col):
    if product_col is None:
        return None
    temp = df.copy()

    agg = {"__revenue__": "sum"}
    if cost_col is not None:
        agg["__cost__"] = "sum"

    g = temp.groupby(product_col, dropna=False).agg(agg).reset_index()
    g.rename(columns={product_col: "Product"}, inplace=True)

    g["Revenue"] = g["__revenue__"]
    if cost_col is not None:
        g["Cost"] = g["__cost__"]
        g["Profit"] = g["Revenue"] - g["Cost"]
    else:
        g["Cost"] = 0.0
        g["Profit"] = g["Revenue"]

    g["Margin %"] = np.where(
        g["Revenue"].abs() > 1e-9, g["Profit"] / g["Revenue"] * 100, np.nan
    )

    g = g.sort_values("Profit", ascending=False)
    return g[["Product", "Revenue", "Cost", "Profit", "Margin %"]]


def build_waterfall_latest_month(monthly_df: pd.DataFrame):
    """Waterfall chart using graph_objects (works without px.waterfall)."""
    if monthly_df is None or monthly_df.empty:
        return None

    ms = monthly_df.sort_values("Month")
    row = ms.iloc[-1]
    rev, cost, prof = row["Revenue"], row["Cost"], row["Profit"]

    labels = ["Revenue", "Cost", "Profit"]
    measures = ["relative", "relative", "total"]
    values = [rev, -cost, prof]

    fig = go.Figure(
        go.Waterfall(
            x=labels,
            measure=measures,
            y=values,
            connector={"line": {"color": "rgb(120,120,120)"}},
        )
    )

    fig.update_layout(
        title=f"Revenue → Cost → Profit (Latest Month: {row['Month'].strftime('%b %Y')})",
        showlegend=False,
        yaxis_title="Amount",
        margin=dict(t=40, l=40, r=20, b=40),
    )
    return fig


def generate_pdf_report(
    total_revenue,
    total_cost,
    total_profit,
    margin_pct,
    main_fig,
    monthly_summary: pd.DataFrame,
    product_summary: pd.DataFrame,
    executive_summary: str,
):
    """Build a PDF report with KPIs, chart (if possible), summary and tables."""
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, height - 50, "Business Profit Report")

    # Executive summary
    y = height - 80
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Executive Summary")
    y -= 15
    c.setFont("Helvetica", 10)

    if executive_summary:
        # simple wrap by splitting into lines
        words = executive_summary.split()
        line = ""
        for w in words:
            test = (line + " " + w).strip()
            if c.stringWidth(test, "Helvetica", 10) > width - 80:
                c.drawString(40, y, line)
                y -= 12
                line = w
            else:
                line = test
        if line:
            c.drawString(40, y, line)
            y -= 20
    else:
        c.drawString(40, y, "No summary available.")
        y -= 20

    # KPIs
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Key Metrics")
    y -= 15
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Total Revenue: ${total_revenue:,.0f}")
    y -= 15
    c.drawString(40, y, f"Total Cost:    ${total_cost:,.0f}")
    y -= 15
    c.drawString(40, y, f"Total Profit:  ${total_profit:,.0f}")
    y -= 15
    c.drawString(40, y, f"Profit Margin: {margin_pct:,.1f}%")

    # Try to draw the main chart; skip if kaleido/to_image fails
    y -= 30
    if main_fig is not None:
        try:
            png = main_fig.to_image(format="png")
            img = ImageReader(BytesIO(png))
            c.drawImage(
                img,
                40,
                y - 210,
                width=520,
                height=200,
                preserveAspectRatio=True,
            )
            y -= 230
        except Exception:
            # Skip chart if rendering fails on Streamlit Cloud
            pass

    # New page for tables
    c.showPage()

    # Monthly summary table (if available)
    if monthly_summary is not None and not monthly_summary.empty:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, height - 50, "Monthly Revenue & Profit")
        c.setFont("Helvetica", 9)

        y = height - 80
        headers = ["Month", "Revenue", "Cost", "Profit"]
        col_x = [40, 200, 320, 440]
        for x, h in zip(col_x, headers):
            c.drawString(x, y, h)
        y -= 15
        c.line(40, y, width - 40, y)
        y -= 10

        for _, row in monthly_summary.iterrows():
            if y < 60:
                c.showPage()
                c.setFont("Helvetica-Bold", 14)
                c.drawString(40, height - 50, "Monthly Revenue & Profit (cont.)")
                c.setFont("Helvetica", 9)
                y = height - 80
                for x, h in zip(col_x, headers):
                    c.drawString(x, y, h)
                y -= 15
                c.line(40, y, width - 40, y)
                y -= 10

            c.drawString(
                col_x[0],
                y,
                pd.to_datetime(row["Month"]).strftime("%Y-%m"),
            )
            c.drawRightString(col_x[1] + 60, y, f"${row['Revenue']:,.0f}")
            c.drawRightString(col_x[2] + 60, y, f"${row['Cost']:,.0f}")
            c.drawRightString(col_x[3] + 60, y, f"${row['Profit']:,.0f}")
            y -= 14

        c.showPage()

    # Product summary table
    if product_summary is not None and not product_summary.empty:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, height - 50, "Product Profit Summary")
        c.setFont("Helvetica", 9)

        y = height - 80
        headers = ["Product", "Revenue", "Cost", "Profit", "Margin %"]
        col_x = [40, 200, 320, 430, 510]
        for x, h in zip(col_x, headers):
            c.drawString(x, y, h)
        y -= 15
        c.line(40, y, width - 40, y)
        y -= 10

        for _, row in product_summary.iterrows():
            if y < 60:
                c.showPage()
                c.setFont("Helvetica-Bold", 14)
                c.drawString(40, height - 50, "Product Profit Summary (cont.)")
                c.setFont("Helvetica", 9)
                y = height - 80
                for x, h in zip(col_x, headers):
                    c.drawString(x, y, h)
                y -= 15
                c.line(40, y, width - 40, y)
                y -= 10

            c.drawString(col_x[0], y, str(row["Product"])[:22])
            c.drawRightString(col_x[1] + 60, y, f"${row['Revenue']:,.0f}")
            c.drawRightString(col_x[2] + 60, y, f"${row['Cost']:,.0f}")
            c.drawRightString(col_x[3] + 60, y, f"${row['Profit']:,.0f}")
            c.drawRightString(col_x[4] + 30, y, f"{row['Margin %']:,.1f}%")
            y -= 14

    c.save()
    buf.seek(0)
    return buf.getvalue()


# -------------------------------------------------
# Layout: Sidebar + Header
# -------------------------------------------------
with st.sidebar:
    if logo is not None:
        st.image(logo, use_column_width=True)
    st.markdown("### AnalyticsByJalal")
    st.markdown("Smart Insights. Real Growth.")
    st.markdown("---")
    st.markdown("**Upload your data on the main panel to begin.**")

header_col1, header_col2 = st.columns([1, 5])
with header_col1:
    if logo is not None:
        st.image(logo, width=80)
with header_col2:
    st.markdown(
        "<h1 style='margin-bottom:0px; color:#0A2540;'>AnalyticsByJalal</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='margin-top:4px; font-size:16px; color:#4A5568;'>Smart Insights. Real Growth.</p>",
        unsafe_allow_html=True,
    )

st.markdown("### Business Profit Dashboard")
st.caption(
    "Corporate-style analytics for revenue, cost, profit, and margin performance."
)
st.markdown("---")


# -------------------------------------------------
# File upload section
# -------------------------------------------------
st.subheader("📂 Upload your data")

uploaded_files = st.file_uploader(
    "Upload one or more CSV/XLSX files",
    type=["csv", "xlsx"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("☝️ Upload at least one CSV or Excel file to get started.")
    st.stop()


# -------------------------------------------------
# Read and combine data
# -------------------------------------------------
all_frames = []
for upl in uploaded_files:
    name = upl.name.lower()
    if name.endswith(".csv"):
        df_part = pd.read_csv(upl)
    else:
        # Excel: support multi-sheet workbooks
        xls = pd.read_excel(upl, sheet_name=None)
        df_part = pd.concat(xls.values(), ignore_index=True)
    all_frames.append(df_part)

df_raw = pd.concat(all_frames, ignore_index=True)

st.subheader("🔍 Data preview")
st.dataframe(df_raw.head(200), use_container_width=True, height=300)


# -------------------------------------------------
# Column mapping
# -------------------------------------------------
st.markdown("---")
st.subheader("⚙️ Column Mapping")

all_columns = list(df_raw.columns)


def guess(col_candidates):
    for c in all_columns:
        lc = str(c).lower()
        if any(k in lc for k in col_candidates):
            return c
    return None


rev_guess = guess(["revenue", "sales", "amount", "total"])
cost_guess = guess(["cost", "cogs", "expense"])
date_guess = guess(["date", "day", "month"])
product_guess = guess(["product", "item", "sku", "category"])

revenue_col = st.selectbox(
    "Revenue column",
    all_columns,
    index=all_columns.index(rev_guess) if rev_guess in all_columns else 0,
)
cost_col_opt = st.selectbox(
    "Cost column (optional)",
    ["None"] + all_columns,
    index=(["None"] + all_columns).index(cost_guess) if cost_guess in all_columns else 0,
)
date_col_opt = st.selectbox(
    "Date column (optional)",
    ["None"] + all_columns,
    index=(["None"] + all_columns).index(date_guess) if date_guess in all_columns else 0,
)
product_col_opt = st.selectbox(
    "Product column (optional)",
    ["None"] + all_columns,
    index=(["None"] + all_columns).index(product_guess)
    if product_guess in all_columns
    else 0,
)

cost_col = None if cost_col_opt == "None" else cost_col_opt
date_col = None if date_col_opt == "None" else date_col_opt
product_col = None if product_col_opt == "None" else product_col_opt


# Prepare numeric columns
df = df_raw.copy()
df["__revenue__"] = clean_number(df[revenue_col])
if cost_col is not None:
    df["__cost__"] = clean_number(df[cost_col])
else:
    df["__cost__"] = 0.0
df["__profit__"] = df["__revenue__"] - df["__cost__"]

# Keep mapped names for later insights
if date_col is not None:
    df["__date_mapped__"] = pd.to_datetime(df[date_col], errors="coerce")
if product_col is not None:
    df["__product_mapped__"] = df[product_col]


# -------------------------------------------------
# KPIs
# -------------------------------------------------
total_revenue = df["__revenue__"].sum()
total_cost = df["__cost__"].sum()
total_profit = df["__profit__"].sum()
margin_pct = (total_profit / total_revenue * 100) if abs(total_revenue) > 1e-9 else 0.0

st.markdown("---")
st.subheader("📈 Key Metrics")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Revenue", f"${total_revenue:,.0f}")
kpi2.metric("Total Cost", f"${total_cost:,.0f}")
kpi3.metric("Total Profit", f"${total_profit:,.0f}")
kpi4.metric("Profit Margin", f"{margin_pct:,.1f}%")


# -------------------------------------------------
# Time series & charts
# -------------------------------------------------
monthly_summary = build_monthly_summary(df, date_col, revenue_col, cost_col)

st.markdown("---")
st.subheader("📊 Revenue & Profit Over Time")

main_fig = None
if monthly_summary is not None and not monthly_summary.empty:
    melted = monthly_summary.melt(
        id_vars="Month",
        value_vars=["Revenue", "Profit"],
        var_name="Metric",
        value_name="Amount",
    )
    main_fig = px.line(
        melted,
        x="Month",
        y="Amount",
        color="Metric",
        markers=True,
        title="Monthly Revenue & Profit",
    )
    main_fig.update_layout(margin=dict(t=40, l=40, r=20, b=40))
    st.plotly_chart(main_fig, use_container_width=True)
else:
    st.info(
        "No valid date column selected or all dates are invalid. Time series chart is skipped."
    )

# Waterfall chart (latest month)
st.subheader("📉 Latest Month Waterfall")

wf_fig = build_waterfall_latest_month(monthly_summary)
if wf_fig is not None:
    st.plotly_chart(wf_fig, use_container_width=True)
else:
    st.info("Waterfall chart will appear once a valid date, revenue, and cost are mapped.")


# -------------------------------------------------
# Product summary
# -------------------------------------------------
st.markdown("---")
st.subheader("🏆 Top Products by Profit")

product_summary = build_product_summary(df, product_col, revenue_col, cost_col)
if product_summary is not None and not product_summary.empty:
    st.dataframe(product_summary, use_container_width=True)
else:
    st.info("Select a product column to see per-product profitability.")


# -------------------------------------------------
# Insights & Executive Summary
# -------------------------------------------------
st.markdown("---")
st.subheader("💡 Insights")

# -------------------------------------------------
# Insights & Executive Summary
# -------------------------------------------------
st.markdown("---")
st.subheader("💡 Insights")

insights = []
exec_parts = []


# 1) Most profitable product
if product_summary is not None and not product_summary.empty:
    top_prod = product_summary.iloc[0]

    insights.append(
        f"{top_prod['Product']} is your most profitable product with "
        f"profit of ${top_prod['Profit']:,.0f} and a margin of {top_prod['Margin %']:,.1f}%."
    )

   exec_parts.append(
    f"{top_prod['Product']} is currently your top performer, delivering "
    f"${top_prod['Profit']:,.0f} in profit at a {top_prod['Margin %']:,.1f}% margin."
)



# 2) Monthly performance + MoM + YoY + Trend Analysis
mom_phrase = ""
yoy_phrase = ""
trend_phrase = ""

if monthly_summary is not None and not monthly_summary.empty:
    ms = monthly_summary.sort_values("Month").copy()
    ms_indexed = ms.set_index("Month")

    latest_row = ms.iloc[-1]
    latest_month = latest_row["Month"]
    latest_rev = latest_row["Revenue"]
    latest_prof = latest_row["Profit"]

    # Main statement
    insights.append(
        f"In the latest month ({latest_month.strftime('%B %Y')}), you generated "
        f"${latest_rev:,.0f} in revenue and ${latest_prof:,.0f} in profit."
    )

    # Month-over-month
    prev_month = latest_month - pd.DateOffset(months=1)
    if prev_month in ms_indexed.index:
        prev_rev = ms_indexed.loc[prev_month, "Revenue"]
        if prev_rev != 0:
            mom_rev = (latest_rev - prev_rev) / abs(prev_rev) * 100
            insights.append(
                f"Month-over-month, revenue changed by {mom_rev:+.1f}% compared to {prev_month.strftime('%b %Y')}."
            )
            mom_phrase = f"Revenue changed by {mom_rev:+.1f}% versus {prev_month.strftime('%b %Y')}."

    # Year-over-year
    prev_year = latest_month - pd.DateOffset(years=1)
    if prev_year in ms_indexed.index:
        prev_year_rev = ms_indexed.loc[prev_year, "Revenue"]
        if prev_year_rev != 0:
            yoy_rev = (latest_rev - prev_year_rev) / abs(prev_year_rev) * 100
            insights.append(
                f"Year-over-year, revenue for {latest_month.strftime('%B')} is {yoy_rev:+.1f}% versus {prev_year.strftime('%B %Y')}."
            )
            yoy_phrase = f"Compared with {prev_year.strftime('%B %Y')}, revenue is {yoy_rev:+.1f}%."

    # Trend - last 3 months
    if len(ms) >= 3:
        last3 = ms.tail(3)
        revs = last3["Revenue"].values
        if np.all(np.diff(revs) > 0):
            trend_phrase = "Revenue has been trending up over the last three months."
            insights.append("Revenue has been trending up over the last three months.")
        elif np.all(np.diff(revs) < 0):
            trend_phrase = "Revenue has been trending down over the last three months."
            insights.append("Revenue has been trending down over the last three months.")

    exec_parts.append(
        f"In {latest_month.strftime('%B %Y')}, the business generated "
        f"${latest_rev:,.0f} in revenue and ${latest_prof:,.0f} in profit. "
        f"{mom_phrase} {yoy_phrase} {trend_phrase}".strip()
    )


# 3) Loss warnings
loss_warnings = []

# Products with negative profit
if product_summary is not None and not product_summary.empty:
    loss_products = product_summary[product_summary["Profit"] < 0]
    if not loss_products.empty:
        names = ", ".join(loss_products["Product"].head(3).astype(str))
        loss_warnings.append(f"Loss-making products include: {names}.")

# Months with negative profit
if monthly_summary is not None and not monthly_summary.empty:
    loss_months = monthly_summary[monthly_summary["Profit"] < 0]
    if not loss_months.empty:
        months_list = ", ".join(loss_months["Month"].dt.strftime("%b %Y"))
        loss_warnings.append(f"Some months show negative profit, such as {months_list}.")

for w in loss_warnings:
    insights.append("⚠️ " + w)

if loss_warnings:
    exec_parts.append("Additionally, " + " ".join(w.rstrip('.') + "." for w in loss_warnings))


# Display insights bullets
if not insights:
    st.write("Insights will appear here once data is mapped.")
else:
    for bullet in insights:
        st.markdown(f"- {bullet}")


# Build Executive Summary
executive_summary = " ".join(exec_parts).strip()
if not executive_summary:
    executive_summary = "A summary could not be generated based on available data."

st.markdown("---")
st.subheader("📝 Executive Summary")
st.markdown(executive_summary)


# 3) Loss warnings
loss_warnings = []

# Products with negative profit
if product_summary is not None and not product_summary.empty:
    loss_products = product_summary[product_summary["Profit"] < 0].sort_values(
        "Profit"
    )
    if not loss_products.empty:
        names = ", ".join(loss_products["Product"].head(3).astype(str).tolist())
        loss_warnings.append(
            f"The following products are loss-making: **{names}** "
            f"(negative profit)."
        )

# Months with negative profit
if monthly_summary is not None and not monthly_summary.empty:
    loss_months = monthly_summary[monthly_summary["Profit"] < 0]
    if not loss_months.empty:
        months_str = ", ".join(
            loss_months["Month"].dt.strftime("%b %Y").head(3).tolist()
        )
        loss_warnings.append(
            f"Some months show negative profit, including **{months_str}**."
        )

for warning in loss_warnings:
    insights.append(f"⚠️ {warning}")

if loss_warnings:
    exec_parts.append(
        "However, there are areas of risk: "
        + " ".join(warning.rstrip(".") + "." for warning in loss_warnings)
    )

# Fallback if no insights
if not insights:
    st.write("Insights will appear here once you map your columns.")
else:
    for bullet in insights:
        st.markdown(f"- {bullet}")

# Build executive summary paragraph
executive_summary = " ".join(exec_parts).strip()
if not executive_summary:
    executive_summary = (
        "The uploaded data has been processed, but there was not enough "
        "information to generate a detailed narrative summary."
    )

st.markdown("---")
st.subheader("📝 Executive Summary")
st.markdown(executive_summary)


# -------------------------------------------------
# Export reports
# -------------------------------------------------
st.markdown("---")
st.subheader("⬇️ Export Reports")

# CSV: product summary
if product_summary is not None and not product_summary.empty:
    csv_bytes = product_summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download product summary (CSV)",
        data=csv_bytes,
        file_name="product_profit_summary.csv",
        mime="text/csv",
    )

# Excel report
excel_buf = BytesIO()
with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
    df_raw.to_excel(writer, sheet_name="Raw Data", index=False)
    df[["__revenue__", "__cost__", "__profit__"]].rename(
        columns={"__revenue__": "Revenue", "__cost__": "Cost", "__profit__": "Profit"}
    ).to_excel(writer, sheet_name="Cleaned (R,C,P)", index=False)
    if monthly_summary is not None:
        monthly_summary.to_excel(writer, sheet_name="Monthly Summary", index=False)
    if product_summary is not None:
        product_summary.to_excel(writer, sheet_name="Product Summary", index=False)

excel_buf.seek(0)
st.download_button(
    label="Download full Excel report",
    data=excel_buf,
    file_name="business_profit_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# PDF report
pdf_bytes = generate_pdf_report(
    total_revenue=total_revenue,
    total_cost=total_cost,
    total_profit=total_profit,
    margin_pct=margin_pct,
    main_fig=main_fig,
    monthly_summary=monthly_summary
    if monthly_summary is not None
    else pd.DataFrame(),
    product_summary=product_summary
    if product_summary is not None
    else pd.DataFrame(),
    executive_summary=executive_summary,
)

st.download_button(
    label="Download full PDF report",
    data=pdf_bytes,
    file_name="business_profit_report.pdf",
    mime="application/pdf",
)

st.success(
    "Analysis complete. Adjust mappings or upload new files to refresh the dashboard."
)


