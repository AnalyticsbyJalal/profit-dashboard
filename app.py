from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO, BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
# Load logo for page icon + header
logo = Image.open("logo.png")

# ----------------------------
# Layout & Theme
# ----------------------------
st.set_page_config(
    page_title="Business Profit Dashboard - AnalyticsByJalal",
    page_icon=logo,
    layout="wide"
)
with st.sidebar:
    st.image(logo, use_column_width=True)
    st.markdown("### AnalyticsByJalal")
    st.markdown("Smart Insights. Real Growth.")
    st.markdown("---")


st.markdown("""
<style>
.main {
    background-color: #f4f6fa;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}
[data-testid="stMetric"] {
    background-color: #ffffff;
    padding: 18px 16px;
    border-radius: 10px;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.08);
    border-left: 6px solid #2b6cb0;
}
h1, h2, h3, h4 {
    color: #234a7c !important;
}
hr {
    border: none;
    border-top: 1px solid #d0d7e2;
    margin: 18px 0;
}
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
}
.stDownloadButton button {
    background-color: #2b6cb0 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------
# Helper functions
# ----------------------------
def clean_number(series: pd.Series) -> pd.Series:
    def _clean(x):
        if pd.isna(x):
            return 0.0
        if isinstance(x, (int, float)):
            return float(x)
        x = str(x).replace("$", "").replace(",", "").strip()
        try:
            return float(x)
        except ValueError:
            return 0.0
    return series.apply(_clean)


def read_uploaded_files(files):
    """Read one or many CSV/XLSX files and concat into single DataFrame."""
    frames = []
    for f in files:
        name = f.name.lower()
        if name.endswith(".csv"):
            frames.append(pd.read_csv(f))
        elif name.endswith(".xlsx"):
            # use first sheet if multiple
            frames.append(pd.read_excel(f))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def build_monthly_summary(df: pd.DataFrame):
    if "__date__" not in df.columns or df["__date__"].notna().sum() == 0:
        return None
    temp = df.dropna(subset=["__date__"]).copy()
    temp["Month"] = temp["__date__"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        temp.groupby("Month")[["__revenue__", "__cost__", "__profit__"]]
        .sum()
        .reset_index()
    )
    monthly.rename(
        columns={
            "__revenue__": "Revenue",
            "__cost__": "Cost",
            "__profit__": "Profit",
        },
        inplace=True,
    )
    monthly["Month_str"] = monthly["Month"].dt.strftime("%Y-%m")
    monthly["Margin %"] = (
        (monthly["Profit"] / monthly["Revenue"] * 100)
        .where(monthly["Revenue"] != 0, 0.0)
        .round(1)
    )
    monthly["Year"] = monthly["Month"].dt.year
    monthly["MonthNum"] = monthly["Month"].dt.month
    return monthly


def build_product_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("__product__")[["__revenue__", "__cost__", "__profit__"]]
        .sum()
        .sort_values("__profit__", ascending=False)
        .reset_index()
    )
    summary.rename(
        columns={
            "__product__": "Product",
            "__revenue__": "Revenue",
            "__cost__": "Cost",
            "__profit__": "Profit",
        },
        inplace=True,
    )
    summary["Margin %"] = (
        (summary["Profit"] / summary["Revenue"] * 100)
        .where(summary["Revenue"] != 0, 0.0)
        .round(1)
    )
    return summary


def build_category_summary(df: pd.DataFrame):
    if "__category__" not in df.columns:
        return None
    summary = (
        df.groupby("__category__")[["__revenue__", "__cost__", "__profit__"]]
        .sum()
        .sort_values("__profit__", ascending=False)
        .reset_index()
    )
    summary.rename(
        columns={
            "__category__": "Category",
            "__revenue__": "Revenue",
            "__cost__": "Cost",
            "__profit__": "Profit",
        },
        inplace=True,
    )
    summary["Margin %"] = (
        (summary["Profit"] / summary["Revenue"] * 100)
        .where(summary["Revenue"] != 0, 0.0)
        .round(1)
    )
    return summary


def compute_mom_text(monthly: pd.DataFrame):
    if monthly is None or len(monthly) < 2:
        return "N/A"
    ms = monthly.sort_values("Month")
    last, prev = ms.iloc[-1], ms.iloc[-2]
    if prev["Revenue"] == 0:
        return "N/A"
    growth = (last["Revenue"] - prev["Revenue"]) / prev["Revenue"] * 100
    return f"{growth:+.1f}% vs {prev['Month_str']}"


def compute_yoy_chart(monthly: pd.DataFrame):
    if monthly is None or monthly["Year"].nunique() < 2:
        return None
    # revenue by MonthNum & Year
    data = monthly.copy()
    data["MonthLabel"] = data["MonthNum"].apply(lambda m: f"{m:02d}")
    fig = px.line(
        data,
        x="MonthLabel",
        y="Revenue",
        color="Year",
        markers=True,
        labels={"MonthLabel": "Month", "Revenue": "Revenue", "Year": "Year"},
        title="Year-over-Year Revenue by Month",
    )
    return style_fig(fig)


def forecast_revenue(monthly: pd.DataFrame, periods: int = 3):
    """Simple linear regression forecast on monthly revenue."""
    if monthly is None or len(monthly) < 3:
        return None
    ms = monthly.sort_values("Month").reset_index(drop=True)
    x = np.arange(len(ms))
    y = ms["Revenue"].values
    slope, intercept = np.polyfit(x, y, 1)
    # existing fitted values
    ms["Forecast"] = intercept + slope * x
    # future periods
    last_month = ms["Month"].iloc[-1]
    future_months = [last_month + pd.offsets.MonthBegin(i + 1) for i in range(periods)]
    x_future = np.arange(len(ms), len(ms) + periods)
    future_values = intercept + slope * x_future
    future_df = pd.DataFrame({
        "Month": future_months,
        "Revenue": np.nan,
        "Forecast": future_values,
    })
    result = pd.concat([ms[["Month", "Revenue", "Forecast"]], future_df], ignore_index=True)
    result["Month_str"] = result["Month"].dt.strftime("%Y-%m")
    return result


def build_waterfall_latest_month(monthly: pd.DataFrame):
    """Build a simple waterfall using graph_objects instead of plotly.express."""
    if monthly is None or monthly.empty:
        return None

    ms = monthly.sort_values("Month")
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
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        )
    )

    fig.update_layout(
        title=f"Revenue → Cost → Profit (Latest Month: {row['Month_str']})",
        showlegend=False,
    )

    return style_fig(fig)


def style_fig(fig):
    if fig is None:
        return None
    fig.update_layout(
        template="simple_white",
        title_font_color="#234a7c",
        font_color="#234a7c",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=20, t=50, b=40),
        colorway=["#2b6cb0", "#2f855a", "#718096", "#dd6b20"],
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
    return fig


def generate_excel_report(df_filtered, product_summary, category_summary, monthly_summary):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_filtered.to_excel(writer, sheet_name="Filtered Raw Data", index=False)
        product_summary.to_excel(writer, sheet_name="Product Summary", index=False)
        if category_summary is not None:
            category_summary.to_excel(writer, sheet_name="Category Summary", index=False)
        if monthly_summary is not None:
            monthly_summary.to_excel(writer, sheet_name="Monthly Summary", index=False)
    output.seek(0)
    return output.getvalue()


def generate_pdf_report(total_revenue, total_cost, total_profit, margin_pct,
                        main_fig, yoy_fig, product_summary, category_summary):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    # Title & KPIs
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, height - 50, "Business Profit Report")

    c.setFont("Helvetica", 11)
    y = height - 80
    c.drawString(40, y, f"Total Revenue: ${total_revenue:,.0f}")
    y -= 15
    c.drawString(40, y, f"Total Cost:    ${total_cost:,.0f}")
    y -= 15
    c.drawString(40, y, f"Total Profit:  ${total_profit:,.0f}")
    y -= 15
    c.drawString(40, y, f"Profit Margin: {margin_pct:,.1f}%")

    # Try to draw charts, but don't crash if kaleido / to_image fails
    y -= 30
    if main_fig is not None:
        try:
            png = main_fig.to_image(format="png")
            img = ImageReader(BytesIO(png))
            c.drawImage(img, 40, y - 210, width=520, height=200, preserveAspectRatio=True)
            y -= 230
        except Exception:
            # Skip chart if rendering fails (e.g. on Streamlit Cloud)
            pass

    if yoy_fig is not None:
        try:
            png = yoy_fig.to_image(format="png")
            img = ImageReader(BytesIO(png))
            c.drawImage(img, 40, y - 210, width=520, height=200, preserveAspectRatio=True)
        except Exception:
            pass

    # Product table
    c.showPage()
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
        c.drawString(col_x[0], y, str(row["Product"])[:18])
        c.drawRightString(col_x[1] + 60, y, f"${row['Revenue']:,.0f}")
        c.drawRightString(col_x[2] + 60, y, f"${row['Cost']:,.0f}")
        c.drawRightString(col_x[3] + 60, y, f"${row['Profit']:,.0f}")
        c.drawRightString(col_x[4] + 30, y, f"{row['Margin %']:,.1f}%")
        y -= 14

    # Category table (optional)
    if category_summary is not None and not category_summary.empty:
        c.showPage()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, height - 50, "Category Profit Summary")
        c.setFont("Helvetica", 9)

        y = height - 80
        headers = ["Category", "Revenue", "Cost", "Profit", "Margin %"]
        col_x = [40, 200, 320, 430, 510]
        for x, h in zip(col_x, headers):
            c.drawString(x, y, h)
        y -= 15
        c.line(40, y, width - 40, y)
        y -= 10

        for _, row in category_summary.iterrows():
            if y < 60:
                c.showPage()
                c.setFont("Helvetica-Bold", 14)
                c.drawString(40, height - 50, "Category Profit Summary (cont.)")
                c.setFont("Helvetica", 9)
                y = height - 80
            c.drawString(col_x[0], y, str(row["Category"])[:18])
            c.drawRightString(col_x[1] + 60, y, f"${row['Revenue']:,.0f}")
            c.drawRightString(col_x[2] + 60, y, f"${row['Cost']:,.0f}")
            c.drawRightString(col_x[3] + 60, y, f"${row['Profit']:,.0f}")
            c.drawRightString(col_x[4] + 30, y, f"{row['Margin %']:,.1f}%")
            y -= 14

    c.save()
    buf.seek(0)
    return buf.getvalue()



def generate_insights(monthly, product_summary, category_summary):
    insights = []

    if monthly is not None and not monthly.empty:
        ms = monthly.sort_values("Month")
        last = ms.iloc[-1]
        insights.append(
            f"Latest month ({last['Month_str']}) revenue was ${last['Revenue']:,.0f} "
            f"with profit of ${last['Profit']:,.0f} and margin {last['Margin %']:.1f}%."
        )
        if len(ms) >= 2:
            prev = ms.iloc[-2]
            if prev["Revenue"] != 0:
                growth = (last["Revenue"] - prev["Revenue"]) / prev["Revenue"] * 100
                insights.append(
                    f"Month-over-month revenue changed {growth:+.1f}% vs {prev['Month_str']}."
                )

        max_rev = ms.loc[ms["Revenue"].idxmax()]
        insights.append(
            f"Highest revenue month was {max_rev['Month_str']} at ${max_rev['Revenue']:,.0f}."
        )
        max_prof = ms.loc[ms["Profit"].idxmax()]
        insights.append(
            f"Highest profit month was {max_prof['Month_str']} at ${max_prof['Profit']:,.0f}."
        )

    if not product_summary.empty:
        top_prod = product_summary.iloc[0]
        insights.append(
            f"Top product by profit is {top_prod['Product']} "
            f"(${top_prod['Profit']:,.0f}, margin {top_prod['Margin %']:.1f}%)."
        )
        high_margin = product_summary[product_summary["Margin %"] >= 40]
        if not high_margin.empty:
            names = ", ".join(high_margin["Product"].head(5))
            insights.append(f"High-margin products (≥40%) include: {names}.")

    if category_summary is not None and not category_summary.empty:
        top_cat = category_summary.iloc[0]
        insights.append(
            f"Best category by profit is {top_cat['Category']} "
            f"with ${top_cat['Profit']:,.0f} and margin {top_cat['Margin %']:.1f}%."
        )

    if not insights:
        insights.append("Add date, revenue, cost and product columns for richer insights.")
    return insights


# ----------------------------
# Main UI
# ----------------------------
# --- Branded Header ---
header_col1, header_col2 = st.columns([1, 5])

with header_col1:
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
st.caption("Corporate-style analytics for revenue, cost, profit, and margin performance.")
st.markdown("---")


files = st.file_uploader(
    "Upload one or more CSV/XLSX files",
    type=["csv", "xlsx"],
    accept_multiple_files=True,
)

if files:
    df = read_uploaded_files(files)
    if df is None or df.empty:
        st.error("Could not read any data from the uploaded files.")
    else:
        st.markdown("### 🔍 Data Preview")
        st.dataframe(df.head(100), use_container_width=True, height=350)

        cols = df.columns.tolist()

        st.markdown("---")
        st.markdown("### ⚙️ Column Mapping")

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            revenue_col = st.selectbox("Revenue", cols)
        with c2:
            cost_col = st.selectbox("Cost", ["None"] + cols)
        with c3:
            date_col = st.selectbox("Date", ["None"] + cols)
        with c4:
            product_col = st.selectbox("Product", ["None"] + cols)
        with c5:
            category_col = st.selectbox("Category (optional)", ["None"] + cols)

        # standardized columns
        df["__revenue__"] = clean_number(df[revenue_col])
        df["__cost__"] = clean_number(df[cost_col]) if cost_col != "None" else 0.0
        df["__profit__"] = df["__revenue__"] - df["__cost__"]
        df["__product__"] = df[product_col].astype(str) if product_col != "None" else "Unknown"
        if date_col != "None":
            df["__date__"] = pd.to_datetime(df[date_col], errors="coerce")
        else:
            df["__date__"] = pd.NaT
        if category_col != "None":
            df["__category__"] = df[category_col].astype(str)

        # Date filter
        if df["__date__"].notna().any():
            st.markdown("### 📅 Date Range Filter")
            min_date = df["__date__"].min()
            max_date = df["__date__"].max()
            start_date, end_date = st.date_input(
                "Select date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
            if hasattr(start_date, "to_pydatetime"):
                start_date = start_date.to_pydatetime().date()
            if hasattr(end_date, "to_pydatetime"):
                end_date = end_date.to_pydatetime().date()
            mask = (df["__date__"].dt.date >= start_date) & (df["__date__"].dt.date <= end_date)
            df = df.loc[mask].copy()

        # Product / Category filters
        st.markdown("### 🔎 Filters")
        f1, f2 = st.columns(2)
        with f1:
            prod_options = sorted(df["__product__"].dropna().unique().tolist())
            selected_products = st.multiselect(
                "Products", prod_options, default=prod_options
            )
        if "__category__" in df.columns:
            with f2:
                cat_options = sorted(df["__category__"].dropna().unique().tolist())
                selected_categories = st.multiselect(
                    "Categories", cat_options, default=cat_options
                )
        else:
            selected_categories = None

        mask_prod = df["__product__"].isin(selected_products) if selected_products else True
        if selected_categories is not None:
            mask_cat = df["__category__"].isin(selected_categories) if selected_categories else True
            df_filtered = df.loc[mask_prod & mask_cat].copy()
        else:
            df_filtered = df.loc[mask_prod].copy()

        st.markdown("---")

        # Summaries
        monthly_summary = build_monthly_summary(df_filtered)
        product_summary = build_product_summary(df_filtered)
        category_summary = build_category_summary(df_filtered)

        # KPIs
        total_revenue = df_filtered["__revenue__"].sum()
        total_cost = df_filtered["__cost__"].sum()
        total_profit = df_filtered["__profit__"].sum()
        margin_pct = (total_profit / total_revenue * 100) if total_revenue else 0.0

        st.markdown("### 🧮 Key Metrics")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Revenue", f"${total_revenue:,.0f}")
        k2.metric("Total Cost", f"${total_cost:,.0f}")
        k3.metric("Total Profit", f"${total_profit:,.0f}")
        k4.metric("Profit Margin", f"{margin_pct:.1f}%")
        k5.metric("MoM Revenue Change", compute_mom_text(monthly_summary))

        # Charts
        st.markdown("---")
        st.markdown("### 📈 Performance & Growth")

        c_left, c_right = st.columns(2)

        main_fig = None
        if monthly_summary is not None and not monthly_summary.empty:
            with c_left:
                fig = px.line(
                    monthly_summary,
                    x="Month_str",
                    y=["Revenue", "Profit"],
                    labels={"value": "Amount", "variable": "Metric", "Month_str": "Month"},
                    title="Revenue & Profit Over Time",
                )
                main_fig = style_fig(fig)
                st.plotly_chart(main_fig, use_container_width=True)

        with c_right:
            fig2 = px.bar(
                product_summary.head(10),
                x="Product",
                y="Profit",
                title="Top 10 Products by Profit",
            )
            fig2 = style_fig(fig2)
            st.plotly_chart(fig2, use_container_width=True)

        # YoY & Forecast & Waterfall
        st.markdown("### 📊 Advanced Analytics")

        a1, a2, a3 = st.columns(3)

        # YoY
        with a1:
            yoy_fig = compute_yoy_chart(monthly_summary)
            if yoy_fig is not None:
                st.plotly_chart(yoy_fig, use_container_width=True)
            else:
                st.info("Upload at least 2 years of data for YoY.")

        # Forecast
        with a2:
            forecast_df = forecast_revenue(monthly_summary)
            if forecast_df is not None:
                fig_f = px.line(
                    forecast_df,
                    x="Month_str",
                    y=["Revenue", "Forecast"],
                    title="3-Month Revenue Forecast",
                    labels={"value": "Amount", "Month_str": "Month"},
                )
                fig_f = style_fig(fig_f)
                st.plotly_chart(fig_f, use_container_width=True)
            else:
                st.info("Need at least 3 months of data for forecasting.")

        # Waterfall
        with a3:
            wf_fig = build_waterfall_latest_month(monthly_summary)
            if wf_fig is not None:
                st.plotly_chart(wf_fig, use_container_width=True)
            else:
                st.info("Need at least 1 month of summarized data for waterfall.")

        # Category visuals
        if category_summary is not None and not category_summary.empty:
            st.markdown("### 🧩 Category Performance")
            cc1, cc2 = st.columns(2)
            with cc1:
                fig_cat1 = px.bar(
                    category_summary,
                    x="Category",
                    y="Profit",
                    title="Profit by Category",
                )
                fig_cat1 = style_fig(fig_cat1)
                st.plotly_chart(fig_cat1, use_container_width=True)
            with cc2:
                fig_cat2 = px.bar(
                    category_summary,
                    x="Category",
                    y="Margin %",
                    title="Margin % by Category",
                )
                fig_cat2 = style_fig(fig_cat2)
                st.plotly_chart(fig_cat2, use_container_width=True)

        # Drill-down
        st.markdown("### 🔍 Drill-down by Product")
        drill_products = product_summary["Product"].tolist()
        if drill_products:
            selected_drill = st.selectbox("Select product", drill_products)
            ddf = df_filtered[df_filtered["__product__"] == selected_drill].copy()
            if ddf["__date__"].notna().any():
                ddf = ddf.dropna(subset=["__date__"])
                ddf["Date_str"] = ddf["__date__"].dt.strftime("%Y-%m-%d")
                dfig = px.bar(
                    ddf,
                    x="Date_str",
                    y="__profit__",
                    title=f"Daily Profit for {selected_drill}",
                    labels={"Date_str": "Date", "__profit__": "Profit"},
                )
                dfig = style_fig(dfig)
                st.plotly_chart(dfig, use_container_width=True)
            st.dataframe(
                ddf[["__date__", "__revenue__", "__cost__", "__profit__"]]
                .rename(columns={
                    "__date__": "Date",
                    "__revenue__": "Revenue",
                    "__cost__": "Cost",
                    "__profit__": "Profit",
                }),
                use_container_width=True,
                height=260,
            )

        # Product table
        st.markdown("### 📦 Product Profit Detail")
        st.dataframe(product_summary, use_container_width=True, height=340)

        # Exports
        st.markdown("---")
        st.markdown("### ⬇️ Export Reports")

        csv_buffer = StringIO()
        product_summary.to_csv(csv_buffer, index=False)
        st.download_button(
            "Download product summary (CSV)",
            data=csv_buffer.getvalue(),
            file_name="product_profit_summary.csv",
            mime="text/csv",
        )

        excel_bytes = generate_excel_report(
            df_filtered, product_summary, category_summary, monthly_summary
        )
        st.download_button(
            "Download full Excel report",
            data=excel_bytes,
            file_name="business_profit_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        pdf_bytes = generate_pdf_report(
            total_revenue,
            total_cost,
            total_profit,
            margin_pct,
            main_fig,
            compute_yoy_chart(monthly_summary),
            product_summary,
            category_summary,
        )
        st.download_button(
            "Download full PDF report",
            data=pdf_bytes,
            file_name="business_profit_report.pdf",
            mime="application/pdf",
        )

        # Insights
        st.markdown("---")
        st.markdown("### 💡 Insights")
        for txt in generate_insights(monthly_summary, product_summary, category_summary):
            st.markdown(f"- {txt}")

        st.success("Analysis complete. Adjust filters or upload new files to refresh.")
else:
    st.info("📁 Upload one or more CSV/XLSX files to get started.")


