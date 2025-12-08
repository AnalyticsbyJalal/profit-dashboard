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
                f"Month-over-month, revenue is **{mom_rev:+.1f}%** ({direction} vs. the prior month)."
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
