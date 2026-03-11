from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_dashboard_figure(
    processed: pd.DataFrame,
    forecast: pd.DataFrame,
    datetime_col: str = "datetime",
    target_col: str = "global_active_power",
) -> go.Figure:
    """Create a portfolio-grade analytics dashboard.

    Sections
    --------
    1) Key metrics summary
    2) Historical consumption with anomalies
    3) Short-term forecast (next 24h)
    4) Consumption behavior analytics (hourly, weekday, distribution, heatmap)
    5) Automated analytical insights
    """

    def _safe_float(x: float) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    def _fmt(x: float, digits: int = 3) -> str:
        xf = _safe_float(x)
        if pd.isna(xf):
            return "n/a"
        return f"{xf:.{digits}f}"

    # --- Prepare data
    df = processed.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.dropna(subset=[datetime_col]).sort_values(datetime_col)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    if "hour" not in df.columns:
        df["hour"] = df[datetime_col].dt.hour
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df[datetime_col].dt.dayofweek

    last_ts = df[datetime_col].max()
    time_min = df[datetime_col].min()
    time_max = df[datetime_col].max()

    # Forecast df
    fc = forecast.copy()
    fc["ds"] = pd.to_datetime(fc["ds"], errors="coerce")
    fc = fc.dropna(subset=["ds"]).sort_values("ds")
    fc_future = fc[fc["ds"] > last_ts].copy()

    # Recent context window for the forecast view
    recent_start = last_ts - pd.Timedelta(days=14)
    df_recent = df[df[datetime_col] >= recent_start].copy()

    forecast_horizon_end = last_ts + pd.Timedelta(hours=24)

    # --- Key metrics
    n_total = int(len(df))
    avg_cons = _safe_float(df[target_col].mean())
    min_cons = _safe_float(df[target_col].min())
    max_cons = _safe_float(df[target_col].max())

    n_anoms = int((df.get("anomaly", pd.Series([1] * len(df))) == -1).sum()) if n_total else 0
    anom_pct = (n_anoms / n_total * 100.0) if n_total else 0.0

    # --- Behavior analytics
    hourly_avg = df.groupby("hour", as_index=True)[target_col].mean().reindex(range(24))
    dow_order = [0, 1, 2, 3, 4, 5, 6]
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_avg = df.groupby("day_of_week", as_index=True)[target_col].mean().reindex(dow_order)

    heat = (
        df.groupby(["day_of_week", "hour"], as_index=False)[target_col]
        .mean()
        .pivot(index="day_of_week", columns="hour", values=target_col)
        .reindex(index=dow_order, columns=range(24))
    )

    # --- Extra stats for readability
    p50 = _safe_float(df[target_col].quantile(0.50)) if n_total else float("nan")
    p95 = _safe_float(df[target_col].quantile(0.95)) if n_total else float("nan")
    p99 = _safe_float(df[target_col].quantile(0.99)) if n_total else float("nan")

    recent_mean = _safe_float(df_recent[target_col].mean()) if not df_recent.empty else float("nan")
    recent_min = _safe_float(df_recent[target_col].min()) if not df_recent.empty else float("nan")
    recent_max = _safe_float(df_recent[target_col].max()) if not df_recent.empty else float("nan")

    fc_mean = _safe_float(fc_future["yhat"].mean()) if not fc_future.empty else float("nan")
    fc_max = _safe_float(fc_future["yhat"].max()) if not fc_future.empty else float("nan")
    fc_band = (
        _safe_float((fc_future["yhat_upper"] - fc_future["yhat_lower"]).mean()) if not fc_future.empty else float("nan")
    )

    # Anomaly breakdown (if available)
    anom_hour = None
    anom_dow = None
    anom_volt_delta = None
    anom_int_delta = None
    if "anomaly" in df.columns and n_anoms:
        anom_df = df[df["anomaly"] == -1].copy()
        if not anom_df.empty:
            by_hour = anom_df.groupby("hour")[target_col].agg(["count", "mean"]).sort_values("count", ascending=False)
            by_dow = anom_df.groupby("day_of_week")[target_col].agg(["count", "mean"]).sort_values("count", ascending=False)
            anom_hour = (int(by_hour.index[0]), int(by_hour.iloc[0]["count"]), _safe_float(by_hour.iloc[0]["mean"]))
            anom_dow = (int(by_dow.index[0]), int(by_dow.iloc[0]["count"]), _safe_float(by_dow.iloc[0]["mean"]))

            # Compare anomalies vs normal for available sensors
            if "voltage" in df.columns:
                v_an = pd.to_numeric(anom_df["voltage"], errors="coerce").dropna()
                v_no = pd.to_numeric(df[df["anomaly"] != -1]["voltage"], errors="coerce").dropna()
                if not v_an.empty and not v_no.empty:
                    anom_volt_delta = _safe_float(v_an.mean() - v_no.mean())
            if "global_intensity" in df.columns:
                i_an = pd.to_numeric(anom_df["global_intensity"], errors="coerce").dropna()
                i_no = pd.to_numeric(df[df["anomaly"] != -1]["global_intensity"], errors="coerce").dropna()
                if not i_an.empty and not i_no.empty:
                    anom_int_delta = _safe_float(i_an.mean() - i_no.mean())

    # --- Automated insights
    peak_hour = int(hourly_avg.idxmax()) if hourly_avg.notna().any() else None
    peak_dow = int(dow_avg.idxmax()) if dow_avg.notna().any() else None
    avg_daily = _safe_float(df.set_index(datetime_col)[target_col].resample("D").mean().mean())

    insights_lines = [
        f"Most common peak hour (avg): {peak_hour:02d}:00" if peak_hour is not None else "Most common peak hour (avg): n/a",
        f"Peak usage day of week (avg): {dow_labels[peak_dow]}" if peak_dow is not None else "Peak usage day of week (avg): n/a",
        f"Average daily consumption (avg of daily means): {avg_daily:.3f}",
        f"Anomaly percentage: {anom_pct:.2f}% ({n_anoms} / {n_total})",
        (
            f"Anomalies most frequent at hour: {anom_hour[0]:02d}:00 (count={anom_hour[1]}, avg={_fmt(anom_hour[2])})"
            if anom_hour is not None
            else "Anomalies most frequent at hour: n/a"
        ),
        (
            f"Anomalies most frequent on: {dow_labels[anom_dow[0]]} (count={anom_dow[1]}, avg={_fmt(anom_dow[2])})"
            if anom_dow is not None
            else "Anomalies most frequent on: n/a"
        ),
        (f"Voltage shift during anomalies (mean anomaly − mean normal): {_fmt(anom_volt_delta, 2)}" if anom_volt_delta is not None else None),
        (f"Intensity shift during anomalies (mean anomaly − mean normal): {_fmt(anom_int_delta, 2)}" if anom_int_delta is not None else None),
        f"Data coverage: {time_min:%Y-%m-%d} → {time_max:%Y-%m-%d}",
    ]

    insights_lines = [line for line in insights_lines if line]

    # --- Layout
    subplot_title_texts = (
        "Key Metrics Summary",
        "Historical Consumption with Anomalies",
        "Short-term Forecast (Next 24 Hours)",
        "Avg Consumption by Hour of Day",
        "Avg Consumption by Day of Week",
        "Consumption Distribution",
        "Heatmap: Hour vs Day of Week",
        "Analytical Insights",
    )

    fig = make_subplots(
        rows=6,
        cols=2,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
        # Give tables more breathing room (top metrics + bottom insights)
        row_heights=[0.16, 0.26, 0.22, 0.15, 0.15, 0.14],
        specs=[
            [{"type": "table", "colspan": 2}, None],
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "table", "colspan": 2}, None],
        ],
        subplot_titles=subplot_title_texts,
    )

    # Section 1: Key metrics table
    metrics = [
        ("Total records", f"{n_total:,}"),
        ("Average consumption", f"{avg_cons:.3f}"),
        ("Minimum consumption", f"{min_cons:.3f}"),
        ("Maximum consumption", f"{max_cons:.3f}"),
        ("Anomalies detected", f"{n_anoms:,} ({anom_pct:.2f}%)"),
        ("Time range", f"{time_min:%Y-%m-%d} → {time_max:%Y-%m-%d}"),
    ]
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Metric", "Value"],
                align="left",
                fill_color="rgba(31, 119, 180, 0.12)",
                height=34,
                font=dict(size=12),
            ),
            cells=dict(
                values=[[m[0] for m in metrics], [m[1] for m in metrics]],
                align="left",
                height=32,
                font=dict(size=12),
            ),
            columnwidth=[0.35, 0.65],
        ),
        row=1,
        col=1,
    )

    # Section 2: Historical consumption + anomalies
    fig.add_trace(
        go.Scatter(
            x=df[datetime_col],
            y=df[target_col],
            mode="lines",
            name="Consumption",
            line=dict(width=2, color="#1F77B4"),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Consumption=%{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    if "anomaly" in df.columns:
        anoms = df[df["anomaly"] == -1]
        fig.add_trace(
            go.Scatter(
                x=anoms[datetime_col],
                y=anoms[target_col],
                mode="markers",
                name="Anomaly (-1)",
                marker=dict(size=10, color="#D62728", line=dict(width=1, color="#FFFFFF")),
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br><b>Anomaly</b><br>Consumption=%{y:.3f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # Section 3: Forecast
    # Shade the future region (forecast horizon) so it doesn't look like the historical panel.
    if not pd.isna(recent_min) and not pd.isna(recent_max):
        band_min = recent_min
        band_max = recent_max
        if not fc_future.empty:
            band_min = min(band_min, _safe_float(fc_future["yhat_lower"].min()))
            band_max = max(band_max, _safe_float(fc_future["yhat_upper"].max()))

        if not pd.isna(band_min) and not pd.isna(band_max):
            fig.add_trace(
                go.Scatter(
                    x=[last_ts, forecast_horizon_end, forecast_horizon_end, last_ts],
                    y=[band_min, band_min, band_max, band_max],
                    mode="lines",
                    line=dict(width=0),
                    fill="toself",
                    fillcolor="rgba(255, 127, 14, 0.06)",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=3,
                col=1,
            )

    if not df_recent.empty:
        fig.add_trace(
            go.Scatter(
                x=df_recent[datetime_col],
                y=df_recent[target_col],
                mode="lines",
                name="Historical (recent 14d)",
                line=dict(width=2, color="rgba(0, 0, 0, 0.35)"),
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Consumption=%{y:.3f}<extra></extra>",
            ),
            row=3,
            col=1,
        )

    if not fc_future.empty:
        fig.add_trace(
            go.Scatter(
                x=fc_future["ds"],
                y=fc_future["yhat_upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=fc_future["ds"],
                y=fc_future["yhat_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(255, 127, 14, 0.18)",
                name="Forecast interval",
                hoverinfo="skip",
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=fc_future["ds"],
                y=fc_future["yhat"],
                mode="lines+markers",
                name="Forecast",
                line=dict(width=3, color="#FF7F0E"),
                marker=dict(size=4, color="#FF7F0E", opacity=0.8),
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Forecast=%{y:.3f}<extra></extra>",
            ),
            row=3,
            col=1,
        )

        # Explicit split marker between history and forecast (avoid add_vline due to table subplots)
        panel_min = min(recent_min, _safe_float(fc_future["yhat_lower"].min()))
        panel_max = max(recent_max, _safe_float(fc_future["yhat_upper"].max()))
        if not pd.isna(panel_min) and not pd.isna(panel_max):
            fig.add_trace(
                go.Scatter(
                    x=[last_ts, last_ts],
                    y=[panel_min, panel_max],
                    mode="lines",
                    line=dict(width=1, dash="dot", color="rgba(0,0,0,0.5)"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=3,
                col=1,
            )

            fig.add_annotation(
                row=3,
                col=1,
                x=last_ts,
                y=panel_max,
                xref="x",
                yref="y",
                text="Forecast starts",
                showarrow=False,
                yshift=12,
                font=dict(size=10, color="rgba(0,0,0,0.65)"),
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="rgba(0,0,0,0.12)",
                borderwidth=1,
            )

    # Section 4: Behavior analytics
    fig.add_trace(
        go.Bar(
            x=list(range(24)),
            y=hourly_avg.values,
            name="Hourly avg",
            marker_color="rgba(44, 160, 44, 0.85)",
            hovertemplate="Hour=%{x}:00<br>Avg=%{y:.3f}<extra></extra>",
        ),
        row=4,
        col=1,
    )
    fig.update_xaxes(title_text="Hour of day", row=4, col=1)

    fig.add_trace(
        go.Bar(
            x=dow_labels,
            y=dow_avg.values,
            name="Weekday avg",
            marker_color="rgba(255, 127, 14, 0.85)",
            hovertemplate="%{x}<br>Avg=%{y:.3f}<extra></extra>",
        ),
        row=4,
        col=2,
    )
    fig.update_xaxes(title_text="Day of week", row=4, col=2)

    fig.add_trace(
        go.Histogram(
            x=df[target_col],
            nbinsx=60,
            name="Consumption distribution",
            marker_color="rgba(148, 103, 189, 0.75)",
            hovertemplate="Consumption=%{x:.3f}<br>Count=%{y}<extra></extra>",
        ),
        row=5,
        col=1,
    )
    fig.update_xaxes(title_text="Consumption", row=5, col=1)
    fig.update_yaxes(title_text="Count", row=5, col=1)

    fig.add_trace(
        go.Heatmap(
            z=heat.values,
            x=[f"{h:02d}" for h in range(24)],
            y=dow_labels,
            colorscale="Blues",
            colorbar=dict(title="Avg"),
            hovertemplate="Day=%{y}<br>Hour=%{x}:00<br>Avg=%{z:.3f}<extra></extra>",
        ),
        row=5,
        col=2,
    )
    fig.update_xaxes(title_text="Hour of day", row=5, col=2)
    fig.update_yaxes(title_text="Day of week", row=5, col=2)

    # Section 5: Analytical insights
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Automatically generated insights"],
                align="left",
                fill_color="rgba(31, 119, 180, 0.12)",
                height=34,
                font=dict(size=12),
            ),
            cells=dict(values=[insights_lines], align="left", height=32, font=dict(size=12)),
        ),
        row=6,
        col=1,
    )

    # Global styling
    fig.update_layout(
        template="plotly_white",
        title=dict(text="Energy Consumption Analytics Dashboard", x=0.5, y=0.985),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0),
        margin=dict(l=70, r=35, t=150, b=70),
        height=1900,
    )

    # Make subplot titles and axis labels readable
    fig.update_annotations(font=dict(size=12))
    fig.update_xaxes(automargin=True, tickfont=dict(size=10))
    fig.update_yaxes(automargin=True, tickfont=dict(size=10))

    # Lift subplot titles a bit so they don't overlap with traces/legend
    try:
        for ann in list(fig.layout.annotations or []):
            if getattr(ann, "xref", None) == "paper" and getattr(ann, "yref", None) == "paper":
                if getattr(ann, "text", None) in set(subplot_title_texts):
                    ann.yshift = 14
    except Exception:
        pass

    # Axes labels for the main panels
    fig.update_yaxes(title_text="Consumption", title_standoff=10, row=2, col=1)
    fig.update_xaxes(title_text="Time", title_standoff=8, row=2, col=1, showticklabels=True, tickformat="%Y-%m-%d")
    fig.update_yaxes(title_text="Consumption", title_standoff=10, row=3, col=1)
    fig.update_xaxes(title_text="Time", title_standoff=8, row=3, col=1, showticklabels=True, tickformat="%Y-%m-%d")

    # Make it easy to read values on hover in the forecast panel
    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="rgba(0,0,0,0.35)",
        spikethickness=1,
        row=3,
        col=1,
    )
    fig.update_yaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="rgba(0,0,0,0.20)",
        spikethickness=1,
        row=3,
        col=1,
    )

    # Range slider + selector on historical panel
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.06),
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all", label="All"),
                ]
            )
        ),
        row=2,
        col=1,
    )

    # Focus forecast panel on recent + future
    x_start = recent_start
    x_end = fc_future["ds"].max() if not fc_future.empty else forecast_horizon_end
    fig.update_xaxes(range=[x_start, x_end], row=3, col=1)

    # --- Non-invasive KPI annotations per panel
    def _add_panel_kpi(row: int, col: int, lines: list[str]) -> None:
        text = "<br>".join(lines)
        fig.add_annotation(
            row=row,
            col=col,
            x=0.01,
            y=0.99,
            xref="x domain",
            yref="y domain",
            text=text,
            showarrow=False,
            align="left",
            font=dict(size=11, color="rgba(0,0,0,0.80)"),
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
        )

    _add_panel_kpi(
        2,
        1,
        [
            f"Mean={_fmt(avg_cons)} | Min={_fmt(min_cons)} | Max={_fmt(max_cons)}",
            f"P95={_fmt(p95)} | P99={_fmt(p99)} | Anoms={n_anoms} ({anom_pct:.2f}%)",
        ],
    )

    if not df_recent.empty or not fc_future.empty:
        _add_panel_kpi(
            3,
            1,
            [
                f"Recent 14d: mean={_fmt(recent_mean)} min={_fmt(recent_min)} max={_fmt(recent_max)}",
                f"Next 24h: mean={_fmt(fc_mean)} peak={_fmt(fc_max)} | avg band width={_fmt(fc_band)}",
            ],
        )

    if hourly_avg.notna().any():
        h_peak = int(hourly_avg.idxmax())
        h_low = int(hourly_avg.idxmin())
        _add_panel_kpi(
            4,
            1,
            [
                f"Peak hour: {h_peak:02d}:00 (avg={_fmt(hourly_avg.loc[h_peak])})",
                f"Lowest hour: {h_low:02d}:00 (avg={_fmt(hourly_avg.loc[h_low])})",
            ],
        )

    if dow_avg.notna().any():
        d_peak = int(dow_avg.idxmax())
        d_low = int(dow_avg.idxmin())
        _add_panel_kpi(
            4,
            2,
            [
                f"Peak day: {dow_labels[d_peak]} (avg={_fmt(dow_avg.loc[d_peak])})",
                f"Lowest day: {dow_labels[d_low]} (avg={_fmt(dow_avg.loc[d_low])})",
            ],
        )

    _add_panel_kpi(
        5,
        1,
        [
            f"Median (P50)={_fmt(p50)} | P95={_fmt(p95)} | P99={_fmt(p99)}",
        ],
    )

    # Heatmap peak cell
    try:
        flat = heat.to_numpy(dtype=float)
        idx = int(pd.Series(flat.ravel()).idxmax())
        d = int(idx // 24)
        h = int(idx % 24)
        peak_val = _safe_float(flat[d, h])
        _add_panel_kpi(
            5,
            2,
            [
                f"Peak cell: {dow_labels[d]} @ {h:02d}:00 (avg={_fmt(peak_val)})",
            ],
        )
    except Exception:
        pass

    return fig


def save_dashboard_html(
    processed: pd.DataFrame,
    forecast: pd.DataFrame,
    output_path: Path,
    datetime_col: str = "datetime",
    target_col: str = "global_active_power",
) -> Path:
    """Build and save the interactive Plotly dashboard as a standalone HTML file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = build_dashboard_figure(
        processed=processed,
        forecast=forecast,
        datetime_col=datetime_col,
        target_col=target_col,
    )

    # Self-contained HTML is easier for reviewers/clients to open without internet access.
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
    return output_path
