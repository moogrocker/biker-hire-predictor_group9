import base64
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from dash import Dash, Input, Output, dash_table, dcc, html

from data.open_meteo import open_meteo, open_meteo_history

BIKES_URL = "https://raw.githubusercontent.com/kostis-christodoulou/am01-code-sep2026/main/data/london_bikes.csv"
COEF_PATH = os.path.join(os.path.dirname(__file__), "data", "model_coefficients.csv")

WEATHER_VARS = ["temp", "tempmax", "humidity", "precip", "windspeed", "cloudcover", "visibility", "solarenergy"]
DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]
DAY_TERM_PREFIX = "day_"
SEASON_TERM_PREFIX = "season_"
PRICE_TERM = "after_price_inc"

# The model's after_price_inc dummy has no matching column anywhere in the
# public dataset — it's a feature engineered for the model fit that never
# made it into the CSV this app reads. Both prediction windows (the Jan
# 2026 history week and the live forecast) fall after the entire 2010-2025
# training range, so "after the increase" (=1) is the only sane constant
# for both. See PRICE_TERM in predict().
ASSUME_AFTER_PRICE_INCREASE = 1

# Palette — a night ride through London: navy sky, sodium-lamp amber, wet-asphalt
# teal. Shared between assets/style.css and the Plotly chart theme below so the
# charts sit inside the glass panels instead of showing up as white boxes.
INK = "#0a0f1e"
INK_2 = "#0d1526"
SURFACE = "#121b30"
GRID = "#24304a"
TEXT = "#edf1f7"
TEXT_MUTED = "#8b96ac"
AMBER = "#f2a649"
TEAL = "#46c2b9"
CATEGORICAL = [AMBER, TEAL, "#e1573c", "#6c8eef"]

# --- hand-built weather icons for the forecast strip -----------------------
# Simple primitive shapes (circles/rects/lines), no icon library. Colours are
# baked in — these ship as <img> data URIs, which don't inherit page CSS.


def _svg_data_uri(svg):
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def _cloud_shapes(fill, y_offset=0):
    return f"""
      <circle cx="12" cy="{14 + y_offset}" r="6" fill="{fill}"/>
      <circle cx="18" cy="{10 + y_offset}" r="7.5" fill="{fill}"/>
      <circle cx="23" cy="{15 + y_offset}" r="5" fill="{fill}"/>
      <rect x="7" y="{14 + y_offset}" width="20" height="8" rx="4" fill="{fill}"/>
    """


SUN_SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <circle cx="16" cy="16" r="7" fill="{AMBER}"/>
  <g stroke="{AMBER}" stroke-width="2" stroke-linecap="round">
    <line x1="16" y1="2" x2="16" y2="6"/><line x1="16" y1="26" x2="16" y2="30"/>
    <line x1="2" y1="16" x2="6" y2="16"/><line x1="26" y1="16" x2="30" y2="16"/>
    <line x1="5.8" y1="5.8" x2="8.6" y2="8.6"/><line x1="23.4" y1="23.4" x2="26.2" y2="26.2"/>
    <line x1="5.8" y1="26.2" x2="8.6" y2="23.4"/><line x1="23.4" y1="8.6" x2="26.2" y2="5.8"/>
  </g>
</svg>"""

PARTLY_CLOUDY_SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <circle cx="12" cy="11" r="5.5" fill="{AMBER}"/>
  <g stroke="{AMBER}" stroke-width="1.8" stroke-linecap="round">
    <line x1="12" y1="1" x2="12" y2="3.5"/>
    <line x1="4.5" y1="5.5" x2="6.3" y2="7.3"/>
    <line x1="3" y1="11" x2="5.5" y2="11"/>
  </g>
  <g>{_cloud_shapes(TEXT_MUTED, y_offset=6)}</g>
</svg>"""

CLOUDY_SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">{_cloud_shapes(TEXT_MUTED)}</svg>"""

RAIN_SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  {_cloud_shapes(TEAL)}
  <g stroke="{TEAL}" stroke-width="2" stroke-linecap="round">
    <line x1="12" y1="25" x2="10.5" y2="29"/>
    <line x1="18" y1="25" x2="16.5" y2="29"/>
    <line x1="24" y1="25" x2="22.5" y2="29"/>
  </g>
</svg>"""


def classify_weather(precip, cloudcover):
    """Map forecast fields to a (label, icon svg) pair — no condition field
    comes back from Open-Meteo, so this is derived from precip/cloudcover."""
    precip = precip or 0
    cloudcover = cloudcover if pd.notna(cloudcover) else 0
    if precip >= 0.5:
        return "Rain", RAIN_SVG
    if cloudcover >= 70:
        return "Cloudy", CLOUDY_SVG
    if cloudcover >= 30:
        return "Partly cloudy", PARTLY_CLOUDY_SVG
    return "Clear", SUN_SVG


def load_bikes():
    df = pd.read_csv(BIKES_URL)
    if "day_of_week" in df.columns:
        df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=DAY_ORDER, ordered=True)
    if "season_name" in df.columns:
        order = [s for s in SEASON_ORDER if s in set(df["season_name"])]
        df["season_name"] = pd.Categorical(df["season_name"], categories=order, ordered=True)
    return df


def load_coefficients():
    coef = pd.read_csv(COEF_PATH)
    return dict(zip(coef["term"], coef["coefficient"]))


bikes_df = load_bikes()
coefficients = load_coefficients()
numeric_terms = [
    t
    for t in coefficients
    if t != "Intercept"
    and not t.startswith(DAY_TERM_PREFIX)
    and not t.startswith(SEASON_TERM_PREFIX)
    and t != PRICE_TERM
]
seasons = [s for s in SEASON_ORDER if s in set(bikes_df.get("season_name", []))]


def predict(weather_df, after_price_inc=ASSUME_AFTER_PRICE_INCREASE):
    """weather_df needs a numeric column per entry in numeric_terms, plus
    day_of_week and (if the model has season_* terms) season_name. Terms
    the loaded model doesn't have (e.g. an older coefficients file with no
    season_/after_price_inc terms) contribute 0 automatically via
    coefficients.get(..., 0.0), so this stays compatible with either model."""
    preds = []
    for _, row in weather_df.iterrows():
        total = coefficients.get("Intercept", 0.0)
        for term in numeric_terms:
            total += coefficients.get(term, 0.0) * row.get(term, 0.0)
        total += coefficients.get(f"{DAY_TERM_PREFIX}{row['day_of_week']}", 0.0)
        if "season_name" in row.index:
            total += coefficients.get(f"{SEASON_TERM_PREFIX}{row['season_name']}", 0.0)
        total += coefficients.get(PRICE_TERM, 0.0) * after_price_inc
        preds.append(max(total, 0.0))
    result = weather_df.copy()
    result["bikes_hired"] = preds
    return result


def fetch_weather_frames():
    """Fetch both weather windows, returning (frame, error_message) for each."""
    history, history_err = None, None
    forecast, forecast_err = None, None
    try:
        history = open_meteo_history("London", "2026-01-01", "2026-01-07")
    except (requests.RequestException, ValueError, KeyError) as exc:
        history_err = str(exc)
    try:
        forecast = open_meteo("London", 5)
    except (requests.RequestException, ValueError, KeyError) as exc:
        forecast_err = str(exc)
    return history, history_err, forecast, forecast_err


def filter_by_season(seasons_selected):
    if "season_name" not in bikes_df.columns:
        return bikes_df
    if not seasons_selected:
        return bikes_df.iloc[0:0]
    return bikes_df[bikes_df["season_name"].isin(seasons_selected)]


def empty_message_fig(text):
    fig = go.Figure()
    fig.update_layout(
        annotations=[dict(text=text, showarrow=False, font=dict(size=14, color=TEXT_MUTED))],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return style_fig(fig, height=360)


def style_fig(fig, height=420):
    """Apply the dashboard's dark theme to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT_MUTED, size=13),
        title=dict(font=dict(family="Space Grotesk, sans-serif", color=TEXT, size=16)),
        margin=dict(l=10, r=10, t=48, b=10),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_MUTED)),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=GRID, font=dict(color=TEXT, family="Inter, sans-serif")),
        coloraxis_showscale=False,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, color=TEXT_MUTED)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, color=TEXT_MUTED)
    return fig


def build_sparkline(values, width=320, height=32):
    """A thin trend line under the forecast strip — folds the demand trend
    into the strip itself instead of a separate chart below it."""
    n = len(values)
    vmin, vmax = min(values), max(values)
    span = (vmax - vmin) or 1
    xs = [width * i / (n - 1) for i in range(n)] if n > 1 else [width / 2]
    ys = [height - 5 - (v - vmin) / span * (height - 10) for v in values]
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{AMBER}"/>' for x, y in zip(xs, ys))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">
      <polyline points="{points}" fill="none" stroke="{TEAL}" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round" opacity="0.85"/>
      {dots}
    </svg>"""
    return html.Img(
        src=_svg_data_uri(svg),
        alt="Predicted bike-hire trend across the five days",
        className="forecast-sparkline",
    )


def build_forecast_strip(pred_df):
    """One glass strip with a column per day, divided by hairlines, rather
    than five separate cards — the days are a real sequence, so a single
    connected strip (with a trend line threaded underneath) reads as one
    forecast rather than a repeated card kit."""
    rows = pred_df.reset_index(drop=True)
    peak_idx = rows["bikes_hired"].idxmax()
    columns = []
    for i, row in rows.iterrows():
        condition, icon_svg = classify_weather(row.get("precip"), row.get("cloudcover"))
        is_peak = i == peak_idx
        bikes_class = "forecast-bikes forecast-bikes--peak" if is_peak else "forecast-bikes"
        children = [
            html.Span(row["day_of_week"], className="forecast-day"),
            html.Span(pd.to_datetime(row["date"]).strftime("%-d %b"), className="forecast-date"),
            html.Img(src=_svg_data_uri(icon_svg), alt=condition, className="forecast-icon"),
            html.Span(condition, className="forecast-condition"),
            html.Span(f"{row['temp']:.0f}°C", className="forecast-temp"),
            html.Span(f"{row['bikes_hired']:,.0f}", className=bikes_class),
        ]
        if is_peak:
            children.append(html.Span("busiest day", className="forecast-peak-label"))
        columns.append(html.Div(children, className="forecast-col"))
    return html.Div(
        [
            html.Div(columns, className="forecast-cols"),
            build_sparkline(rows["bikes_hired"].tolist()),
        ],
        className="forecast-strip",
    )


TABLE_STYLE = dict(
    style_table={"overflowX": "auto"},
    style_header={
        "backgroundColor": SURFACE,
        "color": TEXT,
        "fontFamily": "Space Grotesk, sans-serif",
        "fontWeight": "600",
        "border": "none",
        "borderBottom": f"1px solid {GRID}",
    },
    style_cell={
        "backgroundColor": "transparent",
        "color": TEXT_MUTED,
        "fontFamily": "Inter, sans-serif",
        "fontSize": "13.5px",
        "border": "none",
        "borderBottom": f"1px solid {GRID}",
        "padding": "10px 14px",
    },
    style_data={"backgroundColor": "transparent"},
    style_as_list_view=True,
)

app = Dash(__name__)
server = app.server
app.title = "London Bikes Dashboard"

app.index_string = """<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link
        href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap"
        rel="stylesheet">
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>"""

kpi_total_days = len(bikes_df)
kpi_avg_hires = bikes_df["bikes_hired"].mean()
kpi_avg_temp = bikes_df["temp"].mean() if "temp" in bikes_df.columns else None

hero = html.Div(
    [
        html.Div(
            [
                html.P("Weather-driven demand for London's cycle hire scheme", className="hero-kicker"),
                html.H1("London Bikes", className="hero-title"),
            ],
            className="hero-text",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Span(f"{kpi_total_days:,}", className="kpi-value"),
                        html.Span("days of hire records", className="kpi-label"),
                    ],
                    className="kpi-card",
                ),
                html.Div(
                    [
                        html.Span(f"{kpi_avg_hires:,.0f}", className="kpi-value"),
                        html.Span("average daily hires", className="kpi-label"),
                    ],
                    className="kpi-card",
                ),
                html.Div(
                    [
                        html.Span(f"{kpi_avg_temp:.1f}°C", className="kpi-value"),
                        html.Span("average daily temperature", className="kpi-label"),
                    ],
                    className="kpi-card",
                ),
            ],
            className="kpi-row",
        ),
    ],
    className="hero",
)

season_options = [{"label": s, "value": s} for s in seasons]

controls_panel = html.Div(
    [
        html.Div(
            [
                html.Label("Weather variable", className="control-label"),
                dcc.Dropdown(
                    id="weather-var",
                    options=[{"label": v.capitalize(), "value": v} for v in WEATHER_VARS if v in bikes_df.columns],
                    value="temp",
                    clearable=False,
                    className="dash-dropdown-dark",
                ),
            ],
            className="control-block",
        ),
        html.Div(
            [
                html.Label("Colour points by", className="control-label"),
                dcc.RadioItems(
                    id="colour-by",
                    options=[
                        {"label": "Weekend", "value": "weekend"},
                        {"label": "Season", "value": "season_name"},
                    ],
                    value="weekend",
                    className="pill-group",
                ),
            ],
            className="control-block",
        ),
        html.Div(
            [
                html.Label("Season", className="control-label"),
                dcc.Checklist(
                    id="season-filter",
                    options=season_options,
                    value=[o["value"] for o in season_options],
                    className="pill-group",
                ),
            ],
            className="control-block",
        ),
    ],
    className="panel controls-panel",
)

explore_tab = html.Div(
    [
        html.Div(
            [
                controls_panel,
                html.Div(
                    [
                        html.Div(dcc.Graph(id="scatter-plot", config={"displaylogo": False}), className="panel chart-panel"),
                        html.Div(dcc.Graph(id="day-bar-plot", config={"displaylogo": False}), className="panel chart-panel"),
                    ],
                    className="chart-stack",
                ),
            ],
            className="explore-grid",
        ),
    ],
    style={"marginTop": "8px"},
)

predict_tab = html.Div(
    [
        html.P(
            "Both windows below assume current, post fare-increase pricing.",
            className="predict-note",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("First week of January 2026", className="panel-title"),
                        html.P("Historical weather, same model", className="panel-sub"),
                        html.Div(id="history-status"),
                        dash_table.DataTable(id="history-table", page_size=7, **TABLE_STYLE),
                        dcc.Graph(id="history-bar", config={"displaylogo": False}),
                    ],
                    className="panel predict-panel",
                ),
                html.Div(
                    [
                        html.H3("Next five days", className="panel-title"),
                        html.P("Live forecast from Open-Meteo", className="panel-sub"),
                        html.Div(id="forecast-status"),
                        html.Div(id="forecast-strip"),
                    ],
                    className="panel predict-panel",
                ),
            ],
            className="predict-grid",
        ),
    ],
    style={"marginTop": "8px"},
)

app.layout = html.Div(
    [
        hero,
        dcc.Tabs(
            id="tabs",
            value="explore",
            parent_className="tabs-nav",
            className="tabs-track",
            children=[
                dcc.Tab(
                    label="Explore the data",
                    value="explore",
                    className="tab-pill",
                    selected_className="tab-pill--selected",
                    children=[explore_tab],
                ),
                dcc.Tab(
                    label="Predict",
                    value="predict",
                    className="tab-pill",
                    selected_className="tab-pill--selected",
                    children=[predict_tab],
                ),
            ],
        ),
    ],
    className="page",
)


@app.callback(
    Output("scatter-plot", "figure"),
    Input("weather-var", "value"),
    Input("colour-by", "value"),
    Input("season-filter", "value"),
)
def update_scatter(weather_var, colour_by, seasons_selected):
    df = filter_by_season(seasons_selected)
    if df.empty:
        return empty_message_fig("Select at least one season to see the data")
    fig = px.scatter(
        df,
        x=weather_var,
        y="bikes_hired",
        color=colour_by,
        color_discrete_sequence=CATEGORICAL,
        title=f"Daily bikes hired vs {weather_var}",
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=0)))
    return style_fig(fig)


@app.callback(Output("day-bar-plot", "figure"), Input("season-filter", "value"))
def update_day_bar(seasons_selected):
    df = filter_by_season(seasons_selected)
    if df.empty:
        return empty_message_fig("Select at least one season to see the data")
    avg_by_day = df.groupby("day_of_week", observed=True)["bikes_hired"].mean().reindex(DAY_ORDER)
    fig = px.bar(
        x=avg_by_day.index,
        y=avg_by_day.values,
        labels={"x": "Day of week", "y": "Average bikes hired"},
        title="Average bikes hired by day of week",
        color=avg_by_day.values,
        color_continuous_scale=[TEAL, AMBER],
    )
    fig.update_traces(marker_line_width=0)
    return style_fig(fig)


@app.callback(
    Output("history-status", "children"),
    Output("history-table", "data"),
    Output("history-table", "columns"),
    Output("history-bar", "figure"),
    Output("forecast-status", "children"),
    Output("forecast-strip", "children"),
    Input("tabs", "value"),
)
def update_predictions(tab):
    placeholder_fig = empty_message_fig("No data available")
    if tab != "predict":
        return "", [], [], placeholder_fig, "", []

    history, history_err, forecast, forecast_err = fetch_weather_frames()

    if history_err is not None or history is None:
        history_status = html.Div(f"Could not load weather data: {history_err}", className="status-error")
        history_data, history_cols, history_fig = [], [], placeholder_fig
    else:
        pred_history = predict(history)
        display_df = pred_history.copy()
        display_df["date"] = display_df["date"].astype(str)
        display_df["bikes_hired"] = display_df["bikes_hired"].round(0)
        fig = px.bar(
            pred_history,
            x="date",
            y="bikes_hired",
            title="Predicted bikes hired",
            color="bikes_hired",
            color_continuous_scale=[TEAL, AMBER],
        )
        fig.update_traces(marker_line_width=0)
        history_status = ""
        history_data = display_df.to_dict("records")
        history_cols = [{"name": c, "id": c} for c in display_df.columns]
        history_fig = style_fig(fig, height=320)

    if forecast_err is not None or forecast is None:
        forecast_status = html.Div(f"Could not load weather data: {forecast_err}", className="status-error")
        forecast_children = []
    else:
        forecast_status = ""
        forecast_children = build_forecast_strip(predict(forecast))

    return history_status, history_data, history_cols, history_fig, forecast_status, forecast_children


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
