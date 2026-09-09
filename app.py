import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from dash import Dash, Input, Output, dash_table, dcc, html

from data.open_meteo import open_meteo, open_meteo_history

BIKES_URL = "https://raw.githubusercontent.com/kostis-christodoulou/am01-code-sep2026/main/data/london_bikes.csv"
COEF_PATH = os.path.join(os.path.dirname(__file__), "data", "model_coefficients.csv")

WEATHER_VARS = ["temp", "humidity", "precip", "windspeed", "cloudcover"]
DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]

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
numeric_terms = [t for t in coefficients if t != "Intercept" and not t.startswith("day_")]
seasons = [s for s in SEASON_ORDER if s in set(bikes_df.get("season_name", []))]


def predict(weather_df):
    preds = []
    for _, row in weather_df.iterrows():
        total = coefficients.get("Intercept", 0.0)
        for term in numeric_terms:
            total += coefficients.get(term, 0.0) * row.get(term, 0.0)
        total += coefficients.get(f"day_{row['day_of_week']}", 0.0)
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
                        dash_table.DataTable(id="forecast-table", page_size=5, **TABLE_STYLE),
                        dcc.Graph(id="forecast-bar", config={"displaylogo": False}),
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
    Output("forecast-table", "data"),
    Output("forecast-table", "columns"),
    Output("forecast-bar", "figure"),
    Input("tabs", "value"),
)
def update_predictions(tab):
    placeholder_fig = empty_message_fig("No data available")
    if tab != "predict":
        return "", [], [], placeholder_fig, "", [], [], placeholder_fig

    history, history_err, forecast, forecast_err = fetch_weather_frames()

    def build_outputs(df, err, title):
        if err is not None or df is None:
            status = html.Div(f"Could not load weather data: {err}", className="status-error")
            return status, [], [], placeholder_fig
        pred_df = predict(df)
        display_df = pred_df.copy()
        display_df["date"] = display_df["date"].astype(str)
        display_df["bikes_hired"] = display_df["bikes_hired"].round(0)
        columns = [{"name": c, "id": c} for c in display_df.columns]
        fig = px.bar(
            pred_df,
            x="date",
            y="bikes_hired",
            title=title,
            color="bikes_hired",
            color_continuous_scale=[TEAL, AMBER],
        )
        fig.update_traces(marker_line_width=0)
        return "", display_df.to_dict("records"), columns, style_fig(fig, height=320)

    history_status, history_data, history_cols, history_fig = build_outputs(
        history, history_err, "Predicted bikes hired"
    )
    forecast_status, forecast_data, forecast_cols, forecast_fig = build_outputs(
        forecast, forecast_err, "Predicted bikes hired"
    )

    return (
        history_status,
        history_data,
        history_cols,
        history_fig,
        forecast_status,
        forecast_data,
        forecast_cols,
        forecast_fig,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
