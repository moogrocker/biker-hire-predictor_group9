import os

import pandas as pd
import plotly.express as px
import requests
from dash import Dash, Input, Output, dash_table, dcc, html

from data.open_meteo import open_meteo, open_meteo_history

BIKES_URL = "https://raw.githubusercontent.com/kostis-christodoulou/am01-code-sep2026/main/data/london_bikes.csv"
COEF_PATH = os.path.join(os.path.dirname(__file__), "data", "model_coefficients.csv")

WEATHER_VARS = ["temp", "humidity", "precip", "windspeed", "cloudcover"]
DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_bikes():
    df = pd.read_csv(BIKES_URL)
    if "day_of_week" in df.columns:
        df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=DAY_ORDER, ordered=True)
    return df


def load_coefficients():
    coef = pd.read_csv(COEF_PATH)
    return dict(zip(coef["term"], coef["coefficient"]))


bikes_df = load_bikes()
coefficients = load_coefficients()
numeric_terms = [t for t in coefficients if t != "Intercept" and not t.startswith("day_")]


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


app = Dash(__name__)
server = app.server
app.title = "London Bikes Dashboard"

explore_tab = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Weather variable"),
                        dcc.Dropdown(
                            id="weather-var",
                            options=[{"label": v, "value": v} for v in WEATHER_VARS if v in bikes_df.columns],
                            value="temp",
                            clearable=False,
                        ),
                    ],
                    style={"width": "45%", "display": "inline-block", "marginRight": "5%"},
                ),
                html.Div(
                    [
                        html.Label("Colour by"),
                        dcc.RadioItems(
                            id="colour-by",
                            options=[
                                {"label": "Weekend", "value": "weekend"},
                                {"label": "Season", "value": "season_name"},
                            ],
                            value="weekend",
                            inline=True,
                        ),
                    ],
                    style={"width": "45%", "display": "inline-block"},
                ),
            ],
            style={"marginTop": "20px", "marginBottom": "10px"},
        ),
        dcc.Graph(id="scatter-plot"),
        html.H3("Average hires by day of week"),
        dcc.Graph(id="day-bar-plot"),
    ]
)

predict_tab = html.Div(
    [
        html.H3("First week of January 2026 (history)"),
        html.Div(id="history-status"),
        dash_table.DataTable(id="history-table", page_size=7, style_table={"overflowX": "auto"}),
        dcc.Graph(id="history-bar"),
        html.H3("Next five days (forecast)", style={"marginTop": "30px"}),
        html.Div(id="forecast-status"),
        dash_table.DataTable(id="forecast-table", page_size=5, style_table={"overflowX": "auto"}),
        dcc.Graph(id="forecast-bar"),
    ],
    style={"marginTop": "20px"},
)

app.layout = html.Div(
    [
        html.H1("London Bikes Dashboard"),
        dcc.Tabs(
            id="tabs",
            value="explore",
            children=[
                dcc.Tab(label="Explore the data", value="explore", children=[explore_tab]),
                dcc.Tab(label="Predict", value="predict", children=[predict_tab]),
            ],
        ),
    ],
    style={"maxWidth": "1000px", "margin": "0 auto", "fontFamily": "Arial, sans-serif"},
)


@app.callback(
    Output("scatter-plot", "figure"),
    Input("weather-var", "value"),
    Input("colour-by", "value"),
)
def update_scatter(weather_var, colour_by):
    fig = px.scatter(
        bikes_df,
        x=weather_var,
        y="bikes_hired",
        color=colour_by,
        title=f"Daily bikes hired vs {weather_var}",
        opacity=0.7,
    )
    return fig


@app.callback(Output("day-bar-plot", "figure"), Input("weather-var", "value"))
def update_day_bar(_weather_var):
    avg_by_day = bikes_df.groupby("day_of_week", observed=True)["bikes_hired"].mean().reindex(DAY_ORDER)
    fig = px.bar(
        x=avg_by_day.index,
        y=avg_by_day.values,
        labels={"x": "Day of week", "y": "Average bikes hired"},
        title="Average bikes hired by day of week",
    )
    return fig


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
    empty_fig = px.bar(title="No data available")
    if tab != "predict":
        return "", [], [], empty_fig, "", [], [], empty_fig

    history, history_err, forecast, forecast_err = fetch_weather_frames()

    def build_outputs(df, err, title):
        if err is not None or df is None:
            status = html.Div(
                f"Could not load weather data: {err}",
                style={"color": "red"},
            )
            return status, [], [], empty_fig
        pred_df = predict(df)
        display_df = pred_df.copy()
        display_df["date"] = display_df["date"].astype(str)
        display_df["bikes_hired"] = display_df["bikes_hired"].round(0)
        columns = [{"name": c, "id": c} for c in display_df.columns]
        fig = px.bar(pred_df, x="date", y="bikes_hired", title=title)
        return "", display_df.to_dict("records"), columns, fig

    history_status, history_data, history_cols, history_fig = build_outputs(
        history, history_err, "Predicted bikes hired (first week of Jan 2026)"
    )
    forecast_status, forecast_data, forecast_cols, forecast_fig = build_outputs(
        forecast, forecast_err, "Predicted bikes hired (next 5 days)"
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
