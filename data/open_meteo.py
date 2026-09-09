"""
open_meteo.py - fetch daily weather from Open-Meteo.

Open-Meteo (https://open-meteo.com) is free for non-commercial use and needs no
API key. This module has two functions, both returning the same tidy shape:

    from open_meteo import open_meteo, open_meteo_history

    open_meteo("London", 5)                              # next few days (forecast)
    open_meteo_history("London", "2026-01-01", "2026-01-07")  # a past date range

Both return a pandas DataFrame with one row per day and the columns:

    date, day_of_week, season_name, temp, tempmax, humidity, precip,
    windspeed, cloudcover, visibility, solarenergy

which line up with the model's predictors (tempmax, precip, windspeed,
visibility, solarenergy, plus the day-of-week, season, and price-period
effects — see predict() in app.py). temp/humidity/cloudcover are also
included for display and weather-icon classification even though the
current model doesn't use them directly. Temperature is in degrees
Celsius, wind in km/h, precipitation in mm, humidity and cloud cover in
percent, visibility in km (Open-Meteo reports metres; converted here to
match the training data's units), and solarenergy in MJ/m².

The forecast reaches about 7 days ahead. For a date in the recent past
(roughly the last four years — Open-Meteo's historical-forecast archive
starts around 2021-2022) use open_meteo_history, which reads Open-Meteo's
historical *forecast-model* archive rather than its ERA5 reanalysis
archive: the reanalysis archive (archive-api.open-meteo.com) never
reports visibility for any date, so it can't supply one of the model's
predictors. historical-forecast-api.open-meteo.com is built from the same
forecast models as the live forecast, does report visibility, and easily
covers a recent date range like the first week of January 2026.
"""

import pandas as pd
import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Daily aggregates Open-Meteo computes for us directly.
DAILY_FIELDS = {
    "temperature_2m_mean": "temp",
    "temperature_2m_max": "tempmax",
    "relative_humidity_2m_mean": "humidity",
    "precipitation_sum": "precip",
    "wind_speed_10m_mean": "windspeed",
    "cloud_cover_mean": "cloudcover",
    "shortwave_radiation_sum": "solarenergy",  # MJ/m^2, matches the training data's units
}

# Visibility has no daily aggregate on Open-Meteo, so it's requested hourly
# and averaged to a daily figure ourselves, same as the fields below.
HOURLY_ONLY_FIELDS = {
    "visibility": "visibility",  # metres; converted to km below
}

# Fields used when aggregating the historical-forecast archive, which (unlike
# the live forecast endpoint) has no daily-aggregate support at all — the
# whole row is built by hand from hourly values.
HOURLY_FIELDS = {
    "temperature_2m": "temp",
    "relative_humidity_2m": "humidity",
    "precipitation": "precip",
    "wind_speed_10m": "windspeed",
    "cloud_cover": "cloudcover",
    "visibility": "visibility",
    "shortwave_radiation": "solarenergy",
}

SEASON_BY_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}


def _add_calendar_columns(df, date_col="date"):
    df.insert(1, "day_of_week", df[date_col].dt.strftime("%a"))
    df.insert(2, "season_name", df[date_col].dt.month.map(SEASON_BY_MONTH))
    return df


def geocode(location):
    """Turn a place name into (latitude, longitude, label) using Open-Meteo."""
    resp = requests.get(
        GEOCODE_URL,
        params={"name": location, "count": 1, "language": "en"},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        raise ValueError(f"Open-Meteo could not find a location called {location!r}.")
    top = results[0]
    label = ", ".join(p for p in [top.get("name"), top.get("country")] if p)
    return top["latitude"], top["longitude"], label


def open_meteo(location="London", days_to_forecast=5):
    """Return a daily weather forecast for a location as a tidy DataFrame.

    Args:
        location (str): a place name, e.g. "London" or "Paris".
        days_to_forecast (int): number of days ahead, from 1 to 7
            (Open-Meteo's forecast does not go beyond 7 days).

    Returns:
        pandas.DataFrame — see module docstring for columns. The resolved
        place name is stored in df.attrs["location"].
    """
    days_to_forecast = int(days_to_forecast)
    if not 1 <= days_to_forecast <= 7:
        raise ValueError("days_to_forecast must be between 1 and 7 (Open-Meteo's forecast limit).")

    lat, lon, label = geocode(location)

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(DAILY_FIELDS),
        "hourly": ",".join(HOURLY_ONLY_FIELDS),
        "forecast_days": days_to_forecast,
        "timezone": "auto",
        "wind_speed_unit": "kmh",  # matches the training data units
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    daily = payload["daily"]
    df = pd.DataFrame({col: daily[field] for field, col in DAILY_FIELDS.items()})
    df.insert(0, "date", pd.to_datetime(daily["time"]))

    visibility_m = _daily_mean_from_hourly(payload["hourly"], "visibility", df["date"])
    df["visibility"] = visibility_m / 1000.0  # metres -> km, matches training data

    df = _add_calendar_columns(df)
    df.attrs["location"] = label
    return df


def open_meteo_history(location, start_date, end_date):
    """Return daily weather for a recent past date range from Open-Meteo's
    historical-forecast archive.

    Use this for dates the live forecast can't reach but that are still
    recent enough for the forecast-model archive to cover — roughly the
    last few years, which comfortably includes a window like the first
    week of January 2026. For much older dates, Open-Meteo's separate
    ERA5-based archive (archive-api.open-meteo.com) reaches back to the
    1940s, but it never reports visibility, so it can't feed this app's
    model — this function deliberately doesn't fall back to it.

    Args:
        location (str): a place name, e.g. "London".
        start_date (str): first day, "YYYY-MM-DD".
        end_date (str): last day, "YYYY-MM-DD".

    Returns:
        pandas.DataFrame — see module docstring for columns.
    """
    lat, lon, label = geocode(location)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_FIELDS),
        "timezone": "auto",
        "wind_speed_unit": "kmh",  # matches the training data units
    }
    resp = requests.get(HISTORICAL_FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]

    hf = pd.DataFrame({col: hourly[field] for field, col in HOURLY_FIELDS.items()})
    hf["date"] = pd.to_datetime(hourly["time"]).normalize()

    daily = hf.groupby("date").agg(
        temp=("temp", "mean"),
        tempmax=("temp", "max"),
        humidity=("humidity", "mean"),
        precip=("precip", "sum"),
        windspeed=("windspeed", "mean"),
        cloudcover=("cloudcover", "mean"),
        visibility=("visibility", "mean"),
        solarenergy=("solarenergy", lambda s: s.sum() * 3600 / 1_000_000),  # W/m^2 -> MJ/m^2
    ).reset_index()

    daily["visibility"] = daily["visibility"] / 1000.0  # metres -> km, matches training data
    daily = _add_calendar_columns(daily)
    daily.attrs["location"] = label
    return daily


def _daily_mean_from_hourly(hourly_payload, field, dates):
    """Average an hourly Open-Meteo series to one value per day in `dates`."""
    series = pd.Series(hourly_payload[field], index=pd.to_datetime(hourly_payload["time"]))
    daily_mean = series.groupby(series.index.normalize()).mean()
    return daily_mean.reindex(dates.dt.normalize()).to_numpy()


if __name__ == "__main__":
    # Quick manual checks (need internet)
    print("Forecast (next 5 days):")
    print(open_meteo("London", 5).to_string(index=False))
    print("\nHistory (first week of January 2026):")
    print(open_meteo_history("London", "2026-01-01", "2026-01-07").to_string(index=False))
