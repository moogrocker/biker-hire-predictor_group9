# London Bikes Dashboard

A [Dash](https://dash.plotly.com/) app that explores London's Santander Cycles hire
data against weather, and predicts daily bike-hire demand from a linear model.

**Live:** https://biker-hire-predictor-group9.onrender.com/

## What it does

**Explore the data** — scatter of daily `bikes_hired` against a chosen weather
variable (temperature, precipitation, wind speed, visibility, solar energy, ...),
coloured by weekend/season, plus average hires by day of week. A season filter
restricts both charts to the seasons you pick.

**Predict** — for the first week of January 2026 and the next five days, fetches
real weather from Open-Meteo, runs it through the fitted linear model, and shows
each day as a card (day, weather icon, temperature, predicted hires — busiest day
highlighted) with a bar chart of the same predictions ticked by day of week.

## Data & model

- Historical bike-hire and weather data: [`london_bikes.csv`](https://github.com/kostis-christodoulou/am01-code-sep2026/blob/main/data/london_bikes.csv),
  loaded directly from GitHub at startup.
- `data/model_coefficients.csv` — a fitted linear model (`term, coefficient` pairs).
  A prediction is `Intercept + Σ(coefficient × value)` over the numeric weather
  terms, plus whichever `day_<Weekday>` and `season_<Name>` terms match the date,
  plus `after_price_inc` (see below).
- `data/open_meteo.py` — fetches live weather from [Open-Meteo](https://open-meteo.com)
  (no API key needed): `open_meteo("London", 5)` for a forecast, and
  `open_meteo_history("London", start, end)` for a recent past window. The latter
  uses Open-Meteo's *historical-forecast* archive rather than its ERA5 reanalysis
  archive, because the reanalysis archive never reports visibility — one of the
  model's predictors — for any date.
- `after_price_inc` isn't in the public dataset (it was engineered for the model
  fit but never made it into the CSV above), so the app assumes it's always `1`:
  both prediction windows fall after the model's entire 2010–2025 training range,
  so "after the fare increase" is the only sane constant. This is noted in the
  Predict tab itself.

Swapping in a different `model_coefficients.csv` mostly just works: any numeric
term is read straight off the weather data, `day_*`/`season_*` terms are matched
by name against the date, and terms the new model doesn't have contribute nothing.

## Tech stack

Dash + Plotly for the app and charts, pandas for data wrangling, a hand-written
dark theme (`assets/style.css`, no component library) for the look, and
hand-built SVG icons for the weather cards (no icon library, no external CDN).

## Local development

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python app.py       # http://localhost:8050
```

## Deployment

Deployed on [Render](https://render.com) from `render.yaml`: build with
`pip install uv && uv sync`, serve with `uv run gunicorn app:server`. Pushing to
`main` triggers a redeploy.
