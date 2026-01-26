# Copilot instructions

## Project overview
- Single-file Streamlit app in [weather_app.py](../weather_app.py) that renders "Project Helios" UI and all data pipelines.
- Data sources are external HTTP APIs: NWS points/forecast + station observations, AWC METAR/TAF, Open-Meteo HRRR proxy, and Kalshi markets.
- Local timezone is hardcoded to Miami: `TZ_MIAMI = ZoneInfo("America/New_York")` and reused across time/forecast logic.

## Architecture & data flow
- Network access uses a shared `requests.Session` from `make_session()` and `safe_get()` with retries; avoid ad‑hoc `requests.get()`.
- Forecast pipeline: `fetch_forecast_data()` pulls NWS point/forecast + HRRR(Open‑Meteo) + TAF, and returns a single dict consumed by UI.
- Live observations pipeline: `fetch_live_history()` merges NWS station obs with AWC METAR, then `merge_and_dedupe()` buckets to 5‑minute bins with source priority.
- Kalshi integration: `KalshiAuth` loads secrets from `st.secrets` and signs requests; `fetch_market_data()` is cached and expects Kalshi event tickers `KXHIGHMIA-<YYMonDD>`.

## Caching & refresh patterns
- Heavy calls are wrapped with `@st.cache_data` and short TTLs (e.g., 5s for Kalshi, 300s for forecasts, 3600s for sunrise/sunset).
- Manual refresh is via `st.cache_data.clear()` followed by `st.rerun()` in the Live dashboard.
- Auto-refresh is implemented with `components.html()` in `render_autorefresh()`; avoid `time.sleep()` in the UI thread.

## UI conventions
- Main rendering is in `render_live_dashboard()` using Streamlit columns/metrics and data warnings inside expanders.
- Weather “AI” summaries and confidence are in pure functions (`get_agent_analysis()`, `forecast_trade_confidence()`, `forecast_ai_summary()`); keep them side‑effect free.
- Icons are derived via `condition_icon_from_sky_wx()` and `icon_from_short_forecast()`; reuse these instead of inline emoji logic.

## External dependencies & secrets
- Dependencies are listed in [requirements.txt](../requirements.txt); `cryptography` is required for Kalshi auth.
- Secrets must be provided via `st.secrets` keys `KALSHI_KEY_ID` and `KALSHI_PRIVATE_KEY` (PEM string with \n escapes).

## Example touchpoints
- NWS + HRRR merge: `fetch_forecast_data()` in [weather_app.py](../weather_app.py).
- Observation dedupe logic: `merge_and_dedupe()` in [weather_app.py](../weather_app.py).
- Kalshi label/strike rules: `kalshi_label_and_strike()` in [weather_app.py](../weather_app.py).
