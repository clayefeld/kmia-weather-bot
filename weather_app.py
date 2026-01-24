import streamlit as st
import streamlit.components.v1 as components
import requests
import base64
import time
import pandas as pd
import math
import re
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Tuple, Any

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(page_title="Project Helios", page_icon="☀️", layout="wide")

TZ_MIAMI = ZoneInfo("America/New_York")

# --- DATA SOURCES ---
NWS_API_HISTORY = "https://api.weather.gov/stations/KMIA/observations"
NWS_POINT_URL = "https://api.weather.gov/points/25.7954,-80.2901"

AWC_METAR_URL = "https://aviationweather.gov/api/data/metar?ids=KMIA&format=raw&hours=12"
AWC_TAF_URL = "https://aviationweather.gov/api/data/taf?ids=KMIA&format=raw"

KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2"

OM_API_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=25.7954&longitude=-80.2901"
    "&hourly=temperature_2m,precipitation_probability,shortwave_radiation,cloud_cover"
    "&timezone=America%2FNew_York&forecast_days=2&models=hrrr_north_america"
)

# --- GLOBAL STYLES ---
HIDE_INDEX_CSS = """
<style>
thead tr th:first-child {display:none}
tbody th {display:none}
div.stButton > button {width: 100%;}
@media (max-width: 640px) {
    [data-testid="column"] {width: 50% !important; flex: 0 0 50% !important; min-width: 0 !important; padding: 0 0.2rem !important;}
    .block-container {padding-top: 1rem !important; padding-bottom: 2rem !important;}
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    header {visibility: hidden;}
}
</style>
"""

# --- LOGGING ---
logger = logging.getLogger("helios")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# =============================================================================
# HTTP SESSION + RETRIES
# =============================================================================
def make_session() -> requests.Session:
    s = requests.Session()
    # NWS expects a User-Agent identifying the application
    s.headers.update(
        {
            "User-Agent": "ProjectHelios/1.4 (Streamlit app)",
            "Accept": "application/geo+json, application/json;q=0.9, */*;q=0.8",
        }
    )
    return s


SESSION = make_session()


def safe_get(url: str, timeout: float = 4.0) -> requests.Response:
    last_exc = None
    for attempt in range(3):
        try:
            return SESSION.get(url, timeout=timeout)
        except Exception as e:
            last_exc = e
            time.sleep(0.25 * (attempt + 1))
    raise last_exc  # type: ignore[misc]


# =============================================================================
# TIME + PARSING UTILS
# =============================================================================
def get_miami_time() -> datetime:
    return datetime.now(TZ_MIAMI)


def parse_iso_time(iso_str: Optional[str]) -> Optional[datetime]:
    if not iso_str:
        return None
    try:
        s = iso_str.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def get_display_time(dt_aware: datetime) -> str:
    return dt_aware.astimezone(TZ_MIAMI).strftime("%I:%M %p")


def c_to_f(c: float) -> float:
    return (c * 1.8) + 32.0


def mps_to_kt(mps: float) -> float:
    return mps * 1.94384


def calculate_heat_index(temp_f: float, humidity: float) -> float:
    if temp_f < 80:
        return temp_f
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = (
        -42.379,
        2.04901523,
        10.14333127,
        -0.22475541,
        -6.83783e-3,
        -5.481717e-2,
        1.22874e-3,
        8.5282e-4,
        -1.99e-6,
    )
    T, R = temp_f, humidity
    return c1 + (c2 * T) + (c3 * R) + (c4 * T * R) + (c5 * T ** 2) + (c6 * R ** 2) + (c7 * T ** 2 * R) + (c8 * T * R ** 2) + (c9 * T ** 2 * R ** 2)


def condition_icon(sky: str, wx: Optional[str] = None) -> str:
    s = (sky or "").upper()
    w = (wx or "").upper()

    # Wx overrides
    if "TS" in w:
        return "⛈️"
    if "RA" in w or "SH" in w:
        return "🌧️"
    if "FG" in w or "BR" in w or "HZ" in w:
        return "🌫️"

    if s in ("CLR", "SKC"):
        return "☀️"
    if s == "FEW":
        return "🌤️"
    if s == "SCT":
        return "⛅"
    if s in ("BKN", "OVC"):
        return "☁️"
    return "☁️" if s else "—"


# =============================================================================
# KALSHI AUTH
# =============================================================================
class KalshiAuth:
    def __init__(self):
        self.ready = False
        self.key_id = None
        self.private_key = None
        self.error = None

        try:
            self.key_id = st.secrets["KALSHI_KEY_ID"]
            pk_raw = st.secrets["KALSHI_PRIVATE_KEY"]
            private_key_str = pk_raw.replace("\\n", "\n").strip()

            from cryptography.hazmat.primitives import serialization

            self.private_key = serialization.load_pem_private_key(
                private_key_str.encode("utf-8"), password=None
            )
            self.ready = True
        except Exception as e:
            self.error = str(e)
            self.ready = False

    def sign_request(self, method: str, path: str, timestamp_ms: str) -> Optional[str]:
        if not self.ready:
            return None

        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        msg = f"{timestamp_ms}{method}{path}".encode("utf-8")
        sig = self.private_key.sign(  # type: ignore[union-attr]
            msg,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode("utf-8")


# =============================================================================
# KALSHI MARKET FETCHER (BRACKETS MATCH WEBSITE)
# =============================================================================
def kalshi_bounds_for_display(
    floor_raw: Optional[float], cap_raw: Optional[float]
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Kalshi strikes can be boundary values; website shows integer *official* temps.

    Display rules that match Kalshi UI for these temp buckets:
      - Open-low  ("X° or below"): X = floor(cap_raw - eps)
      - Middle    ("A° - B°"):     A = floor(floor_raw) + 1,  B = floor(cap_raw)
      - Open-high ("Y° or above"): Y = floor(floor_raw) + 1

    Returns:
      (display_floor_for_mid_high, display_cap_for_mid, display_cap_for_open_low)
    """
    eps = 1e-6
    display_floor = None if floor_raw is None else int(math.floor(float(floor_raw)) + 1)
    display_cap_mid = None if cap_raw is None else int(math.floor(float(cap_raw)))
    display_cap_openlow = None if cap_raw is None else int(math.floor(float(cap_raw) - eps))
    return display_floor, display_cap_mid, display_cap_openlow


def format_kalshi_label(
    display_floor: Optional[int], display_cap_mid: Optional[int], display_cap_openlow: Optional[int]
) -> str:
    if display_floor is None and display_cap_openlow is not None:
        return f"{display_cap_openlow}° or below"
    if display_cap_mid is None and display_floor is not None:
        return f"{display_floor}° or above"
    if display_floor is not None and display_cap_mid is not None:
        return f"{display_floor}° - {display_cap_mid}°"
    return "Unknown"


@st.cache_data(ttl=5)
def fetch_market_data() -> Tuple[List[Dict[str, Any]], str]:
    # cryptography is required for Kalshi signing
    try:
        from cryptography.hazmat.primitives import serialization  # noqa: F401
    except Exception:
        return [], "🔴 Crypto lib missing"

    auth = KalshiAuth()
    if not auth.ready:
        return [], "🔴 Kalshi key error"

    try:
        now_miami = get_miami_time()
        date_str = now_miami.strftime("%y%b%d").upper()
        event_ticker = f"KXHIGHMIA-{date_str}"

        path = f"/events/{event_ticker}"
        ts = str(int(time.time() * 1000))
        sig = auth.sign_request("GET", path, ts)

        headers = {
            "KALSHI-ACCESS-KEY": auth.key_id,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "Content-Type": "application/json",
        }

        r = SESSION.get(KALSHI_API_URL + path, headers=headers, timeout=4)
        r.raise_for_status()
        data = r.json()

        raw_markets = data.get("markets", [])
        parsed: List[Dict[str, Any]] = []

        for m in raw_markets:
            floor_raw = m.get("floor_strike")
            cap_raw = m.get("cap_strike")
            ask = int(m.get("yes_ask", 0) or 0)

            floor_f = float(floor_raw) if floor_raw is not None else None
            cap_f = float(cap_raw) if cap_raw is not None else None

            # Sort: open-low first, then by increasing floor, open-high last
            if floor_f is None and cap_f is not None:
                sort_key = -1_000_000 + cap_f
            elif floor_f is not None:
                sort_key = floor_f
            else:
                sort_key = 1_000_000

            display_floor, display_cap_mid, display_cap_openlow = kalshi_bounds_for_display(floor_f, cap_f)
            label = format_kalshi_label(display_floor, display_cap_mid, display_cap_openlow)

            # Strike used for UI selection:
            # - open-low: use displayed cap (e.g., 76)
            # - middle/open-high: use displayed floor (e.g., 81, 85)
            if floor_f is None and cap_f is not None:
                strike = float(display_cap_openlow) if display_cap_openlow is not None else float("nan")
                cap_for_bust = float(display_cap_openlow) if display_cap_openlow is not None else None
            else:
                strike = float(display_floor) if display_floor is not None else float("nan")
                cap_for_bust = float(display_cap_mid) if display_cap_mid is not None else None  # None for open-high

            parsed.append(
                {
                    "label": label,
                    "strike": strike,
                    "price": ask,
                    "floor_raw": floor_f,
                    "cap_raw": cap_f,
                    "floor": display_floor,
                    "cap": cap_for_bust,
                    "sort": sort_key,
                }
            )

        parsed.sort(key=lambda x: x["sort"])
        return parsed, "🟢 Live"

    except Exception:
        logger.exception("Kalshi fetch error")
        return [], "🔴 Kalshi API error"


# =============================================================================
# FORECAST FETCHER (NWS + HRRR(Open-Meteo) + TAF)
# =============================================================================
@st.cache_data(ttl=300)
def fetch_forecast_data() -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "today_daily": None,
        "tomorrow_daily": None,
        "all_hourly": [],
        "hrrr_now": None,
        "taf_raw": None,
        "status": [],
    }

    # NWS point -> forecast URLs
    try:
        r = safe_get(NWS_POINT_URL, timeout=6)
        r.raise_for_status()
        urls = r.json().get("properties", {})

        r_d = safe_get(urls.get("forecast"), timeout=6)
        if r_d.status_code == 200:
            periods = r_d.json().get("properties", {}).get("periods", [])
            daytimes = [p for p in periods if p.get("isDaytime")]
            if len(daytimes) >= 1:
                data["today_daily"] = daytimes[0]
            if len(daytimes) >= 2:
                data["tomorrow_daily"] = daytimes[1]
        else:
            data["status"].append("NWS daily forecast unavailable")

        r_h = safe_get(urls.get("forecastHourly"), timeout=6)
        if r_h.status_code == 200:
            data["all_hourly"] = r_h.json().get("properties", {}).get("periods", [])
        else:
            data["status"].append("NWS hourly forecast unavailable")

    except Exception:
        logger.exception("NWS forecast error")
        data["status"].append("NWS forecast error")

    # HRRR via Open-Meteo: match by local date+hour (fixes 0 W/m² from wrong index)
    try:
        r = safe_get(OM_API_URL, timeout=5)
        if r.status_code == 200:
            hrrr = r.json()
            h = hrrr.get("hourly", {})
            times = h.get("time", [])

            if times:
                now_local = get_miami_time().replace(minute=0, second=0, microsecond=0)
                target_date = now_local.date()
                target_hour = now_local.hour

                best_i = None
                for i, t in enumerate(times):
                    dt = parse_iso_time(t)
                    if dt is None:
                        continue
                    # Open-Meteo with timezone=America/New_York often returns naive local times
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=TZ_MIAMI)
                    dt_local = dt.astimezone(TZ_MIAMI)
                    if dt_local.date() == target_date and dt_local.hour == target_hour:
                        best_i = i
                        break

                # Fallback: nearest
                if best_i is None:
                    best_i = 0
                    best_delta = None
                    for i, t in enumerate(times):
                        dt = parse_iso_time(t)
                        if dt is None:
                            continue
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=TZ_MIAMI)
                        dt_local = dt.astimezone(TZ_MIAMI)
                        delta = abs((dt_local - now_local).total_seconds())
                        if best_delta is None or delta < best_delta:
                            best_delta = delta
                            best_i = i

                def at(arr, idx):
                    if not isinstance(arr, list) or idx >= len(arr):
                        return None
                    return arr[idx]

                data["hrrr_now"] = {
                    "rad": at(h.get("shortwave_radiation", []), best_i),
                    "precip": at(h.get("precipitation_probability", []), best_i),
                    "temp_c": at(h.get("temperature_2m", []), best_i),
                    "cloud": at(h.get("cloud_cover", []), best_i),
                    "time": times[best_i],
                    "index": best_i,
                }
    except Exception:
        logger.exception("HRRR(Open-Meteo) fetch error")
        data["status"].append("HRRR(Open-Meteo) unavailable")

    # TAF (AWC)
    try:
        r = safe_get(AWC_TAF_URL, timeout=5)
        if r.status_code == 200:
            lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
            data["taf_raw"] = "\n".join(lines[:8]) if lines else None
    except Exception:
        logger.exception("TAF fetch error")
        data["status"].append("TAF unavailable")

    return data


# =============================================================================
# LIVE OBSERVATIONS (NWS station obs + METAR), with dedupe
# =============================================================================
def _metar_month_rollover(dt: datetime, now_utc: datetime) -> datetime:
    # If parsed METAR timestamp looks in the future (month boundary), roll back a month.
    if dt > now_utc + timedelta(hours=2):
        first = datetime(now_utc.year, now_utc.month, 1, tzinfo=timezone.utc)
        prev_last = first - timedelta(days=1)
        day = min(dt.day, prev_last.day)
        return datetime(prev_last.year, prev_last.month, day, dt.hour, dt.minute, tzinfo=timezone.utc)
    return dt


def _bucket_key(dt: datetime, minutes: int = 5) -> Tuple[int, int, int, int, int]:
    dt = dt.astimezone(timezone.utc)
    bucket_min = (dt.minute // minutes) * minutes
    return (dt.year, dt.month, dt.day, dt.hour, bucket_min)


def _source_priority(src: str) -> int:
    # Prefer METAR over NWS when deduping
    return {"AWC": 0, "NWS": 1}.get(src, 9)


def merge_and_dedupe(observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[int, int, int, int, int], Dict[str, Any]] = {}
    for o in observations:
        k = _bucket_key(o["dt_utc"], minutes=5)
        cur = buckets.get(k)
        if cur is None:
            buckets[k] = o
            continue

        if _source_priority(o["Source"]) < _source_priority(cur["Source"]):
            buckets[k] = o
        elif _source_priority(o["Source"]) == _source_priority(cur["Source"]):
            if o["dt_utc"] > cur["dt_utc"]:
                buckets[k] = o

    merged = list(buckets.values())
    merged.sort(key=lambda x: x["dt_utc"], reverse=True)
    return merged


def fetch_live_history() -> Tuple[List[Dict[str, Any]], List[str]]:
    data_list: List[Dict[str, Any]] = []
    status: List[str] = []

    # --- NWS station observations ---
    try:
        r = safe_get(NWS_API_HISTORY, timeout=5)
        if r.status_code == 200:
            for item in r.json().get("features", []):
                p = item.get("properties", {})

                t_c = (p.get("temperature") or {}).get("value")
                ts = p.get("timestamp")
                dt = parse_iso_time(ts) if ts else None
                if t_c is None or dt is None:
                    continue

                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                dt_utc = dt.astimezone(timezone.utc)

                # Wind (NWS windSpeed usually m/s)
                wdir = (p.get("windDirection") or {}).get("value")
                wspd = (p.get("windSpeed") or {}).get("value")
                w_str = "--"
                w_val = -1
                if wdir is not None:
                    w_val = int(round(float(wdir)))
                if wdir is not None and wspd is not None:
                    try:
                        kt = mps_to_kt(float(wspd))
                        w_str = f"{int(round(float(wdir))):03d} @ {int(round(kt))}kt"
                    except Exception:
                        w_str = "--"

                clouds = p.get("cloudLayers") or []
                sky_str = clouds[0].get("amount", "--") if clouds else "--"

                hum = (p.get("relativeHumidity") or {}).get("value")
                hum = float(hum) if hum is not None else 0.0

                dew_c = (p.get("dewpoint") or {}).get("value")
                dew_f = c_to_f(float(dew_c)) if dew_c is not None else 0.0

                press_pa = (p.get("barometricPressure") or {}).get("value")
                press_in = (float(press_pa) * 0.0002953) if press_pa is not None else 0.0

                temp_f = c_to_f(float(t_c))

                data_list.append(
                    {
                        "dt_utc": dt_utc,
                        "Source": "NWS",
                        "Temp": temp_f,
                        "Wind": w_str,
                        "Sky": sky_str,
                        "Wx": None,
                        "WindVal": w_val,
                        "Hum": hum,
                        "Dew": dew_f,
                        "Press": press_in,
                    }
                )
        else:
            status.append("NWS observations unavailable")
    except Exception:
        logger.exception("NWS observations error")
        status.append("NWS observations error")

    # --- AWC METAR ---
    try:
        r = safe_get(AWC_METAR_URL, timeout=5)
        if r.status_code == 200:
            for line in r.text.splitlines():
                line = line.strip()
                if not line or not line.startswith("KMIA"):
                    continue

                tm_m = re.search(r"\b(\d{2})(\d{4})Z\b", line)
                tp_m = re.search(r"\b(M?\d{2})/(M?\d{2})\b", line)
                pr_m = re.search(r"\bA(\d{4})\b", line)
                wind_m = re.search(r"\b(\d{3}|VRB)(\d{2,3})(G(\d{2,3}))?KT\b", line)

                if not (tm_m and tp_m):
                    continue

                day = int(tm_m.group(1))
                hhmm = tm_m.group(2)
                now_utc = datetime.now(timezone.utc)
                dt = datetime(now_utc.year, now_utc.month, day, int(hhmm[:2]), int(hhmm[2:]), tzinfo=timezone.utc)
                dt = _metar_month_rollover(dt, now_utc)

                tc = int(tp_m.group(1).replace("M", "-"))
                dc = int(tp_m.group(2).replace("M", "-"))

                press_in = int(pr_m.group(1)) / 100.0 if pr_m else 0.0

                w_str = "--"
                w_val = -1
                if wind_m:
                    wd = wind_m.group(1)
                    ws = wind_m.group(2)
                    w_str = f"{wd} @ {ws}kt"
                    if wd.isdigit():
                        w_val = int(wd)

                # Cloud amount (simple)
                sky = "CLR"
                if "OVC" in line:
                    sky = "OVC"
                elif "BKN" in line:
                    sky = "BKN"
                elif "SCT" in line:
                    sky = "SCT"
                elif "FEW" in line:
                    sky = "FEW"

                # Simple wx token detection
                wx = None
                if re.search(r"\bTS\b", line):
                    wx = "TS"
                elif re.search(r"\b(SH)?RA\b", line):
                    wx = "RA"
                elif re.search(r"\bFG\b|\bBR\b|\bHZ\b", line):
                    wx = "FG"

                try:
                    hum = 100.0 * (
                        math.exp((17.625 * dc) / (243.04 + dc))
                        / math.exp((17.625 * tc) / (243.04 + tc))
                    )
                except Exception:
                    hum = 0.0

                temp_f = c_to_f(float(tc))
                dew_f = c_to_f(float(dc))

                data_list.append(
                    {
                        "dt_utc": dt,
                        "Source": "AWC",
                        "Temp": temp_f,
                        "Wind": w_str,
                        "Sky": sky,
                        "Wx": wx,
                        "WindVal": w_val,
                        "Hum": float(hum),
                        "Dew": dew_f,
                        "Press": press_in,
                    }
                )
        else:
            status.append("AWC METAR unavailable")
    except Exception:
        logger.exception("AWC METAR error")
        status.append("AWC METAR error")

    merged = merge_and_dedupe(data_list)
    return merged, status


# =============================================================================
# TREND + "AGENT"
# =============================================================================
def calculate_smart_trend(master_list: List[Dict[str, Any]]) -> float:
    if len(master_list) < 2:
        return 0.0

    now = master_list[0]["dt_utc"]
    one_hr_ago = now - timedelta(hours=1)
    points = [p for p in master_list if p["dt_utc"] >= one_hr_ago]
    if len(points) < 2:
        return 0.0

    x = [(p["dt_utc"] - one_hr_ago).total_seconds() / 60.0 for p in points]  # minutes
    y = [p["Temp"] for p in points]

    N = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(i * j for i, j in zip(x, y))
    sum_xx = sum(i * i for i in x)
    den = (N * sum_xx - sum_x * sum_x)
    if den == 0:
        return 0.0

    slope_per_min = (N * sum_xy - sum_x * sum_y) / den
    return slope_per_min * 60.0  # °F/hr


def get_agent_analysis(
    trend_f_per_hr: float,
    wind_dir: int,
    dew_f: float,
    temp_f: float,
    rad_watts: Optional[float],
    precip_prob: Optional[float],
    current_hour: int,
) -> Tuple[str, str, int]:
    reasons: List[str] = []
    confidence = 50
    sentiment = "NEUTRAL"

    # Diurnal
    if current_hour < 11:
        reasons.append("Morning ramp")
        confidence += 20
    elif current_hour < 15:
        reasons.append("Peak heating window")
        confidence += 10
    else:
        reasons.append("Late day")
        confidence -= 15

    # Solar / precip
    if rad_watts is not None:
        if rad_watts >= 700:
            reasons.append("Strong sun")
            confidence += 15
        elif rad_watts <= 200 and current_hour < 17:
            reasons.append("Weak sun")
            confidence -= 15
    else:
        reasons.append("Solar unavailable")

    if precip_prob is not None and precip_prob >= 50:
        reasons.append("Rain risk")
        confidence -= 10

    # Moisture
    dew_dep = temp_f - dew_f
    if dew_dep < 3:
        reasons.append("Near-saturated air")
        sentiment = "TRAP"
        confidence = min(confidence, 25)
    elif dew_dep > 12:
        reasons.append("Drier air")
        confidence += 8

    # Wind
    if 0 <= wind_dir <= 180:
        reasons.append("Ocean breeze bias")
        confidence -= 8
    elif wind_dir > 180:
        reasons.append("Land breeze bias")
        confidence += 5

    # Trend
    if trend_f_per_hr > 0.7:
        reasons.append("Fast rise")
        confidence += 12
    elif trend_f_per_hr > 0.2:
        reasons.append("Rising")
        confidence += 6
    elif trend_f_per_hr < -0.5 and current_hour < 17:
        reasons.append("Dropping")
        confidence -= 12

    confidence = max(0, min(100, int(round(confidence))))
    if sentiment == "NEUTRAL":
        if confidence >= 70:
            sentiment = "BULLISH"
        elif confidence <= 35:
            sentiment = "BEARISH"

    return sentiment, " + ".join(reasons), confidence


# =============================================================================
# AUTO-REFRESH (no time.sleep)
# =============================================================================
def render_autorefresh(interval_seconds: int) -> None:
    interval_ms = max(1, int(interval_seconds)) * 1000
    components.html(
        """
        <script>
        setTimeout(function() {
            window.location.reload();
        }, %d);
        </script>
        """ % interval_ms,
        height=0,
    )


# =============================================================================
# UI RENDERING
# =============================================================================
def render_live_dashboard(target_temp: float, bracket_label: str, live_price: int, bracket_cap: Optional[float]) -> None:
    st.title("🔴 Project Helios: Live Feed")

    if st.button("🔄 Refresh System", type="primary"):
        st.cache_data.clear()
        st.rerun()

    history, obs_status = fetch_live_history()
    f_data = fetch_forecast_data()

    if obs_status or (f_data.get("status") or []):
        with st.expander("Data status / warnings", expanded=False):
            for s in obs_status:
                st.warning(s)
            for s in (f_data.get("status") or []):
                st.warning(s)

    if not history:
        st.error("No live observations available right now.")
        return

    latest = history[0]
    now_miami = get_miami_time()

    # Day high
    today_recs = [x for x in history if x["dt_utc"].astimezone(TZ_MIAMI).date() == now_miami.date()]
    high_mark = max(today_recs, key=lambda x: x["Temp"]) if today_recs else latest
    high_round = int(round(high_mark["Temp"]))

    # HRRR now
    hrrr_rad = None
    hrrr_precip = None
    if f_data.get("hrrr_now"):
        try:
            val = f_data["hrrr_now"].get("rad")
            hrrr_rad = None if val is None else float(val)
        except Exception:
            hrrr_rad = None
        try:
            valp = f_data["hrrr_now"].get("precip")
            hrrr_precip = None if valp is None else float(valp)
        except Exception:
            hrrr_precip = None

    smart_trend = calculate_smart_trend(history)
    safe_trend = smart_trend
    if now_miami.hour > 17 and safe_trend < -0.5:
        safe_trend = -0.5

    ai_sent, ai_reason, ai_conf = get_agent_analysis(
        trend_f_per_hr=safe_trend,
        wind_dir=latest["WindVal"],
        dew_f=latest["Dew"],
        temp_f=latest["Temp"],
        rad_watts=hrrr_rad,
        precip_prob=hrrr_precip,
        current_hour=now_miami.hour,
    )

    # Bust / ITM logic
    ref_msg = None
    if bracket_cap is not None and high_round > bracket_cap:
        ai_conf = 0
        ai_sent = "DEAD"
        ref_msg = "💀 BUSTED: Day high %d° > cap %d°" % (high_round, int(round(bracket_cap)))
    elif high_round >= target_temp and ai_sent == "BULLISH":
        ai_conf = max(0, ai_conf - 40)
        ref_msg = "⚠️ ITM already, but conditions still support further heating."

    # Forecast high (NWS daytime period)
    forecast_high = high_round
    if f_data.get("today_daily"):
        nws_high = f_data["today_daily"].get("temperature")
        if isinstance(nws_high, (int, float)):
            forecast_high = max(forecast_high, int(nws_high))

    st.markdown(HIDE_INDEX_CSS, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Temp",
        "%.2f°F" % latest["Temp"],
        "Feels %.0f°" % calculate_heat_index(latest["Temp"], latest["Hum"]),
    )
    c2.metric("Proj. High", "%d°F" % forecast_high, "NWS daytime", delta_color="off")
    c3.metric("Day High", "%.2f°F" % high_mark["Temp"], "Rounded %d°F" % high_round, delta_color="off")
    c4.metric("Solar (HRRR)", "—" if hrrr_rad is None else "%d W/m²" % int(round(hrrr_rad)))

    st.markdown("---")

    m1, m2 = st.columns([2, 1])
    color = "grey"
    if ai_sent == "BULLISH":
        color = "green"
    elif ai_sent == "BEARISH":
        color = "red"
    elif ai_sent == "TRAP":
        color = "orange"
    elif ai_sent == "DEAD":
        color = "grey"

    m1.info("🤖 **PHYSICS:** :%s[**%s**] (%d%%)\n\n%s" % (color, ai_sent, ai_conf, (ref_msg if ref_msg else ai_reason)))

    edge = ai_conf - (live_price or 0)
    m2.metric("Kalshi (%s)" % bracket_label, "%d¢" % live_price, "%+d%% Edge" % int(round(edge)), delta_color="off")

    # Next 3 hours projection (simple)
    next_3_hours = []
    current_utc = datetime.now(timezone.utc)
    for p in f_data.get("all_hourly", []):
        p_dt = parse_iso_time(p.get("startTime"))
        if p_dt and p_dt.tzinfo is None:
            p_dt = p_dt.replace(tzinfo=timezone.utc)
        if p_dt and p_dt.astimezone(timezone.utc) > current_utc:
            next_3_hours.append(p)
            if len(next_3_hours) >= 3:
                break

    if len(next_3_hours) < 3:
        proj_str = "⚠️ Forecast Unavailable"
    else:
        proj_vals = []
        curr_temp = latest["Temp"]
        for i, f in enumerate(next_3_hours):
            try:
                nws_temp = float(f.get("temperature"))
            except Exception:
                nws_temp = curr_temp

            trend_weight = 0.45 / (i + 1)
            model_weight = 1.0 - trend_weight
            raw_proj = (curr_temp + (safe_trend * (i + 1))) * trend_weight + (nws_temp * model_weight)

            if 0 <= latest["WindVal"] <= 180:
                raw_proj -= 0.4 * (i + 1)

            icon = "☁️"
            sf = f.get("shortForecast", "") or ""
            if "Rain" in sf or "Thunder" in sf:
                icon = "🌧️"
            elif "Sunny" in sf or "Clear" in sf:
                icon = "☀️"

            proj_vals.append("**+%dh:** %.1f°F %s" % (i + 1, raw_proj, icon))
        proj_str = " | ".join(proj_vals)

    st.success("**🔮 AI PROJECTION:** %s" % proj_str)

    # TAF display
    if f_data.get("taf_raw"):
        st.caption("TAF (AWC)")
        st.code(f_data["taf_raw"], language="text")

    # Markets buttons
    st.subheader("🎯 Select Bracket (Live Markets)")
    markets, m_status = fetch_market_data()
    if m_status != "🟢 Live":
        st.warning(m_status)

    if markets:
        cols = st.columns(len(markets))
        for i, m in enumerate(markets):
            label = "%s\n(%d¢)" % (m["label"], m["price"])
            is_selected = (not math.isnan(m["strike"])) and (float(m["strike"]) == float(target_temp))
            if cols[i].button(label, key="mkt_%d" % i, type="primary" if is_selected else "secondary"):
                st.query_params["target"] = str(m["strike"])
                st.rerun()
    else:
        st.info("No markets returned for today (ticker may be unavailable yet).")

    # Sensor log (requested columns)
    st.subheader("Sensor Log (Miami Time)")
    clean_rows = []
    for i, row in enumerate(history[:20]):
        vel = None
        vel_arrow = "—"
        if i < len(history) - 1:
            dt1, dt2 = row["dt_utc"], history[i + 1]["dt_utc"]
            diff_hr = (dt1 - dt2).total_seconds() / 3600.0
            if diff_hr > 0:
                vel = (row["Temp"] - history[i + 1]["Temp"]) / diff_hr
                if vel > 0.5:
                    vel_arrow = "⬆️"
                elif vel > 0.1:
                    vel_arrow = "↗️"
                elif vel < -0.5:
                    vel_arrow = "⬇️"
                elif vel < -0.1:
                    vel_arrow = "↘️"

        vel_str = "—" if vel is None else "%+.2f°F/hr %s" % (vel, vel_arrow)
        sky = row.get("Sky", "--")
        wx = row.get("Wx")
        cond = "%s %s" % (condition_icon(sky, wx), sky)
        hi = calculate_heat_index(row["Temp"], row["Hum"])

        clean_rows.append(
            {
                "Time": get_display_time(row["dt_utc"]),
                "Source": row["Source"],
                "Condition": cond,
                "Temp": "%.2f°F" % row["Temp"],
                "Heat Index": "%.1f°F" % hi,
                "Dew Point": "%.1f°F" % row["Dew"],
                "Humidity": "%d%%" % int(round(row["Hum"])),
                "Pressure": "%.2f inHg" % row["Press"],
                "Wind": row["Wind"],
                "Velocity": vel_str,
            }
        )

    st.table(pd.DataFrame(clean_rows))

    # Debug panel (helpful if brackets or solar still odd)
    with st.expander("Debug", expanded=False):
        st.write("Selected target:", target_temp)
        if f_data.get("hrrr_now"):
            st.json(f_data["hrrr_now"])
        if markets:
            st.write("Kalshi markets (raw strikes):")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "label": m["label"],
                            "strike": m["strike"],
                            "price": m["price"],
                            "floor_raw": m.get("floor_raw"),
                            "cap_raw": m.get("cap_raw"),
                            "cap_for_bust": m.get("cap"),
                        }
                        for m in markets
                    ]
                )
            )


def render_forecast_generic(daily: Optional[Dict[str, Any]], hourly: List[Dict[str, Any]], date_label: str) -> None:
    st.title("Forecast: %s" % date_label)

    if st.button("🔄 Refresh Forecast"):
        st.cache_data.clear()
        st.rerun()

    if daily:
        st.success(daily.get("detailedForecast", ""))
        st.caption("NWS Period: %s, Temp: %s°" % (daily.get("name", ""), daily.get("temperature", "")))
    else:
        st.warning("Daily forecast not available for this view.")

    if hourly:
        h_data = []
        for h in hourly[:24]:
            dt = parse_iso_time(h.get("startTime"))
            if not dt:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt_local = dt.astimezone(TZ_MIAMI)
            h_data.append(
                {
                    "Time": dt_local.strftime("%a %I %p"),
                    "Temp": h.get("temperature"),
                    "Cond": h.get("shortForecast"),
                    "Wind": "%s %s" % (h.get("windDirection", ""), h.get("windSpeed", "")),
                }
            )
        st.table(pd.DataFrame(h_data))
    else:
        st.warning("⚠️ Hourly forecast data temporarily unavailable from NWS.")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    if "target" not in st.query_params:
        st.query_params["target"] = "81.0"

    try:
        tgt = float(st.query_params["target"])
    except Exception:
        tgt = 81.0
        st.query_params["target"] = "81.0"

    view_mode = st.sidebar.radio("Deck:", ["Live Monitor", "Today's Forecast", "Tomorrow's Forecast"])
    st.sidebar.divider()

    auto = st.sidebar.checkbox("⚡ Auto-Refresh (10s)", value=True)
    if auto:
        render_autorefresh(10)

    # Resolve selected market label/price/cap
    markets, _ = fetch_market_data()
    lbl, price, cap = "Target", 0, None
    for m in markets:
        if (not math.isnan(m["strike"])) and float(m["strike"]) == float(tgt):
            lbl = m["label"]
            price = m["price"]
            cap = m.get("cap")  # integer cap for bust checks (None for open-high)
            break

    st.markdown(HIDE_INDEX_CSS, unsafe_allow_html=True)

    if view_mode == "Live Monitor":
        render_live_dashboard(tgt, lbl, price, cap)
    else:
        f_data = fetch_forecast_data()
        now = get_miami_time()

        if view_mode == "Today's Forecast":
            render_forecast_generic(
                daily=f_data.get("today_daily"),
                hourly=f_data.get("all_hourly", []),
                date_label=now.strftime("%A"),
            )
        else:
            tomorrow = (now + timedelta(days=1)).date()
            hourly = []
            for h in (f_data.get("all_hourly") or []):
                dt = parse_iso_time(h.get("startTime"))
                if not dt:
                    continue
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt.astimezone(TZ_MIAMI).date() == tomorrow:
                    hourly.append(h)

            render_forecast_generic(
                daily=f_data.get("tomorrow_daily"),
                hourly=hourly if hourly else (f_data.get("all_hourly") or []),
                date_label=(now + timedelta(days=1)).strftime("%A"),
            )


if __name__ == "__main__":
    st.sidebar.caption("Miami time: %s" % get_miami_time().strftime("%Y-%m-%d %I:%M:%S %p"))
    main()
