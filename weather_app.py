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

# Sunrise/Sunset calculation
from astral import LocationInfo
from astral.sun import sun

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
    "https://api.open-meteo.com/v1/gfs"
    "?latitude=25.7954&longitude=-80.2901"
    "&hourly=temperature_2m,precipitation_probability,shortwave_radiation,cloud_cover"
    "&timezone=America%2FNew_York&forecast_days=2"
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
    s.headers.update(
        {
            "User-Agent": "ProjectHelios/1.6 (Streamlit app)",
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


def format_ordinal_day(dt_obj: datetime) -> str:
    day = dt_obj.day
    if 11 <= day <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{dt_obj.strftime('%A')} {day}{suffix}"


@st.cache_data(ttl=3600)
def get_sun_times(local_date) -> Tuple[datetime, datetime]:
    city = LocationInfo(name="Miami", region="USA", timezone="America/New_York", latitude=25.7954, longitude=-80.2901)
    s = sun(city.observer, date=local_date, tzinfo=TZ_MIAMI)
    return s["sunrise"], s["sunset"]


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


def is_nan(x: Any) -> bool:
    try:
        return math.isnan(float(x))
    except Exception:
        return False


def condition_icon_from_sky_wx(sky: str, wx: Optional[str] = None) -> str:
    s = (sky or "").upper()
    w = (wx or "").upper()

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


def icon_from_short_forecast(text: str) -> str:
    t = (text or "").lower()
    # This function will be wrapped for day/night context
    if "thunder" in t or "t-storm" in t or "storm" in t:
        return "⛈️"
    if "rain" in t or "showers" in t:
        return "🌧️"
    if "fog" in t or "haze" in t or "mist" in t:
        return "🌫️"
    if "snow" in t or "sleet" in t:
        return "🌨️"
    if "sunny" in t or "clear" in t:
        return "☀️"  # Will be replaced with moon at night
    if "partly" in t and "cloud" in t:
        return "⛅"
    if "cloud" in t or "overcast" in t:
        return "☁️"
    return "☁️"


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
# KALSHI MARKET FETCHER (GENERIC)
# =============================================================================
def kalshi_label_and_strike(floor_raw: Optional[float], cap_raw: Optional[float]) -> Tuple[str, float, Optional[float]]:
    """
    Generic labeling rules matching Kalshi UI:
      - floor=None, cap=X      -> (X-1) or below
      - floor=A, cap=B         -> A to B
      - floor=X, cap=None      -> (X+1) or above

    strike_for_ui:
      - open-low: cap-1
      - middle: cap  (so selecting 83-84 uses target=84.0)
      - open-high: floor+1
    """
    if cap_raw is not None and floor_raw is None:
        cap_i = int(round(float(cap_raw)))
        label = f"{cap_i - 1}° or below"
        strike = float(cap_i - 1)
        cap_for_bust = float(cap_i - 1)
        return label, strike, cap_for_bust

    if floor_raw is not None and cap_raw is not None:
        a = int(round(float(floor_raw)))
        b = int(round(float(cap_raw)))
        label = f"{a}° - {b}°"
        strike = float(b)
        cap_for_bust = float(b)
        return label, strike, cap_for_bust

    if floor_raw is not None and cap_raw is None:
        a = int(round(float(floor_raw))) + 1
        label = f"{a}° or above"
        strike = float(a)
        cap_for_bust = None
        return label, strike, cap_for_bust

    return "Unknown", float("nan"), None


@st.cache_data(ttl=5)
def fetch_market_data() -> Tuple[List[Dict[str, Any]], str]:
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

            label, strike, cap_for_bust = kalshi_label_and_strike(floor_f, cap_f)

            if floor_f is None and cap_f is not None:
                sort_key = -1_000_000 + (cap_f - 0.5)
            elif floor_f is not None and cap_f is not None:
                sort_key = strike
            else:
                sort_key = 1_000_000 + strike

            parsed.append(
                {
                    "label": label,
                    "strike": strike,
                    "price": ask,
                    "floor_raw": floor_f,
                    "cap_raw": cap_f,
                    "cap_for_bust": cap_for_bust,
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
def _nearest_index_for_local_hour(times: List[str], target_local: datetime) -> Optional[int]:
    best_i = None
    best_delta = None
    for i, t in enumerate(times):
        dt = parse_iso_time(t)
        if dt is None:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TZ_MIAMI)
        dt_local = dt.astimezone(TZ_MIAMI)
        delta = abs((dt_local - target_local).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_i = i
    return best_i


def _hrrr_peak_for_date(h: Dict[str, Any], times: List[str], target_date) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (peak_rad, peak_precip) for 09-16 local on the given date.
    """
    rad_arr = h.get("shortwave_radiation", []) or []
    precip_arr = h.get("precipitation_probability", []) or []

    peak_rad = None
    peak_precip = None

    for i, t in enumerate(times):
        dt = parse_iso_time(t)
        if dt is None:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TZ_MIAMI)
        dt_local = dt.astimezone(TZ_MIAMI)

        if dt_local.date() != target_date:
            continue
        if not (9 <= dt_local.hour <= 16):
            continue

        if i < len(rad_arr):
            try:
                v = float(rad_arr[i])
                if peak_rad is None or v > peak_rad:
                    peak_rad = v
            except Exception:
                pass

        if i < len(precip_arr):
            try:
                v = float(precip_arr[i])
                if peak_precip is None or v > peak_precip:
                    peak_precip = v
            except Exception:
                pass

    return peak_rad, peak_precip


@st.cache_data(ttl=300)
def fetch_forecast_data() -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "today_daily": None,
        "tomorrow_daily": None,
        "all_hourly": [],
        "hrrr_now": None,
        "hrrr_today_peak": None,
        "hrrr_tomorrow_peak": None,
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

    # HRRR via Open-Meteo (GFS/HRRR endpoint)
    try:
        r = safe_get(OM_API_URL, timeout=5)
        if r.status_code == 200:
            hrrr = r.json()
            h = hrrr.get("hourly", {})
            times = h.get("time", []) or []
            if times:
                now_local = get_miami_time()
                best_i = _nearest_index_for_local_hour(times, now_local)

                def at(arr, idx):
                    if not isinstance(arr, list) or idx is None or idx < 0 or idx >= len(arr):
                        return None
                    return arr[idx]

                rad_arr = h.get("shortwave_radiation", []) or []
                precip_arr = h.get("precipitation_probability", []) or []
                temp_arr = h.get("temperature_2m", []) or []
                cloud_arr = h.get("cloud_cover", []) or []

                rad = at(rad_arr, best_i)
                precip = at(precip_arr, best_i)
                temp_c = at(temp_arr, best_i)
                cloud = at(cloud_arr, best_i)

                # If rad looks too low during daytime, take max in ±2h window
                try:
                    if best_i is not None and 9 <= now_local.hour <= 16:
                        candidates = []
                        for j in range(best_i - 2, best_i + 3):
                            v = at(rad_arr, j)
                            if v is not None:
                                candidates.append(float(v))
                        if candidates:
                            rad = max(candidates)
                except Exception:
                    pass

                data["hrrr_now"] = {
                    "rad": rad,
                    "precip": precip,
                    "temp_c": temp_c,
                    "cloud": cloud,
                    "time": times[best_i] if best_i is not None else None,
                    "index": best_i,
                }

                # Peak solar/precip for today & tomorrow (09-16)
                today = now_local.date()
                tomorrow = (now_local + timedelta(days=1)).date()
                peak_rad_today, peak_pp_today = _hrrr_peak_for_date(h, times, today)
                peak_rad_tmr, peak_pp_tmr = _hrrr_peak_for_date(h, times, tomorrow)

                data["hrrr_today_peak"] = {"rad": peak_rad_today, "precip": peak_pp_today}
                data["hrrr_tomorrow_peak"] = {"rad": peak_rad_tmr, "precip": peak_pp_tmr}
        else:
            data["status"].append("HRRR(Open-Meteo) unavailable")

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
                if not line or " KMIA " not in f" {line} ":
                    continue

                tm_m = re.search(r"\b(\d{2})(\d{4})Z\b", line)
                tp_m = re.search(r"\b(M?\d{2})/(M?\d{2})\b", line)
                tg_m = re.search(r"\bT([01])(\d{3})([01])(\d{3})\b", line)
                pr_m = re.search(r"\bA(\d{4})\b", line)
                wind_m = re.search(r"\b(\d{3}|VRB)(\d{2,3})(G(\d{2,3}))?KT\b", line)

                if not (tm_m and (tp_m or tg_m)):
                    continue

                day = int(tm_m.group(1))
                hhmm = tm_m.group(2)
                now_utc = datetime.now(timezone.utc)
                dt = datetime(now_utc.year, now_utc.month, day, int(hhmm[:2]), int(hhmm[2:]), tzinfo=timezone.utc)
                dt = _metar_month_rollover(dt, now_utc)

                tc: Optional[float] = None
                dc: Optional[float] = None
                if tg_m:
                    t_sign = -1 if tg_m.group(1) == "1" else 1
                    d_sign = -1 if tg_m.group(3) == "1" else 1
                    tc = t_sign * (int(tg_m.group(2)) / 10.0)
                    dc = d_sign * (int(tg_m.group(4)) / 10.0)
                elif tp_m:
                    tc = float(int(tp_m.group(1).replace("M", "-")))
                    dc = float(int(tp_m.group(2).replace("M", "-")))

                press_in = int(pr_m.group(1)) / 100.0 if pr_m else 0.0

                w_str = "--"
                w_val = -1
                if wind_m:
                    wd = wind_m.group(1)
                    ws = wind_m.group(2)
                    w_str = f"{wd} @ {ws}kt"
                    if wd.isdigit():
                        w_val = int(wd)

                sky = "CLR"
                if "OVC" in line:
                    sky = "OVC"
                elif "BKN" in line:
                    sky = "BKN"
                elif "SCT" in line:
                    sky = "SCT"
                elif "FEW" in line:
                    sky = "FEW"

                wx = None
                if re.search(r"\bTS\b", line):
                    wx = "TS"
                elif re.search(r"\b(SH)?RA\b", line):
                    wx = "RA"
                elif re.search(r"\bFG\b|\bBR\b|\bHZ\b", line):
                    wx = "FG"

                hum = 0.0
                temp_f = 0.0
                dew_f = 0.0
                if tc is not None and dc is not None:
                    try:
                        hum = 100.0 * (
                            math.exp((17.625 * dc) / (243.04 + dc))
                            / math.exp((17.625 * tc) / (243.04 + tc))
                        )
                    except Exception:
                        hum = 0.0

                    temp_f = c_to_f(tc)
                    dew_f = c_to_f(dc)

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
                        "Raw": line,
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
# TREND + VELOCITY (SMOOTHED)
# =============================================================================
def linear_regression_slope_f_per_hr(points: List[Dict[str, Any]], anchor_dt: datetime) -> Optional[float]:
    """
    Slope in °F/hr using linear regression over points.
    x = minutes from anchor_dt (older points have negative x if anchor is newest)
    """
    if len(points) < 3:
        return None

    x = []
    y = []
    for p in points:
        dt = p["dt_utc"]
        minutes = (dt - anchor_dt).total_seconds() / 60.0
        x.append(minutes)
        y.append(p["Temp"])

    N = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(i * j for i, j in zip(x, y))
    sum_xx = sum(i * i for i in x)
    den = (N * sum_xx - sum_x * sum_x)
    if den == 0:
        return None

    slope_per_min = (N * sum_xy - sum_x * sum_y) / den
    return slope_per_min * 60.0


def compute_smoothed_velocity(history: List[Dict[str, Any]], i: int, window_minutes: int = 30) -> Optional[float]:
    """
    Use a regression over ~window_minutes starting at row i and going older.
    This avoids insane spikes when consecutive points are only 2–5 minutes apart.
    """
    if i >= len(history):
        return None
    anchor = history[i]["dt_utc"]
    window = []
    for j in range(i, len(history)):
        if (anchor - history[j]["dt_utc"]).total_seconds() / 60.0 <= window_minutes:
            window.append(history[j])
        else:
            break

    slope = linear_regression_slope_f_per_hr(window, anchor_dt=anchor)
    if slope is not None:
        return slope

    # Fallback: find a pair at least 10 minutes apart
    for j in range(i + 1, len(history)):
        dt2 = history[j]["dt_utc"]
        diff_hr = (anchor - dt2).total_seconds() / 3600.0
        if diff_hr >= (10 / 60):
            return (history[i]["Temp"] - history[j]["Temp"]) / diff_hr
    return None


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
    is_daylight: bool,
) -> Tuple[str, str, int]:
    reasons: List[str] = []
    confidence = 50
    sentiment = "NEUTRAL"

    if current_hour < 11:
        reasons.append("Morning ramp")
        confidence += 20
    elif current_hour < 15:
        reasons.append("Peak heating window")
        confidence += 10
    else:
        reasons.append("Late day")
        confidence -= 15

    if not is_daylight:
        reasons.append("Nighttime cooling")
        confidence -= 20

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

    dew_dep = temp_f - dew_f
    if dew_dep < 3:
        reasons.append("Near-saturated air")
        sentiment = "TRAP"
        confidence = min(confidence, 25)
    elif dew_dep > 12:
        reasons.append("Drier air")
        confidence += 8

    if 0 <= wind_dir <= 180:
        reasons.append("Ocean breeze bias")
        confidence -= 8
    elif wind_dir > 180:
        reasons.append("Land breeze bias")
        confidence += 5

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
# FORECAST CONFIDENCE (for forecast pages)
# =============================================================================
def forecast_trade_confidence(
    daily_text: str,
    peak_rad: Optional[float],
    peak_precip: Optional[float],
    is_daylight_period: Optional[bool],
) -> Tuple[str, int, str]:
    """
    Simple day-ahead "trade confidence" score using:
      - daily forecast text keywords
      - HRRR peak solar (09-16)
      - HRRR peak precip (09-16)
    """
    text = (daily_text or "").lower()
    reasons = []
    score = 55

    if any(k in text for k in ["sunny", "clear"]):
        score += 15
        reasons.append("Sunny/clear signal")
    if "mostly sunny" in text:
        score += 8
        reasons.append("Mostly sunny")
    if any(k in text for k in ["cloudy", "overcast"]):
        score -= 10
        reasons.append("Cloud cover risk")
    if any(k in text for k in ["rain", "showers"]):
        score -= 15
        reasons.append("Rain risk")
    if any(k in text for k in ["thunder", "t-storm", "storm"]):
        score -= 20
        reasons.append("Thunder risk")

    if is_daylight_period is False:
        score -= 10
        reasons.append("Nighttime cooling")

    if peak_rad is not None:
        if peak_rad >= 650:
            score += 10
            reasons.append(f"Strong peak solar ({int(peak_rad)}W)")
        elif peak_rad <= 250:
            score -= 10
            reasons.append(f"Weak peak solar ({int(peak_rad)}W)")
    else:
        reasons.append("HRRR solar unavailable")

    if peak_precip is not None:
        if peak_precip >= 70:
            score -= 20
            reasons.append(f"High precip chance ({int(peak_precip)}%)")
        elif peak_precip >= 50:
            score -= 12
            reasons.append(f"Elevated precip chance ({int(peak_precip)}%)")

    score = max(0, min(100, int(round(score))))
    sentiment = "NEUTRAL"
    if score >= 70:
        sentiment = "BULLISH"
    elif score <= 35:
        sentiment = "BEARISH"
    if any(k in text for k in ["thunder", "t-storm"]) and score < 50:
        sentiment = "TRAP"

    return sentiment, score, " + ".join(reasons[:5]) if reasons else "—"


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

    today_recs = [x for x in history if x["dt_utc"].astimezone(TZ_MIAMI).date() == now_miami.date()]
    high_mark = max(today_recs, key=lambda x: x["Temp"]) if today_recs else latest
    high_round = int(round(high_mark["Temp"]))

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

    sunrise_now, sunset_now = get_sun_times(now_miami.date())
    is_daylight_now = sunrise_now <= now_miami <= sunset_now

    ai_sent, ai_reason, ai_conf = get_agent_analysis(
        trend_f_per_hr=safe_trend,
        wind_dir=latest["WindVal"],
        dew_f=latest["Dew"],
        temp_f=latest["Temp"],
        rad_watts=hrrr_rad,
        precip_prob=hrrr_precip,
        current_hour=now_miami.hour,
        is_daylight=is_daylight_now,
    )

    ref_msg = None
    if bracket_cap is not None and high_round > bracket_cap:
        ai_conf = 0
        ai_sent = "DEAD"
        ref_msg = "💀 BUSTED: Day high %d° > cap %d°" % (high_round, int(round(bracket_cap)))
    elif high_round >= target_temp and ai_sent == "BULLISH":
        ai_conf = max(0, ai_conf - 40)
        ref_msg = "⚠️ ITM already, but conditions still support further heating."

    forecast_high = high_round
    if f_data.get("today_daily"):
        nws_high = f_data["today_daily"].get("temperature")
        if isinstance(nws_high, (int, float)):
            forecast_high = max(forecast_high, int(nws_high))

    st.markdown(HIDE_INDEX_CSS, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temp", "%.2f°F" % latest["Temp"], "Feels %.0f°" % calculate_heat_index(latest["Temp"], latest["Hum"]))
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
            is_selected = (not is_nan(m["strike"])) and (float(m["strike"]) == float(target_temp))
            if cols[i].button(label, key="mkt_%d" % i, type="primary" if is_selected else "secondary"):
                st.query_params["target"] = str(m["strike"])
                st.rerun()
    else:
        st.info("No markets returned for today (ticker may be unavailable yet).")

    # Sensor log (Velocity now smoothed)
    st.subheader("Sensor Log (Miami Time)")
    clean_rows = []
    for i, row in enumerate(history[:25]):
        vel = compute_smoothed_velocity(history, i, window_minutes=30)
        vel_arrow = "—"
        if vel is not None:
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
        cond = "%s %s" % (condition_icon_from_sky_wx(sky, wx), sky)
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

    # Debug
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
                            "cap_for_bust": m.get("cap_for_bust"),
                        }
                        for m in markets
                    ]
                )
            )


def render_forecast_generic(
    daily: Optional[Dict[str, Any]],
    hourly: List[Dict[str, Any]],
    date_label: str,
    f_data: Dict[str, Any],
    which_day: str,  # "today" or "tomorrow"
) -> None:
    st.title("📅 Forecast: %s" % date_label)

    if st.button("🔄 Refresh Forecast"):
        st.cache_data.clear()
        st.rerun()

    # Confidence gauge
    daily_text = ""
    if daily:
        daily_text = daily.get("detailedForecast", "") or ""

    peak = f_data.get("hrrr_today_peak") if which_day == "today" else f_data.get("hrrr_tomorrow_peak")
    peak_rad = None
    peak_precip = None
    if isinstance(peak, dict):
        peak_rad = peak.get("rad")
        peak_precip = peak.get("precip")

    period_time = None
    if daily and daily.get("startTime"):
        period_time = parse_iso_time(daily["startTime"])
    if not period_time:
        period_time = get_miami_time()
    sunrise_p, sunset_p = get_sun_times(period_time.astimezone(TZ_MIAMI).date())
    is_daylight_period = sunrise_p <= period_time.astimezone(TZ_MIAMI) <= sunset_p

    sent, conf, reasons = forecast_trade_confidence(daily_text, peak_rad, peak_precip, is_daylight_period)

    gauge_col1, gauge_col2 = st.columns([1, 3])
    with gauge_col1:
        st.metric("Trade Bias", sent)
    with gauge_col2:
        st.progress(conf / 100.0)
        st.caption(f"Confidence: **{conf}%** — {reasons}")

    # Daily text
    if daily:
        # Use sunrise/sunset to determine if the period is day or night
        sunrise, sunset = get_sun_times(period_time.astimezone(TZ_MIAMI).date())
        is_night = period_time < sunrise or period_time > sunset
        icon = icon_from_short_forecast(daily.get("shortForecast", "") or "")
        if is_night and icon == "☀️":
            icon = "🌙"
        st.success(f"{icon} {daily_text}")
        st.caption(
            "NWS Period: %s | Temp: %s° | Wind: %s"
            % (daily.get("name", ""), daily.get("temperature", ""), daily.get("windSpeed", ""))
        )
    else:
        st.warning("Daily forecast not available for this view.")

    # Hourly table with icons
    if hourly:
        # Get sunrise/sunset for each hour's local date
        h_data = []
        for h in hourly[:24]:
            dt = parse_iso_time(h.get("startTime"))
            if not dt:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt_local = dt.astimezone(TZ_MIAMI)
            sunrise, sunset = get_sun_times(dt_local.date())
            is_night = dt_local < sunrise or dt_local > sunset

            sf = h.get("shortForecast", "") or ""
            icon = icon_from_short_forecast(sf)
            if is_night and icon == "☀️":
                icon = "🌙"

            pop = None
            try:
                pop_val = (h.get("probabilityOfPrecipitation") or {}).get("value")
                pop = None if pop_val is None else int(round(float(pop_val)))
            except Exception:
                pop = None

            h_data.append(
                {
                    "Time": dt_local.strftime("%a %I %p"),
                    "Cond": f"{icon} {sf}",
                    "Temp": h.get("temperature"),
                    "PoP": "—" if pop is None else f"{pop}%",
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

    # Sunrise/Sunset calculation for Miami
    sunrise_dt, sunset_dt = get_sun_times(get_miami_time().date())
    sunrise = sunrise_dt.strftime("%I:%M %p")
    sunset = sunset_dt.strftime("%I:%M %p")
    st.sidebar.markdown(f"☀️ **Sunrise:** {sunrise}")
    st.sidebar.markdown(f"🌙 **Sunset:** {sunset}")

    auto = st.sidebar.checkbox("⚡ Auto-Refresh (10s)", value=True)
    if auto:
        render_autorefresh(10)

    markets, _ = fetch_market_data()
    lbl, price, cap = "Target", 0, None
    for m in markets:
        if (not is_nan(m["strike"])) and float(m["strike"]) == float(tgt):
            lbl = m["label"]
            price = int(m["price"])
            cap = m.get("cap_for_bust")
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
                date_label=format_ordinal_day(now),
                f_data=f_data,
                which_day="today",
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
                date_label=format_ordinal_day(now + timedelta(days=1)),
                f_data=f_data,
                which_day="tomorrow",
            )


if __name__ == "__main__":
    st.sidebar.caption("Miami time: %s" % get_miami_time().strftime("%Y-%m-%d %I:%M:%S %p"))
    main()
