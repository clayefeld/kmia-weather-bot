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
# HTTP SESSION
# =============================================================================
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "ProjectHelios/1.0 (Streamlit app)",
        "Accept": "application/geo+json, application/json;q=0.9, */*;q=0.8",
    })
    return s

SESSION = make_session()

def safe_get(url: str, timeout: float = 4.0) -> requests.Response:
    last_exc = None
    for attempt in range(3):
        try:
            r = SESSION.get(url, timeout=timeout)
            return r
        except Exception as e:
            last_exc = e
            time.sleep(0.25 * (attempt + 1))
    raise last_exc


# =============================================================================
# TIME + PARSING UTILS
# =============================================================================
def get_miami_time() -> datetime:
    return datetime.now(TZ_MIAMI)

def parse_iso_time(iso_str: str) -> datetime | None:
    if not iso_str: return None
    try:
        s = iso_str.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception: return None

def get_display_time(dt_aware: datetime) -> str:
    return dt_aware.astimezone(TZ_MIAMI).strftime("%I:%M %p")

def c_to_f(c: float) -> float:
    return (c * 1.8) + 32.0

def mps_to_kt(mps: float) -> float:
    return mps * 1.94384

def calculate_heat_index(temp_f: float, humidity: float) -> float:
    if temp_f < 80: return temp_f
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = -42.379, 2.04901523, 10.14333127, -0.22475541, -6.83783e-3, -5.481717e-2, 1.22874e-3, 8.5282e-4, -1.99e-6
    T, R = temp_f, humidity
    return c1 + (c2 * T) + (c3 * R) + (c4 * T * R) + (c5 * T**2) + (c6 * R**2) + (c7 * T**2 * R) + (c8 * T * R**2) + (c9 * T**2 * R**2)

def format_strike_label(floor: float | None, cap: float | None) -> str:
    if floor is None and cap is not None: return f"{int(round(cap))}° or below"
    if cap is None and floor is not None: return f"{int(round(floor))}° or above"
    if floor is not None and cap is not None: return f"{int(round(floor))}° – {int(round(cap))}°"
    return "Unknown"

# --- ICON HELPER ---
def get_icon(condition_text: str, is_day: bool = True) -> str:
    s = condition_text.upper()
    icon = "☁️" # Default
    
    if "CLR" in s or "SKC" in s or "FEW" in s or "SUNNY" in s or "CLEAR" in s:
        icon = "☀️" if is_day else "🌙"
    elif "SCT" in s or "PARTLY" in s:
        icon = "⛅"
    elif "BKN" in s:
        icon = "🌥️"
    elif "OVC" in s or "CLOUDY" in s:
        icon = "☁️"
    
    if "RAIN" in s or "SHOWER" in s: icon = "🌧️"
    if "THUNDER" in s or "TS" in s: icon = "⛈️"
    if "FOG" in s or "FG" in s: icon = "🌫️"
    
    return icon

# =============================================================================
# KALSHI AUTH & FETCH
# =============================================================================
class KalshiAuth:
    def __init__(self):
        self.ready = False
        try:
            self.key_id = st.secrets["KALSHI_KEY_ID"]
            pk_raw = st.secrets["KALSHI_PRIVATE_KEY"]
            private_key_str = pk_raw.replace("\\n", "\n").strip()
            from cryptography.hazmat.primitives import serialization
            self.private_key = serialization.load_pem_private_key(private_key_str.encode("utf-8"), password=None)
            self.ready = True
        except Exception as e:
            self.error = str(e)
            self.ready = False

    def sign_request(self, method: str, path: str, timestamp_ms: str) -> str | None:
        if not self.ready: return None
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        msg = f"{timestamp_ms}{method}{path}".encode("utf-8")
        sig = self.private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode("utf-8")

@st.cache_data(ttl=5)
def fetch_market_data() -> tuple[list[dict], str]:
    try: from cryptography.hazmat.primitives import serialization # noqa
    except Exception: return [], "🔴 Crypto lib missing"

    auth = KalshiAuth()
    if not auth.ready: return [], f"🔴 Kalshi auth error"

    try:
        now_miami = get_miami_time()
        date_str = now_miami.strftime("%y%b%d").upper()
        event_ticker = f"KXHIGHMIA-{date_str}"
        path = f"/events/{event_ticker}"
        ts = str(int(time.time() * 1000))
        sig = auth.sign_request("GET", path, ts)
        
        headers = {"KALSHI-ACCESS-KEY": auth.key_id, "KALSHI-ACCESS-SIGNATURE": sig, "KALSHI-ACCESS-TIMESTAMP": ts, "Content-Type": "application/json"}
        r = SESSION.get(KALSHI_API_URL + path, headers=headers, timeout=4)
        r.raise_for_status()
        
        raw_markets = r.json().get("markets", [])
        parsed = []
        for m in raw_markets:
            floor, cap = m.get("floor_strike"), m.get("cap_strike")
            ask = m.get("yes_ask", 0) or 0
            floor_f = float(floor) if floor is not None else None
            cap_f = float(cap) if cap is not None else None
            
            if floor_f is None and cap_f is not None: sort_key = cap_f - 10000
            elif floor_f is not None: sort_key = floor_f
            else: sort_key = 10000000
            
            parsed.append({"floor": floor_f, "cap": cap_f, "price": int(ask), "sort": sort_key})

        parsed.sort(key=lambda x: x["sort"])
        final_list = []
        for m in parsed:
            label = format_strike_label(m["floor"], m["cap"])
            strike = m["cap"] if (m["floor"] is None and m["cap"] is not None) else m["floor"]
            final_list.append({
                "label": label, 
                "strike": float(strike) if strike is not None else float("nan"), 
                "price": m["price"], 
                "floor": m["floor"], 
                "cap": m["cap"]
            })
        return final_list, "🟢 Live"
    except Exception:
        return [], "🔴 Kalshi API error"

# =============================================================================
# FORECAST FETCHER
# =============================================================================
@st.cache_data(ttl=300)
def fetch_forecast_data() -> dict:
    data = {"today_daily": None, "tomorrow_daily": None, "all_hourly": [], "hrrr_now": None, "taf_raw": None, "status": []}
    
    # NWS
    try:
        r = safe_get(NWS_POINT_URL, timeout=6)
        if r.status_code == 200:
            urls = r.json().get("properties", {})
            r_d = safe_get(urls.get("forecast"), timeout=6)
            if r_d and r_d.status_code == 200:
                periods = r_d.json().get("properties", {}).get("periods", [])
                daytimes = [p for p in periods if p.get("isDaytime")]
                if len(daytimes) >= 1: data["today_daily"] = daytimes[0]
                if len(daytimes) >= 2: data["tomorrow_daily"] = daytimes[1]
            
            r_h = safe_get(urls.get("forecastHourly"), timeout=6)
            if r_h and r_h.status_code == 200:
                data["all_hourly"] = r_h.json().get("properties", {}).get("periods", [])
    except: data["status"].append("NWS Error")

    # HRRR
    try:
        r = safe_get(OM_API_URL, timeout=5)
        if r and r.status_code == 200:
            hrrr = r.json()
            h = hrrr.get("hourly", {})
            times = h.get("time", [])
            if times:
                # String Match Strategy for Timezone Safety
                now_str = get_miami_time().strftime("%Y-%m-%dT%H:00")
                idx = -1
                if now_str in times:
                    idx = times.index(now_str)
                else:
                    # Fallback prefix match
                    prefix = get_miami_time().strftime("%Y-%m-%dT%H")
                    for i, t in enumerate(times): 
                        if t.startswith(prefix): idx = i; break
                
                if idx != -1:
                    data["hrrr_now"] = {
                        "rad": h.get("shortwave_radiation", [0])[idx],
                        "precip": h.get("precipitation_probability", [0])[idx],
                        "temp_c": h.get("temperature_2m", [0])[idx],
                        "cloud": h.get("cloud_cover", [0])[idx],
                    }
    except: data["status"].append("HRRR Error")

    # TAF
    try:
        r = safe_get(AWC_TAF_URL, timeout=5)
        if r and r.status_code == 200:
            lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
            data["taf_raw"] = "\n".join(lines[:6]) if lines else None
    except: pass

    return data

# =============================================================================
# OBSERVATIONS FETCHER
# =============================================================================
def _metar_month_rollover(dt: datetime, now_utc: datetime) -> datetime:
    if dt > now_utc + timedelta(hours=2):
        first = datetime(now_utc.year, now_utc.month, 1, tzinfo=timezone.utc)
        prev = first - timedelta(days=1)
        day = min(dt.day, prev.day)
        return datetime(prev.year, prev.month, day, dt.hour, dt.minute, tzinfo=timezone.utc)
    return dt

def fetch_live_history() -> tuple[list[dict], list[str]]:
    data_list = []
    status = []

    # NWS
    try:
        r = safe_get(NWS_API_HISTORY, timeout=5)
        if r and r.status_code == 200:
            for item in r.json().get("features", []):
                p = item.get("properties", {})
                t_c = (p.get("temperature") or {}).get("value")
                ts = p.get("timestamp")
                if t_c is None or not ts: continue
                
                dt = parse_iso_time(ts)
                if not dt: continue
                if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
                
                wdir = (p.get("windDirection") or {}).get("value")
                wspd = (p.get("windSpeed") or {}).get("value")
                w_str = "--"
                w_val = -1
                if wdir is not None: w_val = int(round(wdir))
                if wdir is not None and wspd is not None:
                    w_str = f"{int(round(wdir)):03d} @ {int(round(mps_to_kt(float(wspd))))}kt"
                
                clouds = p.get("cloudLayers") or []
                sky_str = clouds[0].get("amount", "--") if clouds else "--"
                
                hum = (p.get("relativeHumidity") or {}).get("value") or 0.0
                dew_c = (p.get("dewpoint") or {}).get("value")
                dew_f = c_to_f(float(dew_c)) if dew_c is not None else 0.0
                press_pa = (p.get("barometricPressure") or {}).get("value")
                press_in = (float(press_pa) * 0.0002953) if press_pa is not None else 0.0

                data_list.append({
                    "dt_utc": dt, "Source": "NWS", "Temp": c_to_f(float(t_c)),
                    "Official": int(round(c_to_f(float(t_c)))), "Wind": w_str,
                    "Sky": sky_str, "WindVal": w_val, "Hum": float(hum),
                    "Dew": dew_f, "Press": press_in
                })
    except: status.append("NWS Obs Error")

    # AWC
    try:
        r = safe_get(AWC_METAR_URL, timeout=5)
        if r and r.status_code == 200:
            for line in r.text.splitlines():
                if "KMIA" not in line: continue
                tm_m = re.search(r"\b(\d{2})(\d{4})Z\b", line)
                tp_m = re.search(r"\b(M?\d{2})/(M?\d{2})\b", line)
                pr_m = re.search(r"\bA(\d{4})\b", line)
                wd_m = re.search(r"\b(\d{3}|VRB)(\d{2,3})G?(\d{2,3})?KT\b", line)
                
                if tm_m and tp_m:
                    day, hhmm = int(tm_m.group(1)), tm_m.group(2)
                    now_utc = datetime.now(timezone.utc)
                    dt = datetime(now_utc.year, now_utc.month, day, int(hhmm[:2]), int(hhmm[2:]), tzinfo=timezone.utc)
                    dt = _metar_month_rollover(dt, now_utc)
                    
                    tc = int(tp_m.group(1).replace("M","-"))
                    dc = int(tp_m.group(2).replace("M","-"))
                    press_in = int(pr_m.group(1))/100.0 if pr_m else 0.0
                    
                    w_str, w_val = "--", -1
                    if wd_m:
                        w_str = f"{wd_m.group(1)} @ {wd_m.group(2)}kt"
                        if wd_m.group(1).isdigit(): w_val = int(wd_m.group(1))
                    
                    sky = "CLR"
                    if "OVC" in line: sky = "OVC"
                    elif "BKN" in line: sky = "BKN"
                    elif "SCT" in line: sky = "SCT"
                    elif "FEW" in line: sky = "FEW"
                    
                    try: hum = 100 * (math.exp((17.625*dc)/(243.04+dc))/math.exp((17.625*tc)/(243.04+tc)))
                    except: hum = 0.0
                    
                    data_list.append({
                        "dt_utc": dt, "Source": "AWC", "Temp": c_to_f(tc),
                        "Official": int(round(c_to_f(tc))), "Wind": w_str,
                        "Sky": sky, "WindVal": w_val, "Hum": hum,
                        "Dew": c_to_f(dc), "Press": press_in
                    })
    except: status.append("METAR Error")

    # Simple Dedupe
    merged = {}
    for x in data_list:
        k = x["dt_utc"].isoformat()
        if k not in merged: merged[k] = x
    
    final = sorted(merged.values(), key=lambda x: x["dt_utc"], reverse=True)
    return final, status

# =============================================================================
# LOGIC
# =============================================================================
def calculate_smart_trend(history):
    if len(history) < 2: return 0.0
    now = history[0]["dt_utc"]
    one_hr = now - timedelta(hours=1)
    pts = [p for p in history if p["dt_utc"] >= one_hr]
    if len(pts) < 2: return 0.0
    
    x = [(p["dt_utc"] - one_hr).total_seconds()/60 for p in pts]
    y = [p["Temp"] for p in pts]
    N, sx, sy = len(x), sum(x), sum(y)
    sxy = sum(i*j for i,j in zip(x,y))
    sxx = sum(i*i for i in x)
    den = (N*sxx - sx*sx)
    return ((N*sxy - sx*sy)/den)*60 if den != 0 else 0.0

def get_agent_analysis(trend, wind_dir, sky, dew_f, temp_f, rad, precip, hour):
    reasons, conf, sent = [], 50, "NEUTRAL"
    
    # 1. Diurnal
    if hour < 11: reasons.append("Morning Ramp"); conf += 20
    elif hour < 15: reasons.append("Peak Heat"); conf += 10
    else: reasons.append(f"Late Day ({hour}:00)"); conf -= 15
    
    # 2. Solar
    if rad > 700: reasons.append(f"Strong Sun ({int(rad)} W)"); conf += 15
    elif rad < 200 and hour < 17: reasons.append("Low Sun"); conf -= 15
    
    # 3. Moisture
    if (temp_f - dew_f) < 3: reasons.append("Saturated"); sent="TRAP"; conf=min(conf, 25)
    elif (temp_f - dew_f) > 12: reasons.append("Dry Air"); conf += 8
    
    # 4. Wind
    if 0 <= wind_dir <= 180: reasons.append("Ocean Breeze"); conf -= 8
    
    # 5. Trend
    if trend > 0.5: reasons.append(f"Rising Fast ({trend:.1f}/hr)"); conf += 10
    
    conf = max(0, min(100, conf))
    if sent == "NEUTRAL":
        if conf >= 70: sent = "BULLISH"
        elif conf <= 35: sent = "BEARISH"
    
    return sent, " + ".join(reasons), conf

# =============================================================================
# UI
# =============================================================================
def render_live_dashboard(tgt, lbl, price, cap):
    st.title("🔴 Project Helios: Live Feed")
    if st.button("🔄 Refresh System", type="primary"): st.cache_data.clear(); st.rerun()
    
    history, _ = fetch_live_history()
    f_data = fetch_forecast_data()
    
    if not history: st.error("No Data"); return
    latest = history[0]
    now_miami = get_miami_time()
    
    today_recs = [x for x in history if x["dt_utc"].astimezone(TZ_MIAMI).date() == now_miami.date()]
    high_mark = max(today_recs, key=lambda x: x["Temp"]) if today_recs else latest
    high_round = int(round(high_mark["Temp"]))
    
    hrrr_rad = f_data.get("hrrr_now", {}).get("rad", 0) or 0
    hrrr_precip = f_data.get("hrrr_now", {}).get("precip", 0) or 0
    
    safe_trend = calculate_smart_trend(history)
    if now_miami.hour > 17 and safe_trend < -0.5: safe_trend = -0.5
    
    ai_sent, ai_reason, ai_conf = get_agent_analysis(
        safe_trend, latest["WindVal"], latest["Sky"], latest["Dew"], 
        latest["Temp"], hrrr_rad, hrrr_precip, now_miami.hour
    )
    
    ref_msg = None
    if cap and high_round > cap:
        ai_conf, ai_sent, ref_msg = 0, "DEAD", f"💀 BUSTED: High {high_round}° > Cap {int(cap)}°"
    elif high_round >= tgt and ai_sent == "BULLISH":
        ai_conf = max(0, ai_conf - 40)
        ref_msg = "⚠️ ITM but heating continues."
    
    fc_high = high_round
    if f_data.get("today_daily"):
        n = f_data["today_daily"].get("temperature")
        if n: fc_high = max(fc_high, int(n))
        
    st.markdown(HIDE_INDEX_CSS, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temp", f"{latest['Temp']:.2f}°F", f"Feels {calculate_heat_index(latest['Temp'], latest['Hum']):.0f}°")
    c2.metric("Proj. High", f"{fc_high}°F", "NWS Daytime", delta_color="off")
    c3.metric("Day High", f"{high_mark['Temp']:.2f}°F", f"Rounded {high_round}°F", delta_color="off")
    c4.metric("Solar (HRRR)", f"{int(hrrr_rad)} W/m²")
    
    st.markdown("---")
    m1, m2 = st.columns([2, 1])
    color = {"BULLISH": "green", "BEARISH": "red", "TRAP": "orange", "DEAD": "grey", "NEUTRAL": "grey"}.get(ai_sent, "grey")
    m1.info(f"🤖 **PHYSICS:** :{color}[**{ai_sent}**] ({ai_conf}%)\n\n{ref_msg if ref_msg else ai_reason}")
    edge = ai_conf - (price or 0)
    m2.metric(f"Kalshi ({lbl})", f"{price}¢", f"{edge:+.0f}% Edge", delta_color="off")
    
    # Projection
    next_3 = []
    curr_utc = datetime.now(timezone.utc)
    for p in f_data.get("all_hourly", []):
        dt = parse_iso_time(p.get("startTime"))
        if dt and dt.astimezone(timezone.utc) > curr_utc:
            next_3.append(p)
            if len(next_3) >= 3: break
            
    if len(next_3) < 3: st.warning("Forecast Unavailable")
    else:
        p_vals = []
        curr = latest["Temp"]
        for i, f in enumerate(next_3):
            nws = float(f.get("temperature"))
            tw = 0.45/(i+1)
            raw = (curr + (safe_trend*(i+1)))*tw + (nws*(1-tw))
            if 0 <= latest["WindVal"] <= 180: raw -= 0.4*(i+1)
            
            icon = "☁️"
            sf = f.get("shortForecast", "")
            if "Rain" in sf: icon = "🌧️"
            elif "Sunny" in sf or "Clear" in sf: icon = "☀️"
            p_vals.append(f"**+{i+1}h:** {raw:.1f}°F {icon}")
        st.success(f"**🔮 AI PROJECTION:** {' | '.join(p_vals)}")
        
    st.subheader("🎯 Select Bracket (Live Markets)")
    markets, _ = fetch_market_data()
    cols = st.columns(len(markets) if markets else 1)
    if markets:
        for i, m in enumerate(markets):
            sel = (not math.isnan(m["strike"]) and float(m["strike"]) == float(tgt))
            if cols[i].button(f"{m['label']}\n({m['price']}¢)", key=i, type="primary" if sel else "secondary"):
                st.query_params["target"] = str(m["strike"])
                st.rerun()
                
    st.subheader("Sensor Log (Miami Time)")
    rows = []
    for i, r in enumerate(history[:15]):
        vel = "—"
        if i < len(history)-1:
            d = (r["dt_utc"] - history[i+1]["dt_utc"]).total_seconds()/3600
            if d > 0:
                v = (r["Temp"] - history[i+1]["Temp"])/d
                if v > 0.5: vel = "⬆️ Fast"
                elif v > 0.1: vel = "↗️ Rising"
                elif v < -0.5: vel = "⬇️ Drop"
                elif v < -0.1: vel = "↘️ Falling"
        
        # ICON LOGIC
        is_day = 7 <= r["dt_utc"].astimezone(TZ_MIAMI).hour < 20
        icon = get_icon(r["Sky"], is_day)
        
        rows.append({
            "Time": get_display_time(r["dt_utc"]),
            "Src": r["Source"],
            "Condition": f"{icon} {r['Sky']}",
            "Temp (°F)": f"{r['Temp']:.2f}",
            "Dew (°F)": f"{r['Dew']:.1f}",
            "Hum": f"{int(r['Hum'])}%",
            "Press": f"{r['Press']:.2f}",
            "Velocity": vel,
            "Wind": r["Wind"]
        })
    st.table(pd.DataFrame(rows))

def render_forecast_generic(daily, hourly, date_label):
    st.title(f"Forecast: {date_label}")
    if st.button("🔄 Refresh"): st.cache_data.clear(); st.rerun()
    if daily: st.success(daily.get("detailedForecast"))
    
    rows = []
    for h in hourly[:24]:
        dt = parse_iso_time(h.get("startTime"))
        if not dt: continue
        is_day = 7 <= dt.astimezone(TZ_MIAMI).hour < 20
        cond = h.get("shortForecast", "")
        rows.append({
            "Time": dt.astimezone(TZ_MIAMI).strftime("%a %I %p"),
            "Temp": h.get("temperature"),
            "Cond": f"{get_icon(cond, is_day)} {cond}",
            "Wind": f"{h.get('windDirection')} {h.get('windSpeed')}"
        })
    st.table(pd.DataFrame(rows))

def main():
    if "target" not in st.query_params: st.query_params["target"] = "81.0"
    try: tgt = float(st.query_params["target"])
    except: tgt = 81.0
    
    view = st.sidebar.radio("Deck", ["Live Monitor", "Today's Forecast", "Tomorrow's Forecast"])
    st.sidebar.divider()
    
    if st.sidebar.checkbox("⚡ Auto-Refresh", value=True):
        components.html(f"<script>setTimeout(function(){{window.location.reload();}}, 10000);</script>", height=0)
        
    markets, _ = fetch_market_data()
    lbl, price, cap = "Target", 0, None
    for m in markets:
        if not math.isnan(m["strike"]) and float(m["strike"]) == tgt:
            lbl, price, cap = m["label"], m["price"], m["cap"]
            
    if view == "Live Monitor": render_live_dashboard(tgt, lbl, price, cap)
    else:
        f_data = fetch_forecast_data()
        now = get_miami_time()
        if view == "Today's Forecast": render_forecast_generic(f_data.get("today_daily"), f_data.get("all_hourly"), now.strftime("%A"))
        else:
            tom = (now + timedelta(days=1)).date()
            h_tom = [h for h in f_data.get("all_hourly", []) if parse_iso_time(h.get("startTime")).astimezone(TZ_MIAMI).date() == tom]
            render_forecast_generic(f_data.get("tomorrow_daily"), h_tom, (now + timedelta(days=1)).strftime("%A"))

if __name__ == "__main__":
    main()
