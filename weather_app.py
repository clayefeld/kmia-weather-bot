import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import base64
import time
import pandas as pd
import math
import re
from datetime import datetime, timedelta, timezone 
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# --- CONFIGURATION ---
st.set_page_config(page_title="Project Helios", page_icon="☀️", layout="wide")

# --- DATA SOURCES ---
NWS_API_HISTORY = "https://api.weather.gov/stations/KMIA/observations"
AWC_METAR_URL = "https://aviationweather.gov/api/data/metar?ids=KMIA&format=raw&hours=12"
NWS_POINT_URL = "https://api.weather.gov/points/25.7954,-80.2901"
AWC_TAF_URL = "https://aviationweather.gov/api/data/taf?ids=KMIA&format=raw"
KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2"
# Force API to EST so index 0 is always Midnight EST
OM_API_URL = "https://api.open-meteo.com/v1/forecast?latitude=25.7954&longitude=-80.2901&hourly=temperature_2m,precipitation_probability,shortwave_radiation,cloud_cover&timezone=America%2FNew_York&forecast_days=2&models=hrrr_north_america"

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

# --- UTILS (Defined Top-Level to prevent NameError) ---
def get_miami_time():
    try: return datetime.now(ZoneInfo("US/Eastern"))
    except: return datetime.now(timezone(timedelta(hours=-5)))

def parse_iso_time(iso_str):
    try: return datetime.fromisoformat(iso_str)
    except: return None

def get_display_time(dt_utc):
    try:
        return dt_utc.astimezone(ZoneInfo("US/Eastern")).strftime("%I:%M %p")
    except:
        return dt_utc.strftime("%H:%M")

def calculate_heat_index(temp_f, humidity):
    if temp_f < 80: return temp_f 
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = -42.379, 2.04901523, 10.14333127, -0.22475541, -6.83783e-3, -5.481717e-2, 1.22874e-3, 8.5282e-4, -1.99e-6
    T, R = temp_f, humidity
    return c1 + (c2 * T) + (c3 * R) + (c4 * T * R) + (c5 * T**2) + (c6 * R**2) + (c7 * T**2 * R) + (c8 * T * R**2) + (c9 * T**2 * R**2)

# --- KALSHI AUTH ---
class KalshiAuth:
    def __init__(self):
        try:
            self.key_id = st.secrets["KALSHI_KEY_ID"]
            pk_raw = st.secrets["KALSHI_PRIVATE_KEY"]
            self.private_key_str = pk_raw.replace("\\n", "\n").strip()
            from cryptography.hazmat.primitives import serialization
            self.private_key = serialization.load_pem_private_key(
                self.private_key_str.encode(), password=None
            )
            self.ready = True
        except:
            self.ready = False

    def sign_request(self, method, path, timestamp):
        if not self.ready: return None
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        msg = f"{timestamp}{method}{path}".encode('utf-8')
        signature = self.private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')

# --- MARKET FETCHER ---
@st.cache_data(ttl=5)
def fetch_market_data():
    try: from cryptography.hazmat.primitives import serialization
    except ImportError: return [], "🔴 Crypto Lib Missing"

    auth = KalshiAuth()
    if not auth.ready: return [], "🔴 Key Error"

    try:
        now_miami = get_miami_time()
        date_str = now_miami.strftime("%y%b%d").upper() 
        event_ticker = f"KXHIGHMIA-{date_str}"
        path = f"/events/{event_ticker}"
        ts = str(int(time.time() * 1000))
        sig = auth.sign_request("GET", path, ts)
        
        headers = {"KALSHI-ACCESS-KEY": auth.key_id, "KALSHI-ACCESS-SIGNATURE": sig, "KALSHI-ACCESS-TIMESTAMP": ts, "Content-Type": "application/json"}
        r = requests.get(KALSHI_API_URL + path, headers=headers, timeout=3)
        
        data = r.json()
        raw_markets = data.get('markets', [])
        
        parsed_markets = []
        for m in raw_markets:
            floor = m.get('floor_strike')
            cap = m.get('cap_strike')
            ask = m.get('yes_ask', 0)
            sort_key = floor if floor is not None else -999.0
            parsed_markets.append({"floor": floor, "cap": cap, "price": ask, "sort": sort_key})
            
        parsed_markets.sort(key=lambda x: x['sort'])
        
        final_list = []
        count = len(parsed_markets)
        for i, m in enumerate(parsed_markets):
            label = "Unknown"
            strike_val = m['floor'] if m['floor'] else m['cap']
            if i == 0:
                val = m['cap'] if m['cap'] else m['floor']
                if count > 1 and parsed_markets[1]['floor'] == val: val = val - 1
                label = f"{int(val)} or below"
                strike_val = val 
            elif i == count - 1:
                val = m['floor'] if m['floor'] else m['cap']
                if count > 1 and parsed_markets[i-1]['cap'] == val: val = val + 1
                label = f"{int(val)} or above"
                strike_val = val
            else:
                label = f"{m['floor']} - {m['cap']}"
            
            logic_cap = m['cap']
            if i == count - 1: logic_cap = 999 
            if i == 0: logic_cap = val 
            
            final_list.append({"label": label, "strike": float(strike_val), "price": m['price'], "cap": logic_cap})
            
        return final_list, "🟢 Live"
    except:
        return [], "🔴 API Error"

# --- AI AGENT ---
def get_agent_analysis(trend, hum, wind_dir, solar_min, sky, dew_f, temp_f, press_in, rad_watts, precip_prob, current_hour):
    reasons = []
    sentiment = "NEUTRAL"
    confidence = 50 
    
    # 1. DIURNAL CURVE
    if current_hour >= 15: 
        reasons.append(f"Late Day ({current_hour}:00) - Cooling Bias")
        confidence = 30 
        if trend > 0: reasons.append("Ignoring late-day noise")
    elif current_hour >= 12: 
        reasons.append("Peak Heating Window")
        confidence = 60
    elif current_hour < 11: 
        reasons.append("Morning Ramp-Up")
        confidence = 80

    # 2. Solar (HRRR)
    if rad_watts > 700:
        reasons.append(f"High Solar ({int(rad_watts)} W/m²)")
        if current_hour < 14: confidence += 15
    elif rad_watts < 200 and current_hour < 17:
        reasons.append(f"Low Solar ({int(rad_watts)} W/m²)")
        confidence -= 15
        if sentiment == "NEUTRAL": sentiment = "BEARISH"

    # 3. Moisture
    dew_dep = temp_f - dew_f
    if dew_dep < 3:
        reasons.append("Air Saturated")
        sentiment = "TRAP"
        confidence = 10
    elif dew_dep > 12:
        reasons.append("Dry Air (Heating Possible)")
        if current_hour < 14: confidence += 10

    # 4. Wind
    if 0 <= wind_dir <= 180:
        reasons.append("Ocean Breeze (Cooling)")
        if sentiment == "NEUTRAL": confidence = 30
    elif wind_dir > 180:
        reasons.append("Land Breeze (Heating)")
        confidence += 5

    # 5. Radar
    if precip_prob > 40:
        reasons.append(f"Rain Risk {precip_prob}%")
        sentiment = "BEARISH"
        confidence = 5

    if confidence > 75 and sentiment == "NEUTRAL": sentiment = "BULLISH"
    if confidence < 35 and sentiment == "NEUTRAL": sentiment = "BEARISH"

    return sentiment, " + ".join(reasons), confidence

# --- FETCHERS ---
def fetch_live_history():
    data_list = []
    # NWS
    try:
        r = requests.get(NWS_API_HISTORY, timeout=4)
        if r.status_code == 200:
            for item in r.json().get('features', []):
                p = item.get('properties', {})
                t_c = p.get('temperature', {}).get('value')
                if t_c is None: continue
                ts = p.get('timestamp')
                dt = datetime.fromisoformat(ts.split('+')[0]).replace(tzinfo=timezone.utc)
                data_list.append({
                    "dt_utc": dt, "Source": "NWS",
                    "Temp": (t_c * 1.8) + 32,
                    "Hum": p.get('relativeHumidity', {}).get('value') or 0,
                    "Dew": (p.get('dewpoint', {}).get('value') or 0) * 1.8 + 32,
                    "Press": (p.get('barometricPressure', {}).get('value') or 0) * 0.0002953,
                    "WindVal": p.get('windDirection', {}).get('value') or -1,
                    "Sky": (p.get('cloudLayers', []) or [{}])[0].get('amount', '--')
                })
    except: pass
    
    # AWC
    try:
        r = requests.get(AWC_METAR_URL, timeout=4)
        for line in r.text.split('\n'):
            if "KMIA" in line:
                tm_m = re.search(r"\b(\d{2})(\d{4})Z\b", line)
                tp_m = re.search(r"\b(M?\d{2})/(M?\d{2})\b", line)
                pr_m = re.search(r"\bA(\d{4})\b", line)
                if tm_m and tp_m:
                    day, tm = int(tm_m.group(1)), tm_m.group(2)
                    now = datetime.now(timezone.utc)
                    dt = datetime(now.year, now.month, day, int(tm[:2]), int(tm[2:]), tzinfo=timezone.utc)
                    tc = int(tp_m.group(1).replace('M','-'))
                    dc = int(tp_m.group(2).replace('M','-'))
                    data_list.append({
                        "dt_utc": dt, "Source": "AWC",
                        "Temp": (tc * 1.8) + 32,
                        "Hum": 0, 
                        "Dew": (dc * 1.8) + 32,
                        "Press": int(pr_m.group(1))/100.0 if pr_m else 0,
                        "WindVal": -1,
                        "Sky": "METAR"
                    })
    except: pass
    return sorted(data_list, key=lambda x: x['dt_utc'], reverse=True)

@st.cache_data(ttl=300)
def fetch_forecast_data():
    data = {"today_daily": None, "all_hourly": [], "hrrr_now": None}
    try:
        r = requests.get(NWS_POINT_URL, timeout=5)
        if r.status_code == 200:
            urls = r.json().get('properties', {})
            r_d = requests.get(urls.get('forecast'), timeout=5)
            if r_d.status_code == 200:
                for p in r_d.json()['properties']['periods']:
                    if p['isDaytime']: data["today_daily"] = p; break
            r_h = requests.get(urls.get('forecastHourly'), timeout=5)
            if r_h.status_code == 200:
                data["all_hourly"] = r_h.json()['properties']['periods']
    except: pass
    
    # HRRR FIX: DIRECT INDEXING (Because we forced API to NY Time)
    try:
        r = requests.get(OM_API_URL, timeout=3)
        if r.status_code == 200:
            hrrr = r.json()
            # Index 0 = 00:00 Today. Index 13 = 13:00 Today.
            # Get Miami Hour directly (0-23)
            idx = get_miami_time().hour
            
            h = hrrr['hourly']
            data["hrrr_now"] = {
                "rad": h['shortwave_radiation'][idx],
                "precip": h['precipitation_probability'][idx],
                "temp": h['temperature_2m'][idx],
                "cloud": h['cloud_cover'][idx]
            }
    except: pass
    return data

def calculate_smart_trend(master_list):
    if len(master_list) < 2: return 0.0
    now = master_list[0]['dt_utc']
    one_hr_ago = now - timedelta(hours=1)
    points = [p for p in master_list if p['dt_utc'] >= one_hr_ago]
    if len(points) < 2: return 0.0
    x = [(p['dt_utc'] - one_hr_ago).total_seconds()/60 for p in points]
    y = [p['Temp'] for p in points]
    N = len(x)
    sum_x, sum_y = sum(x), sum(y)
    sum_xy = sum(i*j for i, j in zip(x,y))
    sum_xx = sum(i*i for i in x)
    den = (N*sum_xx - sum_x*sum_x)
    if den == 0: return 0.0
    return ((N*sum_xy - sum_x*sum_y) / den) * 60

# --- RENDERER ---
def render_live_dashboard(target_temp, bracket_label, live_price, bracket_cap):
    st.title("🔴 Project Helios: Live Feed")
    if st.button("🔄 Refresh System", type="primary"): st.cache_data.clear(); st.rerun()
    
    history = fetch_live_history()
    f_data = fetch_forecast_data()
    
    if not history: st.error("No Data"); return
    latest = history[0]
    
    now_miami = get_miami_time()
    today_recs = [x for x in history if x['dt_utc'].astimezone(ZoneInfo("US/Eastern")).date() == now_miami.date()]
    high_mark = max(today_recs, key=lambda x: x['Temp']) if today_recs else latest
    high_round = int(round(high_mark['Temp']))
    
    # Physics Extraction
    hrrr_rad = f_data['hrrr_now']['rad'] if f_data['hrrr_now'] else 0
    hrrr_precip = f_data['hrrr_now']['precip'] if f_data['hrrr_now'] else 0
    
    # Define safe_trend before usage (CRITICAL FIX)
    smart_trend = calculate_smart_trend(history)
    safe_trend = smart_trend
    if now_miami.hour > 17 and safe_trend < -0.5: safe_trend = -0.5

    # AI Analysis
    ai_sent, ai_reason, ai_conf = get_agent_analysis(
        safe_trend, latest['Hum'], latest['WindVal'], 999, latest['Sky'], 
        latest['Dew'], latest['Temp'], latest['Press'], hrrr_rad, hrrr_precip, now_miami.hour
    )

    # Referee Logic
    ref_msg = None
    if bracket_cap and high_round > bracket_cap:
        ai_conf = 0
        ai_sent = "DEAD"
        ref_msg = f"💀 BUSTED: High {high_round}° > Cap {bracket_cap}°"
    elif high_round >= target_temp and ai_sent == "BULLISH":
        ai_conf = max(0, ai_conf - 50)
        ref_msg = f"⚠️ HOLDING RISK: ITM, but heating continues."

    # Forecast High
    forecast_high = high_round 
    if f_data['today_daily']:
        nws_high = f_data['today_daily'].get('temperature')
        if nws_high: forecast_high = max(high_round, nws_high)

    st.markdown(HIDE_INDEX_CSS, unsafe_allow_html=True)
    
    # METRICS (CLASSIC LAYOUT RESTORED)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temp", f"{latest['Temp']:.2f}°F", f"Feels {calculate_heat_index(latest['Temp'], latest['Hum']):.0f}")
    c2.metric("Proj. High", f"{forecast_high}°F", "Forecast", delta_color="off")
    c3.metric("Day High", f"{high_mark['Temp']:.2f}°F", f"Officially {high_round}°F", delta_color="off")
    c4.metric("Solar (HRRR)", f"{hrrr_rad} W/m²")

    st.markdown("---")
    m1, m2 = st.columns([2,1])
    color = "grey"
    if ai_sent == "BULLISH": color = "green"
    if ai_sent == "BEARISH": color = "red"
    if ai_sent == "TRAP": color = "orange"
    if ai_sent == "DEAD": color = "grey"
    
    m1.info(f"🤖 **PHYSICS:** :{color}[**{ai_sent}**] ({ai_conf}%)\n\n{ref_msg if ref_msg else ai_reason}")
    
    edge = ai_conf - (live_price or 0)
    m2.metric(f"Kalshi ({bracket_label})", f"{live_price}¢", f"{edge:+.0f}% Edge", delta_color="off")

    # PROJECTION
    next_3_hours = []
    current_utc = datetime.now(timezone.utc)
    for p in f_data['all_hourly']:
        p_dt = parse_iso_time(p['startTime'])
        if p_dt > current_utc:
            next_3_hours.append(p)
            if len(next_3_hours) >= 3: break
    if len(next_3_hours) < 3: proj_str = "⚠️ Forecast Unavailable"
    else:
        proj_vals = []
        curr_temp = latest['Temp']
        for i, f in enumerate(next_3_hours):
            nws_temp = f['temperature']
            trend_weight = 0.4 / (i + 1)
            model_weight = 1.0 - trend_weight
            raw_proj = (curr_temp + (safe_trend * (i+1))) * trend_weight + (nws_temp * model_weight)
            if 0 <= latest['WindVal'] <= 180: raw_proj -= (0.5 * (i+1))
            icon = "🌧️" if "Rain" in f['shortForecast'] else "☁️"
            if "Sunny" in f['shortForecast']: icon = "☀️"
            proj_vals.append(f"**+{i+1}h:** {raw_proj:.1f}°F {icon}")
        proj_str = " | ".join(proj_vals)
    st.success(f"**🔮 AI PROJECTION:** {proj_str}")

    # BRACKETS (CLASSIC LAYOUT)
    st.subheader("🎯 Select Bracket (Live Markets)")
    markets, _ = fetch_market_data()
    cols = st.columns(len(markets) if markets else 1)
    if markets:
        for i, m in enumerate(markets):
            if cols[i].button(f"{m['label']}\n({m['price']}¢)", key=i, type="primary" if target_temp==m['strike'] else "secondary"):
                st.query_params["target"] = str(m['strike'])
                st.rerun()

    # LOG
    st.subheader("Sensor Log (Miami Time)")
    df = pd.DataFrame(history[:15])
    if not df.empty:
        df['Time'] = df['dt_utc'].apply(get_display_time)
        df['Temp'] = df['Temp'].apply(lambda x: f"{x:.2f}")
        df['Dew'] = df['Dew'].apply(lambda x: f"{x:.1f}")
        st.table(df[['Time', 'Source', 'Temp', 'Dew', 'Hum', 'Press']])

# --- FORECAST VIEW ---
def render_forecast_generic(daily, hourly, taf, date_label):
    st.title(f"Forecast: {date_label}")
    if st.button(f"🔄 Refresh"): st.cache_data.clear(); st.rerun()
    if daily: st.success(f"{daily['detailedForecast']}")
    if hourly:
        h_data = []
        for h in hourly:
            dt = parse_iso_time(h['startTime'])
            h_data.append({"Time": dt.strftime("%I %p"), "Temp": h['temperature'], "Cond": h['shortForecast'], "Wind": f"{h['windDirection']} {h['windSpeed']}"})
        st.table(pd.DataFrame(h_data))

# --- MAIN ---
def main():
    if "target" not in st.query_params: st.query_params["target"] = "81.0"
    tgt = float(st.query_params["target"])
    
    view_mode = st.sidebar.radio("Deck:", ["Live Monitor", "Today's Forecast", "Tomorrow's Forecast"])
    st.sidebar.divider()
    
    if "auto" not in st.query_params: st.query_params["auto"] = "true"
    components.html(f"""<script>setTimeout(function(){{window.parent.location.reload();}}, 10000);</script>""", height=0)

    markets, _ = fetch_market_data()
    lbl, price, cap = "Target", 0, None
    for m in markets:
        if m['strike'] == tgt: lbl, price, cap = m['label'], m['price'], m['cap']
            
    if view_mode == "Live Monitor": render_live_dashboard(tgt, lbl, price, cap)
    else:
        f_data = fetch_forecast_data()
        now = get_miami_time()
        if view_mode == "Today's Forecast": render_forecast_generic(f_data['today_daily'], f_data['all_hourly'], f_data['taf'], now.strftime("%A"))
        else: render_forecast_generic(None, f_data['all_hourly'], f_data['taf'], (now + timedelta(days=1)).strftime("%A"))

if __name__ == "__main__":
    main()
