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
# Note: forecast_days=2 ensures we have enough buffer for timezone shifts
OM_API_URL = "https://api.open-meteo.com/v1/forecast?latitude=25.7954&longitude=-80.2901&hourly=temperature_2m,precipitation_probability,shortwave_radiation,cloud_cover&forecast_days=2&models=hrrr_north_america"

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

# --- KALSHI AUTH ---
class KalshiAuth:
    def __init__(self):
        try:
            self.key_id = st.secrets["KALSHI_KEY_ID"]
            pk_raw = st.secrets["KALSHI_PRIVATE_KEY"]
            self.private_key_str = pk_raw.replace("\\n", "\n").strip()
            # Dynamic import to prevent crash if library missing
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
    auth = KalshiAuth()
    if not auth.ready: return [], "🔴 Auth Fail"

    try:
        now_miami = datetime.now(ZoneInfo("US/Eastern"))
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

# --- UTILS ---
def get_miami_time():
    return datetime.now(ZoneInfo("US/Eastern"))

def get_display_time(dt_utc):
    return dt_utc.astimezone(ZoneInfo("US/Eastern")).strftime("%I:%M %p")

def calculate_heat_index(temp_f, humidity):
    if temp_f < 80: return temp_f 
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = -42.379, 2.04901523, 10.14333127, -0.22475541, -6.83783e-3, -5.481717e-2, 1.22874e-3, 8.5282e-4, -1.99e-6
    T, R = temp_f, humidity
    return c1 + (c2 * T) + (c3 * R) + (c4 * T * R) + (c5 * T**2) + (c6 * R**2) + (c7 * T**2 * R) + (c8 * T * R**2) + (c9 * T**2 * R**2)

# --- AI AGENT (DIURNAL LOGIC) ---
def get_agent_analysis(trend, hum, wind_dir, solar_min, sky, dew_f, temp_f, press_in, rad_watts, precip_prob, current_hour):
    reasons = []
    sentiment = "NEUTRAL"
    confidence = 50 
    
    # 1. TIME OF DAY PHYSICS (The "Diurnal Curve")
    if current_hour >= 15: # After 3 PM
        reasons.append(f"Late Afternoon ({current_hour}:00) - Cooling Bias")
        if trend > 0: 
            reasons.append("Ignoring Fake Bullish Trend")
            trend = 0 # Force neutral trend
        confidence = 20 # Very hard to hit NEW highs
    elif current_hour >= 13: # 1 PM - 3 PM (Peak)
        reasons.append("Peak Heating Window")
        confidence = 60
    elif current_hour < 10:
        reasons.append("Morning Ramp-Up")
        confidence = 80

    # 2. Solar (HRRR)
    if rad_watts > 700:
        reasons.append(f"High Solar ({int(rad_watts)} W/m²)")
        if current_hour < 14: confidence += 15
    elif rad_watts < 200 and current_hour < 17:
        reasons.append("Cloud Shadow / Low Energy")
        confidence -= 20
        sentiment = "BEARISH"

    # 3. Moisture
    dew_dep = temp_f - dew_f
    if dew_dep < 3:
        reasons.append("Air Saturated")
        sentiment = "TRAP"
        confidence = 10
    elif dew_dep > 12:
        reasons.append("Dry Air (Spike Potential)")
        if current_hour < 14: confidence += 10

    # 4. Wind
    if 0 <= wind_dir <= 180:
        reasons.append("Ocean Breeze")
        if sentiment == "NEUTRAL": confidence = 30
    elif wind_dir > 180:
        reasons.append("Land Breeze")
        confidence += 5

    # 5. Radar
    if precip_prob > 40:
        reasons.append(f"Rain Risk {precip_prob}%")
        sentiment = "BEARISH"
        confidence = 5

    # Final Sentiment Calculation
    if confidence > 70 and sentiment == "NEUTRAL": sentiment = "BULLISH"
    if confidence < 30 and sentiment == "NEUTRAL": sentiment = "BEARISH"

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
                        "Hum": 0, # Calculated later if needed
                        "Dew": (dc * 1.8) + 32,
                        "Press": int(pr_m.group(1))/100.0 if pr_m else 0,
                        "WindVal": -1, # Simplification
                        "Sky": "METAR"
                    })
    except: pass
    return sorted(data_list, key=lambda x: x['dt_utc'], reverse=True)

@st.cache_data(ttl=300)
def fetch_forecast_data():
    data = {"today_daily": None, "all_hourly": [], "hrrr_now": None}
    # NWS
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
    
    # HRRR (Fixed Indexing)
    try:
        r = requests.get(OM_API_URL, timeout=3)
        if r.status_code == 200:
            hrrr = r.json()
            # Find index matching current hour
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00")
            times = hrrr.get('hourly', {}).get('time', [])
            try:
                idx = -1
                # Open-Meteo returns ISO strings. Match the hour.
                # Simplest robust way: find entry starting with YYYY-MM-DDTHH
                target_h = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")
                for i, t in enumerate(times):
                    if t.startswith(target_h):
                        idx = i
                        break
                
                if idx != -1:
                    h = hrrr['hourly']
                    data["hrrr_now"] = {
                        "rad": h['shortwave_radiation'][idx],
                        "precip": h['precipitation_probability'][idx],
                        "temp": h['temperature_2m'][idx],
                        "cloud": h['cloud_cover'][idx]
                    }
            except: pass
    except: pass
    return data

# --- RENDERER ---
def render_dashboard(target_temp, bracket_label, live_price, bracket_cap):
    st.title("🔴 Project Helios: Live Feed")
    if st.button("🔄 Refresh System", type="primary"): st.cache_data.clear(); st.rerun()
    
    history = fetch_live_history()
    f_data = fetch_forecast_data()
    
    if not history: st.error("No Data"); return
    latest = history[0]
    
    # Time Context
    now_miami = get_miami_time()
    today_recs = [x for x in history if x['dt_utc'].astimezone(ZoneInfo("US/Eastern")).date() == now_miami.date()]
    high_mark = max(today_recs, key=lambda x: x['Temp']) if today_recs else latest
    high_round = int(round(high_mark['Temp']))
    
    # Physics Extraction
    hrrr_rad = f_data['hrrr_now']['rad'] if f_data['hrrr_now'] else 0
    hrrr_precip = f_data['hrrr_now']['precip'] if f_data['hrrr_now'] else 0
    
    # AI Analysis
    ai_sent, ai_reason, ai_conf = get_agent_analysis(
        0, latest['Hum'], latest['WindVal'], 999, latest['Sky'], 
        latest['Dew'], latest['Temp'], latest['Press'], hrrr_rad, hrrr_precip, now_miami.hour
    )

    # Referee Logic
    ref_msg = None
    delta_needed = target_temp - high_round
    
    if bracket_cap and high_round > bracket_cap:
        ai_conf = 0
        ai_sent = "DEAD"
        ref_msg = f"💀 BUSTED: High {high_round}° > Cap {bracket_cap}°"
    elif delta_needed <= 0:
        # We are ITM
        if ai_sent == "BULLISH":
            ai_conf = max(0, ai_conf - 50)
            ref_msg = f"⚠️ HOLDING RISK: ITM, but heating continues."
    elif delta_needed > 0:
        # We need to climb
        if now_miami.hour >= 15:
            ai_conf = 0
            ref_msg = f"🛑 TOO LATE: Need +{delta_needed:.1f}°F but sun is setting."
        elif now_miami.hour >= 13 and delta_needed > 2:
            ai_conf = 10
            ref_msg = f"📉 UNLIKELY: Need +{delta_needed:.1f}°F in peak heat."

    # UI
    st.markdown(HIDE_INDEX_CSS, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temp", f"{latest['Temp']:.2f}°", f"Feels {calculate_heat_index(latest['Temp'], latest['Hum']):.0f}")
    c2.metric("Needed for Win", f"+{max(0, delta_needed):.1f}°F" if delta_needed > 0 else "✅ ITM", "Delta", delta_color="off")
    c3.metric("Day High", f"{high_mark['Temp']:.2f}°", f"Off: {high_round}°")
    c4.metric("Solar (HRRR)", f"{hrrr_rad} W/m²")

    st.markdown("---")
    m1, m2 = st.columns([2,1])
    color = "grey"
    if ai_sent == "BULLISH": color = "green"
    if ai_sent == "BEARISH": color = "red"
    if ai_sent == "TRAP": color = "orange"
    
    m1.info(f"🤖 **PHYSICS:** :{color}[**{ai_sent}**] ({ai_conf}%)\n\n{ref_msg if ref_msg else ai_reason}")
    
    edge = ai_conf - (live_price or 0)
    m2.metric(f"Kalshi ({bracket_label})", f"{live_price}¢", f"{edge:+.0f}% Edge", delta_color="off")

    st.subheader("🎯 Select Bracket")
    markets, _ = fetch_market_data()
    cols = st.columns(len(markets) if markets else 1)
    if markets:
        for i, m in enumerate(markets):
            if cols[i].button(f"{m['label']}\n({m['price']}¢)", key=i, type="primary" if target_temp==m['strike'] else "secondary"):
                st.query_params["target"] = str(m['strike'])
                st.rerun()

    st.subheader("Sensor Log")
    df = pd.DataFrame(history[:10])
    if not df.empty:
        df['Time'] = df['dt_utc'].apply(get_display_time)
        df['Temp'] = df['Temp'].apply(lambda x: f"{x:.2f}")
        df['Dew'] = df['Dew'].apply(lambda x: f"{x:.1f}")
        st.table(df[['Time', 'Source', 'Temp', 'Dew', 'Hum', 'Press']])

# --- MAIN ---
def main():
    if "target" not in st.query_params: st.query_params["target"] = "81.0"
    tgt = float(st.query_params["target"])
    
    markets, _ = fetch_market_data()
    lbl, price, cap = "Target", 0, None
    for m in markets:
        if m['strike'] == tgt: lbl, price, cap = m['label'], m['price'], m['cap']
            
    render_dashboard(tgt, lbl, price, cap)

if __name__ == "__main__":
    main()
