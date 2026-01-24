import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import base64
import time
import pandas as pd
from datetime import datetime, timedelta, timezone 
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# --- LIBRARY CHECK ---
try:
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# --- CONFIGURATION ---
st.set_page_config(page_title="Project Helios", page_icon="☀️", layout="wide")

# URLs
NWS_API_HISTORY = "https://api.weather.gov/stations/KMIA/observations"
AWC_METAR_URL = "https://aviationweather.gov/api/data/metar?ids=KMIA&format=raw&hours=12"
NWS_POINT_URL = "https://api.weather.gov/points/25.7906,-80.3164"
AWC_TAF_URL = "https://aviationweather.gov/api/data/taf?ids=KMIA&format=raw"
KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2"

# --- GLOBAL STYLES ---
HIDE_INDEX_CSS = """
    <style>
    thead tr th:first-child {display:none}
    tbody th {display:none}
    div.stButton > button {width: 100%;}
    </style>
    """

# --- KALSHI CLIENT ---
class KalshiAuth:
    def __init__(self):
        try:
            self.key_id = st.secrets["KALSHI_KEY_ID"]
            pk_raw = st.secrets["KALSHI_PRIVATE_KEY"]
            self.private_key_str = pk_raw.replace("\\n", "\n").strip()
            self.private_key = serialization.load_pem_private_key(
                self.private_key_str.encode(),
                password=None
            )
            self.ready = True
        except Exception as e:
            self.ready = False
            self.error = str(e)

    def sign_request(self, method, path, timestamp):
        if not self.ready: return None
        msg = f"{timestamp}{method}{path}".encode('utf-8')
        signature = self.private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')

# --- SMART MARKET FETCHER (TURBO MODE) ---
# CHANGED: TTL reduced from 60s to 5s for near-real-time updates
@st.cache_data(ttl=5)
def fetch_market_data():
    """
    Finds the active Miami High Temp event dynamically and returns bracket prices.
    Includes logic to fix 1-degree offset on outer brackets.
    """
    if not CRYPTO_AVAILABLE: return [], "🔴 Crypto Lib Missing"
    auth = KalshiAuth()
    if not auth.ready: return [], f"🔴 Key Error"

    try:
        now_miami = datetime.now(ZoneInfo("US/Eastern"))
        date_str = now_miami.strftime("%y%b%d").upper() 
        event_ticker = f"KXHIGHMIA-{date_str}"
        
        path = f"/events/{event_ticker}"
        ts = str(int(time.time() * 1000))
        sig = auth.sign_request("GET", path, ts)
        
        headers = {
            "KALSHI-ACCESS-KEY": auth.key_id,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "Content-Type": "application/json"
        }
        
        r = requests.get(KALSHI_API_URL + path, headers=headers, timeout=3)
        if r.status_code != 200: return [], f"🔴 API {r.status_code}"
            
        data = r.json()
        raw_markets = data.get('markets', [])
        if not raw_markets: return [], "🔴 No Markets"
        
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
            
            final_list.append({"label": label, "strike": float(strike_val), "price": m['price']})
            
        return final_list, "🟢 Live"

    except Exception as e:
        return [], f"🔴 Error: {str(e)}"

# --- STYLING & UTILS ---
def get_headers():
    return {'User-Agent': '(project_helios_v48_turbo, myemail@example.com)'}

def get_miami_time():
    try:
        return datetime.now(ZoneInfo("US/Eastern"))
    except:
        return datetime.now(timezone(timedelta(hours=-5)))

def parse_iso_time(iso_str):
    try:
        return datetime.fromisoformat(iso_str)
    except:
        return None

def get_display_time(dt_utc):
    try:
        dt_miami = dt_utc.astimezone(ZoneInfo("US/Eastern"))
    except:
        dt_miami = dt_utc.astimezone(timezone(timedelta(hours=-5)))
    return dt_miami.strftime("%I:%M %p")

# --- AI AGENT ---
def get_agent_analysis(trend, hum, wind_dir, solar_min, sky):
    reasons = []
    sentiment = "NEUTRAL"
    confidence = 50 
    
    if solar_min <= 0:
        reasons.append("Night mode (No solar fuel)")
        sentiment = "BEARISH"
        confidence = 20
    elif solar_min < 120:
        reasons.append("Low solar angle (<2h left)")
        confidence = 40
        
    if hum > 85:
        reasons.append("Atmosphere Saturated")
        if trend > 1.0: 
            reasons.append("⚠️ RALLY SUSPECT")
            sentiment = "TRAP"
            confidence = 15
    elif hum < 50:
        reasons.append("Dry Air (Heating)")
        confidence += 10
    
    if 0 <= wind_dir <= 180:
        reasons.append("Ocean Breeze (Cooling)")
        if sentiment == "NEUTRAL": confidence = 30
    elif wind_dir > 180:
        reasons.append("Land Breeze (Warming)")
        if sentiment == "NEUTRAL": confidence = 60
        
    if "OVC" in sky or "BKN" in sky:
        reasons.append("Clouds blocking sun")
        if sentiment != "TRAP": sentiment = "BEARISH"
        confidence -= 10
    elif "CLR" in sky or "FEW" in sky:
        if solar_min > 120:
            reasons.append("Clear Skies (Heating)")
            if sentiment == "NEUTRAL": sentiment = "BULLISH"; confidence = 85

    confidence = max(1, min(99, confidence))
    if not reasons: reasons.append("Nominal.")
    summary = " + ".join(reasons)
    return sentiment, summary, confidence

# --- FETCHERS ---
def fetch_live_history():
    data_list = []
    try:
        r = requests.get(NWS_API_HISTORY, headers=get_headers(), timeout=4)
        if r.status_code == 200:
            for item in r.json().get('features', []):
                props = item.get('properties', {})
                temp_c = props.get('temperature', {}).get('value')
                if temp_c is None: continue
                
                ts = props.get('timestamp')
                if not ts: continue
                dt_utc = datetime.fromisoformat(ts.split('+')[0]).replace(tzinfo=timezone.utc)
                
                # Wind/Sky
                wdir = props.get('windDirection', {}).get('value')
                wspd = props.get('windSpeed', {}).get('value')
                w_str = f"{int(wdir):03d} @ {int(wspd/1.852)}kt" if wdir and wspd else "--"
                clouds = props.get('cloudLayers', [])
                sky_str = clouds[0].get('amount', '--') if clouds else "--"
                
                data_list.append({
                    "dt_utc": dt_utc,
                    "Source": "NWS",
                    "Temp": (temp_c * 1.8) + 32,
                    "Official": int(round((temp_c * 1.8) + 32)),
                    "Wind": w_str,
                    "Sky": sky_str,
                    "WindVal": int(wdir) if wdir else -1,
                    "Hum": props.get('relativeHumidity', {}).get('value') or 0
                })
    except: pass

    try:
        r = requests.get(AWC_METAR_URL, timeout=4)
        for line in r.text.split('\n'):
            if "KMIA" in line:
                pass 
    except: pass
    
    return sorted(data_list, key=lambda x: x['dt_utc'], reverse=True)

@st.cache_data(ttl=300)
def fetch_forecast_data():
    data = {"all_hourly": []}
    try:
        r = requests.get(NWS_POINT_URL, headers=get_headers(), timeout=5)
        if r.status_code == 200:
            hourly_url = r.json().get('properties', {}).get('forecastHourly')
            r_h = requests.get(hourly_url, headers=get_headers(), timeout=5)
            if r_h.status_code == 200:
                data["all_hourly"] = r_h.json().get('properties', {}).get('periods', [])
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

# --- VIEW: LIVE MONITOR ---
def render_live_dashboard(target_temp, bracket_label, live_price):
    st.title("🔴 Project Helios: Live Feed")
    
    if st.button("🔄 Refresh System", type="primary"):
        st.cache_data.clear()
        st.rerun()
        
    history = fetch_live_history()
    f_data = fetch_forecast_data()
    
    if not history:
        st.error("Connection Failed: No Data Available")
        return

    latest = history[0]
    
    # --- DATE FIX ---
    now_miami = get_miami_time()
    today_date = now_miami.date()
    today_records = [x for x in history if x['dt_utc'].astimezone(ZoneInfo("US/Eastern")).date() == today_date]
    high_mark = max(today_records, key=lambda x: x['Temp']) if today_records else latest
    high_round = int(round(high_mark['Temp']))
    smart_trend = calculate_smart_trend(history)

    sunrise_miami = now_miami.replace(hour=7, minute=0, second=0, microsecond=0)
    sunset_miami = now_miami.replace(hour=17, minute=55, second=0, microsecond=0)
    
    is_night = False
    solar_fuel = "NIGHT"
    solar_min = 0
    
    if now_miami < sunrise_miami or now_miami > sunset_miami:
        is_night = True
    else:
        time_left = sunset_miami - now_miami
        solar_min = time_left.total_seconds() / 60
        hrs, rem = divmod(time_left.seconds, 3600)
        mins = rem // 60
        solar_fuel = f"{hrs}h {mins}m"

    safe_trend = smart_trend
    if is_night and safe_trend < -0.5: safe_trend = -0.5
    
    hum = latest.get('Hum', 0)
    wind_dir = latest.get('WindVal', -1)
    if wind_dir == -1:
        for h in history[1:5]:
            if h.get('WindVal', -1) != -1: wind_dir = h['WindVal']; break
                
    ai_sent, ai_reason, ai_conf = get_agent_analysis(safe_trend, hum, wind_dir, solar_min, latest['Sky'])

    # --- METRICS ---
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Current Temp", f"{latest['Temp']:.2f}°F", f"{smart_trend:+.2f}/hr")
    with c2: st.metric("Official Round", f"{latest['Official']}°F")
    with c3:
        st.metric("Day High (Today)", f"{high_mark['Temp']:.2f}°F", f"Officially {high_round}°F", delta_color="off")
    with c4: st.metric("Solar Fuel", solar_fuel)

    # --- AI ANALYSIS & ARB CALCULATOR ---
    st.markdown("---")
    m_col1, m_col2 = st.columns([2, 1])
    
    with m_col1:
        sentiment_color = "blue"
        if ai_sent == "BULLISH": sentiment_color = "green"
        if ai_sent == "BEARISH": sentiment_color = "red"
        if ai_sent == "TRAP": sentiment_color = "orange"
        st.info(f"🤖 **PHYSICS ENGINE:** :{sentiment_color}[**{ai_sent}**] ({ai_conf}% Conf)\n\n{ai_reason}")

    with m_col2:
        if live_price is not None:
            edge = ai_conf - live_price
            edge_label = "Fair Value"
            edge_color = "off"
            if edge > 15: edge_color = "normal"; edge_label = "🔥 BUY Signal"
            elif edge < -15: edge_color = "inverse"; edge_label = "🛑 OVERPRICED"
            st.metric(f"Kalshi ({bracket_label})", f"{live_price}¢", f"{edge:+.0f}% Edge ({edge_label})", delta_color=edge_color)
            st.caption(f"Last Update: {now_miami.strftime('%H:%M:%S')}") # Live Timestamp
        else:
            st.metric(f"Kalshi ({bracket_label})", "--", "API Error")

    # --- PROJECTION BOARD ---
    next_3_hours = []
    current_utc = datetime.now(timezone.utc)
    for p in f_data['all_hourly']:
        p_dt = parse_iso_time(p['startTime'])
        if p_dt > current_utc:
            next_3_hours.append(p)
            if len(next_3_hours) >= 3: break
    
    if len(next_3_hours) < 3:
        proj_str = "⚠️ Forecast Data Unavailable for Projection"
    else:
        proj_vals = []
        curr_temp = latest['Temp']
        for i, f in enumerate(next_3_hours):
            nws_temp = f['temperature']
            trend_weight = 0.4 / (i + 1)
            model_weight = 1.0 - trend_weight
            raw_proj = (curr_temp + (safe_trend * (i+1))) * trend_weight + (nws_temp * model_weight)
            if 0 <= wind_dir <= 180: raw_proj -= (0.5 * (i+1))
            if is_night: raw_proj -= (0.3 * (i+1))
            icon = "🌧️" if "Rain" in f['shortForecast'] else "☁️"
            if "Sunny" in f['shortForecast']: icon = "☀️"
            proj_vals.append(f"**+{i+1}h:** {raw_proj:.1f}°F {icon}")
        proj_str = " | ".join(proj_vals)

    st.success(f"**🔮 AI PROJECTION:** {proj_str}")

    # --- BRACKET SELECTOR ---
    st.subheader("🎯 Select Bracket (Live Markets)")
    def set_target(val): st.query_params["target"] = str(val)

    markets, m_status = fetch_market_data()
    
    if not markets:
        st.warning(f"Market Connection Status: {m_status}")
    else:
        cols = st.columns(len(markets))
        for i, m in enumerate(markets):
            label = f"{m['label']}\n({m['price']}¢)"
            is_active = (target_temp == m['strike'])
            if cols[i].button(label, use_container_width=True, key=f"btn_{i}", type="primary" if is_active else "secondary"):
                set_target(m['strike'])
                st.rerun()

    # --- SENSOR LOG ---
    st.subheader("Sensor Log (Miami Time)")
    clean_rows = []
    for i, row in enumerate(history[:15]):
        vel_str = "—"
        if i < len(history) - 1:
            dt1, dt2 = row['dt_utc'], history[i+1]['dt_utc']
            diff = (dt1 - dt2).total_seconds()/3600.0
            if diff > 0:
                v = (row['Temp'] - history[i+1]['Temp']) / diff
                if v > 0.5: vel_str = "⬆️ Fast"
                elif v > 0.1: vel_str = "↗️ Rising"
                elif v < -0.5: vel_str = "⬇️ Drop"
                elif v < -0.1: vel_str = "↘️ Falling"
        
        sky_code = row['Sky']
        icon = "☁️" 
        if "CLR" in sky_code or "SKC" in sky_code: icon = "🌙" if is_night else "☀️"
        elif "FEW" in sky_code: icon = "🌤️"
        elif "SCT" in sky_code: icon = "⛅"
        
        clean_rows.append({
            "Time": get_display_time(row['dt_utc']),
            "Src": row['Source'],
            "Condition": f"{icon} {sky_code}",
            "Temp": row['Temp'],
            "Official": row['Official'],
            "Hum": f"{int(row['Hum'])}%" if row['Hum'] > 0 else "--",
            "Velocity": vel_str,
            "Wind": row['Wind']
        })
        
    df = pd.DataFrame(clean_rows)
    df['Temp'] = df['Temp'].apply(lambda x: f"{x:.2f}")
    df = df.rename(columns={"Temp": "Temp (°F)", "Official": "Official (Rnd)"})
    st.markdown(HIDE_INDEX_CSS, unsafe_allow_html=True)
    st.table(df)

# --- MAIN APP ---
def main():
    st.sidebar.header("PROJECT HELIOS ☀️")
    st.sidebar.caption("High-Frequency Weather Algo")
    view_mode = st.sidebar.radio("Command Deck:", ["Live Monitor", "Today's Forecast"])
    st.sidebar.divider()
    
    # Auto-Refresh (Changed to 10s default logic if active)
    default_auto = False
    if "auto" in st.query_params and st.query_params["auto"] == "true": default_auto = True
    auto_refresh = st.sidebar.checkbox("⚡ Turbo Refresh (Every 10s)", value=default_auto)
    if auto_refresh:
        st.query_params["auto"] = "true"
        # CHANGED: 10000ms = 10s
        components.html(f"""<script>setTimeout(function(){{window.parent.location.reload();}}, 10000);</script>""", height=0)
    else:
        if "auto" in st.query_params: del st.query_params["auto"]

    default_target = 81.0
    if "target" in st.query_params:
        try: default_target = float(st.query_params["target"])
        except: pass

    markets, _ = fetch_market_data()
    current_label = f"{default_target}"
    current_price = None
    for m in markets:
        if m['strike'] == default_target:
            current_label = m['label']
            current_price = m['price']
            break

    now_miami = get_miami_time()
    st.sidebar.caption(f"System Time: {now_miami.strftime('%I:%M:%S %p')}")
    
    if view_mode == "Live Monitor": render_live_dashboard(default_target, current_label, current_price)
    elif view_mode == "Today's Forecast": 
        f_data = fetch_forecast_data()
        render_forecast_generic(None, f_data['all_hourly'], None, now_miami.strftime("%A, %b %d"))

if __name__ == "__main__":
    main()
