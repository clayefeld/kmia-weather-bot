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

# --- LIBRARY CHECK ---
try:
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# --- CONFIGURATION ---
st.set_page_config(page_title="Project Helios", page_icon="☀️", layout="wide")

# --- DATA SOURCES (VERIFIED KMIA) ---
NWS_API_HISTORY = "https://api.weather.gov/stations/KMIA/observations"
AWC_METAR_URL = "https://aviationweather.gov/api/data/metar?ids=KMIA&format=raw&hours=12"
NWS_POINT_URL = "https://api.weather.gov/points/25.7954,-80.2901"
AWC_TAF_URL = "https://aviationweather.gov/api/data/taf?ids=KMIA&format=raw"
KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2"
OM_API_URL = "https://api.open-meteo.com/v1/forecast?latitude=25.7954&longitude=-80.2901&hourly=temperature_2m,precipitation_probability,shortwave_radiation,cloud_cover&timezone=America%2FNew_York&forecast_days=1&models=hrrr_north_america"

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

# --- SMART MARKET FETCHER ---
@st.cache_data(ttl=5)
def fetch_market_data():
    if not CRYPTO_AVAILABLE: return [], "🔴 Crypto Lib Missing"
    auth = KalshiAuth()
    if not auth.ready: return [], "🔴 Key Error"

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
            
            final_list.append({
                "label": label, 
                "strike": float(strike_val), 
                "price": m['price'],
                "cap": logic_cap
            })
            
        return final_list, "🟢 Live"

    except Exception as e:
        return [], f"🔴 Error: {str(e)}"

# --- UTILS ---
def get_headers():
    return {'User-Agent': '(project_helios_v59_timezone_fix, myemail@example.com)'}

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

def calculate_heat_index(temp_f, humidity):
    if temp_f < 80: return temp_f 
    c1 = -42.379
    c2 = 2.04901523
    c3 = 10.14333127
    c4 = -0.22475541
    c5 = -6.83783e-3
    c6 = -5.481717e-2
    c7 = 1.22874e-3
    c8 = 8.5282e-4
    c9 = -1.99e-6
    T = temp_f
    R = humidity
    hi = c1 + (c2 * T) + (c3 * R) + (c4 * T * R) + (c5 * T**2) + (c6 * R**2) + (c7 * T**2 * R) + (c8 * T * R**2) + (c9 * T**2 * R**2)
    return hi

# --- AI AGENT ---
def get_agent_analysis(trend, hum, wind_dir, solar_min, sky, dew_f, temp_f, press_in, rad_watts, precip_prob):
    reasons = []
    sentiment = "NEUTRAL"
    confidence = 50 
    
    # 1. Solar Physics (HRRR GHI)
    if solar_min <= 0:
        reasons.append("Night mode")
        sentiment = "BEARISH"
        confidence = 20
    elif rad_watts > 600:
        reasons.append(f"High Solar Energy ({int(rad_watts)} W/m²)")
        confidence += 15
    elif rad_watts < 200 and solar_min > 60:
        reasons.append(f"Low Solar ({int(rad_watts)} W/m²) - Clouds?")
        confidence -= 15
        
    # 2. Dew Point Physics
    dew_depression = temp_f - dew_f
    if dew_depression < 3:
        reasons.append("Air Saturated")
        sentiment = "TRAP"
        confidence = 10
    elif dew_depression > 10:
        reasons.append("Dry Air (Heating Possible)")
        confidence += 10
        
    # 3. Precip Risk (Radar Data)
    if precip_prob > 30:
        reasons.append(f"Rain Risk {precip_prob}%")
        sentiment = "BEARISH"
        confidence = 25
        
    # 4. Wind Physics
    if 0 <= wind_dir <= 180:
        reasons.append("Ocean Breeze (Cooling)")
        if sentiment == "NEUTRAL": confidence = 30
    elif wind_dir > 180:
        reasons.append("Land Breeze (Warming)")
        if sentiment == "NEUTRAL": confidence = 60
        
    # 5. Pressure
    if press_in > 30.05: confidence += 5

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
                dew_c = props.get('dewpoint', {}).get('value')
                dew_f = (dew_c * 1.8) + 32 if dew_c is not None else 0.0
                rel_hum = props.get('relativeHumidity', {}).get('value')
                humidity = rel_hum if rel_hum else 0
                press_pa = props.get('barometricPressure', {}).get('value')
                press_in = (press_pa * 0.0002953) if press_pa else 0.0
                ts = props.get('timestamp')
                if not ts: continue
                dt_utc = datetime.fromisoformat(ts.split('+')[0]).replace(tzinfo=timezone.utc)
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
                    "Hum": humidity,
                    "Dew": dew_f,
                    "Press": press_in
                })
    except: pass

    try:
        r = requests.get(AWC_METAR_URL, timeout=4)
        for line in r.text.split('\n'):
            if "KMIA" in line:
                time_match = re.search(r"\b(\d{2})(\d{4})Z\b", line)
                temp_match = re.search(r"\b(M?\d{2})/(M?\d{2})\b", line)
                press_match = re.search(r"\bA(\d{4})\b", line)
                wind_match = re.search(r"\b(\d{3}|VRB)(\d{2,3})G?(\d{2,3})?KT\b", line)
                if time_match and temp_match:
                    day, tm = int(time_match.group(1)), time_match.group(2)
                    now = datetime.now(timezone.utc)
                    month, year = now.month, now.year
                    if now.day < 5 and day > 25: month -= 1
                    if month == 0: month, year = 12, year - 1
                    dt_utc = datetime(year, month, day, int(tm[:2]), int(tm[2:]), tzinfo=timezone.utc)
                    t_str, d_str = temp_match.group(1), temp_match.group(2)
                    def parse_t(s): return -int(s[1:]) if 'M' in s else int(s)
                    tc, dc = parse_t(t_str), parse_t(d_str)
                    tf, df = (tc * 1.8) + 32, (dc * 1.8) + 32
                    press_in = float(press_match.group(1))/100.0 if press_match else 0.0
                    w_str = "--"
                    w_val = -1
                    if wind_match:
                        w_str = f"{wind_match.group(1)} @ {wind_match.group(2)}kt"
                        if wind_match.group(1).isdigit(): w_val = int(wind_match.group(1))
                    sky = "CLR"
                    if "OVC" in line: sky = "OVC"
                    elif "BKN" in line: sky = "BKN"
                    elif "SCT" in line: sky = "SCT"
                    elif "FEW" in line: sky = "FEW"
                    hum = 100 * (math.exp((17.625*dc)/(243.04+dc))/math.exp((17.625*tc)/(243.04+tc)))
                    data_list.append({
                        "dt_utc": dt_utc,
                        "Source": "AWC",
                        "Temp": tf,
                        "Official": int(round(tf)),
                        "Wind": w_str,
                        "Sky": sky,
                        "WindVal": w_val,
                        "Hum": hum,
                        "Dew": df,
                        "Press": press_in
                    })
    except: pass
    return sorted(data_list, key=lambda x: x['dt_utc'], reverse=True)

@st.cache_data(ttl=300)
def fetch_forecast_data():
    data = {"today_daily": None, "today_hourly": [], "tomorrow_daily": None, "tomorrow_hourly": [], "taf": None, "all_hourly": [], "hrrr_now": None}
    try:
        r = requests.get(NWS_POINT_URL, headers=get_headers(), timeout=5)
        if r.status_code == 200:
            props = r.json().get('properties', {})
            daily_url = props.get('forecast')
            hourly_url = props.get('forecastHourly')
            r_d = requests.get(daily_url, headers=get_headers(), timeout=5)
            if r_d.status_code == 200:
                periods = r_d.json().get('properties', {}).get('periods', [])
                for p in periods:
                    if p['isDaytime']: data["today_daily"] = p; break 
            r_h = requests.get(hourly_url, headers=get_headers(), timeout=5)
            if r_h.status_code == 200:
                data["all_hourly"] = r_h.json().get('properties', {}).get('periods', [])
        r_t = requests.get(AWC_TAF_URL, timeout=5)
        if r_t.status_code == 200: data["taf"] = r_t.text
        
        # --- NEW: HRRR FETCH (TIMEZONE FIX) ---
        r_om = requests.get(OM_API_URL, timeout=3)
        if r_om.status_code == 200:
            hrrr = r_om.json().get('hourly', {})
            # CRITICAL: Use Miami time index, not server UTC index
            miami_hour = datetime.now(ZoneInfo("US/Eastern")).hour
            
            data["hrrr_now"] = {
                "rad": hrrr['shortwave_radiation'][miami_hour],
                "precip": hrrr['precipitation_probability'][miami_hour],
                "temp": hrrr['temperature_2m'][miami_hour],
                "cloud": hrrr['cloud_cover'][miami_hour]
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

# --- VIEW: LIVE MONITOR ---
def render_live_dashboard(target_temp, bracket_label, live_price, bracket_cap):
    st.title("🔴 Project Helios: Live Feed")
    if st.button("🔄 Refresh System", type="primary"): st.cache_data.clear(); st.rerun()
    history = fetch_live_history()
    f_data = fetch_forecast_data()
    
    if not history: st.error("Connection Failed"); return
    latest = history[0]
    
    now_miami = get_miami_time()
    today_date = now_miami.date()
    today_records = [x for x in history if x['dt_utc'].astimezone(ZoneInfo("US/Eastern")).date() == today_date]
    high_mark = max(today_records, key=lambda x: x['Temp']) if today_records else latest
    high_round = int(round(high_mark['Temp']))
    smart_trend = calculate_smart_trend(history)

    forecast_high = high_round 
    if f_data['today_daily']:
        nws_high = f_data['today_daily'].get('temperature')
        if nws_high: forecast_high = max(high_round, nws_high)

    sunrise_miami = now_miami.replace(hour=7, minute=0, second=0, microsecond=0)
    sunset_miami = now_miami.replace(hour=17, minute=55, second=0, microsecond=0)
    is_night = (now_miami < sunrise_miami or now_miami > sunset_miami)
    
    solar_fuel = "NIGHT"
    solar_min = 0
    if not is_night:
        time_left = sunset_miami - now_miami
        solar_min = time_left.total_seconds() / 60
        hrs, rem = divmod(time_left.seconds, 3600)
        mins = rem // 60
        solar_fuel = f"{hrs}h {mins}m"

    safe_trend = smart_trend
    if is_night and safe_trend < -0.5: safe_trend = -0.5
    
    hum = latest.get('Hum', 0)
    dew = latest.get('Dew', 0)
    press = latest.get('Press', 0)
    feels_like = calculate_heat_index(latest['Temp'], hum)
    wind_dir = latest.get('WindVal', -1)
    if wind_dir == -1:
        for h in history[1:5]:
            if h.get('WindVal', -1) != -1: wind_dir = h['WindVal']; break
    
    hrrr_rad = 0
    hrrr_precip = 0
    if f_data['hrrr_now']:
        hrrr_rad = f_data['hrrr_now']['rad']
        hrrr_precip = f_data['hrrr_now']['precip']
                
    ai_sent, ai_reason, ai_conf = get_agent_analysis(safe_trend, hum, wind_dir, solar_min, latest['Sky'], dew, latest['Temp'], press, hrrr_rad, hrrr_precip)

    referee_msg = None
    if bracket_cap is not None:
        if high_round > bracket_cap:
            ai_conf = 0
            ai_sent = "DEAD"
            referee_msg = f"💀 BUSTED: High ({high_round}°F) > Cap ({bracket_cap}°F)"
        elif high_round >= target_temp and ai_sent == "BULLISH":
            ai_conf = max(0, ai_conf - 40)
            referee_msg = f"⚠️ OVERSHOOT RISK: ITM ({high_round}°), but heating continues."

    st.markdown(HIDE_INDEX_CSS, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Temp", f"{latest['Temp']:.2f}°F", f"Feels {feels_like:.1f}°")
    with c2: st.metric("Proj. High", f"{forecast_high}°F", "Forecast", delta_color="off")
    with c3: st.metric("Day High", f"{high_mark['Temp']:.2f}°F", f"Official: {high_round}°F", delta_color="off")
    with c4: st.metric("Solar (HRRR)", f"{hrrr_rad} W/m²")

    st.markdown("---")
    m_col1, m_col2 = st.columns([2, 1])
    with m_col1:
        sentiment_color = "blue"
        if ai_sent == "BULLISH": sentiment_color = "green"
        if ai_sent == "BEARISH": sentiment_color = "red"
        if ai_sent == "TRAP": sentiment_color = "orange"
        if ai_sent == "DEAD": sentiment_color = "grey"
        
        display_msg = referee_msg if referee_msg else ai_reason
        st.info(f"🤖 **PHYSICS ENGINE:** :{sentiment_color}[**{ai_sent}**] ({ai_conf}% Conf)\n\n{display_msg}")

    with m_col2:
        if live_price is not None:
            edge = ai_conf - live_price
            edge_label = "Fair Value"
            edge_color = "off"
            if edge > 15: edge_color = "normal"; edge_label = "🔥 BUY"
            elif edge < -15: edge_color = "inverse"; edge_label = "🛑 OVER"
            st.metric(f"Kalshi ({bracket_label})", f"{live_price}¢", f"{edge:+.0f}% Edge ({edge_label})", delta_color=edge_color)
            st.caption(f"Upd: {now_miami.strftime('%H:%M:%S')}")
        else:
            st.metric(f"Kalshi ({bracket_label})", "--", "API Error")

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
            if 0 <= wind_dir <= 180: raw_proj -= (0.5 * (i+1))
            if is_night: raw_proj -= (0.3 * (i+1))
            icon = "🌧️" if "Rain" in f['shortForecast'] else "☁️"
            if "Sunny" in f['shortForecast']: icon = "☀️"
            proj_vals.append(f"**+{i+1}h:** {raw_proj:.1f}°F {icon}")
        proj_str = " | ".join(proj_vals)
    st.success(f"**🔮 AI PROJECTION:** {proj_str}")

    st.subheader("🎯 Select Bracket (Live Markets)")
    def set_target(val): st.query_params["target"] = str(val)
    markets, m_status = fetch_market_data()
    if not markets: st.warning(f"Market Connection Status: {m_status}")
    else:
        cols = st.columns(len(markets))
        for i, m in enumerate(markets):
            label = f"{m['label']}\n({m['price']}¢)"
            is_active = (target_temp == m['strike'])
            if cols[i].button(label, use_container_width=True, key=f"btn_{i}", type="primary" if is_active else "secondary"):
                set_target(m['strike'])
                st.rerun()

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
            "Dew": row['Dew'], 
            "Hum": f"{int(row['Hum'])}%",
            "Press": f"{row['Press']:.2f}" if row['Press'] > 0 else "--",
            "Velocity": vel_str,
            "Wind": row['Wind']
        })
        
    df = pd.DataFrame(clean_rows)
    df['Temp'] = df['Temp'].apply(lambda x: f"{x:.2f}")
    df['Dew'] = df['Dew'].apply(lambda x: f"{x:.1f}")
    df = df.rename(columns={"Temp": "Temp (°F)", "Dew": "Dew (°F)", "Press": "Press (inHg)"})
    st.table(df)

# --- VIEW: FORECAST RENDERER ---
def render_forecast_generic(daily, hourly, taf, date_label):
    st.title(f"☀️ Helios Forecast: {date_label}")
    if st.button(f"🔄 Refresh {date_label}"): st.cache_data.clear(); st.rerun()
    if not hourly: st.warning(f"Forecast data unavailable for {date_label}."); return

    score = 10
    rain_hours = 0
    for h in hourly:
        s = h['shortForecast'].lower()
        if "rain" in s or "shower" in s: rain_hours += 1
        if "thunder" in s: rain_hours += 2 
    if rain_hours > 0: score -= 2        
    if rain_hours > 4: score -= 2        
    score = max(1, min(10, score))
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Confidence Score", f"{score}/10")
        st.progress(score/10)
    with c2:
        if daily:
            st.success(f"**Analyst Note:** {daily['detailedForecast']}")
            st.caption(f"Winds: {daily['windSpeed']} • Temp: {daily['temperature']}°F")

    st.subheader("Hourly Breakdown (Miami Time)")
    h_data = []
    for h in hourly:
        dt = parse_iso_time(h['startTime'])
        short = h['shortForecast']
        icon = "☁️"
        if "Sunny" in short: icon = "☀️"
        if "Rain" in short: icon = "🌧️"
        if "Thunder" in short: icon = "⛈️"
        if "Clear" in short: icon = "🌙"
        risk = "Safe"
        if "Rain" in short or "Thunder" in short: risk = "⚠️ RISK"
        h_data.append({"Time": dt.strftime("%I %p"), "Temp": h['temperature'], "Condition": f"{icon} {short}", "Wind": f"{h['windDirection']} {h['windSpeed']}", "Trade Risk": risk})

    df_h = pd.DataFrame(h_data)
    df_h['Temp'] = df_h['Temp'].apply(lambda x: f"{x:.0f}")
    df_h = df_h.rename(columns={"Temp": "Temp (°F)"})
    st.table(df_h)
    if taf: st.divider(); st.caption("✈️ AVIATION TAF (PILOT DATA)"); st.code(taf, language="text")

# --- MAIN APP ---
def main():
    st.sidebar.header("PROJECT HELIOS ☀️")
    st.sidebar.caption("High-Frequency Weather Algo")
    view_mode = st.sidebar.radio("Command Deck:", ["Live Monitor", "Today's Forecast", "Tomorrow's Forecast"])
    st.sidebar.divider()
    
    default_auto = False
    if "auto" in st.query_params and st.query_params["auto"] == "true": default_auto = True
    auto_refresh = st.sidebar.checkbox("⚡ Turbo Refresh (Every 10s)", value=default_auto)
    if auto_refresh:
        st.query_params["auto"] = "true"
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
    current_cap = None
    for m in markets:
        if m['strike'] == default_target:
            current_label = m['label']
            current_price = m['price']
            current_cap = m['cap']
            break

    now_miami = get_miami_time()
    st.sidebar.caption(f"System Time: {now_miami.strftime('%I:%M:%S %p')}")
    
    if view_mode == "Live Monitor": render_live_dashboard(default_target, current_label, current_price, current_cap)
    elif view_mode == "Today's Forecast": 
        f_data = fetch_forecast_data()
        render_forecast_generic(f_data['today_daily'], f_data['today_hourly'], f_data['taf'], now_miami.strftime("%A, %b %d"))
    elif view_mode == "Tomorrow's Forecast": 
        f_data = fetch_forecast_data()
        render_forecast_generic(f_data['tomorrow_daily'], f_data['tomorrow_hourly'], f_data['taf'], (now_miami + timedelta(days=1)).strftime("%A, %b %d"))

if __name__ == "__main__":
    main()
