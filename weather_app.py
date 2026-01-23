import streamlit as st
import streamlit.components.v1 as components
import requests
import re
import json
import base64
import time
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
    </style>
    """

# --- KALSHI CLIENT (SECURE AUTH) ---
class KalshiAuth:
    def __init__(self):
        # SECURE FETCH: Look for keys in Streamlit Secrets
        try:
            self.key_id = st.secrets["KALSHI_KEY_ID"]
            # Convert string newline characters back to real newlines if needed
            self.private_key_str = st.secrets["KALSHI_PRIVATE_KEY"].replace("\\n", "\n")
            
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
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')

def fetch_kalshi_market(target_strike):
    """
    Fetches the specific contract for today's High Temp that matches the target_strike.
    """
    if not CRYPTO_AVAILABLE:
        return 0, "Crypto Lib Missing", "Install 'cryptography'"
        
    auth = KalshiAuth()
    if not auth.ready:
        return 0, "No Keys", "Add to .streamlit/secrets.toml"

    try:
        # 1. Generate Ticker Context
        now_miami = datetime.now(ZoneInfo("US/Eastern"))
        # Kalshi High Temp Event Ticker format: KXHIGHMIA-YYMMDD
        date_str = now_miami.strftime("%y%b%d").upper() # e.g. 26JAN23
        event_ticker = f"KXHIGHMIA-{date_str}"
        
        # 2. Prepare Request
        path = f"/events/{event_ticker}"
        ts = str(int(time.time() * 1000))
        sig = auth.sign_request("GET", path, ts)
        
        headers = {
            "KALSHI-ACCESS-KEY": auth.key_id,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "Content-Type": "application/json"
        }
        
        # 3. Execute
        r = requests.get(KALSHI_API_URL + path, headers=headers, timeout=3)
        
        if r.status_code != 200:
            return 0, "API Error", f"Status: {r.status_code}"
            
        data = r.json()
        markets = data.get('markets', [])
        
        # 4. Find Strike Match
        best_market = None
        for m in markets:
            if m.get('floor_strike') == target_strike:
                best_market = m
                break
        
        if not best_market:
            return 0, "Strike Not Found", f"No {target_strike} market"
            
        yes_ask = best_market.get('yes_ask')
        if not yes_ask: return 0, "No Liquidity", "No Asks"
        
        return yes_ask, best_market.get('ticker'), None

    except Exception as e:
        return 0, "Error", str(e)

# --- STYLING & UTILS ---
def get_headers():
    return {'User-Agent': '(project_helios_v35_secure, myemail@example.com)'}

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

# --- AI AGENT LOGIC ---
def get_agent_analysis(trend, hum, wind_dir, solar_min, sky):
    reasons = []
    sentiment = "NEUTRAL"
    confidence = 50 # Base
    
    # 1. Solar
    if solar_min <= 0:
        reasons.append("Night mode (No fuel)")
        sentiment = "BEARISH"
        confidence = 20
    elif solar_min < 120:
        reasons.append("Low solar angle")
        confidence = 40
        
    # 2. Moisture
    if hum > 85:
        reasons.append("Atmosphere Saturated")
        if trend > 1.0: 
            reasons.append("⚠️ RALLY SUSPECT (Divergence)")
            sentiment = "TRAP"
            confidence = 10 
    
    # 3. Wind
    if 0 <= wind_dir <= 180:
        reasons.append("Ocean Breeze (Cooling)")
        if sentiment == "NEUTRAL": confidence = 30
        
    # 4. Sky
    if "OVC" in sky or "BKN" in sky:
        reasons.append("Clouds blocking sun")
        if sentiment != "TRAP": sentiment = "BEARISH"
        confidence = 30
    elif "CLR" in sky or "FEW" in sky:
        if solar_min > 120:
            reasons.append("Clear Skies (Heating)")
            if sentiment == "NEUTRAL": 
                sentiment = "BULLISH"
                confidence = 85

    if not reasons: reasons.append("Nominal.")
    summary = " + ".join(reasons)
    return sentiment, summary, confidence

# --- FETCHERS ---
def fetch_live_history():
    data_list = []
    error_msg = None
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

                ts = props.get('timestamp')
                if not ts: continue
                dt_utc = datetime.fromisoformat(ts.split('+')[0]).replace(tzinfo=timezone.utc)
                f_val = (temp_c * 1.8) + 32
                
                wdir = props.get('windDirection', {}).get('value')
                wspd = props.get('windSpeed', {}).get('value')
                w_str = "--"
                if wdir is not None and wspd is not None:
                     w_str = f"{int(wdir):03d} @ {int(wspd/1.852)}kt"
                
                sky_str = "--"
                clouds = props.get('cloudLayers', [])
                if clouds: sky_str = clouds[0].get('amount', '--')

                data_list.append({
                    "dt_utc": dt_utc,
                    "Source": "NWS",
                    "Temp": f_val,
                    "Official": int(round(f_val)),
                    "Wind": w_str,
                    "Sky": sky_str,
                    "WindVal": int(wdir) if wdir else -1,
                    "Hum": humidity,
                    "Dew": dew_f
                })
    except Exception as e:
        error_msg = str(e)

    try:
        r = requests.get(AWC_METAR_URL, timeout=4)
        for line in r.text.split('\n'):
            if "KMIA" in line:
                time_match = re.search(r"\b(\d{2})(\d{4})Z\b", line)
                t_match = re.search(r" T(\d)(\d{3})", line)
                if t_match and time_match:
                    day, tm = int(time_match.group(1)), time_match.group(2)
                    now = datetime.now(timezone.utc)
                    month, year = now.month, now.year
                    if now.day < 5 and day > 25: month -= 1
                    if month == 0: month, year = 12, year - 1
                    dt_utc = datetime(year, month, day, int(tm[:2]), int(tm[2:]), tzinfo=timezone.utc)
                    
                    sign = -1 if t_match.group(1) == '1' else 1
                    c = sign * (int(t_match.group(2))/10.0)
                    f_val = (c * 1.8) + 32
                    
                    w_m = re.search(r"\b(\d{3}|VRB)(\d{2,3})G?(\d{2,3})?KT\b", line)
                    w_str = "--"
                    w_val = -1
                    if w_m: 
                        w_str = f"{w_m.group(1)} @ {w_m.group(2)}kt"
                        if w_m.group(1).isdigit(): w_val = int(w_m.group(1))

                    s_m = re.search(r"\b(CLR|FEW|SCT|BKN|OVC|VV)\d{3}", line)
                    sky = s_m.group(1) if s_m else "CLR"

                    data_list.append({
                        "dt_utc": dt_utc,
                        "Source": "AWC",
                        "Temp": f_val,
                        "Official": int(round(f_val)),
                        "Wind": w_str,
                        "Sky": sky,
                        "WindVal": w_val,
                        "Hum": 0, 
                        "Dew": 0
                    })
    except Exception as e:
        if not error_msg: error_msg = str(e)
    
    return sorted(data_list, key=lambda x: x['dt_utc'], reverse=True), error_msg

@st.cache_data(ttl=300)
def fetch_forecast_data():
    data = {"today_daily": None, "today_hourly": [], "tomorrow_daily": None, "tomorrow_hourly": [], "taf": None, "all_hourly": []}
    try:
        r = requests.get(NWS_POINT_URL, headers=get_headers(), timeout=5)
        if r.status_code == 200:
            props = r.json().get('properties', {})
            daily_url = props.get('forecast')
            hourly_url = props.get('forecastHourly')
            
            now_miami = get_miami_time()
            today_str = now_miami.strftime("%Y-%m-%d")
            tomorrow_str = (now_miami + timedelta(days=1)).strftime("%Y-%m-%d")
            
            r_d = requests.get(daily_url, headers=get_headers(), timeout=5)
            if r_d.status_code == 200:
                periods = r_d.json().get('properties', {}).get('periods', [])
                for p in periods:
                    if tomorrow_str in p['startTime'] and p['isDaytime']: data["tomorrow_daily"] = p
                    if today_str in p['startTime'] and p['isDaytime']: data["today_daily"] = p
                    if not data["today_daily"] and today_str in p['startTime']: data["today_daily"] = p

            r_h = requests.get(hourly_url, headers=get_headers(), timeout=5)
            if r_h.status_code == 200:
                periods = r_h.json().get('properties', {}).get('periods', [])
                for p in periods:
                    data["all_hourly"].append(p)
                    if tomorrow_str in p['startTime']: data["tomorrow_hourly"].append(p)
                    if today_str in p['startTime']: data["today_hourly"].append(p)
        
        r_t = requests.get(AWC_TAF_URL, timeout=5)
        if r_t.status_code == 200: data["taf"] = r_t.text
    except: pass
    return data

# --- MATH ---
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
def render_live_dashboard(target_temp, show_target, manual_price):
    st.title("🔴 Project Helios: Live Feed")
    
    if st.button("🔄 Refresh System", type="primary"):
        st.cache_data.clear()
        st.rerun()
        
    history, err = fetch_live_history()
    f_data = fetch_forecast_data()
    
    if not history:
        st.error("Connection Failed: No Data Available")
        return

    latest = history[0]
    high_mark = max(history, key=lambda x: x['Temp'])
    high_round = int(round(high_mark['Temp']))
    smart_trend = calculate_smart_trend(history)

    # --- TIME & SOLAR LOGIC ---
    now_miami = get_miami_time()
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

    # --- DAMPENER ---
    safe_trend = smart_trend
    if is_night and safe_trend < -0.5: safe_trend = -0.5
    
    # --- AI AGENT EXECUTION ---
    hum = latest.get('Hum', 0)
    wind_dir = latest.get('WindVal', -1)
    if wind_dir == -1:
        for h in history[1:5]:
            if h.get('WindVal', -1) != -1:
                wind_dir = h['WindVal']
                break
                
    ai_sent, ai_reason, ai_conf = get_agent_analysis(safe_trend, hum, wind_dir, solar_min, latest['Sky'])

    # --- FETCH KALSHI MARKET (SECURELY) ---
    kalshi_price = 0
    kalshi_status = "Manual"
    
    # Try fetching via API first (if keys are in secrets)
    k_price, k_ticker, k_err = fetch_kalshi_market(target_temp)
    if k_price > 0:
        kalshi_price = k_price
        kalshi_status = f"API ({k_ticker})"
    else:
        # Fallback to manual
        kalshi_price = manual_price
        kalshi_status = f"Manual"
        # Optional: display error if we expected API to work
        # if k_err != "No Keys": st.toast(f"API Error: {k_err}")

    # --- METRICS ---
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Current Temp", f"{latest['Temp']:.2f}°F", f"{smart_trend:+.2f}/hr")
    with c2: st.metric("Official Round", f"{latest['Official']}°F")
    with c3:
        high_label = "Day High" if now_miami.hour >= 8 else "24h High"
        st.metric(high_label, f"{high_mark['Temp']:.2f}°F", f"Officially {high_round}°F", delta_color="off")
    with c4: st.metric("Solar Fuel", solar_fuel)

    # --- AI ANALYSIS & MARKET ARB ---
    m_col1, m_col2 = st.columns([2, 1])
    
    with m_col1:
        sentiment_color = "blue"
        if ai_sent == "BULLISH": sentiment_color = "green"
        if ai_sent == "BEARISH": sentiment_color = "red"
        if ai_sent == "TRAP": sentiment_color = "orange"
        st.info(f"🤖 **ANALYST SENTIMENT:** :{sentiment_color}[{ai_sent}]\n\n{ai_reason}")

    with m_col2:
        edge = ai_conf - kalshi_price
        edge_color = "off"
        if edge > 15: edge_color = "normal" 
        if edge < -15: edge_color = "inverse"
        st.metric(f"Market 'Yes' ({kalshi_status})", f"{kalshi_price}¢", f"{edge:+.0f}% Edge", delta_color=edge_color)

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

    if show_target:
        status_msg = "❄️ COLD"
        if target_temp - 0.5 <= latest['Temp'] < target_temp:
            status_msg = f"⚠️ TRAP ZONE (Within 0.5° of {target_temp})"
            st.warning(f"**STATUS:** {status_msg}")
        elif latest['Temp'] >= target_temp:
            status_msg = f"✅ TARGET SECURED (Above {target_temp})"
            st.success(f"**STATUS:** {status_msg}")
        else:
            st.caption(f"**STATUS:** {status_msg} (Gap: {target_temp - latest['Temp']:.2f}°F)")

    # --- SENSOR TABLE ---
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
        if "CLR" in sky_code or "SKC" in sky_code:
            icon = "🌙" if is_night else "☀️"
        elif "FEW" in sky_code: icon = "🌤️"
        elif "SCT" in sky_code: icon = "⛅"
        elif "BKN" in sky_code: icon = "🌥️"
        elif "OVC" in sky_code: icon = "☁️"
        
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
    st.markdown(HIDE_INDEX_CSS, unsafe_allow_html=True)
    st.table(df_h)
    if taf: st.divider(); st.caption("✈️ AVIATION TAF (PILOT DATA)"); st.code(taf, language="text")

# --- MAIN APP ---
def main():
    st.sidebar.header("PROJECT HELIOS ☀️")
    st.sidebar.caption("High-Frequency Weather Algo")
    view_mode = st.sidebar.radio("Command Deck:", ["Live Monitor", "Today's Forecast", "Tomorrow's Forecast"])
    st.sidebar.divider()
    
    auto_refresh = st.sidebar.checkbox("⚡ Auto-Refresh (Every 60s)", value=False)
    if auto_refresh: components.html(f"""<script>setTimeout(function(){{window.parent.location.reload();}}, 60000);</script>""", height=0)

    # DYNAMIC TARGET INPUT
    show_target = st.sidebar.checkbox("🎯 Active Target Line", value=True)
    target_temp = 76.0 
    if show_target:
        target_temp = st.sidebar.number_input("Strike Price", value=76.0, step=0.1, format="%.1f")

    # KALSHI MARKET INPUT
    st.sidebar.divider()
    st.sidebar.markdown("**📉 Market Sentiment**")
    manual_price = st.sidebar.slider("Manual Price Override", 1, 99, 50)

    now_miami = get_miami_time()
    st.sidebar.caption(f"System Time: {now_miami.strftime('%I:%M:%S %p')}")
    f_data = fetch_forecast_data()
    
    if view_mode == "Live Monitor": render_live_dashboard(target_temp, show_target, manual_price)
    elif view_mode == "Today's Forecast": render_forecast_generic(f_data['today_daily'], f_data['today_hourly'], f_data['taf'], now_miami.strftime("%A, %b %d"))
    elif view_mode == "Tomorrow's Forecast": render_forecast_generic(f_data['tomorrow_daily'], f_data['tomorrow_hourly'], f_data['taf'], (now_miami + timedelta(days=1)).strftime("%A, %b %d"))

if __name__ == "__main__":
    main()
