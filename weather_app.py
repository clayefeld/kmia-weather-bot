import streamlit as st
import streamlit.components.v1 as components
import requests
import re
import pandas as pd
from datetime import datetime, timedelta, timezone 
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# --- CONFIGURATION ---
st.set_page_config(page_title="Project Helios", page_icon="☀️", layout="wide")

# URLs
NWS_API_HISTORY = "https://api.weather.gov/stations/KMIA/observations"
AWC_METAR_URL = "https://aviationweather.gov/api/data/metar?ids=KMIA&format=raw&hours=12"
NWS_POINT_URL = "https://api.weather.gov/points/25.7906,-80.3164"
AWC_TAF_URL = "https://aviationweather.gov/api/data/taf?ids=KMIA&format=raw"

# --- STYLING & UTILS ---
def get_headers():
    return {'User-Agent': '(project_helios_v30_ai_hybrid, myemail@example.com)'}

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
                    "WindVal": int(wdir) if wdir else 0
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
                    w_val = 0
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
                        "WindVal": w_val
                    })
    except Exception as e:
        if not error_msg: error_msg = str(e)
    
    return sorted(data_list, key=lambda x: x['dt_utc'], reverse=True), error_msg

@st.cache_data(ttl=300)
def fetch_forecast_data():
    data = {
        "today_daily": None, "today_hourly": [],
        "tomorrow_daily": None, "tomorrow_hourly": [],
        "taf": None, "all_hourly": []
    }
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
                    if tomorrow_str in p['startTime'] and p['isDaytime']:
                        data["tomorrow_daily"] = p
                    if today_str in p['startTime'] and p['isDaytime']:
                        data["today_daily"] = p
                    if not data["today_daily"] and today_str in p['startTime']:
                         data["today_daily"] = p

            r_h = requests.get(hourly_url, headers=get_headers(), timeout=5)
            if r_h.status_code == 200:
                periods = r_h.json().get('properties', {}).get('periods', [])
                for p in periods:
                    data["all_hourly"].append(p)
                    if tomorrow_str in p['startTime']:
                        data["tomorrow_hourly"].append(p)
                    if today_str in p['startTime']:
                        data["today_hourly"].append(p)
        
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
def render_live_dashboard():
    st.title("🔴 Project Helios: Live Feed")
    
    if st.button("🔄 Refresh System", type="primary"):
        st.cache_data.clear()
        st.rerun()
        
    history, err = fetch_live_history()
    f_data = fetch_forecast_data()
    
    if not history:
        st.error("Connection Failed: No Data Available")
        if err: st.warning(f"Debug Error: {err}")
        return

    latest = history[0]
    high_mark = max(history, key=lambda x: x['Temp'])
    high_round = int(round(high_mark['Temp']))
    
    # --- AI AGENT: PHYSICS ENGINE ---
    raw_trend = calculate_smart_trend(history)
    
    # 1. Volatility Dampener (Clamp to ±1.5)
    safe_trend = raw_trend
    if safe_trend > 1.5: safe_trend = 1.5
    if safe_trend < -1.5: safe_trend = -1.5

    # --- TIME LOGIC ---
    now_miami = get_miami_time()
    sunrise_miami = now_miami.replace(hour=7, minute=0, second=0, microsecond=0)
    sunset_miami = now_miami.replace(hour=17, minute=55, second=0, microsecond=0)
    
    is_night = False
    solar_fuel = "NIGHT"
    
    if now_miami < sunrise_miami or now_miami > sunset_miami:
        is_night = True
        # 2. Night Safety Clamp (Prevent panic drops)
        if safe_trend < -0.5: safe_trend = -0.5 
    else:
        time_left = sunset_miami - now_miami
        hrs, rem = divmod(time_left.seconds, 3600)
        mins = rem // 60
        solar_fuel = f"{hrs}h {mins}m"

    # Smart High Label
    high_label = "Day High"
    if now_miami.hour < 8: high_label = "24h High"

    # --- METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Temp", f"{latest['Temp']:.2f}°F", f"{raw_trend:+.2f}/hr")
    with col2:
        st.metric("Official Round", f"{latest['Official']}°F")
    with col3:
        st.metric(high_label, f"{high_mark['Temp']:.2f}°F", f"Officially {high_round}°F", delta_color="off")
    with col4:
        st.metric("Solar Fuel", solar_fuel)

    # --- AI PROJECTION BOARD ---
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
            
            # 3. Weighted Stability (40% Trend / 60% Forecast)
            # This makes the projection "trust" the NWS forecast more than the live sensor
            # which smoothens out short-term sensor noise.
            trend_weight = 0.4 / (i + 1)
            model_weight = 1.0 - trend_weight
            
            # Using SAFE_TREND (Dampened) instead of raw trend
            raw_proj = (curr_temp + (safe_trend * (i+1))) * trend_weight + (nws_temp * model_weight)
            
            if 0 <= latest.get('WindVal', 0) <= 180: raw_proj -= (0.5 * (i+1))
            if is_night: raw_proj -= (0.3 * (i+1))
            
            icon = "🌧️" if "Rain" in f['shortForecast'] else "☁️"
            if "Sunny" in f['shortForecast']: icon = "☀️"
            proj_vals.append(f"**+{i+1}h:** {raw_proj:.1f}°F {icon}")
        proj_str = " | ".join(proj_vals)

    trend_icon = "➡️"
    if raw_trend > 0.5: trend_icon = "🔥 Rising Fast"
    elif raw_trend > 0.1: trend_icon = "↗️ Rising"
    elif raw_trend < -0.5: trend_icon = "❄️ Dropping Fast"
    elif raw_trend < -0.1: trend_icon = "↘️ Falling"
    else: trend_icon = "➡️ Flat"

    st.success(f"**📈 TREND:** {trend_icon} ({raw_trend:+.2f}°F/hr) \n\n **🔮 AI PROJECTION:** {proj_str}")

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
        
        # Icon Logic
        sky_code = row['Sky']
        icon = "☁️" # Default
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
            "Velocity": vel_str,
            "Wind": row['Wind']
        })
        
    df = pd.DataFrame(clean_rows)
    # Formatting
    df['Temp'] = df['Temp'].apply(lambda x: f"{x:.2f}")
    
    # Rename columns for display
    df = df.rename(columns={
        "Temp": "Temp (°F)",
        "Official": "Official (Rnd)"
    })
    
    # CSS HACK to hide the index column (Left Column)
    hide_table_row_index = """
        <style>
        thead tr th:first-child {display:none}
        tbody th {display:none}
        </style>
        """
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    st.table(df)

# --- VIEW: FORECAST RENDERER ---
def render_forecast_generic(daily, hourly, taf, date_label):
    st.title(f"☀️ Helios Forecast: {date_label}")
    
    if st.button(f"🔄 Refresh {date_label}"):
        st.cache_data.clear()
        st.rerun()
    
    if not hourly:
        st.warning(f"Forecast data unavailable for {date_label}.")
        return

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
        
        risk_level = "Safe"
        if "Rain" in short or "Thunder" in short: risk_level = "⚠️ RISK"

        h_data.append({
            "Time": dt.strftime("%I %p"),
            "Temp": h['temperature'],
            "Condition": f"{icon} {short}",
            "Wind": f"{h['windDirection']} {h['windSpeed']}",
            "Trade Risk": risk_level
        })

    df_h = pd.DataFrame(h_data)
    df_h['Temp'] = df_h['Temp'].apply(lambda x: f"{x:.0f}")
    df_h = df_h.rename(columns={"Temp": "Temp (°F)"})
    
    hide_table_row_index = """
        <style>
        thead tr th:first-child {display:none}
        tbody th {display:none}
        </style>
        """
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    st.table(df_h)

    if taf:
        st.divider()
        st.caption("✈️ AVIATION TAF (PILOT DATA)")
        st.code(taf, language="text")

# --- MAIN APP ---
def main():
    st.sidebar.header("PROJECT HELIOS ☀️")
    st.sidebar.caption("High-Frequency Weather Algo")
    
    view_mode = st.sidebar.radio("Command Deck:", [
        "Live Monitor", 
        "Today's Forecast", 
        "Tomorrow's Forecast"
    ])
    
    st.sidebar.divider()
    
    auto_refresh = st.sidebar.checkbox("⚡ Auto-Refresh (Every 60s)", value=False)
    if auto_refresh:
        components.html(
            f"""
                <script>
                    setTimeout(function(){{
                        window.parent.location.reload();
                    }}, 60000);
                </script>
            """,
            height=0
        )

    now_miami = get_miami_time()
    st.sidebar.caption(f"System Time: {now_miami.strftime('%I:%M:%S %p')}")

    f_data = fetch_forecast_data()

    if view_mode == "Live Monitor":
        render_live_dashboard()
        
    elif view_mode == "Today's Forecast":
        today_lbl = now_miami.strftime("%A, %b %d")
        render_forecast_generic(
            f_data['today_daily'], 
            f_data['today_hourly'], 
            f_data['taf'], 
            today_lbl
        )
        
    elif view_mode == "Tomorrow's Forecast":
        tomorrow_lbl = (now_miami + timedelta(days=1)).strftime("%A, %b %d")
        render_forecast_generic(
            f_data['tomorrow_daily'], 
            f_data['tomorrow_hourly'], 
            f_data['taf'], 
            tomorrow_lbl
        )

if __name__ == "__main__":
    main()
