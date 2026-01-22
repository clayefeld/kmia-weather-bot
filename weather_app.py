import streamlit as st
import requests
import re
import pandas as pd
import time
from datetime import datetime, timedelta, timezone

# --- CONFIGURATION ---
st.set_page_config(page_title="KMIA Weather Bot", page_icon="📡", layout="wide")

# URLs
NWS_API_HISTORY = "https://api.weather.gov/stations/KMIA/observations"
AWC_METAR_URL = "https://aviationweather.gov/api/data/metar?ids=KMIA&format=raw&hours=12"
NWS_POINT_URL = "https://api.weather.gov/points/25.7906,-80.3164"
AWC_TAF_URL = "https://aviationweather.gov/api/data/taf?ids=KMIA&format=raw"

# --- STYLING & UTILS ---
def get_headers():
    return {'User-Agent': '(myweatherbot_clean_v4, myemail@example.com)'}

def parse_iso_time(iso_str):
    try:
        return datetime.fromisoformat(iso_str)
    except:
        return None

def get_display_time(dt_utc):
    dt_est = dt_utc.astimezone(timezone(timedelta(hours=-5)))
    return dt_est.strftime("%I:%M %p")

# --- FETCHERS ---
def fetch_live_history():
    data_list = []
    # 1. NWS History
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
                    "Sky": sky_str
                })
    except: pass

    # 2. Aviation METAR
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
                    if w_m: w_str = f"{w_m.group(1)} @ {w_m.group(2)}kt"

                    s_m = re.search(r"\b(CLR|FEW|SCT|BKN|OVC|VV)\d{3}", line)
                    sky = s_m.group(1) if s_m else "CLR"

                    data_list.append({
                        "dt_utc": dt_utc,
                        "Source": "AWC",
                        "Temp": f_val,
                        "Official": int(round(f_val)),
                        "Wind": w_str,
                        "Sky": sky
                    })
    except: pass
    
    return sorted(data_list, key=lambda x: x['dt_utc'], reverse=True)

@st.cache_data(ttl=300)
def fetch_forecast_data():
    daily_data, hourly_data, taf_data = None, [], None
    try:
        r = requests.get(NWS_POINT_URL, headers=get_headers(), timeout=5)
        if r.status_code == 200:
            props = r.json().get('properties', {})
            daily_url = props.get('forecast')
            hourly_url = props.get('forecastHourly')
            
            r_d = requests.get(daily_url, headers=get_headers(), timeout=5)
            if r_d.status_code == 200:
                periods = r_d.json().get('properties', {}).get('periods', [])
                target = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                for p in periods:
                    if target in p['startTime'] and p['isDaytime']:
                        daily_data = p
                        break
            
            r_h = requests.get(hourly_url, headers=get_headers(), timeout=5)
            if r_h.status_code == 200:
                periods = r_h.json().get('properties', {}).get('periods', [])
                target = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                for p in periods:
                    if target in p['startTime']:
                        hourly_data.append(p)
        
        r_t = requests.get(AWC_TAF_URL, timeout=5)
        if r_t.status_code == 200: taf_data = r_t.text
    except: pass
    return daily_data, hourly_data, taf_data

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
    st.title("🔴 Live Market Monitor")
    
    if st.button("🔄 Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()
        
    history = fetch_live_history()
    if not history:
        st.error("Connection Failed: No Data")
        return

    latest = history[0]
    high_mark = max(history, key=lambda x: x['Temp'])
    smart_trend = calculate_smart_trend(history)

    # Solar Calc
    now_est = datetime.now(timezone(timedelta(hours=-5)))
    sunset_est = now_est.replace(hour=17, minute=55, second=0, microsecond=0)
    time_left = sunset_est - now_est
    solar_fuel = "NIGHT"
    if time_left.total_seconds() > 0:
        hrs, rem = divmod(time_left.seconds, 3600)
        mins = rem // 60
        solar_fuel = f"{hrs}h {mins}m"

    # --- BIG METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Temp", f"{latest['Temp']:.1f}°F", f"{smart_trend:+.1f}/hr")
    with col2:
        st.metric("Official Round", f"{latest['Official']}°F")
    with col3:
        st.metric("Day High", f"{high_mark['Temp']:.1f}°F")
    with col4:
        st.metric("Solar Fuel", solar_fuel)

    # --- STATUS BAR ---
    status_msg = "❄️ COLD"
    if 71.5 <= latest['Temp'] < 72.0: status_msg = "⚠️ TRAP ZONE"
    elif latest['Temp'] >= 72.0: status_msg = "✅ TARGET SECURED"
    st.info(f"**STATUS: {status_msg}**")

    # --- CLEAN TABLE ---
    st.subheader("Data Feed")
    
    clean_rows = []
    for i, row in enumerate(history[:15]):
        # Velocity
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
        
        clean_rows.append({
            "Time": get_display_time(row['dt_utc']),
            "Src": row['Source'],
            "Temp": row['Temp'],
            "Trend": vel_str,
            "Wind": row['Wind'],
            "Sky": row['Sky']
        })
        
    df = pd.DataFrame(clean_rows)
    
    # Configure the table to be SUPER readable
    st.dataframe(
        df,
        column_config={
            "Temp": st.column_config.NumberColumn(
                "Temp (°F)",
                format="%.1f",
                step=0.1,
            ),
            "Trend": st.column_config.TextColumn("Velocity"),
            "Src": st.column_config.TextColumn("Source", width="small"),
        },
        use_container_width=True,
        hide_index=True
    )

# --- VIEW: FORECAST ---
def render_forecast_dashboard():
    st.title("📅 Tomorrow's Plan")
    
    if st.button("🔄 Refresh Forecast"):
        st.cache_data.clear()
        st.rerun()

    daily, hourly, taf = fetch_forecast_data()
    
    if not hourly:
        st.warning("Forecast unavailable. Try again later.")
        return

    # --- CORRECTED SCORE LOGIC ---
    score = 10
    rain_hours = 0
    
    for h in hourly:
        s = h['shortForecast'].lower()
        if "rain" in s or "shower" in s: rain_hours += 1
        if "thunder" in s: rain_hours += 2 
    
    # Apply penalties based on THRESHOLDS (not per hour)
    if rain_hours > 0: score -= 2        # Penalty for rain appearing
    if rain_hours > 4: score -= 2        # Extra penalty if it lingers (>4 hours)
    
    score = max(1, min(10, score))
    
    # --- TOP SECTION ---
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Confidence Score", f"{score}/10")
        st.progress(score/10)
    with c2:
        if daily:
            st.success(f"**Analyst Note:** {daily['shortForecast']}")
            st.caption(f"Winds: {daily['windSpeed']} • High: {daily['temperature']}°F")

    # --- HOURLY TABLE ---
    st.subheader("Hourly Risk Breakdown")
    
    h_data = []
    for h in hourly:
        dt = parse_iso_time(h['startTime'])
        short = h['shortForecast']
        
        # Simple Icons
        icon = "☁️"
        if "Sunny" in short: icon = "☀️"
        if "Rain" in short: icon = "🌧️"
        if "Thunder" in short: icon = "⛈️"
        
        # Risk Flag
        risk_level = "Safe"
        if "Rain" in short or "Thunder" in short: risk_level = "⚠️ RISK"

        h_data.append({
            "Time": dt.strftime("%I %p"),
            "Temp": h['temperature'],
            "Condition": f"{icon} {short}",
            "Wind": f"{h['windDirection']} {h['windSpeed']}",
            "Status": risk_level
        })

    df_h = pd.DataFrame(h_data)
    
    st.dataframe(
        df_h,
        column_config={
            "Temp": st.column_config.NumberColumn("Temp (°F)", format="%d"),
            "Status": st.column_config.TextColumn("Trade Risk"),
        },
        use_container_width=True,
        hide_index=True
    )

    if taf:
        st.divider()
        st.caption("✈️ AVIATION TAF (PILOT DATA)")
        st.code(taf, language="text")

# --- MAIN APP ---
def main():
    # SIDEBAR NAVIGATION
    st.sidebar.header("Navigation")
    view_mode = st.sidebar.radio("Select View:", ["Live Monitor", "Tomorrow's Forecast"])
    
    st.sidebar.divider()
    st.sidebar.caption(f"Last Load: {datetime.now().strftime('%H:%M:%S')}")

    if view_mode == "Live Monitor":
        render_live_dashboard()
    else:
        render_forecast_dashboard()

if __name__ == "__main__":
    main()
