import streamlit as st
import requests
import re
import pandas as pd
import time
from datetime import datetime, timedelta, timezone

# --- CONFIGURATION ---
st.set_page_config(page_title="KMIA Unified Command", page_icon="🌤️", layout="wide")

# URLs
NWS_API_HISTORY = "https://api.weather.gov/stations/KMIA/observations"
AWC_METAR_URL = "https://aviationweather.gov/api/data/metar?ids=KMIA&format=raw&hours=12"
NWS_POINT_URL = "https://api.weather.gov/points/25.7906,-80.3164"
AWC_TAF_URL = "https://aviationweather.gov/api/data/taf?ids=KMIA&format=raw"

# --- UTILS ---
def get_headers():
    return {'User-Agent': '(myweatherbot_unified, myemail@example.com)'}

def parse_iso_time(iso_str):
    try:
        return datetime.fromisoformat(iso_str)
    except:
        return None

def get_display_time(dt_utc):
    # Convert UTC to EST (UTC-5)
    dt_est = dt_utc.astimezone(timezone(timedelta(hours=-5)))
    return dt_est.strftime("%I:%M %p")

# --- FETCHERS: LIVE DATA (GOLD MASTER) ---
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
                dew_c = props.get('dewpoint', {}).get('value')
                dew_f = (dew_c * 1.8) + 32 if dew_c else 0.0
                rel_hum = props.get('relativeHumidity', {}).get('value')
                
                wdir = props.get('windDirection', {}).get('value')
                wspd = props.get('windSpeed', {}).get('value')
                w_str = "--"
                if wdir is not None and wspd is not None:
                     w_str = f"{int(wdir):03d}@{int(wspd/1.852):02d}"

                sky_str = "---"
                clouds = props.get('cloudLayers', [])
                if clouds: sky_str = clouds[0].get('amount', '---')

                data_list.append({
                    "dt_utc": dt_utc,
                    "source": "NWS",
                    "f": f_val,
                    "r": int(round(f_val)),
                    "dew": dew_f,
                    "hum": rel_hum if rel_hum else 0,
                    "wind": w_str,
                    "wind_val": int(wdir) if wdir is not None else -1,
                    "sky": sky_str
                })
    except: pass

    # 2. Aviation METAR
    try:
        r = requests.get(AWC_METAR_URL, timeout=4)
        for line in r.text.split('\n'):
            if "KMIA" in line:
                t_match = re.search(r" T(\d)(\d{3})", line)
                time_match = re.search(r"\b(\d{2})(\d{4})Z\b", line)
                if t_match and time_match:
                    day, tm = int(time_match.group(1)), time_match.group(2)
                    now = datetime.now(timezone.utc)
                    month, year = now.month, now.year
                    # Simple rollover logic
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
                        w_str = f"{w_m.group(1)}@{w_m.group(2)}"
                        if w_m.group(1).isdigit(): w_val = int(w_m.group(1))

                    s_m = re.search(r"\b(CLR|FEW|SCT|BKN|OVC|VV)\d{3}", line)
                    sky = s_m.group(1) if s_m else "CLR"

                    data_list.append({
                        "dt_utc": dt_utc,
                        "source": "AWC",
                        "f": f_val,
                        "r": int(round(f_val)),
                        "dew": 0.0,
                        "hum": 0,
                        "wind": w_str,
                        "wind_val": w_val,
                        "sky": sky
                    })
    except: pass
    
    return sorted(data_list, key=lambda x: x['dt_utc'], reverse=True)

# --- FETCHERS: FORECAST DATA (TOMORROW) ---
@st.cache_data(ttl=300)
def fetch_forecast_data():
    daily_data, hourly_data, taf_data = None, [], None
    try:
        # Get URLs
        r = requests.get(NWS_POINT_URL, headers=get_headers(), timeout=5)
        if r.status_code == 200:
            props = r.json().get('properties', {})
            daily_url = props.get('forecast')
            hourly_url = props.get('forecastHourly')
            
            # Daily
            r_d = requests.get(daily_url, headers=get_headers(), timeout=5)
            if r_d.status_code == 200:
                periods = r_d.json().get('properties', {}).get('periods', [])
                target = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                for p in periods:
                    if target in p['startTime'] and p['isDaytime']:
                        daily_data = p
                        break
            
            # Hourly
            r_h = requests.get(hourly_url, headers=get_headers(), timeout=5)
            if r_h.status_code == 200:
                periods = r_h.json().get('properties', {}).get('periods', [])
                target = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                for p in periods:
                    if target in p['startTime']:
                        hourly_data.append(p)
        
        # TAF
        r_t = requests.get(AWC_TAF_URL, timeout=5)
        if r_t.status_code == 200:
            taf_data = r_t.text
            
    except: pass
    return daily_data, hourly_data, taf_data

# --- MATH HELPER ---
def calculate_smart_trend(master_list):
    if len(master_list) < 2: return 0.0
    now = master_list[0]['dt_utc']
    one_hr_ago = now - timedelta(hours=1)
    points = [p for p in master_list if p['dt_utc'] >= one_hr_ago]
    if len(points) < 2: return 0.0
    x = [(p['dt_utc'] - one_hr_ago).total_seconds()/60 for p in points]
    y = [p['f'] for p in points]
    N = len(x)
    sum_x, sum_y = sum(x), sum(y)
    sum_xy = sum(i*j for i, j in zip(x,y))
    sum_xx = sum(i*i for i in x)
    den = (N*sum_xx - sum_x*sum_x)
    if den == 0: return 0.0
    return ((N*sum_xy - sum_x*sum_y) / den) * 60

# --- TAB 1: LIVE DASHBOARD ---
def render_live_dashboard():
    st.header("🔴 Live Market Monitor (Gold Master)")
    
    if st.button("🔄 Refresh Live Data"):
        st.rerun()

    history = fetch_live_history()
    
    if not history:
        st.error("No data available.")
        return

    latest = history[0]
    high_mark = max(history, key=lambda x: x['f'])
    smart_trend = calculate_smart_trend(history)

    # Solar Logic
    now_est = datetime.now(timezone(timedelta(hours=-5)))
    sunset_est = now_est.replace(hour=17, minute=55, second=0, microsecond=0)
    time_left = sunset_est - now_est
    
    solar_fuel = "NIGHT MODE"
    if time_left.total_seconds() > 0:
        hrs, rem = divmod(time_left.seconds, 3600)
        mins = rem // 60
        solar_fuel = f"{hrs}h {mins}m LEFT"
    
    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Current Temp", f"{latest['f']:.2f}°F", f"{smart_trend:+.2f}/hr")
    with c2: st.metric("Official Round", f"{latest['r']}°F")
    with c3: st.metric("Today's High", f"{high_mark['f']:.2f}°F")
    with c4: st.metric("Solar Fuel", solar_fuel)

    # Status Bar
    status_msg = "❄️ COLD"
    status_color = "blue"
    if 71.5 <= latest['f'] < 72.0: 
        status_msg = "⚠️ TRAP ZONE"
        status_color = "orange"
    elif latest['f'] >= 72.0: 
        status_msg = "✅ TARGET SECURED"
        status_color = "green"
    
    st.markdown(f":{status_color}-background[**STATUS: {status_msg}**]")

    # Table
    st.subheader("Observation Log")
    table_data = []
    for i, row in enumerate(history[:15]):
        # Velocity calc
        vel = 0.0
        if i < len(history) - 1:
            prev = history[i+1]
            dt1 = row['dt_utc']
            dt2 = prev['dt_utc']
            diff = (dt1 - dt2).total_seconds()/3600.0
            if diff > 0: vel = (row['f'] - prev['f']) / diff
        
        table_data.append({
            "Time": get_display_time(row['dt_utc']),
            "Source": row['source'],
            "Temp": f"{row['f']:.2f}",
            "Rnd": row['r'],
            "Vel": f"{vel:+.1f}" if abs(vel) > 0.1 else "--",
            "Wind": row['wind'],
            "Sky": row['sky']
        })
    
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

# --- TAB 2: FORECAST ---
def render_forecast_dashboard():
    st.header("📅 Tomorrow's Trading Plan")
    
    daily, hourly, taf = fetch_forecast_data()
    
    if not hourly:
        st.warning("Forecast data unavailable.")
        return

    # Calculate Score
    score = 10
    rain_hours = 0
    wind_hours = 0
    for h in hourly:
        short = h['shortForecast'].lower()
        if "rain" in short or "shower" in short: rain_hours += 1
        if "thunder" in short: rain_hours += 2
    
    score -= (rain_hours * 2)
    score = max(1, min(10, score))
    
    # Header
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Condition Score", f"{score}/10")
    with c2: 
        if daily: st.metric("Proj. High", f"{daily['temperature']}°F")
    with c3:
        vol = "LOW" if score > 7 else "HIGH"
        st.metric("Volatility", vol, delta_color="inverse")

    st.write(f"**Confidence:**")
    st.progress(score/10)
    
    if daily:
        st.info(f"**Analyst Note:** {daily['shortForecast']}. Winds {daily['windSpeed']}.")

    # Hourly Table
    st.subheader("Hourly Breakdown")
    h_data = []
    for h in hourly:
        dt = parse_iso_time(h['startTime'])
        icon = "☁️"
        short = h['shortForecast']
        if "Sunny" in short: icon = "☀️"
        if "Rain" in short: icon = "🌧️"
        
        risk = ""
        if "Rain" in short or "Thunder" in short: risk = "⚠️ RISK"

        h_data.append({
            "Time": dt.strftime("%I %p"),
            "Temp": h['temperature'],
            "Wind": f"{h['windDirection']} {h['windSpeed']}",
            "Condition": f"{icon} {short}",
            "Risk": risk
        })
    
    st.dataframe(pd.DataFrame(h_data), use_container_width=True, hide_index=True)

    if taf:
        with st.expander("✈️ View Pilot TAF Data"):
            st.code(taf)

# --- MAIN APP LAYOUT ---
def main():
    tab1, tab2 = st.tabs(["🔴 Live Dashboard", "📅 Tomorrow's Forecast"])
    
    with tab1:
        render_live_dashboard()
    
    with tab2:
        render_forecast_dashboard()

if __name__ == "__main__":
    main()
