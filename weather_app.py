import streamlit as st
import requests
import re
import pandas as pd
from datetime import datetime, timedelta

# --- CONFIGURATION ---
st.set_page_config(page_title="KMIA Forecast", page_icon="⛈️", layout="centered")

NWS_POINT_URL = "https://api.weather.gov/points/25.7906,-80.3164"
AWC_TAF_URL = "https://aviationweather.gov/api/data/taf?ids=KMIA&format=raw"

# --- UTILS ---
def get_tomorrow_date():
    return datetime.now() + timedelta(days=1)

def get_headers():
    return {'User-Agent': '(myweatherbot_streamlit, myemail@example.com)'}

def parse_iso_time(iso_str):
    try:
        return datetime.fromisoformat(iso_str)
    except:
        return None

# --- FETCHERS ---
@st.cache_data(ttl=300) 
def fetch_all_weather_data():
    try:
        # 1. Get Metadata
        r = requests.get(NWS_POINT_URL, headers=get_headers(), timeout=5)
        if r.status_code != 200: return None, None, None
        
        props = r.json().get('properties', {})
        daily_url = props.get('forecast')
        hourly_url = props.get('forecastHourly')
        
        # 2. Fetch Daily
        daily_data = None
        r_daily = requests.get(daily_url, headers=get_headers(), timeout=5)
        if r_daily.status_code == 200:
            periods = r_daily.json().get('properties', {}).get('periods', [])
            target_date_str = get_tomorrow_date().strftime("%Y-%m-%d")
            for p in periods:
                if target_date_str in p['startTime'] and p['isDaytime']:
                    daily_data = p
                    break
        
        # 3. Fetch Hourly
        hourly_data = []
        r_hourly = requests.get(hourly_url, headers=get_headers(), timeout=5)
        if r_hourly.status_code == 200:
            periods = r_hourly.json().get('properties', {}).get('periods', [])
            target_date_str = get_tomorrow_date().strftime("%Y-%m-%d")
            for p in periods:
                if target_date_str in p['startTime']:
                    hourly_data.append(p)

        # 4. Fetch Aviation
        taf_data = None
        try:
            r_taf = requests.get(AWC_TAF_URL, timeout=5)
            if r_taf.status_code == 200:
                taf_data = r_taf.text
        except: pass

        return daily_data, hourly_data, taf_data

    except Exception as e:
        return None, None, None

def calculate_score(daily, hourly):
    score = 10
    volatility = "LOW"
    
    rain_hours = 0
    high_wind_hours = 0
    
    for h in hourly:
        short = h.get('shortForecast', '').lower()
        wind_speed = h.get('windSpeed', '0 mph')
        w_int = 0
        w_match = re.search(r'\d+', wind_speed)
        if w_match: w_int = int(w_match.group(0))
        
        if "rain" in short or "shower" in short: rain_hours += 1
        if "thunder" in short: rain_hours += 2
        if w_int > 15: high_wind_hours += 1

    if rain_hours > 0: score -= 2
    if rain_hours > 4: score -= 2
    if high_wind_hours > 3: score -= 2
    
    if daily:
        if "thunder" in daily.get('shortForecast', '').lower():
            score -= 1
            volatility = "HIGH"
            
    if score < 5: volatility = "HIGH"
    elif score < 8: volatility = "MED"
    
    return max(1, score), volatility

# --- MAIN APP ---
def main():
    st.title("KMIA Trading Forecast")
    
    target_date = get_tomorrow_date()
    date_label = target_date.strftime('%A, %B %d').upper()
    st.caption(f"TARGETING: {date_label}")

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    daily, hourly, taf = fetch_all_weather_data()
    
    if not hourly:
        st.error("Could not retrieve forecast data. Please try again.")
        return

    score, vol = calculate_score(daily, hourly)

    # --- DASHBOARD HEADER ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Condition Score", value=f"{score}/10")
    with col2:
        st.metric(label="Volatility", value=vol, delta="-High Risk" if vol == "HIGH" else "Normal")
    with col3:
        if daily:
            st.metric(label="Projected High", value=f"{daily['temperature']}°F")

    # --- VISUAL SCORE BAR ---
    score_bar = "█" * score + "░" * (10 - score)
    st.text(f"SCORE: [{score_bar}]")

    # --- NARRATIVE ---
    if daily:
        st.info(f"**Analyst Summary:** {daily['shortForecast']}. Winds {daily['windSpeed']} {daily['windDirection']}.")

    # --- HOURLY TABLE ---
    st.subheader("Hourly Breakdown")
    
    table_data = []
    for h in hourly:
        dt = parse_iso_time(h['startTime'])
        time_str = dt.strftime("%I %p")
        temp = h['temperature']
        wind = f"{h['windDirection']} {h['windSpeed'].replace(' mph','')}"
        short = h['shortForecast']
        
        # Icon logic
        icon = "☁️"
        if "Sunny" in short or "Clear" in short: icon = "☀️"
        elif "Partly" in short: icon = "⛅"
        elif "Rain" in short: icon = "🌧️"
        elif "Thunder" in short: icon = "⛈️"
        
        status = ""
        if "Rain" in short or "Thunder" in short: status = "⚠️ RISK"
        
        table_data.append({
            "Time": time_str,
            "Temp (°F)": temp,
            "Wind": wind,
            "Condition": f"{icon} {short}",
            "Risk": status
        })
    
    # Display as a clean dataframe
    df = pd.DataFrame(table_data)
    st.dataframe(
        df, 
        column_config={
            "Risk": st.column_config.TextColumn(
                "Risk",
                help="Trading Risk Alert",
                validate="^⚠️",
            ),
        },
        use_container_width=True,
        hide_index=True
    )

    # --- AVIATION ---
    st.divider()
    st.caption("✈️ AVIATION (TAF) MONITOR")
    
    if taf:
        keywords = ["TSRA", "SHRA", "GUST", "VRB", "CB"]
        found = [word for word in keywords if word in taf]
        if found:
            st.warning(f"⚠️ PILOT ALERT: Found {', '.join(found)} in TAF report.")
        else:
            st.success("✅ No major aviation hazards (TSRA/GUST) flagged.")
            
        with st.expander("View Raw TAF Data"):
            st.code(taf)

if __name__ == "__main__":
    main()
