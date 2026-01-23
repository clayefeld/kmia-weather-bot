import streamlit as st
import streamlit.components.v1 as components
import requests
import re
import pandas as pd
import altair as alt
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

# --- GLOBAL STYLES ---
HIDE_INDEX_CSS = """
    <style>
    thead tr th:first-child {display:none}
    tbody th {display:none}
    </style>
    """

# --- UTILS ---
def get_headers():
    return {'User-Agent': '(project_helios_v22_layout, myemail@example.com)'}

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
                
                dew_c = props.get('dewpoint', {}).get('value')
                dew_f = (dew_c * 1.8) + 32 if dew_c is not None else None

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
                    "DewPoint": dew_f,
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
                        "DewPoint": None,
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
def render_live_dashboard():
    st.title("🔴 Project Helios: Advanced Feed")
    
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

    now_miami = get_miami_time()
    sunset_miami = now_miami.replace(hour=17, minute=55, second=0, microsecond=0)
    time_left = sunset_miami - now_miami
    is_night = time_left.total_seconds() <= 0
    
    solar_fuel = "NIGHT"
    if not is_night:
        hrs, rem = divmod(time_left.seconds, 3600)
        mins
