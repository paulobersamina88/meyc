from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title='Meycauayan Watershed Monitor', page_icon='🌊', layout='wide')

BASIN_OPTIONS = {
    'Meycauayan River Basin (study basin ~201 km²)': {
        'area_km2': 201.0,
        'center_lat': 14.734,
        'center_lon': 120.952,
        'description': 'Focused Meycauayan River basin for downstream flood monitoring.',
    },
    'Bulacan Basin draining through Meycauayan outlet (~621.4 km²)': {
        'area_km2': 621.4,
        'center_lat': 14.811,
        'center_lon': 121.012,
        'description': 'Broader upstream contributing basin used in some hydrologic studies.',
    },
}

RUNOFF_DEFAULTS = {
    'surface_runoff_pct': 63,
    'evapotranspiration_pct': 21,
    'shallow_gw_pct': 7,
    'deep_percolation_pct': 9,
    'dry_season_mean_cms': 5.6,
    'wet_season_mean_cms': 31.2,
    'annual_mean_cms': 22.3,
}

if 'manual_stage_records' not in st.session_state:
    now = pd.Timestamp.now(tz='Asia/Manila').floor('h').tz_localize(None)
    sample_times = pd.date_range(end=now, periods=8, freq='3h')
    sample_levels = [0.32, 0.41, 0.58, 0.82, 1.04, 1.11, 0.93, 0.71]
    st.session_state.manual_stage_records = pd.DataFrame(
        {'timestamp': sample_times, 'flood_level_m': sample_levels}
    )


@st.cache_data(ttl=1800)
def fetch_open_meteo(lat: float, lon: float, past_days: int = 7, forecast_days: int = 3) -> pd.DataFrame:
    url = 'https://api.open-meteo.com/v1/forecast'
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': ','.join([
            'precipitation',
            'rain',
            'showers',
            'soil_moisture_0_to_1cm',
            'soil_moisture_1_to_3cm',
            'soil_moisture_3_to_9cm',
        ]),
        'timezone': 'Asia/Manila',
        'past_days': past_days,
        'forecast_days': forecast_days,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    hourly = pd.DataFrame(data['hourly'])
    hourly['time'] = pd.to_datetime(hourly['time'])
    hourly = hourly.rename(columns={'time': 'timestamp'})
    return hourly


@st.cache_data(ttl=300)
def fetch_stage_json(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, dict):
        if 'data' in data:
            data = data['data']
        elif 'records' in data:
            data = data['records']
        else:
            data = [data]

    df = pd.DataFrame(data)
    cols = {c.lower().strip(): c for c in df.columns}
    timestamp_key = next((cols[k] for k in cols if k in ['timestamp', 'time', 'datetime', 'date_time']), None)
    level_key = next((cols[k] for k in cols if k in ['flood_level_m', 'water_level_m', 'stage_m', 'level_m', 'level']), None)

    if not timestamp_key or not level_key:
        raise ValueError('JSON endpoint must expose timestamp/time and water-level fields.')

    out = df[[timestamp_key, level_key]].copy()
    out.columns = ['timestamp', 'flood_level_m']
    out['timestamp'] = pd.to_datetime(out['timestamp'], errors='coerce')
    out['flood_level_m'] = pd.to_numeric(out['flood_level_m'], errors='coerce')
    out = out.dropna().sort_values('timestamp')
    return out


def build_stage_from_rainfall(rain_df: pd.DataFrame, base_level=0.28, sensitivity=0.04, lag_hours=4, recession=0.88):
    rain = rain_df['precipitation'].fillna(0).to_numpy()
    stage = np.zeros_like(rain, dtype=float)
    for i in range(len(rain)):
        inflow = rain[i - lag_hours] if i >= lag_hours else 0.0
        prev = stage[i - 1] if i > 0 else 0.0
        stage[i] = max(0, recession * prev + sensitivity * inflow)
    out = rain_df[['timestamp']].copy()
    out['flood_level_m'] = np.round(base_level + stage, 3)
    return out


def merge_rain_stage(rain_df: pd.DataFrame, stage_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge_asof(
        rain_df.sort_values('timestamp'),
        stage_df.sort_values('timestamp'),
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('90min'),
    )
    return merged.dropna(subset=['flood_level_m']).copy()


def add_rainfall_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for hrs in [1, 3, 6, 12, 24]:
        out[f'rain_{hrs}h_mm'] = out['precipitation'].rolling(hrs, min_periods=1).sum()
    return out


def compute_lag_correlation(df: pd.DataFrame, max_lag_hours: int = 24) -> pd.DataFrame:
    rows = []
    for lag in range(0, max_lag_hours + 1):
        shifted = df['precipitation'].shift(lag)
        corr = shifted.corr(df['flood_level_m'])
        rows.append({'lag_hours': lag, 'correlation': corr})
    return pd.DataFrame(rows).dropna()


def compute_runoff_index(area_km2: float, rain_mm_24h: float, runoff_coeff: float) -> float:
    return area_km2 * 1_000_000 * (rain_mm_24h / 1000.0) * runoff_coeff


def stage_band(level_m: float):
    if level_m < 0.5:
        return 'Normal', '🟢'
    if level_m < 1.0:
        return 'Watch', '🟡'
    if level_m < 1.5:
        return 'Warning', '🟠'
    return 'Critical', '🔴'


st.sidebar.title('⚙️ Basin and Data Settings')
basin_name = st.sidebar.selectbox('Analysis boundary', list(BASIN_OPTIONS.keys()))
basin = BASIN_OPTIONS[basin_name]

lat = st.sidebar.number_input('Basin center latitude', value=float(basin['center_lat']), format='%.6f')
lon = st.sidebar.number_input('Basin center longitude', value=float(basin['center_lon']), format='%.6f')
area_km2 = st.sidebar.number_input('Catchment area (km²)', value=float(basin['area_km2']), min_value=1.0)
past_days = st.sidebar.slider('Historical rainfall window (days)', 2, 14, 7)
forecast_days = st.sidebar.slider('Forecast rainfall window (days)', 1, 7, 3)
runoff_coeff = st.sidebar.slider('Runoff coefficient', 0.10, 0.95, 0.63, 0.01)

st.sidebar.markdown('---')
flood_source = st.sidebar.radio(
    'Flood-level source',
    ['Quick manual entry in app', 'Fetch from JSON/API endpoint', 'Auto-estimate from rainfall'],
    index=0,
)

json_url = ''
base_level, sensitivity, lag_hours, recession = 0.28, 0.04, 4, 0.88

if flood_source == 'Fetch from JSON/API endpoint':
    json_url = st.sidebar.text_input('JSON endpoint URL', placeholder='https://example.com/meycauayan-stage.json')
    st.sidebar.caption('Accepted field names include: timestamp/time/datetime and flood_level_m/water_level_m/stage_m/level_m')
elif flood_source == 'Auto-estimate from rainfall':
    base_level = st.sidebar.slider('Base stage (m)', 0.0, 1.5, 0.28, 0.01)
    sensitivity = st.sidebar.slider('Rainfall sensitivity', 0.001, 0.150, 0.040, 0.001)
    lag_hours = st.sidebar.slider('Rainfall-to-stage lag (hours)', 0, 12, 4)
    recession = st.sidebar.slider('Recession factor', 0.50, 0.99, 0.88, 0.01)

show_profile = st.sidebar.checkbox('Show hydrologic profile', value=True)

st.title('🌊 Meycauayan Watershed Rainfall–Flood Monitor')
st.caption('Convenience-first prototype: automatic rainfall, in-app stage entry, optional JSON/API feed, and rainfall–flood correlation.')

with st.expander('How this version avoids CSV', expanded=True):
    st.write(
        'This version is designed for convenience. You can either log water levels directly inside the app, '
        'connect a JSON/API endpoint, or use an auto-estimated stage curve while your live gauge source is being set up.'
    )
    st.info('Best long-term setup: Open-Meteo for rainfall + a small gauge sensor or LGU endpoint that outputs JSON.')

try:
    rain_df = fetch_open_meteo(lat=lat, lon=lon, past_days=past_days, forecast_days=forecast_days)
except Exception as e:
    st.error(f'Rainfall fetch failed: {e}')
    st.stop()

if flood_source == 'Quick manual entry in app':
    st.subheader('Quick water-level log')
    entry_col1, entry_col2, entry_col3, entry_col4 = st.columns([1.4, 1, 1, 1])
    with entry_col1:
        entry_time = st.datetime_input('Timestamp', value=datetime.now())
    with entry_col2:
        entry_level = st.number_input('Flood level (m)', min_value=0.0, max_value=20.0, value=0.75, step=0.01)
    with entry_col3:
        if st.button('Add reading', use_container_width=True):
            new_row = pd.DataFrame({'timestamp': [pd.to_datetime(entry_time)], 'flood_level_m': [entry_level]})
            st.session_state.manual_stage_records = pd.concat(
                [st.session_state.manual_stage_records, new_row], ignore_index=True
            ).drop_duplicates(subset=['timestamp'], keep='last').sort_values('timestamp')
    with entry_col4:
        if st.button('Clear readings', use_container_width=True):
            st.session_state.manual_stage_records = st.session_state.manual_stage_records.iloc[0:0]

    stage_df = st.session_state.manual_stage_records.copy()
    st.dataframe(stage_df.tail(12), use_container_width=True, height=220)

elif flood_source == 'Fetch from JSON/API endpoint':
    if not json_url:
        st.warning('Paste a JSON endpoint URL in the sidebar to load live or semi-live water levels.')
        st.stop()
    try:
        stage_df = fetch_stage_json(json_url)
        st.success(f'Loaded {len(stage_df)} stage records from the endpoint.')
    except Exception as e:
        st.error(f'Could not read the JSON/API endpoint: {e}')
        st.stop()

else:
    stage_df = build_stage_from_rainfall(
        rain_df,
        base_level=base_level,
        sensitivity=sensitivity,
        lag_hours=lag_hours,
        recession=recession,
    )

if stage_df.empty:
    st.warning('No flood-level records available yet.')
    st.stop()

merged = merge_rain_stage(rain_df, stage_df)
merged = add_rainfall_aggregates(merged) if not merged.empty else merged
lag_corr = compute_lag_correlation(merged, max_lag_hours=24) if not merged.empty else pd.DataFrame()

latest_rain = float(rain_df['precipitation'].fillna(0).iloc[-1])
latest_stage = float(stage_df['flood_level_m'].iloc[-1])
status, emoji = stage_band(latest_stage)
rain_24h = float(merged['rain_24h_mm'].iloc[-1]) if not merged.empty else 0.0
runoff_volume = compute_runoff_index(area_km2, rain_24h, runoff_coeff)

c1, c2, c3, c4 = st.columns(4)
c1.metric('Latest hourly rainfall', f'{latest_rain:.1f} mm')
c2.metric('Latest flood level', f'{latest_stage:.2f} m')
c3.metric('24h accumulated rainfall', f'{rain_24h:.1f} mm')
c4.metric('Status', f'{emoji} {status}')

st.markdown('---')
left, right = st.columns([1.6, 1])

with left:
    st.subheader('Rainfall and flood level timeline')
    fig = go.Figure()
    fig.add_trace(go.Bar(x=rain_df['timestamp'], y=rain_df['precipitation'], name='Rainfall (mm/h)', yaxis='y2'))
    fig.add_trace(go.Scatter(x=stage_df['timestamp'], y=stage_df['flood_level_m'], name='Flood level (m)', mode='lines+markers'))
    fig.update_layout(
        height=430,
        xaxis=dict(title='Time'),
        yaxis=dict(title='Flood level (m)'),
        yaxis2=dict(title='Rainfall (mm/h)', overlaying='y', side='right'),
        legend=dict(orientation='h'),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Lead–lag correlation')
    if lag_corr.empty:
        st.warning('Need more overlapping rainfall and stage timestamps to compute correlation.')
    else:
        best = lag_corr.iloc[lag_corr['correlation'].abs().argmax()]
        st.caption(f"Strongest correlation in current window: lag = {int(best['lag_hours'])} h, r = {best['correlation']:.3f}")
        corr_fig = px.line(lag_corr, x='lag_hours', y='correlation', markers=True)
        corr_fig.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(corr_fig, use_container_width=True)

with right:
    st.subheader('Quick watershed diagnostics')
    st.markdown(f'**Selected boundary:** {basin_name}')
    st.write(basin['description'])
    st.write(f'**Catchment area:** {area_km2:,.1f} km²')
    st.write(f'**Estimated runoff volume from last 24 h rainfall:** {runoff_volume:,.0f} m³')

    if show_profile:
        st.markdown('### Hydrologic profile')
        p1, p2 = st.columns(2)
        p1.metric('Annual mean flow', f"{RUNOFF_DEFAULTS['annual_mean_cms']:.1f} cms")
        p2.metric('Wet season mean', f"{RUNOFF_DEFAULTS['wet_season_mean_cms']:.1f} cms")
        p3, p4 = st.columns(2)
        p3.metric('Dry season mean', f"{RUNOFF_DEFAULTS['dry_season_mean_cms']:.1f} cms")
        p4.metric('Surface runoff share', f"{RUNOFF_DEFAULTS['surface_runoff_pct']}%")
        wb = pd.DataFrame({
            'Component': ['Surface runoff', 'Evapotranspiration', 'Shallow subsurface/GW', 'Deep percolation'],
            'Percent': [
                RUNOFF_DEFAULTS['surface_runoff_pct'],
                RUNOFF_DEFAULTS['evapotranspiration_pct'],
                RUNOFF_DEFAULTS['shallow_gw_pct'],
                RUNOFF_DEFAULTS['deep_percolation_pct'],
            ],
        })
        wb_fig = px.pie(wb, names='Component', values='Percent')
        wb_fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(wb_fig, use_container_width=True)

st.markdown('---')

tab1, tab2, tab3 = st.tabs(['Merged data', 'Endpoint format', 'Implementation notes'])

with tab1:
    if merged.empty:
        st.info('Merged rainfall-stage table will appear once timestamps overlap closely enough.')
    else:
        st.dataframe(merged.tail(48), use_container_width=True)
        st.download_button(
            'Download merged rainfall-stage data',
            data=merged.to_csv(index=False).encode('utf-8'),
            file_name='meycauayan_rainfall_stage_merged.csv',
            mime='text/csv',
        )

with tab2:
    st.code(
        '[\n  {"timestamp": "2026-04-19T08:00:00+08:00", "water_level_m": 0.82},\n  {"timestamp": "2026-04-19T09:00:00+08:00", "water_level_m": 0.91}\n]',
        language='json',
    )

with tab3:
    st.markdown(
        '''
        **Convenience-focused upgrades already included**
        - No CSV requirement for normal use.
        - In-app one-click manual stage entry.
        - Optional JSON/API endpoint reader.
        - Automatic rainfall from Open-Meteo.
        - Rainfall–flood lag correlation and runoff estimate.

        **Best next upgrade**
        - Connect a real gauge, Arduino/ESP32 sensor, Google Apps Script JSON feed, or an LGU endpoint.
        - Add real watershed GeoJSON and barangay overlays.
        - Add alerts by threshold with email, Telegram, or SMS.
        '''
    )
