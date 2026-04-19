# Meycauayan Watershed Rainfall–Flood Monitor

A Streamlit starter app for monitoring rainfall and flood levels in the Meycauayan River watershed.

## Features
- Open-Meteo hourly rainfall pull
- CSV/manual flood-level ingestion
- Synthetic stage generator from rainfall for testing
- Rainfall–stage lag correlation
- Quick runoff-volume estimate from 24-hour rainfall
- Hydrologic profile panel for basin context

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Expected flood CSV format
```csv
timestamp,flood_level_m
2026-04-18 00:00:00,0.32
2026-04-18 03:00:00,0.41
```

## Next upgrades
- Plug in live flood-stage data source
- Load Meycauayan basin/subcatchment GeoJSON
- Add tide effects and warnings
- Add alerting and archived event replay
