# Queens Housing Advisor (Instant Start)

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
```

Put your data at `data/rollingsales_queens_clean.csv` (or same-named `.parquet/.feather`).  
To display **all** addresses instantly, either:
- Add `LAT,LON` into the CSV, or
- Run once: `python -m tools.pregeocode_all` to create `data/geocode_cache.parquet` and ship it with the app.
