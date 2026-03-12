# Energy Consumption Analytics (Portfolio Project)

Production-style analytics system for smart meter electricity consumption time-series data.

This repository is designed as a professional **data analytics portfolio project** demonstrating end-to-end skills across **data engineering**, **time-series feature engineering**, **anomaly detection**, **forecasting**, and **interactive visualization**.

## Project Overview

Utilities and energy companies rely on smart meter data to understand consumption patterns, detect abnormal behavior, and forecast short-term demand. This project implements a modular pipeline that:

1. Loads and cleans raw smart meter data
2. Engineers time-series features
3. Detects anomalies with Isolation Forest
4. Forecasts the next 24 hours with Prophet
5. Generates an interactive Plotly dashboard (HTML)

## Key Features

- Robust data loader for common smart-meter CSV conventions (including UCI-style Date/Time columns)
- Time-series features: hour/day/weekend indicators + rolling mean/std + variability
- Anomaly detection: Isolation Forest labeling (`-1` anomalies, `1` normal)
- Short-term forecasting: Prophet forecast with uncertainty interval (with a safe fallback so dashboard generation still works if Prophet isn't available)
- Interactive dashboard (HTML):
	- Key metrics summary (avg/min/max, anomaly %, total records)
	- Historical series with anomaly markers
	- Next-24h forecast view
	- Hourly and weekday pattern charts
	- Consumption distribution histogram
	- Hour × weekday heatmap
	- Automated analytical insights

## Tech Stack

- Python
- pandas, numpy
- scikit-learn (Isolation Forest)
- prophet (time-series forecasting)
- plotly (interactive dashboard)
- matplotlib (EDA)

## Project Architecture

```
energy-consumption-analytics/
	data/
		household_power_consumption.csv
	notebooks/
		exploratory_analysis.ipynb
	src/
		data_loader.py
		feature_engineering.py
		anomaly_detection.py
		forecasting.py
		pipeline.py
	dashboard/
		plotly_dashboard.py
	output/
		dashboard.html
	main.py
	requirements.txt
	README.md
```

### Data Flow

`main.py` runs the pipeline:

Load + clean raw CSV → Resample hourly → Feature engineering → Isolation Forest anomalies → Prophet forecast → Plotly dashboard HTML

## Dataset

Expected dataset path:

- `data/household_power_consumption.csv`

Note: `data/household_power_consumption.csv` is ignored by git by default (see `.gitignore`) to avoid committing large/private datasets.

For reviewers, the repository includes a small demo file:

- `data/sample_household_power_consumption.csv`

## How to Run

From the repo root:

### Windows (PowerShell)

```bash
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python main.py
```

### macOS/Linux

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python main.py
```

### Quick Demo (no private dataset required)

Run the pipeline using the bundled sample dataset:

```bash
.\.venv\Scripts\python main.py --demo
```

Output:

- `output/dashboard.html`

### View Without Installing Anything

This repo ships a pre-generated demo dashboard HTML:

- `output/dashboard_demo.html`

To view it, download the file and open it in your browser.

If your browser blocks local files or you prefer a local URL, serve the repo folder:

```bash
python -m http.server 8000
```

Then open:

- http://localhost:8000/output/dashboard_demo.html

## Live Dashboard link (GitHub Pages)

Once GitHub Pages is enabled for this repository, you can view the pre-generated demo dashboard here:

- https://gera9999.github.io/energy-consumption-analytics/output/dashboard_demo.html

To enable it: GitHub repo → **Settings** → **Pages** → **Build and deployment** → Source: **Deploy from a branch** → Branch: `main` (root).

## Example Outputs

The generated dashboard includes:

- Key metrics summary (avg/min/max, anomaly %, total records)
- Hourly and weekday usage patterns
- Consumption distribution histogram
- Hour × weekday heatmap
- Historical series with anomaly markers
- Next-24h forecast with an uncertainty interval (when available)

## What the Dashboard Shows

- **Key metrics summary:** quick KPI view of the dataset and anomaly rate.
- **Historical consumption + anomalies:** full history with anomaly points overlaid.
- **Forecast view:** recent history alongside the next 24h forecast and uncertainty.
- **Behavior analytics:** hourly and weekday averages, distribution, and an hour × weekday heatmap.
- **Analytical insights:** automatically generated bullets highlighting peak hour/day and anomaly percentage.

## How to Interpret (quick)

- Each chart includes a small KPI box (top-left) with non-invasive numbers (mean/min/max, percentiles, peak hour/day, etc.) so you can interpret the visuals at a glance.
- **Forecast band:** in the forecast panel, the shaded area is the uncertainty interval; if observed values (when you compare with future runs) frequently fall outside the band, it can indicate a behavior change or that the forecast is too simple for that period.
- **Anomalies:** red points are timestamps flagged as statistically unusual versus the overall pattern. This does not automatically mean "bad" behavior; it is a short-list for investigation.
- **Explaining anomalies:** with only smart-meter signals (e.g., consumption/voltage/intensity), you can describe *what changed* (spike/drop, shifted voltage/intensity) but not the real-world cause (occupancy, weather, appliance usage) without extra context.

## Use Cases

- Utility monitoring: identify unusual consumption spikes/drops
- Smart meter analytics: understand seasonality and daily/weekly patterns
- Operational planning: short-term demand forecasting for grid operations
- Client reporting: share interactive dashboards without a server

## Notes (Windows + Prophet)

`prophet` provides pre-built wheels on many platforms, but on some Windows setups it may require additional build tooling.
If you run into installation issues, try upgrading pip first:

```bash
python -m pip install --upgrade pip
```

---

If you want, we can extend this project with model evaluation, richer KPI summaries, and automated reporting while keeping the same architecture.