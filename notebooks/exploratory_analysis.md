"""
This is a plain-text placeholder for notebooks/exploratory_analysis.ipynb.

A real .ipynb is a JSON file; rather than hand-write brittle notebook JSON,
open Jupyter and re-create these cells directly — each one is a natural
notebook cell. This mirrors the analysis worth doing before trusting the
model in models/train.py.

--------------------------------------------------------------------------
Cell 1 (markdown):
# Exploratory analysis — Crop Yield & Irrigation Advisory

Sanity-checks the simulated data before it feeds the modeling pipeline:
rainfall seasonality, yield distributions by crop/district, and the
rainfall-yield relationship the model is expected to learn.

--------------------------------------------------------------------------
Cell 2 (code):
import pandas as pd
import matplotlib.pyplot as plt

weather = pd.read_csv("../data/raw/weather_daily.csv", parse_dates=["obs_date"])
seasons = pd.read_csv("../data/raw/crop_seasons.csv", parse_dates=["sowing_date", "harvest_date"])
soil = pd.read_csv("../data/raw/soil_health.csv")

--------------------------------------------------------------------------
Cell 3 (code):
# Rainfall seasonality — coastal vs Rayalaseema district, monthly totals
coastal_sample = weather[weather.district_name == "Guntur"].copy()
interior_sample = weather[weather.district_name == "Anantapur"].copy()
for df, label in [(coastal_sample, "Guntur (coastal)"), (interior_sample, "Anantapur (Rayalaseema)")]:
    monthly = df.groupby(df.obs_date.dt.month).rainfall_mm.mean() * 30
    monthly.plot(label=label)
plt.legend(); plt.xlabel("Month"); plt.ylabel("Approx. monthly rainfall (mm)")
plt.title("Monsoon rainfall shape: coastal vs interior AP")

--------------------------------------------------------------------------
Cell 4 (code):
# Yield distribution by crop
seasons.boxplot(column="yield_kg_per_ha", by="crop_name", rot=45)
plt.title("Yield distribution by crop"); plt.suptitle("")

--------------------------------------------------------------------------
Cell 5 (code):
# Rainfall vs yield relationship (the core signal the model should learn)
for crop in ["Rice", "Groundnut", "Cotton"]:
    sub = seasons[seasons.crop_name == crop]
    plt.scatter(sub.total_rainfall_mm, sub.yield_kg_per_ha, alpha=0.4, label=crop)
plt.xlabel("Total seasonal rainfall (mm)"); plt.ylabel("Yield (kg/ha)")
plt.legend(); plt.title("Rainfall vs yield by crop")

--------------------------------------------------------------------------
Cell 6 (markdown):
## Takeaways to carry into modeling
- Yield response to rainfall is non-monotonic (too little AND too much both
  hurt) — this is why models/train.py uses a tree-based model (XGBoost)
  rather than plain linear regression, which can't capture that curve.
- Coastal vs interior rainfall regimes differ enough that district identity
  is a meaningful categorical feature, not noise.
- Soil quality and irrigation source visibly shift the yield distribution
  within a single crop/district — both are included in the feature set in
  models/feature_engineering.py.
"""
