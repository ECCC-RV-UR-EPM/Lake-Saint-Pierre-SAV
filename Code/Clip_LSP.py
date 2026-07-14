import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from shapely.geometry import Point, Polygon

# ============================================================
# ============================================================
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "Data" / "01_temperature" / "Temp_prediction_LSP"

file = DATA_DIR / "Temp_training_daily_observe_to_train.xlsx"
df = pd.read_excel(file)

lon = df["Longitude"].values
lat = df["Latitude"].values

# ============================================================
# ============================================================
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(lon, lat, s=1)
ax.set_title("Draw Lake Saint-Pierre boundary")

coords = []

# ============================================================
# ============================================================
def onselect(verts):
    global coords
    coords = verts

    print(" Polygon captured")

    xs, ys = zip(*(coords + [coords[0]]))
    ax.plot(xs, ys, color='red', linewidth=2)
    plt.draw()

polygon_selector = PolygonSelector(ax, onselect)

plt.show()

# ============================================================
# ============================================================
polygon = Polygon(coords)

mask = []
for x, y in zip(lon, lat):
    mask.append(polygon.contains(Point(x, y)))

df_clip = df[mask]

# ============================================================
# ============================================================
out_file = DATA_DIR / "Temp_training_daily_observe_to_train_clip.xlsx"
df_clip.to_excel(out_file, index=False)

print(" ")
