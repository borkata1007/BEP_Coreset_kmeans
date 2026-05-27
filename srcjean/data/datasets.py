import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = {
    "airbnb_sf": {
        "type": "tabular",
        "path": "data/real/airbnb_sf.csv",
        "features": ["latitude", "longitude", "price"],
        "map": True,
        "map_dims": (1, 0),
        "map_labels": ("Longitude", "Latitude"),
    },
    "boris_citibike": {
        "type": "tabular",
        "path": "data/real/boris_citibike_2d.csv",
        "features": ["longitude", "latitude"],
        "map": True,
        "map_dims": (0, 1),
        "map_labels": ("Longitude", "Latitude"),
    },
    "martin_spotify": {
        "type": "tabular",
        "path": "data/real/martin_spotify.csv",
        "features": ["energy", "valence", "danceability", "acousticness"],
        "map": False,
    },
    "uber": {
        "type": "tabular",
        "path": "data/real/uber.csv",
        "features": ["Lon", "Lat"],
        "map": True,
        "map_dims": (0, 1),
        "map_labels": ("Longitude", "Latitude"),
    },
    "donuts": {
        "type": "tabular",
        "path": "data/synthetic/donuts.csv",
        "features": ["x", "y"],
        "map": True,
        "map_dims": (0, 1),
        "map_labels": ("x", "y"),
    },
    "birb": {
        "type": "image",
        "path": "data/image/birb.png",
        "features": ["R", "G", "B"],
    },
    "arhan_sunset": {
        "type": "image",
        "path": "data/image/arhan-sunset.png",
        "features": ["R", "G", "B"],
    },
    "image": {
        "type": "image",
        "path": "data/image/image.png",
        "features": ["R", "G", "B"],
    },
}


def _load_tabular(info):
    df = pd.read_csv(os.path.join(ROOT, info["path"]))
    X = df[info["features"]].to_numpy(dtype=float)
    X = X[~np.isnan(X).any(axis=1)]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X, scaler, info


def _load_image(info):
    img = Image.open(os.path.join(ROOT, info["path"])).convert("RGB")
    arr = np.asarray(img, dtype=float)
    H, W = arr.shape[:2]
    X = arr.reshape(-1, 3)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    info = {**info, "shape": (H, W)}
    return X_scaled, X, scaler, info


def load(name):
    info = DATASETS[name]
    if info.get("type") == "image":
        return _load_image(info)
    return _load_tabular(info)
