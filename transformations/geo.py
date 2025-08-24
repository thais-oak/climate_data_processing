import numpy as np

def fix_longitude(ds):
    return ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")

def select_latam(ds):
    return ds.sel(lat=slice(-60, 15), lon=slice(-120, -30))