def kelvin_to_celsius(ds):
    return ds - 273.15

def remove_outliers(ds, min_val=-90, max_val=60):
    return ds.where((ds > min_val) & (ds < max_val))
