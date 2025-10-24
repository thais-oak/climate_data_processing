import numpy as np
import xarray as xr
from pipeline.logger import configurar_logger
import pytz

# configurações de timezone
br_tz = pytz.timezone("America/Sao_Paulo")

# configuração de log
logger_tranformation = configurar_logger("transformacao_temporal", br_tz)

def add_time_features(df):
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    return df

def add_time_features_teste_freqs(df, frequency="mon"):
    ##########
    logger_tranformation.debug(f">>> dtype da coluna 'time': {df['time'].dtype}<<<")
    logger_tranformation.debug(f"{df['time'].head()}")
    ##########
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    if frequency == "day":
        df["day"] = df["time"].dt.day
        df["dayofyear"] = df["time"].dt.dayofyear
    return df


def convert_datetime(ds: xr.Dataset) -> xr.Dataset:
    """
    garante que a coordenada 'time' seja do tipo datetime64[ns] e
    aplica decodificação CF ou conversão manual se necessário
    """
    try:
        if "time" not in ds:
            logger_tranformation.warning(f"dataset não contém coordenada 'time'")
            return ds

        if not np.issubdtype(ds["time"].dtype, np.datetime64):
            logger_tranformation.warning(f"conversão de 'time' para datetime64[ns] (dtype atual: {ds['time'].dtype})")
            try:
                ds = xr.decode_cf(ds)
            except Exception:
                try:
                    ds["time"] = xr.conventions.times.decode_cf_datetime(
                        ds["time"], ds.time.attrs, use_cftime=False
                    )
                except Exception as e:
                    logger_tranformation.error(f"falha ao converter 'time': {e}")
        ds["time"] = ds["time"].astype("datetime64[ns]")
        return ds

    except Exception as e:
        logger_tranformation.error(f"erro ao padronizar datetime: {e}")
        return ds