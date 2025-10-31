import os
import sys
import pandas as pd
import xarray as xr
import pyspark
import pickle

def s3_join(*parts):
    """
    esta função concatena partes em um path S3 válido.
    Exemplo:
        s3_join("meu-bucket", "datasets", "trusted", "EC-Earth3", "tas")
        -> "s3://meu-bucket/datasets/trusted/EC-Earth3/tas"
    """
    base = "/".join(p.strip("/").replace("\\", "/") for p in parts)
    return f"s3://{base}"

def detect_frequency(input_dir):
    """
    Detecta frequência a partir do caminho da pasta ou nome do arquivo.
    """
    if "freq=day" in input_dir:
        return "day"
    elif "freq=mon" in input_dir:
        return "mon"
    else:
        # fallback simples: diário se o arquivo contiver 'day', mensal se 'Amon'
        return "mon"

# tamanho total em MB (somando todos os arquivos parquet)
def get_dir_size_mb(path):
    total_bytes = 0
    for root, _, files in os.walk(path):
        for f in files:
            
            try:
                total_bytes += os.path.getsize(os.path.join(root, f))
            except OSError:
                continue
            
    return round(total_bytes / (1024 * 1024), 2)

# tamanho aproximado do dataset
def estimate_dataset_size_old(ds):
    """
        calcula o tamanho aproximado do dataset
    """
    total_bytes = 0
    for var in ds.data_vars:
        total_bytes += ds[var].nbytes
    return round(total_bytes / (1024 ** 2), 2)  # em MB


def estimate_dataset_size(ds, spark_sample_rows=1000):
    """
    estima o tamanho de um dataset em MB

    Parâmetros:
    -----------
    ds : xarray.Dataset | pandas.DataFrame | pyspark.sql.DataFrame
        Dataset a ser estimado.
    spark_sample_rows : int
        Número de linhas amostradas para estimativa em Spark DataFrame (default: 1000).

    Retorna:
    --------
    float : tamanho estimado em MB
    """

    if isinstance(ds, xr.Dataset):
        # xarray já possui nbytes
        total_bytes = ds.nbytes
        return round(total_bytes / (1024 ** 2), 2)

    elif isinstance(ds, pd.DataFrame):
        # pandas DataFrame tem memory_usage
        total_bytes = ds.memory_usage(deep=True).sum()
        return round(total_bytes / (1024 ** 2), 2)

    elif isinstance(ds, pyspark.sql.DataFrame):
        # spark dataframe: estima usando amostra + serialização
        count = ds.count()
        if count == 0:
            return 0.0

        # amostra
        sample = ds.limit(spark_sample_rows).collect()  # traz para driver
        if not sample:
            return 0.0

        # tamanho médio da linha serializada
        mem_per_row = sum(len(pickle.dumps(row.asDict())) for row in sample) / len(sample)
        total_bytes = mem_per_row * count

        return round(total_bytes / (1024 ** 2), 2)

    else:
        raise TypeError(f"tipo de dataset não suportado: {type(ds)}")