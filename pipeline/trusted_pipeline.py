from transformations.geo import fix_longitude, select_latam
from transformations.quality import kelvin_to_celsius, remove_outliers
from transformations.temporal import add_time_features
import xarray as xr
import glob
import os
import pandas as pd
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("ClimateData").getOrCreate()

# mapear nomes das funções para objetos Python
transform_funcs = {
    "fix_longitude": fix_longitude,
    "select_latam": select_latam,
    "kelvin_to_celsius": kelvin_to_celsius,
    "remove_outliers": remove_outliers
}

def process_variable_trusted(variable, input_dir, output_dir):
    meta = map_variaveis_meta[variable]
    nc_files = glob.glob(os.path.join(input_dir, "**", "*.nc"), recursive=True)

    for nc_file in nc_files:
        ds = xr.open_dataset(nc_file)[variable]

        # aplica transformações definidas no dicionário
        for transform in meta["transformations"]:
            ds = transform_funcs[transform](ds)

        # dataframe para Spark
        df = ds.to_dataframe().reset_index()
        df["time"] = pd.to_datetime(df["time"])
        df = add_time_features(df)

        df_spark = spark.createDataFrame(df)
        (
            df_spark.write
            .mode("append")
            .partitionBy("year", "month")
            .parquet(output_dir)
        )

        ds.close()
