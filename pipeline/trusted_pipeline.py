from transformations.geo import fix_longitude, select_latam
from transformations.quality import kelvin_to_celsius, remove_outliers
from transformations.temporal import add_time_features
from config.variables_config import map_variaveis_meta
from config.models_config import map_modelos_dir
import xarray as xr
import glob
import os
import pandas as pd
import logging
import pytz
import datetime as dt
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("ClimateData").getOrCreate()
selected_tz = pytz.timezone("America/Sao_Paulo")
####################
# configurações de log
class TZFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, tz=None):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.tz = tz or pytz.UTC

    def formatTime(self, record, datefmt=None):
        date_time = dt.datetime.fromtimestamp(record.created, self.tz)
        if datefmt:
            s = date_time.strftime(datefmt)
        else:
            s = date_time.isoformat()
        return s

# função de configuração do log
def configurar_logger(nome_logger):
    logger = logging.getLogger(nome_logger)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # evita envio ao root logger

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        # Timezone Brasil (Horário de Brasília)
        #tz_brasil = pytz.timezone("America/Sao_Paulo")

        formatter = TZFormatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S %z',
            tz=selected_tz
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

logger_read = configurar_logger("leitura_dados")
logger_ingestion = configurar_logger("ingestao_dados")
####################
logger_read.info(f"inicializando pipeline | trusted")
####################
# diretórios camadas
dir_raiz = r"/home/thais/climate-ingestion/climate_data_processing/datasets"
dir_trusted = os.path.join(dir_raiz, "trusted")
os.makedirs(dir_trusted, exist_ok=True)
####################
# selecionando a variável
variavel_escolhida = map_variaveis_meta["tas"]

logger_read.info(f"variável escolhida: {variavel_escolhida['variable_id']}")
####################
# selecionando o modelo
modelo_escolhido = map_modelos_dir["EC-Earth3"]

logger_read.info(f"modelo escolhido: {modelo_escolhido['nome']}")
####################
# diretório para os datasets na trusted
dir_modelo = modelo_escolhido["nome_dir"]

dir_modelo_trusted = os.path.join(dir_trusted, dir_modelo, variavel_escolhida["variable_id"])
os.makedirs(dir_trusted, exist_ok=True)

logger_read.info(f"diretório trusted para os datsets: {dir_modelo_trusted}")
logger_read.info(f"parâmetros do dataset: source_id={modelo_escolhido['nome']} | experiment_id={variavel_escolhida['experiment_id']} | variable_id={variavel_escolhida['variable_id']} | variant_label={variavel_escolhida['variant_label']}")
####################


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
