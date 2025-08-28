# utils
import os
import logging
import sys

# manipulação de datas e timezone
import pytz
import datetime as dt

# spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, concat_ws

####################
# configurações de timezone
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

def process_variable_delivery(variable_id, input_model, input_experiment_id, dir_trusted, dir_delivery, grid_step=2.0):
    """
    Pipeline da camada Delivery:
    1. Leitura dos dados Trusted (todos os anos)
    2. Reamostragem espacial (redução da resolução)
    3. Flatten (tempo x gridpoints)
    4. Salvar em Parquet particionado

    :param variable_id: variável climática (ex: "tas")
    :param input_model: nome do modelo climático
    :param input_experiment_id: experimento (ex: historical)
    :param dir_trusted: diretório da camada Trusted
    :param dir_delivery: diretório para salvar a camada Delivery
    :param grid_step: resolução em graus (default 2.0°)
    """

    logger_ingestion.info(f"inicializando pipeline | delivery")
    logger_ingestion.info(f"variável: {variable_id} | modelo: {input_model} | experimento: {input_experiment_id}")

    # Iniciar sessão spark
    spark = SparkSession.builder.appName("ClimateData").getOrCreate()

    # 1. Leitura dos dados Trusted (todos os anos)
    logger_ingestion.info(f"lendo todos os arquivos da camada Trusted em {dir_trusted}")
    df_trusted = spark.read.parquet(dir_trusted)

    # Verificar se há dados
    if df_trusted.count() == 0:
        logger_ingestion.warning("nenhum dado encontrado na camada trusted")
        return

    # 2. Reamostragem espacial: arredondando lat/lon para grid_step
    logger_ingestion.info(f"reamostrando para resolução {grid_step}°")
    df_resampled = (
        df_trusted
        .withColumn("lat_grid", (col("lat") / grid_step).cast("int") * grid_step)
        .withColumn("lon_grid", (col("lon") / grid_step).cast("int") * grid_step)
        .groupBy("year", "month", "lat_grid", "lon_grid")
        .agg(avg(variable_id).alias(f"{variable_id}_mean"))
    )

    # 3. Flatten: transformar lat/lon em colunas (tempo x gridpoints)
    logger_ingestion.info("fazendo pivot para formato flatten")
    df_resampled = df_resampled.withColumn("lat_lon", concat_ws("_", col("lat_grid"), col("lon_grid")))

    pivot_values = [row["lat_lon"] for row in df_resampled.select("lat_lon").distinct().collect()]
    df_flatten = (
        df_resampled
        .groupBy("year", "month")
        .pivot("lat_lon", values=pivot_values)
        .agg(avg(f"{variable_id}_mean"))
    )

    # 4. Salvar em Parquet particionado
    output_path = os.path.join(dir_delivery, input_model, variable_id)
    (
        df_flatten.write
        .mode("overwrite")
        .partitionBy("year", "month")
        .parquet(output_path)
    )
    logger_ingestion.info(f"dados salvos em {output_path} particionados por year e month")

    logger_ingestion.info(f"pipeline concluído | delivery")