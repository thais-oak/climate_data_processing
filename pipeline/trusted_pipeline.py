# utils
import glob
import os
import sys
import logging

# manipulação de dados
import xarray as xr
import pandas as pd
from pyspark.sql import SparkSession

# manipulação de datas e timezone
import pytz
import datetime as dt

# configurações de modelos e variáveis
from config.variables_config import map_variaveis_meta

# transformações
from transformations.temporal import add_time_features
from config.variables_config import map_variaveis_meta
####################
# iniciar sessão spark
spark = SparkSession.builder.appName("ClimateData").getOrCreate()
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
####################
logger_ingestion.info(f"inicializando pipeline | trusted")
####################
def process_variable_trusted(variable_id, input_dir, output_dir, map_transform_funcs):
    """
    pipeline para ingestão na camada trusted
    1. leitura dos dados da raw
    2. transformação dos dados de acordo com o mapeamento de map_transform_funcs
    3. armazenamento na camada trusted em parquet
    
    :param variable_id: variável climática (ex: "tas")
    :param input_dir: diretório da camada raw com os datasets
    :param output_dir: diretório da camada trusted para salvar os dataframes processados
    :param map_transform_funcs: dicionário com as variáveis e suas respectivas transformações
    """
    # selecionando a variável
    #meta = map_variaveis_meta[variable_id]
    variavel_escolhida = map_variaveis_meta[variable_id]
    logger_ingestion.info(f"variável escolhida: {variavel_escolhida['variable_id']}")

    # origem dos datasets *.nc | camada raw
    #nc_files = glob.glob(os.path.join(input_dir, "**", "*.nc"), recursive=True)
    nc_files_raw = glob.glob(os.path.join(input_dir, "**", "*.nc"), recursive=True)
    nc_files_raw_shruken = nc_files_raw[:5]  # para teste, remover depois

    ####################
    # lista para armazenar dataframes spark
    list_spark = []
    ####################

    #for nc_file in nc_files_raw:
    for nc_file in nc_files_raw_shruken:

        # carregando dataset
        ds = xr.open_dataset(nc_file)[variable_id]
        logger_ingestion.info(f"arquivo {nc_file} aberto com sucesso")

        # aplica transformações definidas no dicionário
        for transform in variavel_escolhida["transformations"]:
            logger_ingestion.info(f"aplicando transformação {transform}")
            ds = map_transform_funcs[transform](ds)

        # dataframe para Spark
        df = ds.to_dataframe().reset_index()
        df["time"] = pd.to_datetime(df["time"])
        df = add_time_features(df)
        logger_ingestion.info(f"dataframe criado com sucesso")

        # convertendo para spark dataframe e adiciona à lista
        df_spark = spark.createDataFrame(df)
        list_spark.append(df_spark)

        ds.close()
        
    # concatenar todos os dataframes spark
    df_spark_final = list_spark[0]
    for df_s in list_spark[1:]:
        df_spark_final = df_spark_final.union(df_s)
    logger_ingestion.info(f"dataframes concatenados com sucesso")
    
    # reparticionando para evitar small files
    df_spark_final = df_spark_final.repartition(50, "year", "month")
    logger_ingestion.info(f"dataframe reparticionado com sucesso")

    # salvando em parquet particionado na camada trusted
    (
        df_spark_final.write
        .mode("append")
        .partitionBy("year", "month")
        .parquet(output_dir)
    )

    logger_ingestion.info(f"dados salvos em {output_dir} particionado por year e month")
    logger_ingestion.info(f"pipeline concluído | trusted")