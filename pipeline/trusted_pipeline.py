# utils
import glob
import os
import sys
import logging
from collections import defaultdict

# manipulação de dados
import xarray as xr
import pandas as pd
from pyspark.sql import SparkSession
import dask.array as da
import dask.dataframe as dd

# manipulação de datas e timezone
import pytz
import datetime as dt

# configurações de modelos e variáveis
from config.variables_config import map_variaveis_meta

# transformações
from transformations.temporal import add_time_features, add_time_features_teste_freqs
from config.variables_config import map_variaveis_meta
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
    ####################
    logger_ingestion.info(f"inicializando pipeline | trusted")
    ####################
    # iniciar sessão spark
    # glue context
    #from pyspark.context import SparkContext
    #from awsglue.context import GlueContext

    #sc = SparkContext.getOrCreate()
    #glue_context = GlueContext(sc)
    #spark = glue_context.spark_session

    spark = SparkSession.builder.appName("ClimateData").getOrCreate()
    ####################

    # selecionando a variável
    #meta = map_variaveis_meta[variable_id]
    variavel_escolhida = map_variaveis_meta[variable_id]
    logger_ingestion.info(f"variável escolhida: {variavel_escolhida['variable_id']}")

    # origem dos datasets *.nc | camada raw
    #nc_files = glob.glob(os.path.join(input_dir, "**", "*.nc"), recursive=True)
    nc_files_raw = glob.glob(os.path.join(input_dir, "**", "*.nc"), recursive=True)
    nc_files_raw_shruken = nc_files_raw#[:5]  # para teste, remover depois

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

########################
def process_variable_trusted_teste_freqs(variable_id, input_dir, output_dir, map_transform_funcs, frequency="mon", chunk_time=None):
    """
    Pipeline camada trusted.

    Args:
        variable_id (str): variável climática.
        input_dir (str): diretório RAW.
        output_dir (str): diretório TRUSTED.
        map_transform_funcs (dict): dicionário de transformações.
        frequency (str): "mon" ou "day".
        chunk_time (int): tamanho do chunk temporal (opcional, apenas com Dask/xarray).
    """
    logger_ingestion.info(f"Iniciando trusted pipeline | variable={variable_id} | frequency={frequency}")

    spark = SparkSession.builder.appName("ClimateData").getOrCreate()

    variavel_meta = map_variaveis_meta[variable_id]
    nc_files = glob.glob(os.path.join(input_dir, "**", "*.nc"), recursive=True)

    list_spark = []

    for nc_file in nc_files:
        logger_ingestion.info(f"Lendo arquivo {nc_file}")

        # --- Dask opcional para chunking
        if chunk_time:
            ds = xr.open_dataset(nc_file, chunks={"time": chunk_time})[variable_id]
        else:
            ds = xr.open_dataset(nc_file)[variable_id]

        # aplicar transformações
        for transform in variavel_meta["transformations"]:
            ds = map_transform_funcs[transform](ds)

        # converter para dataframe Pandas
        df = ds.to_dataframe().reset_index()
        df["time"] = pd.to_datetime(df["time"])
        df = add_time_features_teste_freqs(df, frequency=frequency)

        # converter para Spark DataFrame
        df_spark = spark.createDataFrame(df)
        list_spark.append(df_spark)
        ds.close()

    # concatenar Spark DataFrames
    df_spark_final = list_spark[0]
    for df_s in list_spark[1:]:
        df_spark_final = df_spark_final.union(df_s)

    # reparticionar e salvar parquet
    partition_cols = ["year", "month"] if frequency == "mon" else ["year", "month", "day"]
    df_spark_final = df_spark_final.repartition(50, *partition_cols)

    (
        df_spark_final.write
        .mode("append")
        .partitionBy(*partition_cols)
        .parquet(output_dir)
    )
    
    logger_ingestion.info(f"pipeline concluído | trusted")
    logger_ingestion.info(f"dados salvos em {output_dir}")

########################
####################
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

####################
#def process_variable_trusted_dask(variable_id, input_dir, output_dir, map_transform_funcs):
def process_variable_trusted_dask(input_model,
                                input_experiment_id,
                                input_variable_id,
                                input_variant_label,
                                input_dir, output_dir, map_transform_funcs, table_id="Amon"):
    """
    
    Pipeline camada trusted.

    Args:
        variable_id (str): variável climática.
        input_dir (str): diretório RAW.
        output_dir (str): diretório TRUSTED.
        map_transform_funcs (dict): dicionário de transformações.
        frequency (str): "mon" ou "day".
        chunk_time (int): tamanho do chunk temporal (opcional, apenas com Dask/xarray).
    
    """

    logger_ingestion.info(f"inicializando pipeline | trusted")

    frequency = detect_frequency(input_dir)
    logger_ingestion.info(f"frequência detectada: {frequency}")
    logger_ingestion.info(f"config: model={input_model}, exp={input_experiment_id}, var={input_variable_id}, table={table_id}, freq={frequency}")

    # spark session
    spark = SparkSession.builder.appName("ClimateData").getOrCreate()

    # selecionando transformações
    variavel_escolhida = map_variaveis_meta[input_variable_id]
    logger_ingestion.info(f"transformações aplicadas: {variavel_escolhida['transformations']}")

    # listando arquivos NetCDF
    nc_files = glob.glob(os.path.join(input_dir, "**", "*.nc"), recursive=True)
    if not nc_files:
        logger_read.warning("nenhum arquivo encontrado na camada raw")
        return

    logger_ingestion.info(f"{len(nc_files)} arquivos encontrados")

    # lista de dataframes dask
    list_dask = []

    # agrupando arquivos por ano para processamento incremental
    files_by_year = {}
    for f in nc_files:
        basename = os.path.basename(f)
        # assume padrão ..._YYYYMM-YYYYMM.nc
        year_str = basename.split("_")[-1][:4]
        files_by_year.setdefault(year_str, []).append(f)

    for year, files in sorted(files_by_year.items()):
        # lista de dataframes dask
        list_dask = []
        logger_ingestion.info(f"processando ano {year} com {len(files)} arquivos")


        for nc_file in nc_files:
            logger_ingestion.info(f"lendo arquivo: {nc_file}")
            #ds = xr.open_dataset(nc_file)[variable_id]
            try:
                ds = xr.open_dataset(nc_file, engine="h5netcdf", chunks={})[input_variable_id]

                # aplicando transformações
                for transform in variavel_escolhida["transformations"]:
                    logger_ingestion.info(f"aplicando {transform}")
                    ds = map_transform_funcs[transform](ds)

                # convertendo para dask dataframe via xarray
                df = ds.to_dataframe().reset_index()
                df = add_time_features_teste_freqs(df, frequency=frequency)

                # particionando em chunks dask
                ddf = dd.from_pandas(df, npartitions=5)
                list_dask.append(ddf)
                ds.close()

            except Exception as e:
                logger_ingestion.error(f"erro ao abrir {nc_file}: {e}")
                continue

        if not list_dask:
            continue

        # concatenando todos os dask dataframes
        ddf_all = dd.concat(list_dask)
        df_spark = spark.createDataFrame(ddf_all.compute())

        # convertendo para spark dataframe
        logger_ingestion.info(f"dask dataframe do ano {year} convertido para spark dataframe")

        # particionamento dinâmico baseado na frequência
        partition_cols = ["year", "month"]
        if frequency == "day":
            partition_cols.append("day")

        # reparticiona e salva
        df_spark = df_spark.repartition(10, *partition_cols)
        df_spark.write.mode("append").partitionBy(*partition_cols).parquet(output_dir)
        logger_ingestion.info(f"ano {year} salvo em {output_dir} particionado por {partition_cols}")

    logger_ingestion.info(f"pipeline concluído | trusted")
    #logger_ingestion.info(f"dados salvos em {output_dir} particionado por {partition_cols}")


######################################
def process_variable_trusted_dask_teste(input_model,
                                input_experiment_id,
                                input_variable_id,
                                input_variant_label,
                                input_dir, output_dir, map_transform_funcs, table_id="Amon"):
    """
    Pipeline camada trusted.

    Args:
        input_model (str): modelo climático.
        input_experiment_id (str): experimento (ex: historical, ssp585).
        input_variable_id (str): variável climática (ex: tas).
        input_variant_label (str): rótulo da simulação (ex: r1i1p1f1).
        input_dir (str): diretório RAW.
        output_dir (str): diretório TRUSTED.
        map_transform_funcs (dict): dicionário de transformações.
        table_id (str): tabela CMIP (default=Amon).
    """

    logger_ingestion.info(f"inicializando pipeline | trusted")

    frequency = detect_frequency(input_dir)
    logger_ingestion.info(f"frequência detectada: {frequency}")
    logger_ingestion.info(f"config: model={input_model}, exp={input_experiment_id}, var={input_variable_id}, table={table_id}, freq={frequency}")

    # spark session
    spark = SparkSession.builder.appName("ClimateData").getOrCreate()

    # selecionando transformações
    variavel_escolhida = map_variaveis_meta[input_variable_id]
    logger_ingestion.info(f"transformações aplicadas: {variavel_escolhida['transformations']}")

    # listando arquivos NetCDF
    nc_files = glob.glob(os.path.join(input_dir, "**", "*.nc"), recursive=True)
    if not nc_files:
        logger_read.warning("nenhum arquivo encontrado na camada raw")
        return

    logger_ingestion.info(f"{len(nc_files)} arquivos encontrados")

    # agrupando arquivos por ano
    files_by_year = {}
    for f in nc_files:
        basename = os.path.basename(f)
        # assume padrão ..._YYYYMM-YYYYMM.nc
        year_str = basename.split("_")[-1][:4]
        files_by_year.setdefault(year_str, []).append(f)

    for year, files in sorted(files_by_year.items()):
        list_dask = []
        logger_ingestion.info(f"processando ano {year} com {len(files)} arquivos")

        for nc_file in files:  # <-- corrigido, antes estava `nc_files`
            logger_ingestion.info(f"lendo arquivo: {nc_file}")
            try:
                ds = xr.open_dataset(nc_file, engine="h5netcdf", chunks={})[input_variable_id]

                # aplicando transformações
                for transform in variavel_escolhida["transformations"]:
                    logger_ingestion.info(f"aplicando {transform}")
                    ds = map_transform_funcs[transform](ds)

                # convertendo para dask dataframe via xarray
                df = ds.to_dataframe().reset_index()
                df = add_time_features_teste_freqs(df, frequency=frequency)

                # particionando em chunks dask
                ddf = dd.from_pandas(df, npartitions=5)
                list_dask.append(ddf)
                ds.close()

            except Exception as e:
                logger_ingestion.error(f"erro ao abrir {nc_file}: {e}")
                continue

        if not list_dask:
            continue

        # concatenando todos os dask dataframes daquele ano
        ddf_all = dd.concat(list_dask)
        df_spark = spark.createDataFrame(ddf_all.compute())

        logger_ingestion.info(f"dask dataframe do ano {year} convertido para spark dataframe")

        # particionamento dinâmico baseado na frequência
        partition_cols = ["year", "month"]
        if frequency == "day":
            partition_cols.append("day")

        # reparticiona e salva
        df_spark = df_spark.repartition(10, *partition_cols)
        df_spark.write.mode("append").partitionBy(*partition_cols).parquet(output_dir)
        logger_ingestion.info(f"ano {year} salvo em {output_dir} particionado por {partition_cols}")

    logger_ingestion.info(f"pipeline concluído | trusted")

######################################
def process_variable_trusted_dask_only(input_model,
                                       input_experiment_id,
                                       input_variable_id,
                                       input_variant_label,
                                       input_dir,
                                       output_dir,
                                       map_transform_funcs,
                                       table_id="Amon"):
    """
    Pipeline camada trusted apenas com xarray + Dask.
    Lê arquivos NetCDF da raw, aplica transformações e salva em Parquet particionado.

    Args:
        input_model (str): modelo GCM.
        input_experiment_id (str): experimento (historical, ssp...).
        input_variable_id (str): variável climática (tas, pr...).
        input_variant_label (str): rótulo do conjunto (r1i1p1f1...).
        input_dir (str): diretório RAW.
        output_dir (str): diretório TRUSTED.
        map_transform_funcs (dict): dicionário de transformações.
        table_id (str): tabela CMIP (ex.: Amon).
    """
    logger_ingestion.info("inicializando pipeline | trusted (xarray + Dask)")

    frequency = detect_frequency(input_dir)
    logger_ingestion.info(f"frequência detectada: {frequency}")
    logger_ingestion.info(f"config: model={input_model}, exp={input_experiment_id}, var={input_variable_id}, table={table_id}, freq={frequency}")

    # selecionando transformações
    variavel_escolhida = map_variaveis_meta[input_variable_id]
    logger_ingestion.info(f"transformações aplicadas: {variavel_escolhida['transformations']}")

    # listando arquivos NetCDF
    nc_files = glob.glob(os.path.join(input_dir, "**", "*.nc"), recursive=True)
    if not nc_files:
        logger_ingestion.warning("nenhum arquivo encontrado na camada raw")
        return

    logger_ingestion.info(f"{len(nc_files)} arquivos encontrados")

    # abre todos os NetCDFs em lazy loading (Dask gerencia memória)
    ds = xr.open_mfdataset(nc_files,
                           engine="h5netcdf",
                           combine="by_coords",
                           parallel=True,
                           chunks={"time": 50})
    da = ds[input_variable_id]

    # aplica transformações
    for transform in variavel_escolhida["transformations"]:
        logger_ingestion.info(f"aplicando {transform}")
        da = map_transform_funcs[transform](da)

    # converte para dataframe com Dask (sem puxar tudo para memória)
    #df = da.to_dataframe().reset_index()
    #ddf = dd.from_pandas(df, npartitions="auto")
    ddf = da.to_dask_dataframe().reset_index()
    ddf = add_time_features_teste_freqs(ddf, frequency=frequency)

    # define partições de saída
    partition_cols = ["year", "month"]
    if frequency == "day":
        partition_cols.append("day")

    # salva em Parquet particionado
    ddf.to_parquet(
        output_dir,
        engine="pyarrow",
        write_index=False,
        partition_on=partition_cols
    )

    logger_ingestion.info(f"pipeline concluído | trusted (dados salvos em {output_dir}, particionado por {partition_cols})")