# utils
import json
import logging
import os
import sys
#from utils.path_utils import detect_frequency

# manipulação de datas e timezone
import pytz
import datetime as dt
import time

# spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import array, avg, col, collect_list, concat_ws, expr, lit, struct, udf, variance

# pca e redução de dimensionalidade
from pyspark.ml.feature import PCA, StandardScaler, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.stat import Summarizer
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.sql.types import ArrayType, DoubleType, StructType, StructField

# computação científica
import numpy as np
from scipy.stats import entropy
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
logger_transform = configurar_logger("transformacao_dados")


####################
def array_to_vector(colname):
    """Converte array<double> em DenseVector (necessário para MLlib)."""
    return col(colname).cast("array<double>").alias(colname)
####################
def compute_entropy(arr):
        hist, _ = np.histogram(arr, bins=20, density=True)
        hist += 1e-12
        return float(entropy(hist))
####################
def process_variable_delivery_teste(
    variable_id,
    input_model,
    input_experiment_id,
    input_dir,
    output_dir,
    grid_step=2.0,
    apply_pca_flag=False,
    apply_lasso_flag=False,
    lasso_target_strategy="mean",
    n_pca_components=3,
    lasso_regularization=0.1,
    frequency="mon"
):
    """
    pipeline da camada delivery
    
    etapas:
    - leitura dos dados da camada trusted (exp/freq/year)
    - reamostragem espacial (redução resolução → grid_step)
    - flatten (tempo x gridpoints) → vetores de features
    - [opcional] PCA (redução dimensionalidade)
    - [opcional] LASSO (seleção de features)
    - cálculo de métricas (variância, entropia, PCA, LASSO)
    - armazenamento dos datasets em parquet + métricas
    
    Args:
        variable_id (str): variável climática (ex: "tas")
        input_model (str): nome do modelo climático
        input_experiment_id (str): experimento (ex: historical)
        input_dir (str): diretório base da camada trusted
        output_dir (str): diretório base da camada delivery
        grid_step (float): resolução espacial em graus
        apply_pca_flag (bool): aplicar PCA?
        apply_lasso_flag (bool): aplicar LASSO?
        lasso_target_strategy (str): estratégia de target do LASSO [mean|max|min]
        n_pca_components (int): nº de componentes principais (PCA)
        lasso_regularization (float): parâmetro de regularização do LASSO
        frequency (str): frequência temporal ("mon" ou "day")
    """
    start_time = time.time()

    logger_ingestion.info(f"inicializando pipeline | delivery")
    logger_ingestion.info(f"config:  var={variable_id} | modelo={input_model} | exp={input_experiment_id} | freq={frequency}")
    
    spark = SparkSession.builder.appName("ClimateData").getOrCreate()

    # UDF para converter array<double> → DenseVector
    array_to_vector_udf = udf(lambda arr: Vectors.dense(arr) if arr is not None else None, VectorUDT())


    # leitura parquet trusted
    trusted_path = os.path.join(
        input_dir,
        f"exp={input_experiment_id}",
        f"freq={frequency}"
        )
    
    df_trusted = spark.read.parquet(trusted_path)
    if df_trusted.count() == 0:
        logger_ingestion.warning(f"nenhum dado encontrado em {trusted_path}")
        return

    # reamostragem espacial
    df_resampled = (
        df_trusted
        .withColumn("lat_grid", (col("lat") / grid_step).cast("int") * grid_step)
        .withColumn("lon_grid", (col("lon") / grid_step).cast("int") * grid_step)
        .groupBy("year", "month", "lat_grid", "lon_grid")
        .agg(avg(variable_id).alias(f"{variable_id}_mean"))
    )
    var_value = df_resampled.select(variance(f"{variable_id}_mean").alias("var")).first()["var"]

    logger_transform.info(f"reamostragem concluída | {df_resampled.count()} registros | variância={var_value:.3f}")

    # flatten → vetor de features
    df_vector = (
        df_resampled
        .groupBy("year", "month")
        .agg(
            collect_list(f"{variable_id}_mean").alias("grid_list"),
            collect_list(struct("lat_grid", "lon_grid")).alias("grid_coords")
        )
    )
    #df_vector = df_vector.withColumn("features", array_to_vector("grid_list"))

    #assembler = VectorAssembler(inputCols=["grid_list"], outputCol="features")
    #df_vector = assembler.transform(df_vector)

    # array<double> → DenseVector
    df_vector = df_vector.withColumn("features", array_to_vector_udf("grid_list"))


    # salvar feature_map
    coords_example = df_vector.select("grid_coords").first()["grid_coords"]
    feature_map = {f"f{i}": (row["lat_grid"], row["lon_grid"]) for i, row in enumerate(coords_example)}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "feature_map.json"), "w", encoding="utf-8") as f:
        json.dump(feature_map, f, indent=2)

    # PCA opcional
    pca_metrics = {}
    if apply_pca_flag:
        logger_transform.info(f"aplicando PCA | k={n_pca_components}")
        pca = PCA(k=n_pca_components, inputCol="features", outputCol="pca_features")
        pca_model = pca.fit(df_vector)
        df_vector = pca_model.transform(df_vector)
        explained_variance = pca_model.explainedVariance.toArray()
        pca_metrics["explained_variance"] = explained_variance.tolist()
        pca_metrics["cumulative_variance"] = np.cumsum(explained_variance).tolist()
    else:
        logger_transform.info("PCA não aplicado")

    # LASSO opcional
    lasso_metrics = {}
    if apply_lasso_flag:
        target_col = f"{variable_id}_{lasso_target_strategy}"
        if lasso_target_strategy == "mean":
            df_vector = df_vector.withColumn(target_col, expr("aggregate(grid_list, 0D, (acc, x) -> acc + x)/size(grid_list)"))
        elif lasso_target_strategy == "max":
            df_vector = df_vector.withColumn(target_col, expr("aggregate(grid_list, -1D/0D, (acc, x) -> IF(x > acc, x, acc))"))
        elif lasso_target_strategy == "min":
            df_vector = df_vector.withColumn(target_col, expr("aggregate(grid_list, 1D/0D, (acc, x) -> IF(x < acc, x, acc))"))
        else:
            raise ValueError(f"target inválido: {lasso_target_strategy}")

        #df_vector = df_vector.withColumn("features_scaled", array_to_vector("grid_list"))
        #assembler_lasso = VectorAssembler(inputCols=["grid_list"], outputCol="features_scaled")
        #df_vector = assembler_lasso.transform(df_vector)

        # array<double> → DenseVector
        df_vector = df_vector.withColumn("features_scaled", array_to_vector_udf("grid_list"))

        lasso = LinearRegression(
            featuresCol="features_scaled",
            labelCol=target_col,
            elasticNetParam=1.0,
            regParam=lasso_regularization
        )

        lasso_model = lasso.fit(df_vector)
        coeffs = lasso_model.coefficients.toArray()
        selected_idx = [i for i, c in enumerate(coeffs) if abs(c) > 1e-6]
        lasso_metrics.update({
            "regParam": lasso_regularization,
            "MSE": lasso_model.summary.meanSquaredError,
            "R2": lasso_model.summary.r2,
            "n_selected_features": len(selected_idx),
            "sparsity": len(selected_idx)/len(coeffs),
            "selected_indices": selected_idx
        })
        logger_transform.info(f"LASSO concluído | n_features={len(selected_idx)} | R2={lasso_model.summary.r2:.3f}")

        if "coeffs" in locals():
            lasso_metrics.update({
            "coefficients": coeffs.tolist(),  # todos os coeficientes
            "selected_features": {
                f"f{i}": {
                    "coord": feature_map[f"f{i}"],
                    "weight": float(coeffs[i])
                }
                for i in selected_idx
            }
            })

    # métricas derivadas: entropia média
    entropy_values = df_vector.select("grid_list").rdd.map(lambda r: compute_entropy(r[0])).collect()
    entropy_mean = float(np.mean(entropy_values))

    # cálculo da quantidade de linhas e particções do dataframe
    n_rows = df_vector.count()
    n_partitions = df_vector.rdd.getNumPartitions()

    # contagem de lat/lon únicos
    n_lat = df_resampled.select("lat_grid").distinct().count()
    n_lon = df_resampled.select("lon_grid").distinct().count()
    n_gridpoints = n_lat * n_lon

    # estatísticas descritivas
    stats_row = (
        df_resampled
        .selectExpr(
            f"mean({variable_id}_mean) as mean",
            f"stddev({variable_id}_mean) as std",
            f"min({variable_id}_mean) as min",
            f"max({variable_id}_mean) as max"
        )
        .first()
    )

    end_time = time.time()
    execution_time_seconds = end_time - start_time

    # construção do dicionário de métricas
    '''
    metrics = {
        "meta": {
            "model": input_model,
            "variable": variable_id,
            "experiment": input_experiment_id,
            "grid_step": grid_step,
            "n_samples": df_vector.count(),
            "n_features": len(coords_example),
            "frequency": frequency
        },
        "pca": pca_metrics,
        "lasso": lasso_metrics,
        "derived": {"entropy_mean": entropy_mean}
    }
    '''

    metrics = {
        "meta": {
            "model": input_model,
            "variable": variable_id,
            "experiment": input_experiment_id,
            "grid_step": grid_step,
            "frequency": frequency
        },

        "time": {
            "n_steps": df_resampled.select("year", "month").distinct().count(),
            "frequency": frequency
        },
        "space": {
            "n_lat": n_lat,
            "n_lon": n_lon,
            "n_gridpoints": n_gridpoints
        },
        "parquet": {
            "n_rows": n_rows,
            "n_partitions": n_partitions
        },
        "stats": {
            "mean": float(stats_row["mean"]),
            "std": float(stats_row["std"]),
            "min": float(stats_row["min"]),
            "max": float(stats_row["max"])
        },
        "n_samples": n_rows,
        "n_features": len(coords_example),
        "n_partitions": n_partitions,
        "entropy_mean": entropy_mean,
        "execution_time_seconds": execution_time_seconds,
        "pca": pca_metrics,
        "lasso": lasso_metrics
    }


    # salvar parquet particionado (exp/freq/year)
    delivery_path = os.path.join(
        output_dir,
        f"exp={input_experiment_id}",
        f"freq={frequency}"
    )

    # salvando o dataframe
    (
        df_vector
        .write
        .mode("overwrite")
        .partitionBy("year", "month")
        .parquet(delivery_path)
    )
    logger_ingestion.info(f"dados salvos em {delivery_path}")

    

    # salvar metrics.json
    metrics_path = os.path.join(delivery_path, f"metrics_delivery_{variable_id}_{input_model.lower()}_{input_experiment_id}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger_ingestion.info(f"métricas salvas em {metrics_path}")

    logger_ingestion.info(f"pipeline concluído | delivery | tempo total de execução da camada delivery: = {execution_time_seconds: .2f}s")
####################

'''
def apply_pca(dataframe_vector, n_components=3):
    """
    Aplica PCA para redução de dimensionalidade
    :param dataframe_vector: DataFrame do Spark com coluna 'features'
    :param n_components: número de componentes principais
    :return: DataFrame com coluna 'pca_features'
    """
    logger_transform.info(f"inicializando PCA com {n_components} componentes principais")
    pca = PCA(k=n_components, inputCol="features", outputCol="pca_features")
    pca_model = pca.fit(dataframe_vector)
    dataframe_pca = pca_model.transform(dataframe_vector)

    explained_variance = pca_model.explainedVariance.toArray()
    logger_transform.info(f"variância explicada pelos {n_components} componentes principais: {explained_variance}")
    logger_transform.info(f"variância acumulada: {explained_variance.sum():.5f}")

    return dataframe_pca
'''
####################
'''
def apply_lasso(df_vector, target_col, use_pca=False, regularization=0.1, feature_map_path=None):
    """
    Aplica regressão LASSO com ou sem PCA.
    
    :param df_vector: DataFrame Spark com coluna 'features'
    :param use_pca: se True, aplica LASSO sobre PCA
    :param target_col: coluna alvo
    :param regularization: parâmetro alpha do LASSO; forã da regularização
    :param output_path: diretório para salvar o resultado
    :param feature_map_path: caminho do feature_map.json
    :return: DataFrame com variáveis explicativas e selecionadas
    """
    
    # input (features originais ou PCA)
    if use_pca:
        logger_transform.info(f"inicializando regressão linear LASSO com regularização {regularization} | input: DenseArray com PCA")
        df_input = df_vector.withColumn("features_lasso", col("pca_features"))
    else:
        logger_transform.info(f"inicializando regressão linear LASSO com regularização {regularization} | input: DenseArray sem PCA")
        df_input = df_vector.withColumn("features_lasso", col("features"))

    # ajustando modelo lasso
    # elasticNetParam deve ser sempre 1.0 para que a regularização LASSO pura seja aplicada
    lasso = LinearRegression(featuresCol="features_lasso", labelCol=target_col, elasticNetParam=1.0, regParam=regularization)

    lasso_model = lasso.fit(df_input)

    # extraindo coeficientes não nulos (seleção de features)
    logger_transform.info(f"extraindo coeficientes não-nulos (seleção de features)")
    coeffs = lasso_model.coefficients.toArray()
    selected_idx = [i for i, c in enumerate(coeffs) if abs(c) > 1e-6]

    mse = lasso_model.summary.meanSquaredError
    r2 = lasso_model.summary.r2

    logger_transform.info(f"LASSO MSE: {mse: .5f}")
    logger_transform.info(f"LASSO R²: {r2: .5f}")

    # se houver feature_map, traduzir índices → (lat, lon)
    feature_names = []
    if feature_map_path and os.path.exists(feature_map_path):
        logger_transform.info(f"lendo feature map para identificar índices")
        with open(feature_map_path, "r", encoding="utf-8") as f:
            feature_map = json.load(f)
        feature_names = [feature_map[f"f{i}"] for i in selected_idx]
    else:
        logger_transform.info(f"varredura para identificar índices")
        feature_names = [f"f{i}" for i in selected_idx]

    # construindo dataframe final
    logger_transform.info(f"construção do dataframe com features selecionadas")
    df_processed = df_input.select("year", "month", target_col, "features")
    
    # converter DenseVector → array<double>
    df_processed = df_processed.withColumn("features_array", vector_to_array("features"))

    # adicionar colunas para features selecionadas
    for idx, name in zip(selected_idx, feature_names):
        #df_processed = df_processed.withColumn(f"sel_{idx}", df_processed.features[idx])
        df_processed = df_processed.withColumn(f"sel_{idx}", df_processed.features_array[idx])

    return df_processed, selected_idx, feature_names
'''
####################
'''   
def process_variable_delivery(variable_id, input_model, input_experiment_id, input_dir, output_dir,
                                grid_step=2.0, apply_pca_flag=False, apply_lasso_flag=False,
                                lasso_target_strategy="mean", n_pca_components=3, lasso_regularization=0.1):

    """
    pipeline da ingestão na delivery:
    1. leitura dos dados da trusted (todos os anos)
    2. reamostragem espacial (redução da resolução)
    3. flatten (tempo x gridpoints)
    4. [opcional] aplicar PCA para redução de dimensionalidade
    5. [opcional] aplicar regressão linear com Lasso para seleção de features
    6. salvar em parquet particionado

    :param variable_id: variável climática (ex: "tas")
    :param input_model: nome do modelo climático
    :param input_experiment_id: experimento (ex: historical)
    :param dir_trusted: diretório da camada Trusted
    :param dir_delivery: diretório para salvar a camada Delivery
    :param grid_step: resolução em graus (default 2.0°)
    :param apply_pca_flag: indica se deve aplicar fase de processamento com PCA
    :param apply_lasso_flag: indica se deve aplicar fase de processamento com LASSO
    :param lasso_target_strategy: tipo de agregação a ser aplicada para as features input do LASSO
    :param n_pca_components: quantidade de componentes principais a serem calculadas para o PCA
    :param lasso_regularization: fator de regularização, que deve ser 0.1 para regularização LASSO
    """

    logger_ingestion.info(f"inicializando pipeline | delivery")
    logger_ingestion.info(f"variável: {variable_id} | modelo: {input_model} | experimento: {input_experiment_id}")

    # iniciar sessão spark
    # glue context
    #from pyspark.context import SparkContext
    #from awsglue.context import GlueContext

    #sc = SparkContext.getOrCreate()
    #glue_context = GlueContext(sc)
    #spark = glue_context.spark_session

    spark = SparkSession.builder.appName("ClimateData").getOrCreate()

    # leitura dos dados da trusted (todos os anos)
    logger_ingestion.info(f"lendo todos os arquivos da camada trusted em {input_dir}")
    df_trusted = spark.read.parquet(input_dir)

    # verificando se há dados
    if df_trusted.count() == 0:
        logger_ingestion.warning("nenhum dado encontrado na camada trusted")
        return

    logger_transform.info("instanciando udf para converter o array<double> em DenseVector")

    # UDF para converter array<double> em DenseVector
    to_vector = udf(lambda x: Vectors.dense(x) if x is not None else Vectors.dense([]))

    # reamostragem espacial: arredondando lat/lon para grid_step
    logger_transform.info(f"reamostrando para resolução {grid_step}°")

    df_trusted = df_trusted.repartition(50)

    df_resampled = (
        df_trusted
        .withColumn("lat_grid", (col("lat") / grid_step).cast("int") * grid_step)
        .withColumn("lon_grid", (col("lon") / grid_step).cast("int") * grid_step)
        .groupBy("year", "month", "lat_grid", "lon_grid")
        .agg(avg(variable_id).alias(f"{variable_id}_mean"))
    )

    df_resampled = df_resampled.repartition(50)

    # calculando a variância do dataframe
    #var_df = df_resampled.select(variance(df_resampled.tas_mean).alias("var"))
    var_df = df_resampled.select(variance(df_resampled[f"{variable_id}_mean"]).alias("var"))

    # valor como variável (float)
    var_value = var_df.first()["var"]

    logger_transform.info(f"reamostragem espacial concluída | {df_resampled.count()} registros | variância: {var_value: .3f}")

    # agrupando todos os valores em uma lista
    logger_transform.info("agrupando todos os gridpoints em uma única lista por ano/mês")
    df_vector_ready = (df_resampled
                    .groupBy("year", "month")
                    .agg(collect_list(f"{variable_id}_mean").alias("grid_list"),
                        collect_list(struct("lat_grid", "lon_grid")).alias("grid_coords"))
                    )
    logger_transform.info(f"agrupamento concluído")

    # convertendo array<double> → DenseVector
    #df_vector = df_vector_ready.withColumn("features", array_to_vector("grid_list"))
    df_vector = (df_vector_ready
                .withColumn("features", array_to_vector("grid_list"))
                .withColumn(f"{variable_id}_mean", expr("AGGREGATE(grid_list, 0D, (acc, x) -> acc + x) / size(grid_list)"))
                )
    target_col_name = f"{variable_id}_{lasso_target_strategy}"

    # nomes das colunas de grid
    n_features = df_vector.selectExpr("size(grid_list) as n").first()["n"]
    feature_names = [f"f{i}" for i in range(n_features)]

    logger_ingestion.info(f"construindo dicionário de mapeamento feature —> (lat, lon)")
    # construindo dicionário de mapeamento feature -> (lat, lon)
    coords_example = df_vector.select("grid_coords").first()["grid_coords"]
    feature_map = {f"f{i}": (row["lat_grid"], row["lon_grid"]) for i, row in enumerate(coords_example)}

    # salvar JSON no diretório de saída
    feature_map_dir = os.path.join(output_dir.lower(),
                                    f"exp={input_experiment_id}",
                                    f"pca={apply_pca_flag}",
                                    f"lasso={apply_lasso_flag}",
                                    "feature_map.json")

    os.makedirs(os.path.dirname(feature_map_dir), exist_ok=True)

    with open(feature_map_dir, "w", encoding="utf-8") as f:
        json.dump(feature_map, f, indent=2)

    logger_ingestion.info(f"feature_map salvo em {feature_map_dir}")

    if apply_lasso_flag or apply_pca_flag:

        if lasso_target_strategy == "mean":
            logger_ingestion.info(f"lasso_target_strategy: {lasso_target_strategy}")
            #df_flatten = df_flatten.withColumn("lasso_target", expr("aggregate(grid_array, 0D, (acc, x) -> acc + x) / size(grid_array)"))
            df_vector = df_vector.withColumn(f"{target_col_name}", expr("aggregate(grid_list, 0D, (acc, x) -> acc + x) / size(grid_list)"))
        elif lasso_target_strategy == "max":
            logger_ingestion.info(f"lasso_target_strategy: {lasso_target_strategy}")
            #df_flatten = df_flatten.withColumn("lasso_target", expr("aggregate(grid_array, -1D/0D, (acc, x) -> IF(x > acc, x, acc))"))
            df_vector = df_vector.withColumn(f"{target_col_name}", expr("aggregate(grid_list, -1D/0D, (acc, x) -> IF(x > acc, x, acc))"))
        elif lasso_target_strategy == "min":
            logger_ingestion.info(f"lasso_target_strategy: {lasso_target_strategy}")
            #df_flatten = df_flatten.withColumn("lasso_target", expr("aggregate(grid_array, 1D/0D, (acc, x) -> IF(x < acc, x, acc))"))
            df_vector = df_vector.withColumn(f"{target_col_name}", expr("aggregate(grid_list, 1D/0D, (acc, x) -> IF(x < acc, x, acc))"))
        else:
            raise ValueError(f"valor inválido para lasso_target_strategy: {lasso_target_strategy}")

    logger_transform.info(f"conversão de array<double> para DenseVector concluída")

    # salvando em outra variável para compatibilidade com o restante do processamento
    df_processed = df_vector

    # [opcional] aplicar PCA para redução de dimensionalidade
    if apply_pca_flag:
        # aplica apenas PCA
        df_processed = apply_pca(df_processed, n_components=n_pca_components)
    else:
        logger_transform.info("PCA não aplicado")
    
    ##########
    if apply_lasso_flag:
        df_processed, selected_idx, feature_names = apply_lasso(df_processed, target_col_name, apply_pca_flag, regularization=lasso_regularization, feature_map_path=feature_map_dir)
    else:
        logger_transform.info("LASSO não aplicado")
    ##########

    # 6. salvando em parquet particionado
    #output_path = os.path.join(dir_delivery, input_model, variable_id)

    
    #output_path = os.path.join(output_dir, f"exp={input_experiment_id}", f"pca={apply_pca_flag}", f"lasso={apply_lasso_flag}")
    #(
    #    df_processed.write
    #    .mode("overwrite")
    #    .partitionBy("year", "month")
    #    .parquet(output_path)
    #)
    
    output_path = os.path.join(output_dir)

    (
        df_processed
        .withColumn("exp", lit(input_experiment_id))
        .withColumn("pca", lit(str(apply_pca_flag)))
        .withColumn("lasso", lit(str(apply_lasso_flag)))
        .write
        .mode("overwrite")
        .partitionBy("exp", "pca", "lasso", "year", "month")
        .parquet(output_path)
    )

    logger_ingestion.info(f"dados salvos em {output_path} particionados por year e month")

    logger_ingestion.info(f"pipeline concluído | delivery")
'''