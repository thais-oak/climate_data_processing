# utils
import os
import logging
import sys

# manipulação de datas e timezone
import pytz
import datetime as dt

# spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import array, avg, col, concat_ws, expr, collect_list, struct, udf, variance

# pca e redução de dimensionalidade
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Summarizer
from pyspark.ml.feature import StandardScaler
from pyspark.ml.functions import array_to_vector
from pyspark.sql.types import ArrayType, DoubleType, StructType, StructField

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

def assemble_features(dataframe, target_col=None):
    """
    Função para montar o vetor de features para PCA e regressão linear
    :param dataframe: DataFrame do Spark
    :param target_col: lista de colunas a serem usadas como features
    :return: DataFrame com coluna 'features' contendo o vetor de features
    """
    numeric_cols = [c for c, dtype in dataframe.dtypes if dtype in ("int", "double", "float")]

    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # escapando nomes de colunas problemáticas com crase
    #numeric_cols_escaped = [f"`{c}`" for c in numeric_cols]

    # renomear colunas para nomes "seguros"
    safe_cols = [c.replace(" ", "_").replace("(", "").replace(")", "") for c in numeric_cols]
    for old, new in zip(numeric_cols, safe_cols):
        dataframe = dataframe.withColumnRenamed(old, new)
    logger_transform.info(f"colunas numéricas sanitizadas")

    #assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
    assembler = VectorAssembler(inputCols=safe_cols, outputCol="features")
    dataframe_vector = assembler.transform(dataframe)
    logger_transform.info(f"assemble features: colunas numéricas calculadas")

    return dataframe_vector, numeric_cols

def apply_pca(dataframe_vector, n_components=3):
    """
    Aplica PCA para redução de dimensionalidade
    :param dataframe: DataFrame do Spark com coluna 'features'
    :param n_components: número de componentes principais
    :return: DataFrame com coluna 'pca_features'
    """
    logger_transform.info(f"inicializando PCA com {n_components} componentes principais")
    pca = PCA(k=n_components, inputCol="features", outputCol="pca_features")
    pca_model = pca.fit(dataframe_vector)
    dataframe_pca = pca_model.transform(dataframe_vector)

    explained_variance = pca_model.explainedVariance.toArray()
    logger_transform.info(f"variância explicada pelos {n_components} componentes principais: {explained_variance}")
    logger_transform.info(f"variância acumulada: {explained_variance.sum()}")

    return dataframe_pca


def apply_lasso(dataframe_vector, target_col, pca_marker, regularization=0.1):
    """
    Aplica regressão linear com regularização Lasso (L1)
    :param dataframe: DataFrame do Spark com coluna 'features'
    :param target_col: coluna alvo para regressão
    :param regularization: parâmetro de regularização (lambda)
    :return: modelo treinado e DataFrame com previsões
    """
    if pca_marker:
        logger_transform.info(f"inicializando regressão linear LASSO com regularização {regularization} | input: DenseArray sem PCA")
    else:
        logger_transform.info(f"inicializando regressão linear LASSO com regularização {regularization} | input: output do PCA")
    
    # construção do vetor de features
    # definindo o modelo de regressão linear com Lasso
    lasso = LinearRegression(featuresCol="features", labelCol=target_col, regParam=regularization, elasticNetParam=1.0)

    # treinando o modelo
    lasso_model = lasso.fit(dataframe_vector)

    #selected_features = [feature_cols[i] for i, coef in enumerate(lasso_model.coefficients) if coef != 0]
    
    # métricas do modelo
    mse = lasso_model.summary.meanSquaredError
    r2 = lasso_model.summary.r2

    # pares (feature, coeficiente) ≠ 0
    # nomes das features PCA
    coeffs = lasso_model.coefficients.toArray()   # todos os coeficientes
    feature_names = [f"PC{i+1}" for i in range(len(coeffs))]
    selected_features = [(feature_names[i], c) for i, c in enumerate(coeffs) if c != 0.0]
    
    #non_zero = sum([1 for c in lasso_model.coefficients if c != 0])

    # filtrando apenas as colunas selecionadas + target
    #selected_cols = selected_features + [target_col] if target_col else selected_features

    logger_transform.info(f"LASSO MSE: {mse: .5f}")
    logger_transform.info(f"LASSO R²: {r2: .5f}")
    #logger_transform.info(f"variáveis selecionadas pelo LASSO: {non_zero} de {len(feature_cols)}")
    logger_transform.info(f"features selecionadas pelo LASSO: {len(selected_features)}/{len(feature_names)}")


    return dataframe_vector.select(*selected_features)




def process_variable_delivery_with_algorithms(variable_id, input_model, input_experiment_id,
                                              dir_trusted, dir_delivery, target_col, grid_step=2.0,
                                              apply_pca_flag=False, apply_lasso_flag=False,
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
    """

    logger_ingestion.info(f"inicializando pipeline | delivery")
    logger_ingestion.info(f"variável: {variable_id} | modelo: {input_model} | experimento: {input_experiment_id}")

    # Iniciar sessão spark
    spark = SparkSession.builder.appName("ClimateData").getOrCreate()

    # 1. leitura dos dados da trusted (todos os anos)
    logger_ingestion.info(f"lendo todos os arquivos da camada trusted em {dir_trusted}")
    df_trusted = spark.read.parquet(dir_trusted)

    # verificando se há dados
    if df_trusted.count() == 0:
        logger_ingestion.warning("nenhum dado encontrado na camada trusted")
        return

    logger_transform.info("instanciando udf para converter o array<double> em DenseVector")

    # UDF para converter array<double> em DenseVector
    to_vector = udf(lambda x: Vectors.dense(x) if x is not None else Vectors.dense([]))

    # reamostragem espacial: arredondando lat/lon para grid_step
    logger_transform.info(f"reamostrando para resolução {grid_step}°")

    df_resampled = (
        df_trusted
        .withColumn("lat_grid", (col("lat") / grid_step).cast("int") * grid_step)
        .withColumn("lon_grid", (col("lon") / grid_step).cast("int") * grid_step)
        .groupBy("year", "month", "lat_grid", "lon_grid")
        .agg(avg(variable_id).alias(f"{variable_id}_mean"))
    )

    # calculando a variância do dataframe
    var_df = df_resampled.select(variance(df_resampled.tas_mean).alias("var"))

    # valor como variável (float)
    var_value = var_df.first()["var"]

    logger_transform.info(f"reamostragem espacial concluída | {df_resampled.count()} registros | variância: {var_value: .3f}")

    # agrupando todos os valores em uma lista
    logger_transform.info("agrupando todos os gridpoints em uma única lista por ano/mês")
    df_vector_ready = (df_resampled
                    .groupBy("year", "month")
                    .agg(collect_list(f"{variable_id}_mean")
                            .alias("grid_list")))
    logger_transform.info(f"agrupamento concluído")

    # convertendo array<double> → DenseVector
    #df_vector = df_vector_ready.withColumn("features", array_to_vector("grid_list"))
    df_vector = (df_vector_ready
                .withColumn("features", array_to_vector("grid_list"))
                .withColumn("tas_mean", expr("AGGREGATE(grid_list, 0D, (acc, x) -> acc + x) / size(grid_list)"))
                )

    if apply_lasso_flag or apply_pca_flag:

        if lasso_target_strategy == "mean":
            logger_ingestion.info(f"lasso_target_strategy: {lasso_target_strategy}")
            #df_flatten = df_flatten.withColumn("lasso_target", expr("aggregate(grid_array, 0D, (acc, x) -> acc + x) / size(grid_array)"))
            df_vector = df_vector.withColumn(f"{variable_id}_{lasso_target_strategy}", expr("aggregate(grid_list, 0D, (acc, x) -> acc + x) / size(grid_list)"))
        elif lasso_target_strategy == "max":
            logger_ingestion.info(f"lasso_target_strategy: {lasso_target_strategy}")
            #df_flatten = df_flatten.withColumn("lasso_target", expr("aggregate(grid_array, -1D/0D, (acc, x) -> IF(x > acc, x, acc))"))
            df_vector = df_vector.withColumn(f"{variable_id}_{lasso_target_strategy}", expr("aggregate(grid_list, -1D/0D, (acc, x) -> IF(x > acc, x, acc))"))
        elif lasso_target_strategy == "min":
            logger_ingestion.info(f"lasso_target_strategy: {lasso_target_strategy}")
            #df_flatten = df_flatten.withColumn("lasso_target", expr("aggregate(grid_array, 1D/0D, (acc, x) -> IF(x < acc, x, acc))"))
            df_vector = df_vector.withColumn(f"{variable_id}_{lasso_target_strategy}", expr("aggregate(grid_list, 1D/0D, (acc, x) -> IF(x < acc, x, acc))"))
        else:
            raise ValueError(f"valor inválido para lasso_target_strategy: {lasso_target_strategy}")

    logger_transform.info(f"conversão de array<double> para DenseVector concluída")
    df_processed = df_vector

    # [opcional] aplicar PCA para redução de dimensionalidade
    if apply_pca_flag:
        # aplica apenas PCA
        df_processed = apply_pca(df_vector, n_components=n_pca_components)
    else:
        logger_transform.info("PCA não aplicado")

    # aplicar regressão linear com Lasso para seleção de features
    if apply_lasso_flag and apply_pca_flag:
        # PCA e LASSO
        df_processed = apply_lasso(df_processed, pca_marker=apply_pca_flag, target_col=target_col, regularization=lasso_regularization)
    
    elif apply_lasso_flag and not apply_pca_flag:
        # apenas LASSO
        df_processed = apply_lasso(df_vector, pca_marker=apply_pca_flag, target_col=target_col, regularization=lasso_regularization)
    else:
        logger_transform.info("LASSO não aplicado")

    # 6. salvando em parquet particionado
    output_path = os.path.join(dir_delivery, input_model, variable_id)
    output_path = os.path.join(dir_delivery, input_model, variable_id, f"exp={input_experiment_id}", f"pca={apply_pca_flag}", f"lasso={apply_lasso_flag}")
    
    (
        df_processed.write
        .mode("overwrite")
        .partitionBy("year", "month")
        .parquet(output_path)
    )



    logger_ingestion.info(f"dados salvos em {output_path} particionados por year e month")

    logger_ingestion.info(f"pipeline concluído | delivery")


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