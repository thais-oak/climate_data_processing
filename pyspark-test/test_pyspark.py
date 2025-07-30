from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TesteLocal").getOrCreate()

data = [("Ana", 29), ("Carlos", 31), ("João", 35)]
df = spark.createDataFrame(data, ["Nome", "Idade"])
df.show()

spark.stop()
