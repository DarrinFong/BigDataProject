from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession


def initSpark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark


def dataParallelization():
    spark = initSpark()
    df = spark.read.format('csv')\
        .option('encoding', "UTF-8")\
        .option('header', 'true')\
        .load("./data/hotel_bookings.csv")
    return df
