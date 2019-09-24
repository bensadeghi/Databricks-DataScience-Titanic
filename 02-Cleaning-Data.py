# Databricks notebook source
# MAGIC %md
# MAGIC # Cleaning Data
# MAGIC 
# MAGIC In the previous notebook we have seen how we can get data into Spark Dataframes or SQL Tables, and do initial queries and visualizations on it. However, let's dive deeper into our dataset, and create a stable ETL pipeline for our end-users.

# COMMAND ----------

# MAGIC %run ./Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We can read in the data from one of the permanent tables we created earlier.

# COMMAND ----------

titanicDF = spark.table("titanic")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT COUNT(*) FROM titanic

# COMMAND ----------

display(titanicDF)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Renaming Columns

# COMMAND ----------

# MAGIC %md
# MAGIC Let's rename the `Pclass` field name to `PassengerClass`, and `SibSp` to `SiblingsSpouses`
# MAGIC Save the results in a new DataFrame called `titanic_clean_df`

# COMMAND ----------

titanicCleanDF = titanicDF.withColumnRenamed("Pclass", "PassengerClass").withColumnRenamed("SibSp", "SiblingsSpouses")

# COMMAND ----------

titanicCleanDF.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Dropping Columns

# COMMAND ----------

# MAGIC %md
# MAGIC Let's remove the columns `PassengerId`, `Name`, `Ticket` and `Cabin`, since they won't be very useful for our machine learning work

# COMMAND ----------

titanicCleanDF = titanicCleanDF \
                 .drop("PassengerID") \
                 .drop("Name") \
                 .drop("Ticket") \
                 .drop("Cabin")

# COMMAND ----------

titanicCleanDF.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Remove Null Values

# COMMAND ----------

# Drop all rows containing any null or NaN values

titanicCleanDF = titanicCleanDF.na.drop()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Convert Categorical Columns to Numerical
# MAGIC The [StringIndexer](https://spark.apache.org/docs/latest/ml-features.html#stringindexer) function convert a string column to an index column

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

# Convert Sex and Embarked columns to index column
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(titanicCleanDF) for column in ["Sex", "Embarked"]]
pipeline = Pipeline(stages=indexers)
titanic_clean_df = pipeline.fit(titanicCleanDF).transform(titanicCleanDF)

# Drop old Sex and Embarked columns
titanicCleanDF = titanicCleanDF.drop("Sex").drop("Embarked")

# COMMAND ----------

display(titanicCleanDF)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Saving Our Work
# MAGIC Let's register the new cleaned DataFrame as a Table

# COMMAND ----------

titanicCleanDF.write.mode("overwrite").format("delta").saveAsTable("titanic_clean")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT COUNT(*) FROM titanic_clean

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Step
# MAGIC 
# MAGIC [What is Machine Learning]($./03-What-is-ML)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2019 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>