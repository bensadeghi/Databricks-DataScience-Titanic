# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Delete Created Resources
# MAGIC 
# MAGIC Let's do a bit of housekeeping and remove the tables, database, data and model saved to disk.

# COMMAND ----------

# MAGIC %run ./Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### Delete Files, Tables and Database

# COMMAND ----------

# Remove data file, including the created directory
dbutils.fs.rm('dbfs:/Users/' + userName, recurse=True)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- drop the two tables
# MAGIC DROP TABLE titanic;
# MAGIC DROP TABLE titanic_clean;

# COMMAND ----------

# delete created database
spark.sql("DROP DATABASE `{}`".format(databaseName))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Delete Registered Models

# COMMAND ----------

from mlflow.tracking.client import MlflowClient
client = MlflowClient()

modelName = "Titanic-Model__" + userName
models = client.search_model_versions("name='{}'".format(modelName))

# loop over registered models
for model in models:
  try:
    # set model stage to Archive
    client.transition_model_version_stage(name=modelName, version=model.version, stage='Archived')
  except:
    pass
  # delete version of model
  client.delete_model_version(modelName, model.version)

# delete model
client.delete_registered_model(modelName)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>