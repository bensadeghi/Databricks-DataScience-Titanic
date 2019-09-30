# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Delete Created Resources
# MAGIC 
# MAGIC Let's do a bit of housekeeping and remove the tables, database, data and model saved to disk.

# COMMAND ----------

# MAGIC %run ./Classroom-Setup

# COMMAND ----------

# Remove data and model files, including the created directory
dataDir = userName + '/titanic_data'
dbutils.fs.rm(dataDir, recurse=True)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- drop the two tables
# MAGIC DROP TABLE titanic;
# MAGIC DROP TABLE titanic_clean;

# COMMAND ----------

# delete created database
spark.sql("DROP DATABASE `{}`".format(databaseName))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2019 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>