# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Model Serving
# MAGIC 
# MAGIC When we move to talking about actually operationalizing the machine learning models we've built so far, this is where the discussion becomes tricky. Many organizations have yet to reach this step, as it can become quite complex to get here. Depending on the use-case at hand, there are several options for deploying a model and using it to make predictions on new data.
# MAGIC 
# MAGIC ### Agenda:
# MAGIC * Review model serving options
# MAGIC * Load a registered production model
# MAGIC * Perform batch scoring
# MAGIC * Perform stream scoring
# MAGIC * Discuss serving model as a web service

# COMMAND ----------

# MAGIC %md
# MAGIC By using Databricks to create your models, you can then choose your serving layer. Whether that's **batch** (where you score data on a regular interval), **streaming** (scoring non-stop data), or via a **web service** (where you make "random" calls to be scored), you can achieve the first 2 options using Databricks directly (or, for more complex pipelines, using scheduling via [Azure Data Factory](https://docs.microsoft.com/en-us/azure/data-factory/solution-template-databricks-notebook)), while the latter can easily be covered by integrating Databricks with [Azure Machine Learning Service](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-use-mlflow), for an easy way to deploy to an auto-scalable, containerized API.
# MAGIC 
# MAGIC See **Azure Reference Architecture** below:
# MAGIC ![](https://github.com/bensadeghi/Databricks-DataScience-Titanic/raw/master/img/azure_reference_architecture.PNG)

# COMMAND ----------

# MAGIC %run ./Classroom-Setup

# COMMAND ----------

# MAGIC %md ## Load versions of the registered model

# COMMAND ----------

from mlflow.tracking.client import MlflowClient
client = MlflowClient()

modelName = "Titanic-Model__" + userName
latestVersionInfo = client.get_latest_versions(modelName, stages=["Production"])
latestVersion = latestVersionInfo[0].version

print("The latest production version of the model '%s' is '%s'." % (modelName, latestVersion))

# COMMAND ----------

# MAGIC %md
# MAGIC The following cell uses the `mlflow.pyfunc.load_model()` API to load the latest version of production stage model as a generic Python function.

# COMMAND ----------

import mlflow.pyfunc

modelURI = latestVersionInfo[0].source
modelPipeline = mlflow.pyfunc.load_model(modelURI).spark_model

print("Loading registered model version from URI: '{model_uri}'".format(model_uri=modelURI))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Scoring
# MAGIC The most common way of using Data Science models today is by using a batch scoring system. For example, let's say you have implemented a churn system, and every morning you'd like one of your colleagues to go through a list of the top 5 most likely to churn customers, and ring them up. In the backend, that usually means that, the evening before, after the day's data has been collected, a ML model runs on the new information to produce a list of 5 likely churners.
# MAGIC 
# MAGIC In our case, we'd like to run our model once against our test dataset and prediction survival status.

# COMMAND ----------

# Load data

titanicDF = spark.read.table("titanic_clean")

# COMMAND ----------

# Make predictions in batch

predictions = modelPipeline.transform(titanicDF)

display(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stream Scoring
# MAGIC Another option of scoring would be through real-time streams. Imagine a scenario where you have sensor data from different machinery coming in, and, in a predictive maintenance situation, you'd like to be notified on the first occurence of a sensor being out of bounds, to minimise repair times and costs. Due to the throughput of that type of information, a Spark Streaming job is ideal to score data on-the-go, and showcase any potential anomalies.
# MAGIC 
# MAGIC While in our scenario we don't have frequent updates of information, we can still leverage Spark Streaming to score our batch dataset. We only have to read it as a stream!

# COMMAND ----------

# Read data as a stream

streamDF = spark.readStream.table("titanic_clean")

# COMMAND ----------

# Make real-time predictions on streaming data

scoredStream = modelPipeline.transform(streamDF)

display(scoredStream)

# COMMAND ----------

# Stop streaming jobs
for s in spark.streams.active:
    s.stop()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serving Model as a Web Service

# COMMAND ----------

# MAGIC %md
# MAGIC #### [Track model metrics and deploy ML models with MLflow and Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow#deploy-mlflow-models-as-a-web-service)
# MAGIC * Track and log experiment metrics and artifacts in your Azure Machine Learning workspace. If you already use MLflow Tracking for your experiments, the workspace provides a centralized, secure, and scalable location to store training metrics and models.
# MAGIC * Deploy your MLflow experiments as an Azure Machine Learning web service. By deploying as a web service, you can apply the Azure Machine Learning monitoring and data drift detection functionalities to your production models.
# MAGIC * Azure deployment infrastructure options:
# MAGIC   * Azure Container Instance - suitable choice for a quick dev-test deployment
# MAGIC   * Azure Kubernetes Service - suitable for scalable production deployments
# MAGIC 
# MAGIC ![Azure ML Arch](https://raw.githubusercontent.com/bensadeghi/Databricks-DataScience-Titanic/master/img/azure_ml_arch.JPG)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Resources
# MAGIC - [Notebook: Deploy Model to Azure Container Instance](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/azure-databricks/amlsdk/deploy-to-aci-04.ipynb)
# MAGIC - [Deploy a `python_function` model on Microsoft Azure ML](https://www.mlflow.org/docs/latest/models.html#deploy-a-python-function-model-on-microsoft-azure-ml)
# MAGIC - [mlflow.azureml API](https://www.mlflow.org/docs/latest/python_api/mlflow.azureml.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Step
# MAGIC 
# MAGIC [Delete Resources]($./07-Delete-Resources)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>