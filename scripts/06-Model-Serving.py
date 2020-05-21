# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Model Serving Options
# MAGIC 
# MAGIC When we move to talking about actually operationalizing the machine learning models we've build so far is where the discussion becomes tricky. Many organizations have not yet reached this step, as it can become quite complex to get here, and this part is not any easier either.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC When we talk about open source deployment of ML models, the first to stand out is [Docker](https://opensource.com/resources/what-docker). Docker is a tool designed to make it easier to create, deploy, and run applications by using containers. Containers allow a developer to package up an application with all of the parts it needs, such as libraries and other dependencies, and ship it all out as one package. By doing so, thanks to the container, the developer can rest assured that the application will run on any other Linux machine regardless of any customized settings that machine might have that could differ from the machine used for writing and testing the code.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC As the extent of the scope of Docker containers grew, so did the complexity of their deployment. As such, Kubernetes quickly came into the light of many organizations. [Kubernetes](https://kubernetes.io/docs/concepts/overview/what-is-kubernetes/) is a portable, extensible open-source platform for managing containerized workloads and services, that facilitates both declarative configuration and automation. It has a large, rapidly growing ecosystem. Kubernetes services, support, and tools are widely available.

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Using these 2 solutions, some organizations do realise their deployments, however, with a lengthy development cycle, and a rigourous maintainence team behind it. However, to save some of the headaches, make the process smoother for more users (rather than just expert engineering teams), and in some cases save on TCO too, you also have alternatives in Azure.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC By using Databricks to create your models (or, alternatively, some of the other solutions mentioned in the previous section of our workshop), you can then choose your serving layer. Whether that's **batch** (where you scroll data on a regular interval), **streaming** (scoring data non-stop), or via a **web service** (where you make "random" calls to be scored), you can achieve the first 2 options using Databricks directly (or, for more complex pipelines, using scheduling via Azure Data Factory), while the latter can easily be covered by integrating Databricks with [Azure Machine Learning Service](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-use-mlflow), for an easy way to deploy to an auto-scalable, containerized API.
# MAGIC 
# MAGIC See Azure Reference Architecture below:
# MAGIC ![x](https://github.com/bensadeghi/Databricks-DataScience-Titanic/raw/master/img/azure_reference_architecture.png)

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

# MAGIC %md You can also load a specific model stage. The following cell loads the `Production` stage of the model.
# MAGIC 
# MAGIC The following cell uses the `mlflow.pyfunc.load_model()` API to load the latest version of production stage model as a generic Python function.

# COMMAND ----------

import mlflow.pyfunc

modelURI = "models:/{model_name}/production".format(model_name=modelName)

print("Loading registered model version from URI: '{model_uri}'".format(model_uri=modelURI))
modelPipeline = mlflow.pyfunc.load_model(modelURI).spark_model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Scoring
# MAGIC The most common way of using Data Science models today is by using a batch scoring system. For example, let's say you have implemented a churn system, and every morning you'd like one of your colleagues to go through a list of the top 5 most likely to churn customers, and ring them up. In the backend, that usually means that, the evening before, after the day's data has been collected, a ML model runs on the new information to produce a list of 5 likely churners.
# MAGIC 
# MAGIC In our case, we'd like to run our model once against our test dataset and prediction survival status.

# COMMAND ----------

# Load data

titanicDF = spark.read.table("titanic_clean")

trainDF, testDF = titanicDF.randomSplit([0.8, 0.2], seed=1)

# COMMAND ----------

# Make predictions in batch

predictions = modelPipeline.transform(testDF)

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
# MAGIC 
# MAGIC **Azure Machine Learning** provides the following [MLOps capabilities](https://docs.microsoft.com/en-us/azure/machine-learning/concept-model-management-and-deployment):<br><br>
# MAGIC - Deploy ML projects from anywhere
# MAGIC - Monitor ML applications for operational and ML related issues - compare model inputs between training and inference, explore model-specific metrics and provide monitoring and alerts on your ML infrastructure.
# MAGIC - Capture the data required for establishing an end to end audit trail of the ML lifecycle, including who is publishing models, why changes are being made, and when models were deployed or used in production.
# MAGIC - Automate the end to end ML lifecycle with Azure Machine Learning and Azure DevOps to frequently update models, test new models, and continuously roll out new ML models alongside your other applications and services.
# MAGIC 
# MAGIC ![x](https://github.com/bensadeghi/Databricks-DataScience-Titanic/raw/master/img/azure_devops_ml.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Resources
# MAGIC - [MLflow Guide](https://docs.microsoft.com/en-us/azure/databricks/applications/mlflow/)
# MAGIC - [Track models metrics with MLflow and Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow)
# MAGIC - [Notebook: Deploy Model to Azure Container Instance](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/azure-databricks/amlsdk/deploy-to-aci-04.ipynb)
# MAGIC - [mlflow.azureml API](https://www.mlflow.org/docs/latest/python_api/mlflow.azureml.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Step
# MAGIC 
# MAGIC [Delete Resources]($./07-Delete-Resources)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2019 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>