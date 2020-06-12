# Databricks notebook source
# MAGIC %md
# MAGIC # What is Machine Learning?
# MAGIC 
# MAGIC Machine learning discovers patterns within data without being explicitly programmed.  This section introduces machine learning, explores the main topics in the field, and we'll continue to build an end-to-end pipeline in Spark.
# MAGIC 
# MAGIC #### Agenda:
# MAGIC * Define machine learning
# MAGIC * Differentiate supervised and unsupervised tasks
# MAGIC * Identify regression and classification tasks
# MAGIC * Feature Engineering
# MAGIC * Spark Machine Learning Library
# MAGIC * Machine Learning Lifecycle Management - MLflow

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Learning from Data
# MAGIC 
# MAGIC [Machine learning](https://en.wikipedia.org/wiki/Machine_learning) refers to a diverse set of tools for understanding data.  More technically, **machine learning is the process of _learning from data_ without being _explicitly programmed_**.  Let's unpack what that means.
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/bensadeghi/Databricks-DataScience-Titanic/master/img/programming_vs_ML.jpeg" style="height: 250px; margin: 20px"/>
# MAGIC <br>Above image courtesy of François Chollet
# MAGIC   
# MAGIC Take the Titanic dataset for example.  The dataset consists of the passanger details, such as sex, age, and cabin class, along with their survival status.  Here, the survival value is the _output variable_, also known as the _label_.  The other variables are known as _input variables_ or _features_.
# MAGIC 
# MAGIC **Machine learning is the set of approaches for estimating this function `f()` that maps features to an output.**  The inputs to this function can range from stock prices and customer information to images and DNA sequences.  Many of the same statistical techniques apply regardless of the domain.  This makes machine learning a generalizable skill set that drives decision-making in modern businesses.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Supervised vs Unsupervised Learning
# MAGIC 
# MAGIC Machine learning problems are roughly categorized into two main types:<br><br>
# MAGIC 
# MAGIC * **Supervised learning** looks to predict the value of some outcome based on one or more input measures
# MAGIC   - Our example of the Titanic Dataset is an example of supervised learning
# MAGIC   - In this case, the output is the survival status and the input is passanger features, such as age and cabin class
# MAGIC * **Unsupervised learning** describes associations and patterns in data without a known outcome
# MAGIC   - An example of this would be clustering customer data to find the naturally occurring customer segments
# MAGIC   - In this case, no known output is used as an input.  Instead, the goal is to discover how the data are organized into natural segments or clusters
# MAGIC 
# MAGIC Here will cover supervised learning, which is the vast majority of machine learning use cases in industry.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/eLearning/ML-Part-1/regression.png" style="height: 300px; margin: 20px"/><img src="https://files.training.databricks.com/images/eLearning/ML-Part-1/clustering.png" style="height: 300px; margin: 20px"/>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Regression vs Classification
# MAGIC 
# MAGIC Variables can either be quantitative or qualitative:<br><br>
# MAGIC 
# MAGIC * **Quantitative** values are numeric and generally unbounded, taking any positive or negative value
# MAGIC * **Qualitative** values take on a set number of classes or categories
# MAGIC 
# MAGIC | Variable type    | Also known as         | Examples                                                          |
# MAGIC |:-----------------|:----------------------|:------------------------------------------------------------------|
# MAGIC | quantitative     | continuous, numerical | age, salary, temperature                                          |
# MAGIC | qualitative      | categorical, discrete | gender, whether or a not a patient has cancer, state of residence |
# MAGIC 
# MAGIC Machine learning models operate on numbers so a qualitative variable like gender, for instance, would need to be encoded as `0` for male or `1` for female.  In this case, female isn't "one more" than male, so this variable is handled differently compared to a quantitative variable.
# MAGIC 
# MAGIC Generally speaking, **a supervised model learning a quantitative variable is called regression and a model learning a qualitative variable is called classification.**
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-1/classification_v_regression.jpg" style="height: 400px; margin: 20px"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering
# MAGIC 
# MAGIC **What is a feature?**
# MAGIC 
# MAGIC A feature is an attribute or property shared by all of the independent units on which analysis or prediction is to be done. Any attribute could be a feature, as long as it is useful to the model.  
# MAGIC The purpose of a feature, other than being an attribute, would be much easier to understand in the context of a problem. A feature is a characteristic that might help when solving the problem.
# MAGIC 
# MAGIC **What is feature engineering?**  
# MAGIC 
# MAGIC Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work. Feature engineering is fundamental to the application of machine learning, and is both difficult and expensive. The need for manual feature engineering can be obviated by automated feature learning.
# MAGIC 
# MAGIC The features in your data are important to the predictive models you use and will influence the results you are going to achieve. The quality and quantity of the features will have great influence on whether the model is good or not.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Commonly used terms:
# MAGIC 
# MAGIC ** Feature: **<BR>
# MAGIC &emsp;An attribute useful for your modeling task
# MAGIC   
# MAGIC ** Feature Importance: **<BR>
# MAGIC &emsp;An estimate of the usefulness of a feature
# MAGIC 
# MAGIC ** Feature Extraction: **<BR>
# MAGIC &emsp;The automatic construction of new features from raw data
# MAGIC 
# MAGIC ** Feature Selection: **<BR>
# MAGIC &emsp;From many features to a few that are useful
# MAGIC   
# MAGIC ** Feature Construction: **<BR>
# MAGIC &emsp;The manual construction of new features from raw data
# MAGIC   
# MAGIC ** Feature Learning: **<BR>
# MAGIC &emsp;The automatic identification and use of features in raw data

# COMMAND ----------

# MAGIC %md 
# MAGIC ## ![Spark Logo Tiny](https://kpistoropen.blob.core.windows.net/collateral/roadshow/logo_spark_tiny.png) What is Spark MLlib?
# MAGIC 
# MAGIC **MLlib is Spark’s machine learning (ML) library. Its goal is to make practical machine learning scalable and easy.**
# MAGIC 
# MAGIC **At a high level, it provides tools such as:**
# MAGIC * ML Algorithms: common learning algorithms such as classification, regression, clustering, and collaborative filtering
# MAGIC * Featurization: feature extraction, transformation, dimensionality reduction, and selection
# MAGIC * Pipelines: tools for constructing, evaluating, and tuning ML Pipelines
# MAGIC * Persistence: saving and load algorithms, models, and Pipelines
# MAGIC * Utilities: linear algebra, statistics, data handling, etc.
# MAGIC 
# MAGIC See [MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tracking Experiments with [MLflow](https://mlflow.org/)
# MAGIC 
# MAGIC MLflow Tracking is...<br><br>
# MAGIC 
# MAGIC * a logging API specific for machine learning 
# MAGIC * agnostic to libraries and environments that do the training
# MAGIC * organized around the concept of **runs**, which are executions of data science code
# MAGIC * runs are aggregated into **experiments** where many runs can be a part of a given experiment 
# MAGIC * An MLflow server can host many experiments
# MAGIC 
# MAGIC Each run can record the following information:<br><br>
# MAGIC 
# MAGIC - **Parameters:** Key-value pairs of input parameters such as the number of trees in a random forest model
# MAGIC - **Metrics:** Evaluation metrics such as RMSE or Area Under the ROC Curve
# MAGIC - **Artifacts:** Arbitrary output files in any format.  This can include images, pickled models, and data files
# MAGIC - **Source:** The code that originally ran the experiment
# MAGIC 
# MAGIC Experiments can be tracked using libraries in Python, R, and Java as well as by using the CLI and REST calls.
# MAGIC 
# MAGIC See [MLflow Guide](https://docs.microsoft.com/en-us/azure/databricks/applications/mlflow/)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/mlflow-tracking.png" style="height: 250px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Step
# MAGIC 
# MAGIC [ML Workflows]($./04-ML-Workflows)

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>