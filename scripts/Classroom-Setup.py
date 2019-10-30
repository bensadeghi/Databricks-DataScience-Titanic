# Databricks notebook source
import re
# Get user name
userName = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
userName = re.sub(r"[^a-zA-Z0-9]", "_", userName)
# Create database name
databaseName = userName + "_db"
# Create database
spark.sql("CREATE DATABASE IF NOT EXISTS `{}`".format(databaseName))
# Use database
spark.sql("USE `{}`".format(databaseName))
print("Using database :::", databaseName)