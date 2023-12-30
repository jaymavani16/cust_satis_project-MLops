Problem statement: For a given customer's historical data, we are tasked to predict the review score for the next order or purchase. We will be using the Brazilian E-Commerce Public Dataset by Olist. This dataset has information on 100,000 orders from 2016 to 2018 made at multiple marketplaces in Brazil. Its features allow viewing charges from various dimensions: from order status, price, payment, freight performance to customer location, product attributes and finally, reviews written by customers. The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. In order to achieve this in a real-world scenario, we will be using ZenML to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.

The goal of this project was to predict the customer satisfaction score for the next order or purchase based on historical data. This is a useful task for e-commerce businesses that want to improve their customer retention and loyalty. To achieve this, I used features like order status, price, payment, etc. to train a regression model.

One of the challenges of this project was to make the machine learning pipeline production-ready, meaning that it can handle data ingestion, preprocessing, feature engineering, model training, evaluation, and deployment in a scalable and reproducible way. This is where ZenML came in handy, as it allowed me to build and deploy my pipeline easily and efficiently.

ZenML also integrated well with other tools that I used for this project, such as MLflow and Streamlit. MLflow is a platform for managing the end-to-end machine learning lifecycle, and I used it to track the metrics and parameters of my pipeline, as well as to deploy the model as a REST API. Streamlit is a framework for creating interactive web applications, and I used it to create a dashboard that shows how the model works and how it can be used to predict customer satisfaction scores.

pip install zenml["server"]
zenml up
If you are running the run_deployment.py script, you will also need to install some integrations using ZenML:

zenml integration install mlflow -y
The project can only be executed with a ZenML stack that has an MLflow experiment tracker and model deployer as a component. Configuring a new stack with the two components are as follows:

zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set










