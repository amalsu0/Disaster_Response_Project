# Disaster Response Pipeline Project

###Project Description

This project is a part of the Data Scientist Nanodegree Program and 
aims to develop a machine learning pipeline capable of classifying 
disaster messages in real-time during a disaster event. 
The model is trained to categorize messages into multi-labels 
to ensure their prompt delivery to the appropriate disaster response agency. 
Additionally, the project features a web application that enables disaster response workers 
to input messages and obtain classification results.

###File Descriptions

The project consists of three folders:

1. app: includes run.py to launch the app, and the templates folder containing HTML files.

2. data: contains the raw data files disaster_messages.csv and disaster_categories.csv, 
as well as process_data.py that stores the processed data in a SQLite database DisasterResponse.db. 

3. models: contains the machine learning pipeline train_classifier.py that train the model using 
the processed data and store it as a pickle file classifier.pkl. 


###Installation

To run the project, you should have Python 3.5 or higher installed on your machine.


###Instructions

 1. Run the following commands in the project's root directory to set up your database and model.
 • To run ETL pipeline that cleans data and stores in database 
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
 • To run ML pipeline that trains classifier and saves 
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
 2. Run the following command in the app's directory to run your web app. python run.py
 3. Go to http://0.0.0.0:3001/