# NYPD-Criminal-Offense-Classifier

# Quick Facts
 
1. This is a personal side project, which aims to build a classifier to identify and predict the types of criminal offenses in NYC, given the information of occurence time, sectors, borough, jurisdiction, etc. 

2. The classifier is written in Python and it implemented the Maximum Entropy model from scratch. The training method is SGD.
(for another Java implementation using conditional entropy and IIS, please see: https://github.com/lkmcl37/Implementation-of-Machine-Learning-Algorithms/tree/master/maxent)

3. Both training and test data are fetched from New York City's Socrata data portal, each containing crime records between year 2013 and 2015: https://data.cityofnewyork.us/Public-Safety/NYPD-7-Major-Felony-Incidents/hyij-8hr7/data

4. The best performance is around 47% in accuracy while trained on 10,000 crime records and test on 1,000 records.

5. Possible improvements can be made by feature engineering.

# How to use

To run the model and test, please execute run_model.py.

