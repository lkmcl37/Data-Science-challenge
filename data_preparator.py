# -*- mode: Python; coding: utf-8 -*-
#Author: KaMan Leong

from csv import reader as csv_reader

class Document(object):
  
    def __init__(self, label, data):
        
        self.data = data
        self.label = label
        self.feature_vector = None
 
    def features(self):
        return self.data

class Generate_data():
 
    #load csv file
    def load(self, datafile):
        
        def unicode_csv_reader(csvfile):
            
            records = []
            for data in csv_reader(csvfile):
              
                #skip the first line of the csv
                if data[0] == "Identifier":
                    continue
                
                #label = offense_type
                
                #features used include:
                    #Day of Week,Occurrence Month,
                    #Occurrence Day,
                    #Occurrence Hour,CompStat Month,
                    #CompStat Day,
                    #Sector,Precinct,
                    #Borough,Jurisdiction,
                    #XCoordinate, YCoordinate, Location
                    
                label = data[10]
                feature = data[3:5] + data[6:9] + data[11:-1]
                
                records.append([label, feature])

            return records

        labels = set([])
        data = []
        with open(datafile, encoding = 'utf-8') as file:
            for record in unicode_csv_reader(file):
                labels.add(record[0])
                data.append(Document(record[0], record[1]))
        
        return data, labels

