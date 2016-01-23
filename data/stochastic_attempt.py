import numpy as np
import random
import csv
from numpy import matrix
from numpy import linalg

NUM_WINES = 4898 #white wines
FIELDS = ["fixed acidity", "volatile acidity","citric acid","residual sugar",
          "chlorides","free sulfur dioxide","total sulfur dioxide","density",
          "pH","sulphates","alcohol","quality"]

with open('winequality-white.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';', fieldnames = FIELDS)
    reader.next()
    count = 0
    index = 0
    train = [None] * 2938
    validation = [None] * 980
    test = [None] * 980
    train_answer = [0.0] * 2938
    validation_answer = [0.0] * 980
    test_answer = [0.0] * 980
    for row in reader:
        entry = [float(row["fixed acidity"]),float(row["volatile acidity"]),
                float(row["citric acid"]),float(row["residual sugar"]),
                float(row["chlorides"]),float(row["free sulfur dioxide"]),
                float(row["total sulfur dioxide"]),float(row["density"]),
                float(row["pH"]),float(row["sulphates"]),float(row["alcohol"])]        
        
        if count < 2938:
            train[index] = entry          
            train_answer[index] = float(row["quality"])
            
        elif count >= 2938 and count < 3917:
            validation[index] = entry
            validation_answer[index] = float(row["quality"])
        else:
            test[index] = entry
            test_answer[index] = float(row["quality"])
        if (count == 2937) or (count == 3917):
            index = 0
        else:
            index += 1
        count += 1
    
    train_set = np.matrix(train)
    validation_set = np.matrix(validation)
    test_set = np.matrix(test)
    
    train_answer_set = np.matrix(train_answer)
    validation_answer_set = np.matrix(validation_answer)
    test_answer_set = np.matrix(test_answer)
    print train_set
    print train_answer_set
    t=  test_answer_set.transpose()
    print t.item(979)
    
    #attempt to put all in range
    train_min = train_set.min(axis = 0)
    train_max = train_set.max(axis = 0)
    train_set = (train_set - train_min)/(train_max - train_min)

    
   # validation_min = validation_set.min(axis = 0)
   # validation_max = validation_set.max(axis = 0)
   # validation_set = (validation_set - validation_min)/(validation_max - validation_min)    
    
    #validation_min = validation_set.min(axis = 0)
    #validation_max = validation_set.max(axis = 0)
    #validation_set = (validation_set - validation_min)/(validation_max - validation_min)    

    test_min = test_set.min(axis = 0)
    test_max = test_set.max(axis = 0)
    test_set = (test_set - test_min)/(test_max - test_min)  
    
    
    #Building the tests
    for i in range(1, 7): #degree of polynomial
            for j in range(60): #60 random initializations of theta
                thetas = [0.0] * (i + 1) #add one for intercept term
                for k in range(i + 1): #set values of theta
                    thetas[k] = random.random() * random.randrange(50)
                epsilon = 0.1
                alpha = 2
                for l in range(2938):
                    for m in range(i + 1):
                        temp = thetas[m] + 
                    
    