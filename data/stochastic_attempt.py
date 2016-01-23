import numpy
import csv
from numpy import matrix
from numpy import linalg

NUM_WINES = 4898 #white wines
FIELDS = ["fixed acidity", "volatile acidity","citric acid","residual sugar",
          "chlorides","free sulfur dioxide","total sulfur dioxide","density",
          "pH","sulphates","alcohol","quality"]

with open('winequality-red.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';', fieldnames = FIELDS)
    reader.next()
    count = 0
    train = [None] * 2938
    validation = [None] * 20
    test = [None] * 20
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
            train[count] = entry
            train_answer[count] = float(row["quality"])
        elif count < 3917:
            validation[count] = entry
            validation_answer[count] = float(row["quality"])
        else:
            test[count] = entry
            test_answer[count] = float(row["quality"])
        count += 1
    train_set = np.matrix(train)
    validation_set = np.matrix(validation)
    test_set = np.matrix(test)
    train_answer_set = np.matrix(train_answer)
    validation_answer_set = np.matrix(validation_answer)
    test_answer_set = np.matrix(test_answer)