import csv
import numpy as np
from sklearn import datasets, preprocessing,linear_model, grid_search, cross_validation

from sklearn.linear_model import Ridge
from sklearn import metrics
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import uniform as sp_rand
from sklearn.cross_validation import cross_val_predict

FIELDS = ["fixed acidity", "volatile acidity","citric acid","residual sugar",
          "chlorides","free sulfur dioxide","total sulfur dioxide","density",
          "pH","sulphates","alcohol","quality"]

RED = 1599
WHITE = 4898
TEST_UPPER_BOUND = 1279
TEST_LOWER_BOUND = 320

f = open("winequality-red.csv")
f.readline()
data = np.loadtxt(f, delimiter=";")

#SET UP DATA

#NOT HOLDING OUT DATA
#X = data[:,:11]
#y = data[:,11:]

#normalizeX = preprocessing.normalize(X)
#y.reshape(-1,1)
##print normalizeX

#HOLD OUT SOME DATA
X = data[:TEST_UPPER_BOUND, :11] #3919 for white, 1279 for red
y = data[:TEST_UPPER_BOUND, 11:] 
testX = data[TEST_UPPER_BOUND:, :11]
test_y = data[TEST_UPPER_BOUND:, 11:]



normalizeX = preprocessing.normalize(X)
normalize_test = preprocessing.normalize(testX)
y.reshape(-1,1)
test_y.reshape(-1)


av_abs_error = 0
av_sq_error = 0
for value in test_y:
     abs_error = abs(5.66379984363 - value)
     sq_error = ((5.66379984363 - value)**2)
     ##print prediction, test_y[count], abs_error, sq_error
     av_abs_error += abs_error
     av_sq_error += sq_error
     #count += 1

     
c = av_abs_error/TEST_LOWER_BOUND
d = av_sq_error/TEST_LOWER_BOUND

print "White", 5.66379984363, c, d