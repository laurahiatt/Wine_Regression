import csv
import numpy as np
from sklearn import datasets, preprocessing,linear_model, grid_search

from sklearn.linear_model import Ridge
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import uniform as sp_rand

FIELDS = ["fixed acidity", "volatile acidity","citric acid","residual sugar",
          "chlorides","free sulfur dioxide","total sulfur dioxide","density",
          "pH","sulphates","alcohol","quality"]

f = open("winequality-red.csv")
f.readline()
data = np.loadtxt(f, delimiter=";")

X = data[:,:11]
y = data[:,11:]

#print X
#print y

normalizeX = preprocessing.normalize(X)
#print normalizeX

#standardizedX = preprocessing.scale(X)
#print standardizedX

#clf = linear_model.RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True, 
                           #normalize=False, scoring=None, cv=None, gcv_mode=None, 
                           #store_cv_values=False)


#print clf.fit(X,y)



# prepare a uniform distribution to sample for the alpha parameter
param_grid = {'alpha': sp_rand()}
# create and fit a ridge regression model, testing random alpha values
model = Ridge()
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100)
rsearch.fit(X, y)
print "dkfdkj"
print(rsearch)
# summarize the results of the random parameter search
print(rsearch.best_score_)
print(rsearch.best_estimator_.alpha)








#with open("winequality-red.csv") as csvfile:
    
 #   reader = csv.DictReader(csvfile, fieldnames=FIELDS)
    
  #  for i in range(1280):
        
    #1599 Red and 4898 White
 #   clf.fit
    
  #  pass
    #wines = pandas.read_csv(winequality-red.csv)