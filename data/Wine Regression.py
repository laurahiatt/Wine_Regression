import csv
import numpy as np
from sklearn import datasets, preprocessing,linear_model, grid_search

from sklearn.linear_model import Ridge
from sklearn import metrics
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import uniform as sp_rand

FIELDS = ["fixed acidity", "volatile acidity","citric acid","residual sugar",
          "chlorides","free sulfur dioxide","total sulfur dioxide","density",
          "pH","sulphates","alcohol","quality"]

f = open("winequality-white.csv")
f.readline()
data = np.loadtxt(f, delimiter=";")

X = data[:,:11]
y = data[:,11:]

#print X
#print y

normalizeX = preprocessing.normalize(X)
y.reshape(-1,1)
#print normalizeX

#standardizedX = preprocessing.scale(X)
#print standardizedX

alphas = np.array([0.005, 0.001, 0.015, 0.1, 4])
param_distribution = {0.001}
#param_grid = {'C': scipy.stats.expon(scale = 100), 'gamma': scipy.stats.expon(scale=.1), 'kernel': ['rbf'], 'class_weight':['auto', None]}

print 'kfdj'

model = Ridge()

clf = grid_search.RandomizedSearchCV(estimator = model,param_distributions=dict(alpha=alphas), n_iter=5, scoring=None, fit_params=None, n_jobs=1, iid=True, refit=True, cv=10, verbose=0, pre_dispatch='2*n_jobs', random_state=None, error_score='raise')

clf.fit(normalizeX, y)
print 'dkfdjfk'
print clf.grid_scores_
print clf.best_estimator_
print clf.best_score_
print clf.best_params_ 
print model.coef_

#print metrics.classification_report(

predict= [6,0.29,0.21,1.3,0.055,42,168,0.9914,3.32,0.43,11.1]
normalize_predict = preprocessing.normalize(predict)

print clf.predict(normalize_predict)

#params = clf.get_params

#clf = linear_model.RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True, 
                           #normalize=False, scoring=None, cv=None, gcv_mode=None, 
                           #store_cv_values=False)


#print clf.fit(X,y)



## prepare a uniform distribution to sample for the alpha parameter
#param_grid = {{'C': scipy.stats.expon(scale = 100), 'gamma': scipy.stats.expon(scale=.1), 'kernel': ['rbf'], 'class_weight':['auto', None]}}
## create and fit a ridge regression model, testing random alpha values
#model = Ridge()
#rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100)
#rsearch.fit(X, y)
#print "dkfdkj"
#print(rsearch)
## summarize the results of the random parameter search
#print(rsearch.best_score_)
#print(rsearch.best_estimator_.alpha)








#with open("winequality-red.csv") as csvfile:
    
 #   reader = csv.DictReader(csvfile, fieldnames=FIELDS)
    
  #  for i in range(1280):
        
    #1599 Red and 4898 White
 #   clf.fit
    
  #  pass
    #wines = pandas.read_csv(winequality-red.csv)