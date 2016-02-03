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
TEST_UPPER_BOUND = 3919
TEST_LOWER_BOUND = 979

f = open("winequality-white.csv")
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

a = 0.0
b = 0.0
c = 0.0
d = 0.0

so_done = [np.array([0.000001]), np.array([0.00001]), np.array([0.0001]), np.array([0.001]), np.array([0.01]), np.array([0.1]), np.array([1]), np.array([10]), np.array([100]), np.array([1000])]
alphas = np.array([0.001])

#alphas = np.array([0.005])
#alphas = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

model = Ridge(solver = 'sag')
with open('randomized_search_cvWHITETtry.csv', 'wb') as testfile:
     #for j in range(10):
          #alphas = so_done[j]
          #for i in range(500):
     #print i
               
     clf = grid_search.RandomizedSearchCV(estimator = model,param_distributions=dict(alpha=alphas), 
                                          n_iter=1, fit_params=None, n_jobs=1, iid=True, 
                                          refit=True, cv=10,  verbose=0, pre_dispatch='2*n_jobs', 
                                          random_state=None, error_score='raise')
               
     clf.fit(normalizeX, y)
               #print "All scores for all alphas"
               #print clf.grid_scores_
               #print 
               #print "Best Model"
               #print clf.best_estimator_
               #print 
               #print "Best score"
               #print clf.best_score_
               #print 
               #print "Best alpha"
               #print clf.best_params_.get('alpha')
               #print 
               ##print metrics.classification_report()
               
               
     best_model = Ridge(alpha=0.001, copy_X=True, fit_intercept=True, max_iter=None, 
                        normalize=False, random_state=None, solver='sag', tol=0.001)
               
     #print best_model.coef_()
               #for row in textX:
     predictions = clf.predict(normalize_test)
     count = 0
     av_abs_error = 0
     av_sq_error = 0
     for prediction in clf.predict(normalize_test):
          abs_error = abs(prediction - test_y[count])
          sq_error = ((prediction - test_y[count])**2)
          ##print prediction, test_y[count], abs_error, sq_error
          av_abs_error += abs_error
          av_sq_error += sq_error
          count += 1
               
          av_as_error = av_abs_error/float(TEST_LOWER_BOUND)
          av_sq_error = av_sq_error/float(TEST_LOWER_BOUND)
          #print 'average abs error', av_abs_error
          #print 'average sq error', av_sq_error
          a += clf.best_params_.get('alpha')
          b += clf.best_score_
          c += av_sq_error
          d += av_abs_error
          
     a = a/TEST_LOWER_BOUND
     b = b/TEST_WHITE
     c = av_abs_error/WHITE
     d = av_sq_error/WHITE
          
          
     csv_writer = csv.writer(testfile)
          #row = [clf.best_params_.get('alpha'), clf.best_score_, av_sq_error,av_abs_error]
     row = [a, b, c ,d]
     csv_writer.writerow(row)    