import csv
import numpy as np
import sklearn
from sklearn import datasets, preprocessing,linear_model, grid_search, cross_validation

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
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

X = data[:TEST_UPPER_BOUND, :11] #3919 for white, 1279 for red
y = data[:TEST_UPPER_BOUND, 11:] 
testX = data[TEST_UPPER_BOUND:, :11]
test_y = data[TEST_UPPER_BOUND:, 11:]

Z = np.transpose(X)
a = data[:TEST_UPPER_BOUND, :11]
a = np.transpose(a).tolist()
b = Z.tolist()
c = data[:TEST_UPPER_BOUND, :11]
c = np.transpose(c).tolist()
d = data[:TEST_UPPER_BOUND, :11]
d = np.transpose(d).tolist()


for row in range(len(b)):
    for item in range(len(b[row])):
        b[row][item] = b[row][item] * b[row][item]
        
for row in range(len(c)):
    for item in range(len(c[row])):
        c[row][item] = c[row][item] * c[row][item] * c[row][item]
        
for row in range(len(d)):
    for item in range(len(c[row])):
        d[row][item] = d[row][item] * d[row][item] * d[row][item] * d[row][item]       
        
abcd = a + b + c + d
print abcd
abcd = np.array(abcd)
abcd = np.transpose(abcd)

print 

normalize_abcd = preprocessing.normalize(abcd)

#TEST DATA

Z = np.transpose(testX)
a = data[TEST_UPPER_BOUND:, :11]
a = np.transpose(a).tolist()
b = Z.tolist()
c = data[TEST_UPPER_BOUND:, :11]
c = np.transpose(c).tolist()
d = data[TEST_UPPER_BOUND:, :11]
d = np.transpose(d).tolist()


for row in range(len(b)):
    for item in range(len(b[row])):
        b[row][item] = b[row][item] * b[row][item]
        
for row in range(len(c)):
    for item in range(len(c[row])):
        c[row][item] = c[row][item] * c[row][item] * c[row][item]
        
for row in range(len(d)):
    for item in range(len(c[row])):
        d[row][item] = d[row][item] * d[row][item] * d[row][item] * d[row][item]

test_abcd = a + b + c + d
test_abcd = np.array(test_abcd)
test_abcd = np.transpose(test_abcd)
normalize_test_abcd = preprocessing.normalize(test_abcd)
y.reshape(-1,1)
test_y.reshape(-1)



#normalizeX = preprocessing.normalize(X)
#normalize_test = preprocessing.normalize(testX)
#y.reshape(-1,1)
#test_y.reshape(-1)


raveled = y.ravel()
lasso = sklearn.linear_model.Lasso(alpha=0.0001, fit_intercept=True, normalize=False, 
                           precompute=False, copy_X=True, max_iter=1000, 
                           tol=0.0001, warm_start=False, positive=False, 
                           random_state=None, selection='cyclic')

lasso.fit(normalize_abcd, raveled)
print 'coef',lasso.coef_
print 'intercept', lasso.intercept_
print 'score', lasso.score(normalize_abcd, raveled)
#print 'alpha', lasso.alpha_
#print 'mse_path', lasso.mse_path_
#print 'n_iter_', lasso.n_iter_
#print 'alphas', lasso.alphas

with open('lasso_red.csv', 'wb') as testfile:
    
    alphas = [1,.1,.01,.001,.0001,.00001,.000001,.0000001,.00000001,.000000001,.0000000001, .00000000001, .000000000001, .0000000000001, .00000000000001, .000000000000001]
    #alphas = [.001]
    for r in alphas:
        
        lasso = sklearn.linear_model.Lasso(alpha=r, fit_intercept=True, normalize=False, 
                                           precompute=False, copy_X=True, max_iter=1000, 
                                           tol=0.0001, warm_start=False, positive=False, 
                                           random_state=None, selection='random')
        lasso.fit(normalize_abcd, raveled)
        print r
        print 'coef',lasso.coef_
        print 'intercept', lasso.intercept_
        print 'score', lasso.score(normalize_abcd, raveled)        
        print
        print
        print
        
        
        predictions = lasso.predict(test_abcd)
        count = 0
        av_abs_error = 0
        av_sq_error = 0
        for prediction in lasso.predict(test_abcd):
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
           
           
        abs_error = abs_error/TEST_LOWER_BOUND
        sq_error = sq_error/TEST_LOWER_BOUND
    
        csv_writer = csv.writer(testfile)
        row = [r, lasso.score(normalize_abcd, raveled), abs_error ,sq_error]
        csv_writer.writerow(lasso.coef_)
        csv_writer.writerow(row)    
