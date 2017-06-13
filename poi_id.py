#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import numpy 
import matplotlib.pyplot as plt

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','exercised_stock_options','from_poi_to_this_person','assets_total','expenses'] # You will need to use more features
## features_list_1 = ['poi','salary','exercised_stock_options','from_poi_to_this_person']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

## Remove outliers for salary 
outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue 
    
    outliers.append((key,int(val)))
    
outliers_final = (sorted(outliers,key=lambda x:x[1],reverse=True)[:10])
## print outliers_final

## Check outliers for bonus
outliers = []
for key in data_dict:
    val = data_dict[key]['bonus']
    if val == 'NaN':
        continue 
    
    outliers.append((key,int(val)))
    
outliers_final = (sorted(outliers,key=lambda x:x[1],reverse=True)[:10])
## print outliers_final

## "TOTAL" is the biggest outlier in salary and bonus
## Remove "Total" feature in all instances
data_dict.pop("TOTAL",0)

### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict

## find length of dataset
print "Data set includes:" , len(my_dataset) , "items"

## Number of POI's and non poi's
numb_poi = 0
non_poi = 0
for item in my_dataset:
    if my_dataset[item]["poi"] != "NaN":
        if my_dataset[item]['poi'] == 1:
            numb_poi += 1
        else:
            non_poi += 1 
            
print "Data has", numb_poi, "persons of interest"
print "Data has", non_poi, "persons of non-interest"
print "\n" 
    
for item in my_dataset:
    ## Adding new feature of fraction of emails to this person
    current_person = my_dataset[item]
    
    if current_person['from_poi_to_this_person'] != "NaN" and current_person['to_messages'] != 'NaN' :
        fraction_to = float(my_dataset[item]['from_poi_to_this_person']) / float(my_dataset[item]['to_messages']) 
    else:
        fraction_to = 'NaN'
    ## Adding new feature of fraction of emails from this person
    if current_person['from_this_person_to_poi'] != "NaN" and current_person['from_messages'] != 'NaN' :
        fraction_from = float(my_dataset[item]['from_this_person_to_poi']) / float(my_dataset[item]['from_messages'])
    else:
        fraction_from = 'NaN'
    
    ## Ading new feature "Sum of all assets"
    asset_features = ['total_stock_value','deferred_income','long_term_incentive','salary','bonus']
    assets_total = 0
    for asset in asset_features:
        ## Get the best rough total worth of all their assets. Note: Some of the asset_features will not be encountered 
        ## for people that have "NaN" for that feature
        if current_person[asset] != "NaN":
            assets_total += current_person[asset] 
    
    

    my_dataset[item]['fraction_from'] = fraction_from
    my_dataset[item]['fraction_to'] = fraction_to
    my_dataset[item]['assets_total'] = assets_total
    
    if assets_total == 0:
        my_dataset[item][assets_total] = 'NaN'



### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

## Uncomment to run 

## from sklearn import preprocessing
## scaler = preprocessing.MinMaxScaler()
## features = scaler.fit_transform(features) 

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation 
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from time import time 

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from tester import test_classifier 
from sklearn.ensemble import AdaBoostClassifier 
from sklearn import grid_search 

## Do grid search to find best parameters 

parameters = {'C':[1,100]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr,parameters) 
clf.fit(features_train,labels_train) 
print "Best SVC params:", clf.best_params_ 

parameters_ada = {'n_estimators':(10,150,200),'learning_rate':(0.1,0.6,2.0)} 
ada = AdaBoostClassifier()
clf_ada = grid_search.GridSearchCV(ada,parameters_ada) 
clf_ada.fit(features_train,labels_train)
print "Best adaboost params:", clf_ada.best_params_ 
print "\n" 

def clf_sorter(clf,name): 
    print name, ":"
    print "\n"
    t0 = time()
    clf_1 = clf.fit(features_train,labels_train)
    print "training time:", round(time()-t0, 3), "s" 
    pred = clf_1.predict(features_test) 
    print " Accuracy score: ", accuracy_score(pred,labels_test)  
    precision_score_1 = precision_score(labels_test,pred)
    print " Precision score: ", precision_score_1
    recall_score_1 = recall_score(labels_test,pred)
    print " Recall score", recall_score_1 
    print "\n"

## SVM
clf_svm = svm.SVC(C=1,gamma='auto')
clf_sorter(clf_svm,"SVM")

## Decision tree classifier
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf_sorter(clf,"Decision Tree")

## Adaboost 
clf_ad = AdaBoostClassifier(n_estimators = 150, learning_rate=0.1,algorithm="SAMME.R")
clf_sorter(clf_ad,"Adaboost")


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

## Use decision tree classifier for final analysis 

dump_classifier_and_data(clf, my_dataset, features_list)