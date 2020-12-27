import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
## You will need to use more features
features_list = ['poi','salary','from_poi_to_this_person','from_this_person_to_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset_unix.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
### Task 2: Remove outliers
del data_dict['TOTAL']
'''
lt=[]
for i in data_dict:
    if data_dict[i]['from_poi_to_this_person']=='NaN' or data_dict[i]['from_this_person_to_poi']=='NaN':
        lt.append(i)
for i in lt:
    del data_dict[i]
'''
'''
lt=[]
for i in data_dict:
    if data_dict[i]['salary']=='0' or data_dict[i]['salary']=='NaN':
        lt.append(i)
for i in lt:
    del data_dict[i]
'''
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)

from sklearn.metrics import accuracy_score,precision_score,recall_score
print('Bayes')
print('accu=',accuracy_score(pred,labels_test))
print('prec=',precision_score(pred,labels_test,average='weighted'))
print('recall=',recall_score(pred,labels_test,average='weighted'))

from sklearn.svm import SVC
clf=SVC()
'''
from sklearn.model_selection import GridSearchCV
parameter={'C':[1,10],'kernel':('linear','rbf')}
clf=GridSearchCV(vc,parameter)
'''
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)

from sklearn.metrics import accuracy_score,precision_score,recall_score
print('\nSVM')
print('accu=',accuracy_score(pred,labels_test))
print('prec=',precision_score(pred,labels_test,average='weighted'))
try:
    print('recall=',recall_score(pred,labels_test,average='weighted',zero_division=0))
except ZeroDivisionError:
    print('Error')
'''
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)

from sklearn.metrics import accuracy_score,precision_score,recall_score
print('\nDecision Tree')
print('accu=',accuracy_score(pred,labels_test))
print('prec=',precision_score(pred,labels_test,average='weighted'))
print('recall=',recall_score(pred,labels_test,average='weighted'))

from sklearn.linear_model import LinearRegression
import numpy as np
clf=LinearRegression()
clf.fit(features_train,labels_train)

from sklearn.metrics import accuracy_score,precision_score,recall_score
print('\nLinear Regression')
print('accu=',accuracy_score(pred,labels_test))
print('prec=',precision_score(pred,labels_test,average='weighted'))
print('recall=',recall_score(pred,labels_test,average='weighted'))

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
print('\nRandom forest')
print('accu=',accuracy_score(labels_test,pred))
print('prec=',precision_score(pred,labels_test,average='weighted'))
print('recall=',recall_score(pred,labels_test,average='weighted'))

from sklearn.ensemble import AdaBoostClassifier
clf=AdaBoostClassifier()
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
print('\nAdaboost forest')
print('accu=',accuracy_score(labels_test,pred))
print('prec=',precision_score(pred,labels_test,average='weighted'))
print('recall=',recall_score(pred,labels_test,average='weighted'))


print('\nUNSUPERWISED\n')
from sklearn.cluster import KMeans
clf=KMeans(n_clusters=2,random_state=0)
clf.fit(features_train)
pred=clf.predict(features_test)
print('K-Means')
print('accu=',accuracy_score(labels_test,pred))
print('prec=',precision_score(pred,labels_test,average='weighted'))
print('recall=',recall_score(pred,labels_test,average='weighted'))
'''

import matplotlib.pyplot as plt
for feature, target in zip(features_train, labels_train):
    plt.scatter( feature[0], target, color='b' )
for feature, target in zip(features_test, labels_test):
    plt.scatter( feature[0], target, color='r' )  

### labels for the legend
plt.scatter(features_test[0][0],labels_test[0], color='r', label="test")
plt.scatter(features_test[0][0], labels_test[0], color='b', label="train")
plt.show()

for feature, target in zip(features_train, labels_train):
        plt.scatter( feature[0], feature[2], color='b' )

for feature, target in zip(features_test, labels_test):
        plt.scatter( feature[0], feature[2], color='r' )


### labels for the legend
plt.scatter(features_test[0][1],labels_test[0], color='r', label="test")
plt.scatter(features_test[0][1], labels_test[0], color='b', label="train")
plt.show()
'''
'from_poi_to_this_person','from_this_person_to_poi'
import matplotlib.pyplot
for i in data_dict:
    if(data_dict[i]['salary']!='NaN' and data_dict[i]['from_this_person_to_poi']!='NaN'):
        matplotlib.pyplot.scatter( data_dict[i]['salary'], data_dict[i]['from_this_person_to_poi'] )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("from_poi_to this_person")
matplotlib.pyplot.show()
'''
'''
x_min = 0.0; x_max = 1.0
    y_min = 0.0; y_max = 1.0

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)

    # Plot also the test points
    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]

    plt.scatter(grade_sig, bumpy_sig, color = "b", label="fast")
    plt.scatter(grade_bkg, bumpy_bkg, color = "r", label="slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")

    plt.savefig("test.png")
'''
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)