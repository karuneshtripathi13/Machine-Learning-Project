

"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset_unix.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list,sort_keys = '../tools/python2_lesson14_keys_unix.pkl')
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.3,random_state=42)

from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf.fit(features_train,labels_train)
from sklearn.metrics import accuracy_score,precision_score,recall_score
pred=clf.predict(features_test)
print('acc=',accuracy_score(labels_test,pred))
print('prec=',precision_score(labels_test,pred))
print('recl=',recall_score(labels_test,pred))
c=0
for i in pred:
    if(i==1.):
        c+=1
print(c)
print(len(pred))
c=0
for i in range (0,len(pred)):
    if(pred[i]==labels_test[i]) and pred[i]==1.:
        c+=1
print(c)