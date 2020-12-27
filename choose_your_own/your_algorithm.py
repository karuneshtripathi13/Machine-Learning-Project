
import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=8)
from time import time
t0=time()
clf.fit(features_train,labels_train)
print("training time=",round(time()-t0,3),"s")
"""
from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_classification
#features_train,labels_train=make_classification(n_samples=10, n_features=2,n_informative=2, n_redundant=0,random_state=1, shuffle=False)
clf=RandomForestClassifier(max_depth=2, random_state=1)
from time import time
t0=time()
clf.fit(features_train,labels_train)
print("training time=",round(time()-t0,3),"s")


from sklearn.ensemble import AdaBoostClassifier
clf=AdaBoostClassifier(n_estimators=100, random_state=0)
from time import time
t0=time()
clf.fit(features_train,labels_train)
print("training time=",round(time()-t0,3),"s")
"""

t0=time()
pred=clf.predict(features_test)
print("prediction time=",round(time()-t0,3),"s")
from sklearn.metrics import accuracy_score
print("accuracy=",accuracy_score(pred,labels_test)*100,"%")

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
plt.show()
