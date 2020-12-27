
import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset_unix.pkl", "rb") )
data_dict.pop('TOTAL')
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

maxs=0
maxb=0
name=""
for i in data_dict:
    if(data_dict[i]['salary']!='NaN'):
        if(data_dict[i]['salary']>1000000 and data_dict[i]['bonus']>5000000):
            maxs=data_dict[i]['salary']
            maxb=data_dict[i]['bonus']
            name=i
            print(i)
print(name,' ',maxs,' ',maxb)