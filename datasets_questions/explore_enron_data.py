
""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset_unix.pkl", "rb"))

c=0
for i in enron_data:
    if(enron_data[i]['total_payments']=='NaN' and enron_data[i]["poi"]==True):
        c+=1
print(c)
print(c*100/len(enron_data))
"""
import sys
sys.path.append("../final_project/")
from poi_email_addresses import poiEmails
print(poiEmails())
"""
"""
c1=0
file1=open("../final_project/emails_by_address/"+c,'r')
file2=open("../final_project/poi_names.txt",'r')
email=file1.readlines()
names=file2.readlines()
import re
s=[]
for i in email:
    x=re.split(r'[/-]',i)
    s.append(x[2])
for j in names:
    y=re.split(r'[ ,]',j)
    if(len(y)>1):
        if(y[1].lower() in s):
            c1+=s.count(y[1].lower())
print(c1)
"""