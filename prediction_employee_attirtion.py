#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#import data
df = pd.read_csv("D:/R files/alarm/alarm/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.shape
df.info()

#check for null values
df.isnull().values.any()

#view stattistics
df.describe()

#get the count of number that stayed and left the company
df['Attrition'].value_counts()

#visualize the no.of employee who stayed and left the company
sns.countplot(df["Attrition"]);
plt.show()

#age of employee who stayed and left the company
sns.countplot(x="Age",hue="Attrition" , data=df , palette="bright")
plt.show()

#remove useless columns
df = df.drop('Over18',axis=1)
df = df.drop('EmployeeNumber',axis=1)
df = df.drop('StandardHours',axis=1)
df=df.drop("EmployeeCount" , axis=1)

#get the corraletion
df.corr()

#visualize the corraletion
sns.heatmap(df.corr() , annot=True,fmt='.0%')
plt.show()

#transform data from non numerical to numerical data
from sklearn.preprocessing import LabelEncoder
for column in df.columns:
   if df[column].dtype==np.number:
      continue
   df[column] = LabelEncoder().fit_transform(df[column])
   
df.head()

#get new column
df["age_yr"] = df["Age"]

#drop old column
df=df.drop("Age" ,axis=1)

#split the data
x = df.iloc[:,1:df.shape[1]].values
y = df.iloc[:,0].values

#split the data into train and test set
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y ,test_size=0.2 , random_state = 10)

#use the random forest clssificer
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=20 , criterion = "entropy" , random_state = 100)
forest.fit(x_train , y_train)

#get the accuracy of model 
forest.score(x_train , y_train)

#create the confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test , forest.predict(x_test))

true_nag = cm[0][0]
true_pos = cm[1][1]
false_nag = cm[1][0]
false_pos = cm[0][1]

print(cm)
print("model accuracy :{}".format((true_nag+true_pos) / (true_pos+true_nag+false_nag+false_pos)))

#test the model 
test =[[2,312,1,1,0,3,0,1,10,2,0,2,1,1,399,846,9,0,1,0,3,1,6,3,2,2,2,2,2,9]]
test = pd.DataFrame(test)
print(test)

predict = forest.predict(test)
print("prediction :" , predict)











