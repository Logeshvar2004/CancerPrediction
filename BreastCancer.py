import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

data= pd.read_csv(r'D:\Coding\python\BreastCancer Predicion\data.csv')
print('\t\t\t\t-------------Sample Data--------------')
print(data.head())


'''1) Data preprocessing'''

#1- check count(row,col)
print(data.shape)
#2- Null val
print(data.isnull().sum())
#3- drop col with missing val
data=data.dropna(axis=1) 
#4- check Datatypes:
print(data.dtypes)
ct=data['diagnosis'].value_counts()
print(ct)

'''Label Encoding (Diagnosis- M/N)'''
encoder=LabelEncoder()
data.iloc[:,1]=encoder.fit_transform(data.iloc[:,1].values)
print(data['diagnosis'].values)
sb.pairplot(data.iloc[:,1:7], hue='diagnosis')
plt.show()
data.iloc[:,1:11].corr()

'''Splitting data'''
X=data.iloc[:,2:31].values
Y=data.iloc[:,1].values
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=0)

''' Data scaling'''
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

print('Train\n',x_train)
print('Test\n',x_test)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=0)
model.fit(x_train,y_train)
print('Predicted value')
pred=model.predict(x_test)
print(pred)

#Accuracy Testing
print('Accuracy Testing: \n')
print('confusion matrix \n: ',confusion_matrix(y_test,pred))
print('Accuracy Score: ',accuracy_score(y_test,pred))