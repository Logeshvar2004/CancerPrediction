from sklearn import svm
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

bc= load_breast_cancer()
print('features: \n', bc.feature_names)
print('Target labels: \n',bc.target_names)

df=pd.DataFrame(bc.data)
print('Dataset: \n',df.head())
print('Null values: \n',df.isnull().sum())
print('Info about data: \n',df.info())

x_train,x_test,y_train,y_test= train_test_split(bc.data,  bc.target, test_size=0.3, random_state=109)

#Linear support vectors
model=svm.SVC(kernel='linear')
model.fit(x_train,y_train)

pred=model.predict(x_test)

print(pred)
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))

sb.pairplot(pd.DataFrame(pred)).set(title='Pearson\'s Coefficient')
plt.show()