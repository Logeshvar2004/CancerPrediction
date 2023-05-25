from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

bc=load_breast_cancer()
x_train,x_test,y_train,y_test=train_test_split(bc.data,bc.target,test_size=0.3,random_state=48)

model=GaussianNB()
model.fit(x_train,y_train)

pred=model.predict(x_test)
print(pred)

print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))
sb.heatmap(pd.DataFrame(pred))
plt.show()