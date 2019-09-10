import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
s=pd.read_csv('Classified Data',index_col=0)#index_col will remove unamed collumn
df=pd.DataFrame(s)
# print(df.head())
l=StandardScaler()#object creation
l.fit(df.drop("TARGET CLASS",axis=1))
# print(l)
#print(y)

y=l.transform(df.drop("TARGET CLASS",axis=1))

X=y
Y=df["TARGET CLASS"]
x_trained, x_test , y_trained, y_test = train_test_split(X,Y,test_size=.4,random_state=101)
# print(x_test)
err=[]
for i in range(1,40):
    p = KNeighborsClassifier(n_neighbors=i)
    p.fit(x_trained, y_trained)
    pr = p.predict(x_test)
    err.append(np.mean(pr!=y_test))
print(err)



plt.plot(range(1,40),err,marker= "o")
plt.show()
# set=pd.DataFrame(y,columns=df.columns[:-1])
# # print(set.head())
# print(confusion_matrix(y_test,prediction))
# print(classification_report(y_test,prediction))
