from sklearn import tree 
from sklearn.model_selection import train_test_split
from sklearn import metrics 

import pandas as pd 

url = "https://raw.githubusercontent.com/viniciuspaiva334/IoT-Ia/refs/heads/main/iris/iris.data"
dataSet = pd.read_csv(url, header=None, on_bad_lines='skip')


print(dataSet.shape)   
 
 

columns = len(dataSet.columns)

Y = dataSet[4] 
X = dataSet.loc[:,0:columns-2]

print(Y.head())              
print(Y.value_counts())     
print(Y.unique())            


x_train, x_test , y_train , y_test = train_test_split(X, Y , test_size=0.2, random_state=None , stratify=Y )

model = tree.DecisionTreeClassifier(criterion="entropy")
model = model.fit(x_train, y_train)

#predicao e resultados 

result = model.predict(x_test)
accuracy = metrics.accuracy_score(result , y_test)
show = round(accuracy * 100)
print("{}%".format(show))
#print(list(result))