# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import all the packages that helps to implement Decision Tree.
2. Download and upload required csv file or dataset for predecting Employee Churn
3. Initialize variables with required features.
4. And implement Decision tree classifier to predict Employee Churn

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Sai Praneeth K
RegisterNumber:  212222230067
*/
```

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("/content/Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(18, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()
```

## Output:

## Initial Dataset:

![7 1](https://github.com/SaiPraneeth04/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119390353/fa3617ff-cc3a-4b0b-97d1-65b6dac455dc)


## Mean Squared Error:

![7 2](https://github.com/SaiPraneeth04/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119390353/bc017864-4d0d-4d2e-a6a1-f53eae98de3a)


## R2 (variance):

![7 3](https://github.com/SaiPraneeth04/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119390353/fa69958a-f69f-4973-834c-be44f38457af)


## Data prediction:

![7 4](https://github.com/SaiPraneeth04/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119390353/8e9201a1-aeac-4ebd-87ff-a472e33d96e0)


## Decision Tree:

![7 5](https://github.com/SaiPraneeth04/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119390353/7d801b46-8dd5-470b-ac07-3e748cd53b6b)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
