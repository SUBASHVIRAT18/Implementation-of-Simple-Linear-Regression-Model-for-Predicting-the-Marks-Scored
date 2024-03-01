![IMG-20240301-WA0027](https://github.com/SUBASHVIRAT18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473303/f01db5b4-14a8-4604-9bc9-8fe756c738a2)# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas. 
## Program:
```
/*
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SUBASH E
RegisterNumber: 212223040209
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores (1).csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

*/
```

## Output:
## DATASET:
![IMG-20240301-WA0022](https://github.com/SUBASHVIRAT18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473303/fe9c1bdd-be46-4f7c-add7-5ab5c20bc0c2)

## HEAD VALUES:
![IMG-20240301-WA0025](https://github.com/SUBASHVIRAT18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473303/9abe9af6-6dc7-4f40-89be-567f389edd19)

## TAIL VALUES:
![IMG-20240301-WA0026](https://github.com/SUBASHVIRAT18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473303/64f4f15f-95ee-449b-b89c-802194e01804)

## X and Y VALUES:
![IMG-20240301-WA0027](https://github.com/SUBASHVIRAT18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473303/88f8d653-ec41-42bf-91eb-af9890efcdd0)

## Predication values of X and Y:
![IMG-20240301-WA0029](https://github.com/SUBASHVIRAT18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473303/aada83db-6984-418c-b62b-09a36ee7c169)

## MSE,MAE and RMSE:
![IMG-20240301-WA0028](https://github.com/SUBASHVIRAT18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473303/534556fe-70ab-4d70-b020-9ce66f6be260)

## Training Set:
![IMG-20240301-WA0024](https://github.com/SUBASHVIRAT18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473303/fda9d4f0-e613-4b42-9052-93f22eb9486a)

## Testing Set:
![IMG-20240301-WA0023](https://github.com/SUBASHVIRAT18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473303/55b0d97b-85a7-4395-a300-7d6944a6c1fa)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
