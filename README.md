# Implementation-of-SVM-For-Spam-Mail-Detection
### NAME : SABEEHA SHAIK
### REGISTER NUMBER : 212223230176
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the Program.

2.Import the necessary packages.

3.Read the given csv file and display the few contents of the data.

4.Assign the features for x and y respectively.

5.Split the x and y sets into train and test sets.

6.Convert the Alphabetical data to numeric using CountVectorizer

7.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library

8.Find the accuracy of the model.

9.End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SABEEHA SHAIK
Register Number:  212223230176
*/
```
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')


data.head()


data.info()


data.isnull().sum()


x=data["v1"].values
y=data["v2"].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)


from sklearn.feature_extraction.text import CountVectorizer
#countvectorizer is a method to convert text to numerical data. The text is transformed to a sparse matrix
cv=CountVectorizer()


x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)


from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy


from sklearn.metrics import confusion_matrix,classification_report
con=confusion_matrix(y_test,y_pred)
print(con)
cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:

### data.head()

![image](https://github.com/user-attachments/assets/c7939192-e172-4f52-9c7d-dde66aed0e0d)

### data.info():

![image](https://github.com/user-attachments/assets/7549b729-952e-4b86-800a-6f9f42ab42ec)

### data.isnull():

![image](https://github.com/user-attachments/assets/c00bd3ba-1fcc-4ee3-93ce-93b293241f3d)

### Y_prediction:

![image](https://github.com/user-attachments/assets/9991a129-8f75-4d3c-b6f8-b7691af31c5b)


### Accuracy:

![image](https://github.com/user-attachments/assets/da93b683-4193-476d-86c8-a1989793bb87)

### Confusion Matrix:

![image](https://github.com/user-attachments/assets/4a5a5bef-e99c-4cb6-bff4-f74bb32617a1)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
