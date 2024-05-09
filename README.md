# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. .Collect a labeled dataset of emails, distinguishing between spam and non-spam.

2.Preprocess the email data by removing unnecessary characters, converting to lowercase, removing stop words, and performing stemming or lemmatization.

3.Extract features from the preprocessed text using techniques like Bag-of-Words or TF-IDF.

4.Split the dataset into a training set and a test set.

5.Train an SVM model using the training set, selecting the appropriate kernel function and hyperparameters.

6.Evaluate the trained model using the test set, considering metrics such as accuracy, precision, recall, and F1 score.

7.Optimize the model's performance by tuning its hyperparameters through techniques like grid search or random search.

8.Deploy the trained and fine-tuned model for real-world use, integrating it into an email server or application.

9.Monitor the model's performance and periodically update it with new data or adjust hyperparameters as needed.
 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: kavisree.s
RegisterNumber:  212222047001


import chardet   
file='/content/spam.csv'    
with open(file,'rb') as rawdata:    
  result = chardet.detect(rawdata.read(100000))   
result   

import pandas as pd    
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')    

data.head()   

data.info()   

data.isnull().sum()   

x=data["v1"].values   
y=data["v2"].values    

from sklearn.model_selection import train_test_split    
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)   

from sklearn.feature_extraction.text import CountVectorizer   
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

*/
```
## RESULT OUTPUT:

![image](https://github.com/kavisree86/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145759687/2508f152-e04e-4ca5-ba96-323242771075)

## data. head():
![image](https://github.com/kavisree86/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145759687/9406c459-3269-458a-84ac-99d82c610fca)

## data. info():
![image](https://github.com/kavisree86/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145759687/7424634d-f27c-48af-b788-054e7ad47203)


## data.isnull().sum()

![image](https://github.com/kavisree86/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145759687/4b7c8e76-fa0a-42be-ac2a-a80dd22614b6)

## Y_Prediction value
![image](https://github.com/kavisree86/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145759687/226a23ff-4b9e-45aa-b24d-908acd168db7)

## Accuracy value:

![image](https://github.com/kavisree86/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145759687/2f320b48-7017-40c4-8009-b36481a08491)

## Output:
![SVM For Spam Mail Detection](sam.png)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
