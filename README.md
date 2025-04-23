# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries
2. Load and analyse the dataset
3. Preprocess the data, convert the numerical to categorical data and encode this categorial codes to numeric codes using .cat.codes
4. Assign the input features and target variable
5. Define the functions sigmoid, loss, gradient_descent
6. Make predictions on new data and measure the accuracy

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: RADHIMEENA M
RegisterNumber:  212223040159
*/
```
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        data=pd.read_csv("/content/Placement_Data.csv")
        data.info()
        data=data.drop(['sl_no','salary'],axis=1)
        data
        data["gender"]=data["gender"].astype('category')
        data["ssc_b"]=data["ssc_b"].astype('category')
        data["hsc_b"]=data["hsc_b"].astype('category')
        data["hsc_s"]=data["hsc_s"].astype('category')
        data["degree_t"]=data["degree_t"].astype('category')
        data["workex"]=data["workex"].astype('category')
        data["specialisation"]=data["specialisation"].astype('category')
        data["status"]=data["status"].astype('category')
        data.dtypes
        data["gender"]=data["gender"].cat.codes
        data["ssc_b"]=data["ssc_b"].cat.codes
        data["hsc_b"]=data["hsc_b"].cat.codes
        data["hsc_s"]=data["hsc_s"].cat.codes
        data["degree_t"]=data["degree_t"].cat.codes
        data["workex"]=data["workex"].cat.codes
        data["specialisation"]=data["specialisation"].cat.codes
        data["status"]=data["status"].cat.codes
        data
        x=data.iloc[:,:-1].values
        y=data.iloc[:,-1]
        theta = np.random.randn(x.shape[1])
        
        
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))
        
        def loss(theta, x, y):
            h = sigmoid(x.dot(theta))
            return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        
        def gradient_descent(theta, x, y, alpha, num_iterations):
            m = len(y)
            for i in range(num_iterations):
                h = sigmoid(x.dot(theta))
                gradient = x.T.dot(h - y) / m
                theta -= alpha * gradient
            return theta  
        
        theta = gradient_descent(theta, x, y, alpha=0.01, num_iterations=1000)
        
        def predict(theta, x):
            h = sigmoid(x.dot(theta))
            y_pred = np.where(h >= 0.5, 1, 0)
            return y_pred
        
        y_pred = predict(theta, x)
        
        accuracy=np.mean(y_pred.flatten()==y)
        
        print("Acuracy:",accuracy)
        xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
        y_prednew=predict(theta,xnew)
        print(y_prednew)
        xnew=np.array([[0,0,0,5,65,2,8,2,0,0,1,0]])
        y_prednew=predict(theta,xnew)
        print(y_prednew)

## Output:

![image](https://github.com/user-attachments/assets/bd547041-af4f-4267-80ed-a23a66097ec9)
![image](https://github.com/user-attachments/assets/679b76b4-13e8-481d-a4ae-c8296b641737)
![image](https://github.com/user-attachments/assets/2648ca22-1f7a-424e-b2ff-bbd288b071d4)
![image](https://github.com/user-attachments/assets/211be8db-aa49-4680-8d86-e05ad5d57652)
![image](https://github.com/user-attachments/assets/72a5e3e1-bacb-4feb-bf01-ff1eea88fad1)
![image](https://github.com/user-attachments/assets/b103403c-cf54-44e2-880a-6b3a454a4cfa)
![image](https://github.com/user-attachments/assets/37001684-9482-4982-b764-bdd137089468)
![image](https://github.com/user-attachments/assets/a6ef7477-b52e-42de-aba4-fe3fc20be4b4)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

