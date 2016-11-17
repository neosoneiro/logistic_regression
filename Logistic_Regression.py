import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
path=os.getcwd()+'\ex2data1.txt'
data=pd.read_csv(path,header=None,names=['Exam 1','Exam 2','Admitted'])
data.head()
positive=data[data['Admitted'].isin([1])]
negative=data[data['Admitted'].isin([0])]
#fig,ax=plt.plot(figsize=(12,8))
plt.scatter(positive['Exam 1'],positive['Exam 2'],color='blue',marker='o',label='Admitted')
plt.scatter(negative['Exam 1'],negative['Exam 2'],color='red',marker='x',label='Not Admitted')
plt.legend()
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.show()

def sigmoid(z):
	return 1/(1+np.exp(-z))

#nums=np.arrange(-10,10,step=1)

#fig,ax=plt.subplots(figsize=(12,8))
#ax.plot(nums,sigmoid(nums),'r')

def cost(theta,x,y):
   theta=np.matrix(theta)
   x=np.matrix(x)
   y=np.matrix(y)
   first=np.multiply(-y,np.log(sigmoid(x*theta.T)))
   second=np.multiply((1-y),np.log(1-sigmoid(x*theta.T)))
   return np.sum(first-second)/len(x)

data.insert(0,'Ones',1)

cols=data.shape[1]
x=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]
x=np.array(x.values)
y=np.array(y.values)
theta=np.zeros(3)

x.shape,y.shape,theta.shape

cost(theta,x,y)

def gradient(theta,x,y):
  theta=np.matrix(theta)
  x = np.matrix(x)
  y = np.matrix(y)
  parameters = int(theta.ravel().shape[1])
  grad = np.zeros(parameters)
  error = sigmoid(x * theta.T) - y
  for i in range(parameters):
    term = np.multiply(error, x[:,i])
    grad[i] = np.sum(term) / len(x)
  return grad
    
gradient(theta, x, y)

import scipy.optimize as opt
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(x, y))
result

cost(result[0], x, y)

def predict(theta, x1):
    probability = sigmoid(x* theta.T)
    return [1 if x1 >= 0.5 else 0 for x1 in probability]
    
theta_min = np.matrix(result[0])
predictions = predict(theta_min, x)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print 'accuracy = {0}%'.format(accuracy)   	

