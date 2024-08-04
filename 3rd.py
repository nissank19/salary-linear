import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Linear_Regression():

  # initiating the parameters (learning rate & no. of iterations)
  def __init__(self, learning_rate, no_of_iterations):

    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations


  def fit(self, X, Y ):

    # number of training examples & number of features

    self.m, self.n = X.shape  # number of rows & columns

    self.w = np.zeros(self.n)
    self.b = 0
    self.X = X
    self.Y = Y

    # implementing Gradient Descent

    for i in range(self.no_of_iterations):
        self.update_weights()

  def update_weights(self):
      Y_prediction = self.predict(self.X)

      # calculate gradients

      dw = - (2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m

      db = - 2 * np.sum(self.Y - Y_prediction) / self.m

      # upadating the weights

      self.w = self.w - self.learning_rate * dw
      self.b = self.b - self.learning_rate * db

  def predict(self, X):
      return X.dot(self.w) + self.b


salary_data=pd.read_csv('salary_data.csv')
salary_data.isnull().sum()
x=salary_data.iloc[:,:-1].values
y=salary_data.iloc[:,1].values
X_train,X_test,Y_train,Y_test=train_test_split(x,y, test_size=0.33,random_state=2)
model=Linear_Regression(learning_rate=0.01,no_of_iterations=1000)
model.fit(X_train,Y_train)
test_predict=model.predict(X_test)
print(test_predict)
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_test, test_predict, color='blue')
plt.xlabel(' Work Experience')
plt.ylabel('Salary')
plt.title(' Salary vs Experience')
plt.show()


