import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class LinearRegression:
    def __init__(self, learning_rate, epoch):
        self.learning_rate = learning_rate
        self.epoc = epoch

    def predict(self, X):
        return X.dot(self.W) + self.b 

    def fit(self, X , y):
        self.X  = X
        self.y = y
        self.b = 0
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)

        for _ in range(self.epoc):
            self.update_weight() 
        
        return self

    def update_weight(self):
        y_pred = self.predict(self.X)

        #gradient/covergance
        dW = - ( 2 * ( self.X.T ).dot( self.y - y_pred )  ) / self.m 
        db = -2 * np.sum(self.y - y_pred)/self.m

        #updating the weights
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self

def main():
    data = pd.read_csv('dataset/Salary_Data.csv')
    X = data.iloc[:,:-1].values
    y = data.iloc[:,1].values

    #spliting training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)
    lr_model = LinearRegression(learning_rate=0.01,epoch=100)
    lr_model.fit(X=X_train, y=y_train)

    #prediction
    y_prediction = lr_model.predict(X_test)

    return y_prediction


if __name__== "__main__" :
    prediction_op = main()
    print(prediction_op)
