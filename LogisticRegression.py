# Importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split   #Used to split the dataset

# Logistic Regression
class LogitRegression():
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    # Function for model training
    def fit(self, X, Y):
        # no_of_training_examples, no_of_features
        self.m, self.n = X.shape
        # w=weight b=bias
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # gradient descent learning

        for i in range(self.iterations):
            self.update_weights()
        return self

    # Helper function to update weights in gradient descent

    def update_weights(self):
        A = 1 / (1 + np.exp(- (self.X.dot(self.W) + self.b)))

        # calculate gradients
        tmp = (A - self.Y.T)
        tmp = np.reshape(tmp, self.m)
        dW = np.dot(self.X.T, tmp) / self.m
        db = np.sum(tmp) / self.m

        # updating simultaneously
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

        return self

    # Hypothetical function h(x)
    def predict(self, X):
        Z = 1 / (1 + np.exp(- (X.dot(self.W) + self.b)))
        Y = np.where(Z > 0.5, 1, 0)
        return Y

#Main function

# Importing the dataset( i converted my dataset to have only the fare, age and survived beforehand)
df = pd.read_csv("TitanicData.csv")
df=df.dropna()
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1:].values

# Splitting dataset into train and test set, using sklearn since i couldnt figure out ho to do it otherwise
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1 / 3, random_state=0)



# Training the model
model = LogitRegression(learning_rate=0.01, iterations=1000)
model.fit(X_train, Y_train)

# Prediction on test set
Y_pred = model.predict(X_test)


# measuring the performance
Right_classification = 0
#Counter
count = 0
for count in range(np.size(Y_pred)):

    if Y_test[count] == Y_pred[count]:
        Right_classification = Right_classification + 1
    count = count + 1

print("Accuracy on test set by our Logistic Regression model : ", (
        Right_classification / count) * 100)




