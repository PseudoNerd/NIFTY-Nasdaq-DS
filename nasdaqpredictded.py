import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
nifty = pd.read_csv("C:\\Users\\nanda\\OneDrive\\Desktop\\NIFTY-Nasdaq-DS\\clean_nifty.csv")
nasdaq = pd.read_csv("C:\\Users\\nanda\\OneDrive\\Desktop\\NIFTY-Nasdaq-DS\\clean_nasdaq.csv")

#print(nasdaq.head())
predict = "High"
X = np.array(nasdaq.drop(["Date",predict, "Volume"], 1))
Y = np.array(nasdaq[predict])

x_test, x_train, y_test, y_train = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

linear = linear_model.LinearRegression()
linear.fit(x_test, y_test)
accuracy = linear.score(x_train, y_train)
print(accuracy*100,"\n")
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print("Predicted value: ",predictions[x], "Actual: ",y_test[x])