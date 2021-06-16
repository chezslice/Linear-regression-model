import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use("ggplot")

# Loading data into file.

data = pd.read_csv("student-mat.csv", sep=";")

# Trim the data for the resulting attributes. 

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
data = stuffle(data)

# Seperating the data, by using a attribute known as a label and using numpy feature to create two arrays.
# One contain features of the data and the other to conatin labels.

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# Assigning train and test function to split data to apporiate functions.
# 90% data to train the A.i. the other 10% to test the data.

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

# Training the linear model multiple times for the best result.
  
best = 0
for _ in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc))

    if acc > best:
        best = acc
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)

# Load model.

pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)

print("-------------------------")
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print("-------------------------")

# Prediction on the x_test.

predicted = linear.predict(x_test)

# Interate thru the prediction and print out results.

for x in range(len(predicted)):
    print(predictied[x], x_test[x], y_test[x])

# Save the model using the pickle dump functon.
with open("studentgrades.pickle", "wb") as f:
    pickle.dump(linear, f)

# Drawing and plotting model.
plot = "failures"
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()






