
# importing libraries

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import pickle

# importing dataset
df = load_iris()

X = df.data
feature_names = df.feature_names
y = df.target

# train test splitting

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20)

# creating the decision tree classifier

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# prediction

y_pred = model.predict(X_test)
print(y_pred)

# analyzing the results

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_pred, y_test)
ac = accuracy_score(y_pred, y_test)
print(cm)
print(ac)

# saving the results

pickle.dump(model, open("iris.pkl", "wb"))
