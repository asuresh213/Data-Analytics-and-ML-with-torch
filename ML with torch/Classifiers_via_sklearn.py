from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier 
import numpy as np 
import matplotlib.pyplot as plt

# Loading (iris) dataset:
iris = datasets.load_iris()
inputs = iris.data[:,[2,3]]
labels = iris.target

# ----------------------- Pre processing -------------------------------

# Split dataset into training and testing batches.
X_train, X_test,y_train, y_test = train_test_split(inputs, labels, test_size=0.4, random_state=1, stratify=labels)

# Do feature scaling on the data for improved convergence
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# ------------------------ Simple Perceptron -----------------------------
# Pereptron (OvA) training
pr = Perceptron(eta0=0.1, random_state=100)
pr.fit(X_train_std, y_train)

# Predicting on test data
y_pred = pr.predict(X_test_std)
print("Misclassified examples: {}".format((y_test != y_pred).sum()))
print("Accuracy: {}".format(pr.score(X_test_std, y_test)))


#------------------------ Logistic Regression -----------------------------

# LR (OvA) training 

# ---notes---
# C is regularization hyperparameter.
# lbfgs is one of the few built in convex optimizers (better than SGD)
# multi_class defines the multi-lable classification scheme. 
#---

lr = LogisticRegression(C=100.0, solver = 'lbfgs', multi_class='ovr') 
lr.fit(X_train_std, y_train)

# Predicting on test data
y_pred = lr.predict(X_test_std)
print("Misclassified examples: {}".format((y_test != y_pred).sum()))
print("Accuracy: {}".format(lr.score(X_test_std, y_test)))


# ---------------------- Support Vector Machines ------------------------

# Linear kernel
svm = SVC(kernel = 'linear', C=1.0, random_state = 100)
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)
print("Misclassified examples: {}".format((y_test != y_pred).sum()))
print("Accuracy: {}".format(svm.score(X_test_std, y_test)))

# Non-linear (rbf -radial) kernel with low gamma 
svm2 = SVC(kernel = 'rbf', gamma = 0.01, C=1.0, random_state = 100)
svm2.fit(X_train_std, y_train)
y_pred = svm2.predict(X_test_std)
print("Misclassified examples: {}".format((y_test != y_pred).sum()))
print("Accuracy: {}".format(svm2.score(X_test_std, y_test)))


# Non-linear (rbf -radial) kernel with high gamma 
svm3 = SVC(kernel = 'rbf', gamma = 21.0, C=1.0, random_state = 100)
svm3.fit(X_train_std, y_train)
y_pred = svm3.predict(X_test_std)
print("Misclassified examples: {}".format((y_test != y_pred).sum()))
print("Accuracy: {}".format(svm3.score(X_test_std, y_test)))


# -------------------------- Decision Trees ------------------------------

# Decision Trees
 
tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=100)
tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)
print("Misclassified examples: {}".format((y_test != y_pred).sum()))
print("Accuracy: {}".format(tree_model.score(X_test, y_test)))


feature_names = ['S_length', 'S_width', 'P_length', 'P_width']
tree.plot_tree(tree_model, feature_names=feature_names, filled=True)
#plt.show()


# Random Forest
forest = RandomForestClassifier(n_estimators=25, random_state=100, n_jobs=2)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
print("Misclassified examples: {}".format((y_test != y_pred).sum()))
print("Accuracy: {}".format(forest.score(X_test, y_test)))
