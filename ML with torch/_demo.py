from LinearClassifiers import Perceptron, Adaline, LogisticRegression
from TreeClassifiers import DecisionTree
import numpy as np
import matplotlib.pyplot as plt

# Inputs:
# 100 points to that have all negative entries and 
# 100 points that have all positive entries 

inputs_a = np.random.rand(100,3) - 2
inputs_b = np.random.rand(100,3) + 2
inputs = np.concatenate((inputs_a, inputs_b), axis = 0)
labels_a = np.array([0 for i in range(100)])
labels_b = np.array([1 for i in range(100)])
labels = np.concatenate((labels_a, labels_b), axis=0)


# Various classifiers:

# ----------------- Perceptron -------------------------
p = Perceptron(eta = 0.2)
p.train(inputs, labels)

# Plotting perceptron loss 
t=np.linspace(0, p.max_iter, p.max_iter)
plt.plot(t, p.errors, label='single layer perceptron')
plt.title("Single layer perceptron")
plt.legend()
plt.show()

# ------------------ Adaline ---------------------------

# Adaline (GD) with raw input -- no convergence for large eta, and very very slow convergence for big eta
a = Adaline(eta = 0.2, max_iter=50)
a.train(inputs, labels)
t=np.linspace(0, a.max_iter, a.max_iter)
plt.plot(t, a.loss_vals, label='adaline (raw) GD')
plt.title("Adaline GD without feature scaling")
plt.show()

# Normalizing input (feature scaling)!  
input_scaled = np.copy(inputs)
input_scaled[:,0] = (inputs[:,0]-inputs[:,0].mean())/inputs[:,0].std()
input_scaled[:,1] = (inputs[:,1]-inputs[:,1].mean())/inputs[:,1].std()
input_scaled[:,2] = (inputs[:,2]-inputs[:,2].mean())/inputs[:,2].std()

# Adaline (GD) with scaled features
a.train(input_scaled, labels)
t=np.linspace(0, a.max_iter, a.max_iter)
plt.title("Adaline")
plt.plot(t, a.loss_vals, label='adaline (scaled) GD')

# Adaline (GD) with scaled features + reg
areg = Adaline(reg= True, Lambda = 1.0, eta = 0.2, max_iter=50)
areg.train(input_scaled, labels)
plt.plot(t, areg.loss_vals, label = 'adaline (scaled) GD - reg')


#  Adaline (SGD) with scaled input no regularization.
a2 = Adaline(opt="SGD", eta = 0.005, max_iter=50, seed = 100, shuffle = True)
a2.train(input_scaled, labels)
t=np.linspace(0, a2.max_iter, a2.max_iter)
plt.plot(t, a2.loss_vals, label = 'Adaline (scaled) SGD')

# Adaline (SGD) with scaled input and regularization.
a3 = Adaline(reg = True, Lambda = 0.01, opt="SGD", eta = 0.05, max_iter=50, seed = 100, shuffle = True)
a3.train(input_scaled, labels)
t=np.linspace(0, a3.max_iter, a3.max_iter)
plt.plot(t, a3.loss_vals, label = 'adaline (scaled) SGD - reg')
plt.legend()
plt.show()


# ---------------------------- Logistic reg ------------------

# Logistic regression - full batch GD - no regularization 
lg = LogisticRegression()
lg.train(inputs, labels)
t=np.linspace(0, lg.max_iter, lg.max_iter)
plt.title("Logistic Regression")
plt.plot(t, lg.loss_vals, label='logistic GD')

# Logistic regression - full batch GD - with regularization 
lg2 = LogisticRegression(reg = True, Lambda = 1.0)
lg2.train(inputs, labels)
t=np.linspace(0, lg2.max_iter, lg2.max_iter)
plt.plot(t, lg2.loss_vals, label='logistic GD - reg')


# Logistic regression - full batch SGD - no regularization 
lg3 = LogisticRegression(opt = "SGD")
lg3.train(inputs, labels)
t=np.linspace(0, lg3.max_iter, lg3.max_iter)
plt.plot(t, lg3.loss_vals, label='logistic SGD')

# Logistic regression - full batch SGD - with regularization 
lg4 = LogisticRegression(opt = "SGD", reg = True, Lambda = 0.0001)
lg4.train(inputs, labels)
t=np.linspace(0, lg4.max_iter, lg4.max_iter)
plt.plot(t, lg4.loss_vals, label='logistic SGD - reg')
plt.ylim((0,2))
plt.legend()
plt.show()

# ------------------ Decision Tree ------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=100)

bdt = DecisionTree(max_depth=10)
bdt.fit(X_train, y_train)
y_pred = bdt.predict(X_test)

acc = np.sum(y_test == y_pred) / len(y_test)
print(acc)

# -------------------------------------------------