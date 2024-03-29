from sklearn.learning_curve import
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
X = digits.data
y = digits.target

train_sizes,train_loss,test_loss = learning_curve(
    SVC(gamma=0.001,
        X,
        y,
        cv=10,
        scoring='neg_mean_squared_error',
        ,train_sizes=[0.1,0.25,0.75,1]
        )
)
train_loss_mean = -np.mean(train_loss,axis=1)
test_loss_mean = -np.mean(test_loss,axis=1)