import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

# X_train,X_test,y_train,y_test = train_test_split(iris_X,iris_y,random_state=4)
from sklearn.model_selection import cross_val_score
k_range = range(1,31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, iris_X, iris_y, cv=10, scoring='accuracy')
    # loss = -cross_val_score(knn, iris_X, iris_y, cv=10, scoring='mean_squared_error')
    k_scores.append(scores.mean())
plt.plot(k_range,k_scores)
plt.xlabel('横坐标')
plt.ylabel('纵坐标')
plt.show()


# knn.fit(X_train,y_train)
# print(knn.score(X_test,y_test))
# print(knn.predict(X_test))
# print(y_test)