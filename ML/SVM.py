import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn import svm
from sklearn.model_selection import train_test_split
import random
import pylab as pl
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt



#data = pd.read_csv("Iris.csv")
#X = data.iloc[:, :-2]
#y = data.iloc[:, -1]

iris = load_iris()

X_train, X_test , y_train , y_test = train_test_split(iris.data, iris.target, test_size = 0.2,
                                                     random_state = random.randint(1, 100))

pca = PCA(n_components=3).fit(X_train)
#pca_2d = pca.transform(X_train)
pca_2d = pca.transform(X_train)
#print(pca_2d)

svc = svm.LinearSVC(random_state=random.randint(1,1000)).fit(pca_2d, y_train)

fig = plt.figure()
ax = plt.axes(projection='3d')

#print((pca_2d.shape))

for i in range(0, pca_2d.shape[0]):
    if y_train[i] == 0:
        c1 = ax.scatter3D(pca_2d[i,0],pca_2d[i,1], pca_2d[i, 2],c='r',    s=50,marker='+')
    elif y_train[i] == 1:
        c2 = ax.scatter3D(pca_2d[i,0],pca_2d[i,1] , pca_2d[i, 2],c='g',    s=50,marker='o')
    elif y_train[i] == 2:
        c3 = ax.scatter3D(pca_2d[i,0],pca_2d[i,1], pca_2d[i, 2],c='b',    s=50,marker='*')

#pl.legend([c1, c2, c3], ['Setosa', 'Versicolor',   'Virginica'])
#ax.scatter3D(c1)
x_min, x_max = pca_2d[:, 0].min() - 1,   pca_2d[:,0].max() + 1
y_min, y_max = pca_2d[:, 1].min() - 1,   pca_2d[:, 1].max() + 1
z_min, z_max = pca_2d[:, 2].min() - 1,   pca_2d[:, 2].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, .01),   np.arange(y_min, y_max, .01))
#Z = svmClassifier_2d.predict(np.c_[xx.ravel(),  yy.ravel(), zz.ravel()])
z_0 = lambda xx,yy, : (-svc.intercept_[0]-svc.coef_[0][0]*xx-svc.coef_[0][1]*yy) / svc.coef_[0][2]
z_1 = lambda xx,yy, : (-svc.intercept_[1]-svc.coef_[1][0]*xx-svc.coef_[1][1]*yy) / svc.coef_[1][2]
z_2 = lambda xx,yy, : (-svc.intercept_[2]-svc.coef_[2][0]*xx-svc.coef_[2][1]*yy) / svc.coef_[2][2]

print(svc.intercept_)
print(svc.coef_)
print(svc.decision_function)
#print(Z.shape)
##Z = Z.reshape(xx.shape[0], xx.shape[1])
#Z = Z.reshape(xx.shape)
#print(xx.shape)
#print(yy.shape)
print(z_2(xx,yy).shape)

ax.contour3D(xx, yy, z_2(xx, yy))
#a.axis('off')
plt.show()

