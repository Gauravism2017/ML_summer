import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.six import StringIO  
from IPython.core.display import Image  
from sklearn.tree import export_graphviz
#import pydotplus
from sklearn.datasets import load_iris
import time

data = pd.read_csv('processed_credit_train.csv')
X = data.iloc[:, 2:]
y = data.iloc[:,1]
y_train = y.values
X_train = X.values

#X_test = pd.read_csv('mnist_test.csv').iloc[:,:].values
#iris = load_iris()
print('Start')
classifier = RandomForestClassifier(n_estimators = 440,
                                    n_jobs = -1,
                                    oob_score = True)

t = time.time()
classifier.fit(X_train, y_train)

print(time.time() - t)
#print("done")

#estimator = classifier.estimators_[5]
#dot_data = StringIO()

#export_graphviz(estimator, out_file="tree.dot", 
#                #feature_names = X.columns,
#                #class_names = np.array(list(map(int, list(range(10))))),
#                rounded = True, proportion = False, 
#                precision = 2, filled = True)

## Convert to png using system command (requires Graphviz)
#from subprocess import call
#call(['dot', '-Teps', 'tree.dot', '-o', 'tree.eps', '-Gdpi=600'])

### Display in jupyter notebook
##from IPython.display import Image
##Image(filename = 'tree.svg')

##graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
##Image(graph.create_png())

##with open("test.png", "wb") as png:
##    png.write(graph.create_png())
##png.close()

##print(data.columns)