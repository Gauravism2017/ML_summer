import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import random
import numpy as np
from sklearn.externals.six import StringIO  
from IPython.core.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

#X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.2,
#                                    random_state = random.randint(1, 100))

classifier = DecisionTreeClassifier()
classifier.fit(iris.data, iris.target)



dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

with open("GoogleMap.png", "wb") as png:
    png.write(graph.create_png())
png.close()