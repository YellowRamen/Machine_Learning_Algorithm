import pandas as pd
from itertools import izip
import numpy
from DecisionTree import DecisionTree
from RandomForest import RandomForest


def test_tree(filename):
    df = pd.read_csv(filename)
    y = df.pop('Result').values
    X = df.values
    print X

    ran_forest = RandomForest(10, X.shape[1], features = df.columns)
    ran_forest.fit(X, y)

    y_predict = ran_forest.predict(X)
    print y_predict.shape
    print '%26s   %10s   %10s' % ("FEATURES", "ACTUAL", "PREDICTED")
    print '%26s   %10s   %10s' % ("----------", "----------", "----------")
    for features, true, predicted in izip(X, y, y_predict):
        print '%26s   %10s   %10s' % (str(features), str(true), str(predicted))


if __name__ == '__main__':
    test_tree('../data/playgolf.csv')
