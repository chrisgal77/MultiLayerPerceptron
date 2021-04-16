import numpy as np 
from copy import copy

def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def cross_val_score(model, X, y, k = 10):
    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    limit, rest = divmod(len(indices), k)
    start, stop = 0, limit
    
    test = []
    for i in range(k - 1):
        test.append([start + i * limit, stop + i * limit])
    test.append([start + (k-1)*limit, stop +  (k-1)*limit + rest])

    acc = []
    for i in range(k):
        test_unit = copy(model)
        
        test_range = [x for x in range(test[i][0], test[i][1])]
        train_range = [x for x in range(0, test[i][0])] + [x for x in range(test[i][1], X.shape[0])]
        X_train, X_test, y_train, y_test = X[train_range], X[test_range], y[train_range], y[test_range]

        test_unit.fit(X_train, y_train)
        pred = test_unit.predict(X_test)
        acc.append(accuracy_score(y_test, pred))
    
    return acc