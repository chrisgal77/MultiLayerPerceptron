import numpy as np 

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
        test_start, test_stop = test[i]
        X_train, X_test, y_train, y_test = 1,1,1,1
        model.fit(X_train, y_train)
        pred = model.pred(X_test)
        
        acc.append(accuracy_score(y_test, pred))
    
    return acc