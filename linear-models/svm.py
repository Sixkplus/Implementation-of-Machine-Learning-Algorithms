import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

def objective(w):
    return np.sum(w[1:]**2)

def constraint(w, X, y):
    # !!!!!!!!!!!!!
    # RESHAPE TO A VECTOR []
    return (y*(w.T@X)).reshape((y*(w.T@X)).shape[1]) - 1


def svm(X, y):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    w = np.ones(P + 1)*0.5
    num = 0

    # YOUR CODE HERE
    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge wtih any method
    # that support constrain.
    # begin answer

    X_new = np.vstack((np.ones((1, X.shape[1])), X))

    sol=minimize(fun=objective,x0=w,constraints=({'type': 'ineq', 'args': (X_new,y),
                            'fun':lambda w,X,y: constraint(w,X,y)}) )

    #print(sol)
    w = sol['x']
    support_values = y*(w.T@X_new)
    num = np.sum(support_values <= 1.001)

    # end answer
    return w, num


def objective1(w, C, X):
    P = X.shape[0] - 1
    kesei = w[P+1:]
    w = w[:P+1]
    return np.sum(w[1:]**2) + C*np.sum(kesei)

def constraint1(w, X, y):
    # !!!!!!!!!!!!!
    # RESHAPE TO A VECTOR []
    P = X.shape[0] - 1
    kesei = w[P+1:]
    w = w[:P+1]
    return (y*(w.T@X)).reshape((y*(w.T@X)).shape[1]) + kesei - 1


def constraint2(w,X):
    # !!!!!!!!!!!!!
    # RESHAPE TO A VECTOR []
    P = X.shape[0] - 1
    kesei = w[P+1:]
    return kesei


def svm_slack(X, y):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    w = np.ones(P + 1+N)*0.5
    num = 0

    # YOUR CODE HERE
    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge wtih any method
    # that support constrain.
    # begin answer

    X_new = np.vstack((np.ones((1, X.shape[1])), X))

    kesei = np.ones(N)
    C = 10

    sol=minimize(fun=objective1,x0=w, args=(C,X_new), constraints=
        ({'type': 'ineq', 'args': (X_new,y), 'fun':lambda w,X,y: constraint1(w,X,y)},
         {'type': 'ineq', 'args': (X_new,),'fun':lambda w,X: constraint2(w, X)}) )

    #print(sol)
    w = sol['x'][:P+1]
    support_values = y*(w.T@X_new)
    num = np.sum(support_values <= 1.001)

    # end answer
    return w, num