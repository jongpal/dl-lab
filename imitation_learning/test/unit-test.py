import sys
sys.path.append("../../") 
import numpy as np

from imitation_learning.dataset import stack_histories

def test_histories():
    X = np.array([[[1, 2, 3], [3, 4, 5], [5, 6, 7]], [[0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.5, 0.6, 0.7]], \
                  [[0.01, 0.02, 0.03], [0.03, 0.04, 0.05], [0.05, 0.06, 0.07]]]) # 2, 3, 3
    y = np.array([10, 11, 12]) # 2,

    X_histories, y_histories = stack_histories(X, 3)
    print(X_histories.shape)
    print(X_histories) 

if __name__ == "__main__":
    test_histories()
