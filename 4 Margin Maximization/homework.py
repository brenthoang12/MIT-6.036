import numpy as np
import math

def margin(data, lables, th, th0):
    margin_val = []
    for i in range(data.shape[1]):
        sp = (np.dot(th.T, data[:,i:i+1]) + th0) / np.linalg.norm(th)
        margin = lables[:,i:i+1] * sp
        margin_val.append(float(margin))
    return margin_val
    # sum_mar = sum(margin_val)
    # min_mar = min(margin_val)
    # max_mar = max(margin_val)
    # return min_mar


def hinge_loss(margin, y_ref):
    hinge = []
    for x in margin:
        if x < y_ref:
            print(1-x/y_ref)
            hinge.append(1-x/y_ref)
        else:
            hinge.append(0)
    return hinge


# data = np.array([[1, 2, 1, 2, 10, 10.3, 10.5, 10.7],
#                  [1, 1, 2, 2,  2,  2,  2, 2]])
# labels = np.array([[-1, -1, 1, 1, 1, 1, 1, 1]])
# blue_th = np.array([[0, 1]]).T
# blue_th0 = -1.5
# red_th = np.array([[1, 0]]).T
# red_th0 = -2.5



data = np.array([[1.1, 1, 4],[3.1, 1, 2]])
labels = np.array([[1, -1, -1]])
th = np.array([[1, 1]]).T
th0 = -4
y_ref = math.sqrt(2)/2

print(hinge_loss(margin(data, labels, th, th0), y_ref))


