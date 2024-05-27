import numpy as np

data = np.array([[1, 1, 3, 3],[3, 1, 4, 2]])
labels = np.array([[-1, -1, 1, 1]])
th = np.array([[0, 1]]).T
th0 = -3

# max margin separator

data_point = data.shape[1]

margin_acc = []

for i in range(data_point):
    pt = np.array([data[:,i]]).T
    sp = (np.dot(th.T, pt) + th0) / np.linalg.norm(th)
    margin = sp[0,0] * labels[0,i]
    margin_acc.append(margin)

print(margin_acc)


