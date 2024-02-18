import numpy as np

### 1 Perceptron mistake 

# def perceptron(data, labels):
#     dimension = data.shape[0]
#     data_points = data.shape[1]
#     theta = np.zeros((dimension, 1))
#     for k in range(10):
#         change = False
#         for i in range(data_points):
#             tempResult = labels[:, i:i+1] * (theta.T.dot(data[:,i:i+1]))
#             if tempResult <= 0:
#                 theta = theta + labels[:, i:i+1] * data[:,i:i+1]
#                 print(theta)
#                 change = True
#         if not change:
#             break
#     return theta
    

# x = np.array([[1,-1], [0,1], [-1.5,-1]]).T
# y = np.array([[1, -1, 1]])

# perceptron(x,y)


### 3 Dual View (dude wth)
def perceptron(data, labels, T):
    dimension = data.shape[0]
    data_points = data.shape[1]
    theta = np.zeros((dimension, 1))
    theta_0 = np.zeros((1,1))
    for k in range(T):
        change = False
        for i in range(data_points):
            tempResult = labels[:, i:i+1] * (theta.T.dot(data[:,i:i+1]) + theta_0)
            if tempResult <= 0:
                theta = theta + labels[:, i:i+1] * data[:,i:i+1]
                theta_0 = theta_0 + labels[:,i:i+1]
                change = True
        if not change:
            break
    print(theta, theta_0)

# x = np.array([[-3,2],[-1,1],[-1,-1],[2,2],[1,-1]])
# y = np.array([[1,-1,-1,-1,-1]])

# perceptron(x,y, 1000)


### 4 Decision Boundaries

# x = np.array([(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]).T
# y = np.array([[-1,-1,-1,-1,-1,-1,-1,1]])

# perceptron(x,y, 1000)


# x = np.array([[-1,1],[1,-1],[1,1], [2,2]]).T
# y = np.array([[1,1,-1,-1]])

# perceptron(x,y, 1000)
    

### 6 Mistakes and generalization

margin = [.0001, .0001, .01, .1, .2, .5]
upperBoundMistake = [(1/i)** 2 for i in margin]
print(upperBoundMistake)