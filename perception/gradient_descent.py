import numpy as np
import matplotlib.pyplot as plt 

def draw(x1, x2):
    ln = plt.plot(x1, x2, '-')
    #plt.pause(0.0001)
    #ln[0].remove()

def sigmoid(score):
    return 1 / (1 + np.exp(-score))

def calculate_error(line_parameters, points, y):
    m = points.shape[0]
    probabilities = sigmoid(points * line_parameters)
    cross_entropy = -(1 / m) * (np.log(probabilities).T * y + np.log(1 - probabilities).T * (1 - y))
    return cross_entropy

def gradient_descent(line_parameters, points, y, alpha):
    m = points.shape[0]
    for i in range(500):
        p = sigmoid(points * line_parameters)
        gradient = (points.T * (p - y)) * (1 / m) * alpha
        line_parameters = line_parameters - gradient
        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b  = line_parameters.item(2)
        x1 = np.array([points[:, 0].min(), points[:, 0].max()])
        x2 = -b / w2 + x1 * (-w1 / w2)
    draw(x1, x2)
    print((calculate_error(line_parameters, all_points, y)))

n_stp = 100
bias = np.ones(n_stp)
random_x1_values = np.random.normal(10, 2, n_stp)
random_y1_values = np.random.normal(12, 2, n_stp)
top_region = np.array([random_x1_values, random_y1_values, bias]).T
random_x2_values = np.random.normal(5, 2, n_stp)
random_y2_values = np.random.normal(6, 2, n_stp)
bottom_region = np.array([random_x2_values, random_y2_values, bias]).T
all_points = np.vstack((top_region, bottom_region))

line_parameters = np.matrix(np.zeros(3)).T

y = np.array([np.zeros(n_stp), np.ones(n_stp)]).reshape(2 * n_stp, 1)
print((calculate_error(line_parameters, all_points, y)))

_, ax = plt.subplots(figsize=(4, 4))
ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')
gradient_descent(line_parameters, all_points, y, 0.06)
plt.show()