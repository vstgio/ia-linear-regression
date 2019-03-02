import sys
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_gradient_descent(initial_thetas, x, y, learning_rate=0.01, iterations=100):
    total_points = len(y)
    current_thetas = initial_thetas
    cost_history = []

    for iteration in range(iterations):
        thetas_gradient = x.T.dot(x.dot(current_thetas) - y)
        current_thetas = current_thetas - ((learning_rate * (1 / float(total_points))) * thetas_gradient)

    return current_thetas

def main():
    points = pd.read_csv(sys.argv[1], header=None).values
    x, y = np.hsplit(points, 2)
    x = np.hstack([np.ones((len(x), 1)), x])

    initial_thetas = np.array([[0.0], [0.0]])
    learning_rate = 0.01
    iterations = 1500

    thetas = calculate_gradient_descent(initial_thetas, x, y, learning_rate, iterations)

    plt.plot([x[0] for x in points], [y[1] for y in points], 'rx')
    plt.plot(x[:, 1], x.dot(thetas))

    plt.show()

#------------------------#

main()
