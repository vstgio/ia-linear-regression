import sys
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_cost(distances):
    return np.sum(np.square(distances)) / (2 * float(len(distances)))

def calculate_gradient_descent(initial_thetas, x, y, learning_rate=0.01, iterations=100):
    total_points = len(y)
    current_thetas = initial_thetas
    cost_history = np.array([])
    thetas_history = np.array([])

    for iteration in range(iterations):
        hypothesis = x.dot(current_thetas)
        current_thetas = current_thetas - ((learning_rate * (1 / float(total_points))) * x.T.dot(hypothesis - y))

        cost_history = np.append(cost_history, calculate_cost(hypothesis - y))

    return current_thetas, cost_history

def main():
    points = pd.read_csv("sample-data/single-variable.csv", header=None).values
    x, y = np.hsplit(points, 2)
    x = np.hstack([np.ones((len(x), 1)), x])

    initial_thetas = np.array([[0.0], [0.0]])
    learning_rate = 0.01
    iterations = 1500

    thetas, costs = calculate_gradient_descent(initial_thetas, x, y, learning_rate, iterations)

    plt.plot([x[0] for x in points], [y[1] for y in points], 'rx')
    plt.plot(x[:, 1], x.dot(thetas))

    #VISUALIZE THE COST FUNCTION OVER THE NUMBER OF ITERATIONS
    #plt.plot([(iteration+1) for iteration in range(iterations)], costs[:])

    plt.show()

#------------------------#

main()
