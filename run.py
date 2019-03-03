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

    for iteration in range(iterations):
        hypothesis = x.dot(current_thetas)
        current_thetas = current_thetas - ((learning_rate * (1 / float(total_points))) * x.T.dot(hypothesis - y))

        cost_history = np.append(cost_history, calculate_cost(hypothesis - y))

    return current_thetas, cost_history

def normalize_values(values):
    means = values.mean(0)
    sdeviation = np.std(values, axis=0)
    values_normalized = (values - means) / sdeviation

    return values_normalized, means, sdeviation

def main():
    values = pd.read_csv("sample-data/multiple-variables.csv", header=None).values
    values_normalized, means, sdeviation, x, y = None, None, None, None, None

    if len(sys.argv) > 1:
        values_normalized, means, sdeviation = normalize_values(values)
        x = values_normalized[:, :-1]
        y = values_normalized[:, -1]
        y = y.reshape(len(y), 1)
        x = np.hstack([np.ones((len(y), 1)), x])
    else:
        x = values[:, :-1]
        y = values[:, -1]
        y = y.reshape(len(y), 1)
        x = np.hstack([np.ones((len(y), 1)), x])

    initial_thetas = np.zeros((x.shape[1], 1))
    learning_rate = 0.01
    iterations = 1500

    thetas, costs = calculate_gradient_descent(initial_thetas, x, y, learning_rate, iterations)

    #VISUALIZE THE COST FUNCTION OVER THE NUMBER OF ITERATIONS
    plt.plot([(iteration+1) for iteration in range(iterations)], costs[:])

    plt.show()

#------------------------#

main()
