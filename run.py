import sys
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_cost(points, t0, t1):
    distances = 0
    for index in range(len(points)):
        distances += (points[index][1] - (t0 + (t1 * points[index][0]))) ** 2
    return distances / (2 * float(len(points)))

def calculate_gradient_descent(t0, t1, points, learning_rate=0.01, iterations=100):
    total_points = len(points)
    current_t0, current_t1 = t0, t1

    for iteration in range(iterations):
        t0_gradient, t1_gradient = 0, 0
        for index_point in range(len(points)):
            t0_gradient += ((current_t0 + (current_t1 * points[index_point][0])) - points[index_point][1])
            t1_gradient += ((current_t0 + (current_t1 * points[index_point][0])) - points[index_point][1]) * points[index_point][0]

        current_t0 = current_t0 - ((learning_rate * (1 / float(total_points))) * t0_gradient)
        current_t1 = current_t1 - ((learning_rate * (1 / float(total_points))) * t1_gradient)

    return current_t0, current_t1

####

points = pd.read_csv(sys.argv[1], header=None).values

initial_t0 = 0
initial_t1 = 0

t0, t1 = calculate_gradient_descent(initial_t0, initial_t1, points, 0.001, 15000)

plt.plot([x[0] for x in points], [y[1] for y in points], 'rx')
plt.plot([x[0] for x in points], [((points[counter][0] * t1) + t0) for counter in range(len(points))])

plt.show()
