# Importing libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# driver code

def main():
    # Create dataset
    Y = np.array([[250], [255], [210], [225], [500], [430], [300], [420], [422], [320], [480], [490], [440], [580], [590]])
    Y = Y / 25
    X = [x * 2 for x in range(17)]
    X = np.array(X[2:]).reshape(-1, 1)

    # Model training
    model = LinearRegression()
    model.fit(Y, X)
    # Prediction
    Y_pred = model.predict(Y)

    # Visualization
    plt.scatter(X, Y, color='blue')
    plt.plot(Y_pred, Y, color='orange')
    plt.title('X vs Y')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

if __name__ == "__main__":
    main()