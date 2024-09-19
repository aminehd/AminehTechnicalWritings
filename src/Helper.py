import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os

import numpy as np
import matplotlib.pyplot as plt

# Function to generate spiral data using NumPy
def generate_spiral_numpy(n_points_per_class, noise=0.2):
    X = []
    y = []
    for class_number in range(2):
        radius = np.linspace(1, 6, n_points_per_class)
        theta = np.linspace(class_number * np.pi, (class_number + 2) * np.pi, n_points_per_class) + np.random.randn(n_points_per_class) * noise
        X1 = radius * np.sin(theta)
        X2 = radius * np.cos(theta)
        X.append(np.vstack((X1, X2)).T)
        y.append(np.full(n_points_per_class, class_number))
    
    X = np.vstack(X)
    y = np.hstack(y)
    
    return X, y

def transform_features(X):
    X1 = X[:, 0]
    X2 = X[:, 1]
    X_transformed = np.column_stack([X1, X2, X1**2, X2**2, X1*X2, np.sin(X1), np.sin(X2)])
    return X_transformed



def plot_decision_boundary(X, y, model, epoch):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # Transform the grid for model prediction
    grid_transformed = transform_features(np.c_[xx.ravel(), yy.ravel()])
    
    # Predict on the grid
    Z = model.predict(grid_transformed)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary with a gradient colormap
    plt.contourf(xx, yy, Z, levels=np.linspace(0, 1, 50), cmap="plasma", alpha=0.7)
    plt.colorbar(label='Probability of Class 1')
    
    # Scatter the actual points
    # plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors='k', alpha=0.8)
    plt.title(f"Decision Boundary at Epoch {epoch}")
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.savefig(os.path.join(folder_name, f"decision_boundary_epoch_{epoch:03}.png"))
    plt.close()
