from pennylane import numpy as np
import matplotlib as mpl
import pennylane as qml
from sklearn.svm import SVC
import matplotlib.pyplot as plt

np.random.seed(1549)
num_sectors = 3


def _make_circular_data(num_sectors):
    """Generate datapoints arranged in an even circle."""
    center_indices = np.array(range(0, num_sectors))
    sector_angle = 2 * np.pi / num_sectors
    angles = (center_indices + 0.5) * sector_angle
    x = 0.7 * np.cos(angles)
    y = 0.7 * np.sin(angles)
    labels = 2 * np.remainder(np.floor_divide(angles, sector_angle), 2) - 1

    return x, y, labels


def make_double_cake_data(num_sectors):
    x1, y1, labels1 = _make_circular_data(num_sectors)
    x2, y2, labels2 = _make_circular_data(num_sectors)

    # x and y coordinates of the datapoints
    x = np.hstack([x1, 0.5 * x2])
    y = np.hstack([y1, 0.5 * y2])

    # Canonical form of dataset
    X = np.vstack([x, y]).T

    labels = np.hstack([labels1, -1 * labels2])

    # Canonical form of labels
    Y = labels.astype(int)

    return X, Y


def plot_double_cake_data(X, Y, ax, num_sectors=None):
    """Plot double cake data and corresponding sectors."""
    x, y = X.T
    cmap = mpl.colors.ListedColormap(["#a93226", "#148f77"])
    ax.scatter(x, y, c=Y, cmap=cmap, s=25, marker="s")

    if num_sectors is not None:
        sector_angle = 360 / num_sectors
        for i in range(num_sectors):
            color = ["#e74c3c", "#3498db"][(i % 2)]
            other_color = ["#e74c3c", "#3498db"][((i + 1) % 2)]
            ax.add_artist(
                mpl.patches.Wedge(
                    (0, 0),
                    1,
                    i * sector_angle,
                    (i + 1) * sector_angle,
                    lw=0,
                    color=color,
                    alpha=0.4,
                    width=0.5,
                )
            )
            ax.add_artist(
                mpl.patches.Wedge(
                    (0, 0),
                    0.5,
                    i * sector_angle,
                    (i + 1) * sector_angle,
                    lw=0,
                    color=other_color,
                    alpha=0.4,
                )
            )
            ax.set_xlim(-1, 1)

    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    return ax


X, Y = make_double_cake_data(num_sectors)

ax = plot_double_cake_data(X, Y, plt.gca(), num_sectors=num_sectors)
plt.show()


# Quantum Kernel
# ----------------------------------------------------
# ----------------------------------------------------
# ----------------------------------------------------
def layer(x, params, wires, i0=0, inc=1):
    """Building block of the embedding ansatz"""
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[0, j], wires=[wire])

    qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[1])


def ansatz(x, params, wires):
    """The embedding ansatz"""
    for j, layer_params in enumerate(params):
        layer(x, layer_params, wires, i0=j * len(wires))


adjoint_ansatz = qml.adjoint(ansatz)


def random_params(num_wires, num_layers):
    """Generate random variational parameters in the shape for the ansatz."""
    return np.random.uniform(0, 2 * np.pi, (num_layers, 2, num_wires), requires_grad=True)


dev = qml.device("default.qubit", wires=5, shots=None)
wires = dev.wires.tolist()


@qml.qnode(dev)
def kernel_circuit(x1, x2, params):
    ansatz(x1, params, wires=wires)
    adjoint_ansatz(x2, params, wires=wires)
    return qml.probs(wires=wires)


def kernel(x1, x2, params):
    return kernel_circuit(x1, x2, params)[0]


init_params = random_params(num_wires=5, num_layers=6)
kernel_value = kernel(X[0], X[1], init_params)
print(f"The kernel value between the first and second datapoint is {kernel_value:.3f}")
init_kernel = lambda x1, x2: kernel(x1, x2, init_params)
K_init = qml.kernels.square_kernel_matrix(X, init_kernel, assume_normalized_kernel=True)
svm = SVC(kernel=lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, init_kernel)).fit(X, Y)


def accuracy(classifier, X, Y_target):
    return 1 - np.count_nonzero(classifier.predict(X) - Y_target) / len(Y_target)


accuracy_init = accuracy(svm, X, Y)
print(f"The accuracy of the kernel with random parameters is {accuracy_init:.3f}")


def plot_decision_boundaries(classifier, ax, N_gridpoints=22):
    _xx, _yy = np.meshgrid(np.linspace(-1, 1, N_gridpoints), np.linspace(-1, 1, N_gridpoints))

    _zz = np.zeros_like(_xx)
    for idx in np.ndindex(*_xx.shape):
        _zz[idx] = classifier.predict(np.array([_xx[idx], _yy[idx]])[np.newaxis, :])

    plot_data = {"_xx": _xx, "_yy": _yy, "_zz": _zz}
    ax.contourf(
        _xx,
        _yy,
        _zz,
        cmap=mpl.colors.ListedColormap(["#e74c3c", "#3498db"]),
        alpha=0.4,
        levels=[-1, 0, 1],
    )
    plot_double_cake_data(X, Y, ax)
    plt.show()
    return plot_data


init_plot_data = plot_decision_boundaries(svm, plt.gca())

kta_init = qml.kernels.target_alignment(X, Y, init_kernel, assume_normalized_kernel=True)

print(f"The kernel-target alignment for our dataset and random parameters is {kta_init:.3f}")

with np.printoptions(precision=3, suppress=True):
    print(K_init)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Standard Kernels
# ----------------------------------------------------
# ----------------------------------------------------
# ----------------------------------------------------
def perform_svc_with_kernels(X, y, kernels):
    """
    Perform SVC on the dataset with different kernels and return the accuracy scores.

    Parameters:
    - X: Features of the dataset (numpy array or pandas DataFrame)
    - y: Labels of the dataset (numpy array or pandas Series)
    - kernels: List of kernels to use in SVC (list of strings)

    Returns:
    - results: Dictionary with kernels as keys and accuracy scores as values
    """
    results = {}
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    for kernel in kernels:
        # Initialize the SVC model with the given kernel
        model = SVC(kernel=kernel)
        # Train the model
        model.fit(X_train, y_train)
        # Make predictions
        y_pred = model.predict(X_test)
        # Calculate the accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # Store the accuracy in the results dictionary
        results[kernel] = accuracy

    return results


# Example usage:
kernels = ['linear', 'poly', 'rbf']
results = perform_svc_with_kernels(X, Y, kernels)
print(results)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def plot_svc_decision_boundaries(X, y, kernel='linear'):
    """
    Plot the decision boundaries of an SVC model along with the data points.

    Parameters:
    - X: Features of the dataset (2D numpy array or pandas DataFrame)
    - y: Labels of the dataset (numpy array or pandas Series)
    - kernel: Kernel to use in SVC (string)
    """
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the SVC model with the given kernel
    model = SVC(kernel=kernel)
    # Train the model
    model.fit(X_train, y_train)

    # Define the mesh grid for plotting decision boundaries
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Make predictions for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and data points
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"SVC with {kernel} kernel")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


# Example usage:
# X and y should be your dataset features and labels, with X being 2-dimensional for visualization
plot_svc_decision_boundaries(X, Y, kernel='linear')
plot_svc_decision_boundaries(X, Y, kernel='poly')
plot_svc_decision_boundaries(X, Y, kernel='rbf')
