import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_double_cake_data(X, Y, ax, num_sectors=None):
    """Plot double cake data and corresponding sectors.
    
    Args:
        X: Data points (n_samples, 2)
        Y: Labels (n_samples,)
        ax: Matplotlib axis
        num_sectors: Number of sectors used in generation (optional)
    """
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

def plot_decision_boundaries(trainer, X, Y, ax, N_gridpoints=22):
    """Plot decision boundaries for the trained model."""
    _xx, _yy = np.meshgrid(np.linspace(-1, 1, N_gridpoints), np.linspace(-1, 1, N_gridpoints))

    _zz = np.zeros_like(_xx)
    for idx in np.ndindex(*_xx.shape):
        # Predict expects (1, 2) array
        point = np.array([_xx[idx], _yy[idx]])[np.newaxis, :]
        _zz[idx] = trainer.model.predict(point)

    ax.contourf(
        _xx,
        _yy,
        _zz,
        cmap=mpl.colors.ListedColormap(["#e74c3c", "#3498db"]),
        alpha=0.4,
        levels=[-1, 0, 1],
    )
    plot_double_cake_data(X, Y, ax)
    
    return ax
