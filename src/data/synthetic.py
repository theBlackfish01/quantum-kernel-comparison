import numpy as np

def _make_circular_data(num_sectors):
    """Generate datapoints arranged in an even circle.
    
    Args:
        num_sectors (int): Number of sectors.
        
    Returns:
        tuple: (x, y, labels)
    """
    center_indices = np.array(range(0, num_sectors))
    sector_angle = 2 * np.pi / num_sectors
    angles = (center_indices + 0.5) * sector_angle
    x = 0.7 * np.cos(angles)
    y = 0.7 * np.sin(angles)
    labels = 2 * np.remainder(np.floor_divide(angles, sector_angle), 2) - 1

    return x, y, labels


def make_double_cake_data(num_sectors):
    """Generate the double cake dataset.
    
    Args:
        num_sectors (int): Number of sectors per cake.
        
    Returns:
        tuple: (X, Y) where X is (n_samples, 2) and Y is (n_samples,)
    """
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
