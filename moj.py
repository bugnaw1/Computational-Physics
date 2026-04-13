import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter
from matplotlib import cm
from skimage import io
from skimage import color
import numba
from numba import jit



def build_boundaries(n=300):
    edge = np.linspace(-2, 2, n)

    upper_y = np.cos(np.pi * edge / 2)
    lower_y = edge**4
    upper_x = 1 /(np.e**(-1) - np.e) * (np.exp(edge) - np.e)
    lower_x = 0.5 * (edge**2 - edge)

    return edge, upper_y, lower_y, upper_x, lower_x


@jit
def compute_potential(potential, n_iter):
    length = potential.shape[0]

    for _ in range(n_iter):
        for i in range(1, length - 1):
            for j in range(1, length - 1):
                potential[i, j] = (
                    potential[i+1, j] +
                    potential[i-1, j] +
                    potential[i, j+1] +
                    potential[i, j-1]
                ) / 4

    return potential


def main():
    edge, upper_y, lower_y, upper_x, lower_x = build_boundaries()

    potential = np.zeros((len(edge), len(edge)))
    potential[0, :] = lower_y
    potential[-1, :] = upper_y
    potential[:, 0] = lower_x
    potential[:, -1] = upper_x

    result = compute_potential(potential, 1000)

    plt.contourf(edge, edge, result, 100, cmap="coolwarm")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()



