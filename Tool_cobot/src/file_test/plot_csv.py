import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numba import jit

from matplotlib import colormaps
list(colormaps)
import argparse

SUPERFICIE = {
    'altezza_mm': 471.375,
    'larghezza_mm': 1609.857
}
#----insersici qui il csv da plottare 
df = pd.read_csv("/Users/gianmarcomartino/Desktop/Tool_cobot/Tool_cobot/scansioni/test.csv", skiprows=1, index_col=1)
#---------------------------------------------------------------------------
df = df.to_numpy()
df = -df
rows, cols= df.shape

x = np.linspace(0, SUPERFICIE['larghezza_mm'], cols)
y = np.linspace(0, SUPERFICIE['altezza_mm'], rows)
X, Y = np.meshgrid(x, -y)

    # Plot 3D con PyVista
grid = pv.StructuredGrid(X, Y, df)
plotter = pv.Plotter()
plotter.add_mesh(grid, cmap='viridis', show_edges=False)
plotter.set_background('white')
plotter.add_axes()
plotter.show()


    
plt.show()

