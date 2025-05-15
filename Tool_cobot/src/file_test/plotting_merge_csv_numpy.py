import pandas as pd
from config import *
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

# Configura argparse
parser = argparse.ArgumentParser(description="Script per plottare i dati in 2D o 3D.")
parser.add_argument('--plot', choices=['2d', '3d'], default='3d', help="Scegli il tipo di plot: '2d' o '3d'.")
parser.add_argument('--offset', type=int, default=0, help="Offset manuale per la trasposizione dei dataset (default: 0)")
args = parser.parse_args()

#----importo i file di configurazione
OFFSET_TRASP = {
    'optimal_shift': 80
}

SUPERFICIE = {
    'altezza_mm': 471.375,
    'larghezza_mm': 1609.857
}
CROP_PARAMS = {
    'left': 0,    # numero di colonne da rimuovere a sinistra
    'right': 0,   # numero di colonne da rimuovere a destra
    'top': 30,      # numero di righe da rimuovere dall'alto
    'bottom': 30    # numero di righe da rimuovere dal basso
}

#----importo i dataset
df = np.genfromtxt(INPUT_FILES['top'], delimiter=',', skip_header=2)
df = df.transpose()
df = np.nan_to_num(df, nan=0.0)
df1 = np.genfromtxt(INPUT_FILES['bottom'], delimiter=',', skip_header=2)
df1 = df1.transpose()
df1 = np.nan_to_num(df1, nan=0.0)
#ritaglio con numpy


@jit(nopython=True)
def crop_dataset(arr, left=0, right=0, top=0, bottom=0):
    """
    Ritaglia il dataset rimuovendo le righe e colonne specificate
    
    Parametri:
    arr: array da ritagliare
    left: numero di colonne da rimuovere a sinistra
    right: numero di colonne da rimuovere a destra
    top: numero di righe da rimuovere dall'alto
    bottom: numero di righe da rimuovere dal basso
    """
    rows, cols = arr.shape
    return arr[top:rows-bottom, left:cols-right]

df = crop_dataset(df, 0, 0, 0, 0)
df1 = crop_dataset(df1, 0, 0, 0, 0)

@jit(nopython=True)
def delete_zero_rows(arr, threshold):
    """
    Versione ottimizzata per eliminare le righe con troppi zeri
    """
    # Elimina righe con troppi zeri
    row_mask = (arr != 0).sum(axis=1) > threshold
    return arr[row_mask, :]  # Aggiungi return per restituire l'array modificato


# Correggi l'assegnazione di df1
df = df.astype(np.float32)  # Converti prima in float32
df1 = df1.astype(np.float32)  # Converti prima in float32 (era df invece di df1)






# Calcola le threshold correttamente per ciascun dataset
threshold_df = 0.01 * df.shape[1]
threshold_df1 = 0.01 * df1.shape[1]


# Applica delete_zero_rows a entrambi i dataset
df = delete_zero_rows(df, threshold_df)
df1 = delete_zero_rows(df1, threshold_df1)

# Assicurati che i dataset abbiano la stessa larghezza prima di trovare lo shift
min_width = min(df.shape[1], df1.shape[1])
df = df[:, :min_width]
df1 = df1[:, :min_width]

# Trova lo shift ottimale
optimal_shift = OFFSET_TRASP['optimal_shift']

# Applica lo shift con una sovrapposizione graduale
if args.offset != 0:
    # Usa l'offset manuale
    offset = args.offset
    df_end = df[:-offset] if offset > 0 else df
    df1_start = df1[offset:] if offset > 0 else df1
else:
    # Usa l'algoritmo automatico
    offset = optimal_shift
    df_end = df[:-offset]
    df1_start = df1[offset:]

# Unisci i dataset con una transizione graduale
overlap_region = 5  # Numero di righe per la transizione graduale
weights = np.linspace(0, 1, overlap_region)[:, np.newaxis]
overlap = (1 - weights) * df[-overlap_region:] + weights * df1[:overlap_region]

df_merged = np.concatenate([df_end[:-overlap_region], overlap, df1_start[overlap_region:]], axis=0)

# Converti in float32 per efficienza
df_merged = df_merged.astype(np.float32)
df_merged=-df_merged
threshold_merged = 0.1 * df_merged.shape[1]
df_merged = delete_zero_rows(df_merged, threshold_df)
df_merged = crop_dataset(df_merged, CROP_PARAMS['left'], CROP_PARAMS['right'], CROP_PARAMS['top'], CROP_PARAMS['bottom'])

@jit(nopython=True)
def fill_zeros_with_above(arr):
    """
    Sostituisce gli zeri con i valori della riga superiore
    """
    rows, cols = arr.shape
    result = arr.copy()
    
    # Scorri tutte le righe tranne la prima
    for i in range(1, rows):
        for j in range(cols):
            if result[i, j] == 0:
                # Cerca il valore non zero più vicino nella riga superiore
                row_above = i - 1
                while row_above >= 0 and result[row_above, j] == 0:
                    row_above -= 1
                if row_above >= 0:
                    result[i, j] = result[row_above, j]
    
    return result

# Applica la sostituzione dopo la creazione di df_merged
df_merged = fill_zeros_with_above(df_merged)

rows, cols = df_merged.shape


#------creazione plot



#---converto i dati da mm a pollici
larghezza_inch = SUPERFICIE['larghezza_mm'] / 25.4
altezza_inch = SUPERFICIE['altezza_mm'] / 25.4
scala = 0.2

x = np.linspace(0, SUPERFICIE['larghezza_mm'], cols)
y = np.linspace(0, SUPERFICIE['altezza_mm'], rows)
X, Y = np.meshgrid(x, -y)

if args.plot == '3d':
    # Plot 3D con PyVista
    grid = pv.StructuredGrid(X, Y, df_merged)
    plotter = pv.Plotter()
    plotter.add_mesh(grid, cmap='viridis', show_edges=False)
    plotter.set_background('white')
    plotter.add_axes()
    plotter.show()
elif args.plot == '2d':
    # Inverti manualmente l'asse Y

    # # Plot 2D con Matplotlib
    # plt.figure(figsize=(12, 6))
    # plt.imshow(df_merged, extent=[0, SUPERFICIE['larghezza_mm'], 0, SUPERFICIE['altezza_mm']],
    #            origin='lower', cmap='binary', aspect='equal')  # Mantieni 'aspect=equal' per proporzioni reali
    # plt.colorbar(label='Valore Interpolato')  # Barra dei colori
    # plt.title('Mappa di Calore 2D della Superficie')
    # plt.xlabel('Larghezza (mm)')
    # plt.ylabel('Altezza (mm)')
    # plt.grid(True)  # Rimuove la griglia

    plt.show()



