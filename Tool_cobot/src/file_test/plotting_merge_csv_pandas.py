import pandas as pd
from config import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib import colormaps
list(colormaps)
import argparse

# Configura argparse
parser = argparse.ArgumentParser(description="Script per plottare i dati in 2D o 3D.")
parser.add_argument('--plot', choices=['2d', '3d'], default='3d', help="Scegli il tipo di plot: '2d' o '3d'.")
args = parser.parse_args()

#----importo i file di configurazione
CROP_MARGINI = {
    'left': 199,
    'right': 199

}

SUPERFICIE = {
    'altezza_mm': 471.375,
    'larghezza_mm': 1609.857
}

#----importo i dataset
df = pd.read_csv(INPUT_FILES['top'], skiprows=1, index_col=1)
df1 = pd.read_csv(INPUT_FILES['bottom'], skiprows=1, index_col=1)

def crop_dataset(df, left=0, right=0):
    """
    Ritaglia il dataset rimuovendo le righe e le colonne specificate
    parametri:
    df: dataframe da ritagliare
    left: numero di righe da rimuovere a sinistra
    right: numero di righe da rimuovere a destra

    restituisce:
    df: dataframe ritagliato
    """
    if left < 0 or right < 0:
        raise ValueError("I valori left e right devono essere maggiori di 0")
    if left + right >= df.shape[1]:
        raise ValueError("Il numero di righe da rimuovere non può essere maggiore del numero di righe totali")
    df = df.iloc[:, left:df.shape[1]-right]
    return df
#----ritaglio il dataset
df = crop_dataset(df, CROP_MARGINI['left'], CROP_MARGINI['right'])
df1 = crop_dataset(df1, CROP_MARGINI['left'], CROP_MARGINI['right'])



#-----li unisco
df_merged = pd.concat([df, df1], axis=1, ignore_index=True)

#pulisci i datiset dai Nan e 0
df_merged.fillna(0, inplace=True)
df_merged = df_merged.loc[:, (df_merged != 0).any(axis=0)]

#----elimina le colonne con più del 90% di zeri
def delete_zero_cols(df, treshold=0.1):
    """
    Elimina le colonne con più del 70% di zeri
    """
    threshold = treshold * len(df)
    df = df.loc[:, (df != 0).sum(axis=0) > threshold]
    return df
df_merged = delete_zero_cols(df_merged, treshold=0.9)
df_merged = df_merged.T

df_merged=df_merged.to_numpy()
#----plotto i dati

#---converto i dati da mm a pollici
larghezza_inch = SUPERFICIE['larghezza_mm'] / 25.4
altezza_inch = SUPERFICIE['altezza_mm'] / 25.4
scala = 0.2

rows, cols = df_merged.shape

x = np.linspace(0, SUPERFICIE['larghezza_mm'], cols)
y = np.linspace(0, SUPERFICIE['altezza_mm'], rows)
X, Y = np.meshgrid(x, -y)


#Trova i punti validi (non zero) e i valori corrispondenti
valid_points = np.where(df_merged != 0)
valid_values = df_merged[valid_points]

# Crea una griglia completa e interpolala
grid_points = np.array([X.flatten(), Y.flatten()]).T
valid_coords = np.array([X[valid_points], Y[valid_points]]).T
interpolated_values = griddata(valid_coords, valid_values, grid_points, method='cubic')

# Ricostruisci la matrice interpolata
df_merged_interpolated = interpolated_values.reshape(rows, cols)
df_merged = df_merged_interpolated  # Sostituisci con la matrice interpolata

# Salva df_merged in un file CSv
df_merged = pd.DataFrame(df_merged)  # Riconverti in DataFrame
df_merged.to_csv('df_merged_output.csv', index=False,header=False )  # Salva in CSV
df_merged_interpolated = -df_merged_interpolated  # Inverti i valori della matrice

if args.plot == '3d':
    # Plot 3D con PyVista
    grid = pv.StructuredGrid(X, Y, df_merged_interpolated)
    plotter = pv.Plotter()
    plotter.add_mesh(grid, cmap='viridis', show_edges=False)
    plotter.set_background('white')
    plotter.add_axes()
    plotter.show()
elif args.plot == '2d':
    # Inverti manualmente l'asse Y
    df_merged_interpolated_flipped = np.flipud(df_merged_interpolated)

    # Plot 2D con Matplotlib
    plt.figure(figsize=(12, 6))
    plt.imshow(df_merged_interpolated_flipped, extent=[0, SUPERFICIE['larghezza_mm'], 0, SUPERFICIE['altezza_mm']],
               origin='lower', cmap='binary', aspect='equal')  # Mantieni 'aspect=equal' per proporzioni reali
    plt.colorbar(label='Valore Interpolato')  # Barra dei colori
    plt.title('Mappa di Calore 2D della Superficie')
    plt.xlabel('Larghezza (mm)')
    plt.ylabel('Altezza (mm)')
    plt.grid(True)  # Rimuove la griglia

    # Aggiungi un rettangolo per evidenziare l'area di interesse
    
    plt.show()



