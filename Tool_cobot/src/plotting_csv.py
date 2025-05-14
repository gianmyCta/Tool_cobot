import pandas as pd
from config import *
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
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


#--------
plt.figure(figsize=(larghezza_inch * scala, altezza_inch * scala))
plt.imshow(df_merged, cmap='viridis')
plt.title("Heatmap della superficie completa")
plt.xlabel("Larghezza")
plt.ylabel("Altezza")
plt.show()




