import numpy as np
import pyvista as pv
import pandas as pd
from function import cerchio_valido
import json
import logging
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import os
from datetime import datetime

def salva_checkpoint(mappa_lavorata, punto_corrente, path, checkpoint_dir="checkpoints"):
    """Salva lo stato corrente della lavorazione."""
    # Crea la directory dei checkpoint se non esiste
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Genera il nome del file con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{timestamp}.npz")
    
    # Salva i dati
    np.savez_compressed(checkpoint_file,
                       mappa_lavorata=mappa_lavorata,
                       punto_corrente=punto_corrente,
                       path=path)
    
    logging.info(f"Checkpoint salvato in: {checkpoint_file}")
    return checkpoint_file

def carica_checkpoint(checkpoint_file):
    """Carica lo stato da un checkpoint."""
    try:
        data = np.load(checkpoint_file)
        mappa_lavorata = data['mappa_lavorata']
        punto_corrente = data['punto_corrente']
        path = data['path']
        logging.info(f"Checkpoint caricato da: {checkpoint_file}")
        return mappa_lavorata, punto_corrente, path
    except Exception as e:
        logging.error(f"Errore nel caricamento del checkpoint: {str(e)}")
        return None, None, None

def lista_checkpoints(checkpoint_dir="checkpoints"):
    """Lista tutti i checkpoint disponibili."""
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith("checkpoint_") and file.endswith(".npz"):
            path = os.path.join(checkpoint_dir, file)
            timestamp = datetime.strptime(file[11:-4], "%Y%m%d_%H%M%S")
            checkpoints.append({
                "file": path,
                "timestamp": timestamp,
                "name": file
            })
    
    return sorted(checkpoints, key=lambda x: x["timestamp"], reverse=True)

def crea_mappa_lavorazione(path, z_shape, raggio_cm, checkpoint_file=None):
    """Crea una mappa booleana delle aree lavorate, opzionalmente partendo da un checkpoint."""
    if checkpoint_file:
        mappa_lavorata, punto_corrente, _ = carica_checkpoint(checkpoint_file)
        if mappa_lavorata is not None:
            return mappa_lavorata
    
    mappa_lavorata = np.zeros(z_shape, dtype=bool)
    
    # Per ogni punto del percorso, marca l'area circolare come lavorata
    for x, y, z_val in path:
        # Crea una griglia di coordinate relative al centro
        y_grid, x_grid = np.ogrid[-int(raggio_cm):int(raggio_cm)+1, -int(raggio_cm):int(raggio_cm)+1]
        mask = x_grid*x_grid + y_grid*y_grid <= raggio_cm*raggio_cm
        
        # Calcola i limiti della finestra
        row_min = max(0, int(y - raggio_cm))
        row_max = min(z_shape[0], int(y + raggio_cm + 1))
        col_min = max(0, int(x - raggio_cm))
        col_max = min(z_shape[1], int(x + raggio_cm + 1))
        
        # Se la finestra è valida, marca l'area come lavorata
        if row_min < row_max and col_min < col_max:
            window_height = row_max - row_min
            window_width = col_max - col_min
            mask_window = mask[:window_height, :window_width]
            mappa_lavorata[row_min:row_max, col_min:col_max][mask_window] = True
    
    return mappa_lavorata

def visualizza_superficie_lavorata(surface_file, path_file, raggio_cm, checkpoint_file=None):
    """Visualizza la superficie con le aree lavorate evidenziate."""
    # Carica la superficie
    df = pd.read_csv(surface_file, header=None, skiprows=1)
    z = df.to_numpy()
    n_rows, n_cols = z.shape
    
    # Carica il percorso
    with open(path_file, 'r') as f:
        data = json.load(f)
        path = [tuple(p["posizione"]) for p in data["percorso"] if p["tipo"] == "lavorazione"]
    
    # Crea la mappa delle aree lavorate
    mappa_lavorata = crea_mappa_lavorazione(path, z.shape, raggio_cm, checkpoint_file)
    
    # Se specificato un checkpoint, salva lo stato
    if checkpoint_file is None:
        checkpoint_file = salva_checkpoint(mappa_lavorata, len(path)-1, path)
    
    # Crea la mesh della superficie
    x_vals = np.arange(n_cols)
    y_vals = np.arange(n_rows)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    
    # Crea la griglia strutturata
    grid = pv.StructuredGrid()
    grid.points = np.c_[x_grid.ravel(), y_grid.ravel(), z.ravel()]
    grid.dimensions = [n_cols, n_rows, 1]
    
    # Crea una mappa di colori personalizzata
    colors = ['lightgray', 'lightgreen']
    n_bins = 2
    cmap = plt.cm.colors.ListedColormap(colors)
    
    # Aggiungi un array scalare per il colore
    scalars = mappa_lavorata.ravel()
    grid.point_data["lavorata"] = scalars
    
    # Visualizzazione
    plotter = pv.Plotter()
    
    # Aggiungi la superficie con la mappa di colori personalizzata
    plotter.add_mesh(grid, scalars="lavorata", 
                    cmap=cmap,
                    opacity=0.8,
                    show_scalar_bar=False)
    
    # Aggiungi una legenda
    legend_points = [[-50, -50, np.max(z)], [-50, -70, np.max(z)]]
    legend_labels = ["Area lavorata", "Area non lavorata"]
    for point, label, color in zip(legend_points, legend_labels, colors):
        plotter.add_point_labels([point], [label],
                               point_size=20,
                               point_color=color,
                               text_color='black')
    
    # Configura la vista
    plotter.add_axes()
    plotter.set_background('white')
    plotter.show(title="Simulazione superficie lavorata")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simula e visualizza la superficie lavorata.")
    parser.add_argument('--surface', type=str, default='scansioni/complete_surface.csv',
                       help='File CSV della superficie')
    parser.add_argument('--path', type=str, default='coordinate_cobot.json',
                       help='File JSON del percorso')
    parser.add_argument('--raggio', type=float, default=15.0,
                       help='Raggio dell\'utensile in cm')
    parser.add_argument('--checkpoint', type=str,
                       help='File di checkpoint da caricare')
    parser.add_argument('--lista-checkpoints', action='store_true',
                       help='Mostra la lista dei checkpoint disponibili')
    
    args = parser.parse_args()
    
    try:
        if args.lista_checkpoints:
            checkpoints = lista_checkpoints()
            if checkpoints:
                print("\nCheckpoint disponibili:")
                for cp in checkpoints:
                    print(f"- {cp['name']} ({cp['timestamp'].strftime('%Y-%m-%d %H:%M:%S')})")
            else:
                print("\nNessun checkpoint disponibile.")
        else:
            visualizza_superficie_lavorata(args.surface, args.path, args.raggio, args.checkpoint)
    except Exception as e:
        logging.error(f"Errore durante la simulazione: {str(e)}") 