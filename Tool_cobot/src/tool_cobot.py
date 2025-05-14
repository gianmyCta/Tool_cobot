import pandas as pd
import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_gradient_magnitude, gaussian_filter
from function import *
from extended_processing import ExtendedProcessor
from visualization.visualizer import visualizza_2d, visualizza_3d
import sys
sys.path.append('..')
from config import *
import logging
import argparse
import json
from functools import lru_cache
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from datetime import datetime, timedelta

def cerchio_valido(row, col, lavorabile, raggio_cm):
    """Verifica se un cerchio centrato in (row, col) è completamente in area lavorabile."""
    h, w = lavorabile.shape
    
    # Calcola i limiti della finestra considerando i bordi
    row_min = max(0, int(row - raggio_cm))
    row_max = min(h, int(row + raggio_cm + 1))
    col_min = max(0, int(col - raggio_cm))
    col_max = min(w, int(col + raggio_cm + 1))
    
    # Verifica rapida dei bordi
    if row_min >= row_max or col_min >= col_max:
        return False
        
    # Cache della maschera circolare per dimensioni comuni
    mask = _get_circle_mask(row_max - row_min, col_max - col_min, 
                          row - row_min, col - col_min, raggio_cm)
    
    # Usa view invece di copia per la finestra
    window = lavorabile[row_min:row_max, col_min:col_max]
    
    return np.all(window[mask])

@lru_cache(maxsize=128)
def _get_circle_mask(height, width, center_y, center_x, radius):
    """Cache delle maschere circolari per dimensioni comuni."""
    y, x = np.ogrid[:height, :width]
    return (x - center_x)**2 + (y - center_y)**2 <= radius**2

def plot_2d_path(path, z, lavorabile=None):
    """Visualizza il percorso in 2D con matplotlib."""
    plt.figure(figsize=(12, 8))
    
    # Plot della superficie come heatmap
    plt.imshow(z, cmap='viridis', origin='lower', aspect='equal')
    plt.colorbar(label='Altezza')
    
    # Se abbiamo la maschera lavorabile, visualizziamola come overlay trasparente
    if lavorabile is not None:
        mask_overlay = np.ma.masked_where(~lavorabile, np.ones_like(lavorabile))
        plt.imshow(mask_overlay, cmap='RdYlGn', alpha=0.3, origin='lower', aspect='equal')
    
    # Estrai le coordinate x e y dal percorso
    if path:
        x_coords, y_coords, _ = zip(*path)
        
        # Plot del percorso
        plt.plot(x_coords, y_coords, 'r.-', label='Percorso', linewidth=1, markersize=3)
        
        # Evidenzia punto iniziale e finale
        plt.plot(x_coords[0], y_coords[0], 'go', label='Inizio', markersize=8)
        plt.plot(x_coords[-1], y_coords[-1], 'ro', label='Fine', markersize=8)
    
    plt.title('Visualizzazione 2D del Percorso')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

def cleanup_old_logs(log_dir, days=1):
    """Pulisce i file di log più vecchi di X giorni."""
    if not log_dir.exists():
        return
    
    cutoff = datetime.now() - timedelta(days=days)
    for file in log_dir.glob('*'):
        if file.stat().st_mtime < cutoff.timestamp():
            file.unlink()

def setup_logging(no_logs):
    """Configura il sistema di logging."""
    logger = logging.getLogger(__name__)
    
    if no_logs:
        # Configura solo lo stream handler per i messaggi essenziali
        handler = logging.StreamHandler()
        handler.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
    else:
        # Configura logging normale con output su file e console
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Handler per la console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def main():
    # Configura argparse
    parser = argparse.ArgumentParser(description="Genera e visualizza il percorso di levigatura su una superficie.")
    parser.add_argument('--input', type=str, default=INPUT_FILES['output'], help='File CSV della superficie (default: complete_surface.csv)')
    parser.add_argument('--diametro', type=float, default=TOOL_PARAMS['diametro_default'], help='Diametro dell\'utensile in cm')
    parser.add_argument('--step', type=int, default=TOOL_PARAMS['step_default'], help='Passo del percorso in cm (default: 19)')
    parser.add_argument('--pendenza', type=float, default=TOOL_PARAMS['pendenza_max'], help='Pendenza massima accettabile (default: 0.7)')
    parser.add_argument('--output', type=str, default='coordinate_cobot.json', help='File di output JSON (default: coordinate_cobot.json)')
    
    # Opzioni per il controllo del flusso
    parser.add_argument('--skip-path', action='store_true', help='Salta la generazione del percorso')
    parser.add_argument('--skip-visualization', action='store_true', help='Salta la visualizzazione 3D')
    parser.add_argument('--plot-2d', action='store_true', help='Visualizza il percorso in 2D')
    parser.add_argument('--skip-export', action='store_true', help='Salta l\'esportazione del file JSON')
    parser.add_argument('--only-visualization', action='store_true', help='Esegui solo la visualizzazione del percorso esistente')
    parser.add_argument('--debug', action='store_true', help='Attiva la modalità debug con log dettagliati')
    parser.add_argument('--no-logs', action='store_true', help='Disattiva la creazione dei file di log')

    args = parser.parse_args()

    # Configura il logger
    logger = setup_logging(args.no_logs)

    # Parametri utensile da argparse
    diametro_cm = args.diametro
    raggio_cm = diametro_cm / 2
    step_cm = args.step
    pendenza_max = args.pendenza
    input_file = args.input
    output_file = args.output

    # Carica superficie
    logger.info(f"Caricamento della superficie da '{input_file}'")
    df = pd.read_csv(input_file, header=None, skiprows=1)
    z = df.to_numpy()
    n_rows, n_cols = z.shape
    logger.info(f"Superficie caricata: {n_rows} righe, {n_cols} colonne")

    path = []
    
    if not args.skip_path and not args.only_visualization:
        # Calcolo del gradiente con smoothing
        logger.info("Calcolo delle aree lavorabili...")
        z_smooth = gaussian_filter(z, sigma=2)  # Applica smoothing per ridurre il rumore
        gradient = gaussian_gradient_magnitude(z_smooth, sigma=1)
        
        # Aggiungi un margine di sicurezza alla maschera lavorabile
        lavorabile = gradient < pendenza_max 
        
        # Espandi le aree non lavorabili per creare una zona cuscinetto
        from scipy.ndimage import binary_dilation
        non_lavorabile = ~lavorabile
        non_lavorabile = binary_dilation(non_lavorabile, iterations=2)
        lavorabile = ~non_lavorabile

        # Margine di sicurezza
        margine = int(np.ceil(raggio_cm * 1.1))  # Aumenta il margine del 10%

        # Costruzione percorso serpentina verticale
        logger.info("Inizio costruzione del percorso a serpentina verticale")
        
        # Crea griglie di coordinate - Prima le righe per movimento verticale
        rows = np.arange(margine, n_rows - margine, step_cm)  # Dall'alto verso il basso
        cols = np.arange(margine, n_cols - margine, step_cm)  # Da sinistra a destra
        
        # Crea una matrice di validità per tutti i punti
        validita = np.ones((len(cols), len(rows)), dtype=bool)  # Nota: dimensioni invertite
        
        # Crea una matrice di coordinate per tutti i punti
        col_grid, row_grid = np.meshgrid(cols, rows, indexing='ij')  # Nota: cols prima di rows
        
        # Verifica la validità di tutti i punti in modo vettoriale
        for r in range(int(raggio_cm * 1.2)):
            mask = np.zeros_like(validita)
            for i, j in np.ndindex(validita.shape):
                if validita[i, j]:  # Verifica solo i punti ancora validi
                    mask[i, j] = cerchio_valido(int(row_grid[i, j]), int(col_grid[i, j]), lavorabile, r)
            validita &= mask
        
        # Crea una maschera per l'ordine alternato delle colonne
        alternating_mask = np.zeros_like(validita)
        for i in range(len(cols)):  # Per ogni colonna
            if i % 2 == 0:
                alternating_mask[i, :] = True  # Dall'alto verso il basso
            else:
                alternating_mask[i, ::-1] = True  # Dal basso verso l'alto
        
        # Ottieni gli indici ordinati per il percorso
        ordered_indices = np.where(alternating_mask)
        valid_mask = validita[ordered_indices]
        
        # Aggiungi i punti validi al percorso
        valid_rows = row_grid[ordered_indices][valid_mask]
        valid_cols = col_grid[ordered_indices][valid_mask]
        col_indices = ordered_indices[0][valid_mask]  # Nota: ora usiamo l'indice 0 per le colonne
        
        # Crea il percorso iniziale usando le coordinate originali e i valori z corretti
        path = []
        for row, col, col_idx in zip(valid_rows, valid_cols, col_indices):
            x = cols[col_idx]  # Coordinata x originale
            y = row           # Coordinata y originale
            z_val = z[int(y), int(x)]  # Valore z dalla superficie originale
            path.append((x, y, z_val))
            logger.debug(f"Punto valido aggiunto: ({x}, {y}, {z_val})")
        
        # Cerca punti alternativi per i punti non validi
        invalid_rows = row_grid[ordered_indices][~valid_mask]
        invalid_cols = col_grid[ordered_indices][~valid_mask]
        invalid_col_indices = ordered_indices[0][~valid_mask]  # Nota: ora usiamo l'indice 0 per le colonne
        
        for row, col, col_idx in zip(invalid_rows, invalid_cols, invalid_col_indices):
            x = cols[col_idx]  # Coordinata x originale
            y = row           # Coordinata y originale
            nuovo_row = trova_vicino_valido_vettoriale(int(y), int(x), lavorabile, raggio_cm * 1.2)
            if nuovo_row is not None:
                z_val = z[nuovo_row, int(x)]  # Valore z dalla superficie originale
                if len(path) > 0:
                    ultimo_punto = path[-1]
                    if segmento_sicuro(ultimo_punto, (x, nuovo_row, z_val), lavorabile, raggio_cm):
                        path.append((x, nuovo_row, z_val))
                        logger.debug(f"Punto alternativo trovato: ({x}, {nuovo_row}, {z_val})")
                else:
                    path.append((x, nuovo_row, z_val))

        logger.info(f"Percorso costruito con {len(path)} punti")
    elif args.only_visualization:
        # Carica il percorso esistente dal file JSON
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
                path = [tuple(p["posizione"]) for p in data["percorso"]]
                logger.info(f"Percorso caricato dal file: {len(path)} punti")
        except FileNotFoundError:
            logger.error(f"File del percorso non trovato: {output_file}")
            return
        except json.JSONDecodeError:
            logger.error(f"Errore nel parsing del file JSON: {output_file}")
            return

    # Visualizzazione 2D se richiesta
    if args.plot_2d and path:
        logger.info("Generazione visualizzazione 2D del percorso")
        plot_2d_path(path, z, lavorabile if not args.only_visualization else None)

    # Visualizzazione 3D
    if not args.skip_visualization:
        logger.info("Inizio visualizzazione della superficie e del percorso")

        # Mesh superficie
        x_vals = np.arange(n_cols)
        y_vals = np.arange(n_rows)
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)
        grid = pv.StructuredGrid()
        grid.points = np.c_[x_grid.ravel(), y_grid.ravel(), z.ravel()]
        grid.dimensions = [n_cols, n_rows, 1]

        # Visualizzazione
        plotter = pv.Plotter()
        plotter.add_mesh(grid, cmap="bone", opacity=0.5)

        if path and len(path) >= 2:
            # Crea il percorso completo con i punti di sollevamento
            percorso_completo = []
            punto_precedente = None
            
            for punto in path:
                if punto_precedente is not None:
                    # Calcola la distanza dal punto precedente
                    x, y, z_val = punto
                    x_prev, y_prev, z_prev = punto_precedente
                    dist = np.sqrt((x - x_prev)**2 + (y - y_prev)**2)
                    
                    if dist > 30:  # Se c'è un gap significativo
                        # Aggiungi punto di sollevamento
                        punto_up = genera_punto_sollevamento(punto_precedente)
                        percorso_completo.append(punto_precedente)
                        percorso_completo.append(punto_up)
                        
                        # Aggiungi punto di spostamento in alto
                        punto_over = (x, y, punto_up[2])
                        percorso_completo.append(punto_over)
                        
                        # Aggiungi punto di abbassamento
                        percorso_completo.append(punto)
                    else:
                        # Movimento normale sulla superficie
                        percorso_completo.append(punto)
                else:
                    percorso_completo.append(punto)
                
                punto_precedente = punto

            # Converti il percorso completo in array numpy per la visualizzazione
            percorso_array = np.array(percorso_completo)
            
            # Visualizza il percorso completo
            for i in range(len(percorso_array) - 1):
                p1 = percorso_array[i]
                p2 = percorso_array[i + 1]
                
                # Determina il colore in base al tipo di movimento
                if p1[2] == p2[2] and p1[2] > np.max(z):  # Movimento in alto
                    color = "red"
                elif p1[2] != p2[2]:  # Movimento verticale
                    color = "yellow"
                else:  # Movimento di lavorazione
                    color = "blue"
                
                segment = pv.lines_from_points(np.array([p1, p2]), close=False)
                plotter.add_mesh(segment, color=color, line_width=2)

            # Aggiungi i cerchi del passaggio utensile solo per i punti di lavorazione
            cerchi = []
            for punto in path:
                if punto[2] <= np.max(z):  # Solo per punti sulla superficie
                    cerchio = crea_cerchio_con_normale(punto, raggio_cm, z, n_punti=TOOL_PARAMS['n_punti_cerchio'])
                    cerchio = np.vstack([cerchio, cerchio[0]])
                    segments = np.array([[cerchio[i], cerchio[i + 1]] for i in range(len(cerchio) - 1)])
                    segments = segments.reshape(-1, 3)
                    cerchi.append(segments)

            if cerchi:
                tutti_segmenti = np.vstack(cerchi)
                plotter.add_lines(tutti_segmenti, color=TOOL_PARAMS['colore_deviazione'], width=1)

        plotter.add_axes()
        plotter.set_background("white")
        plotter.show(title="Percorso completo della levigatrice")
        logger.info("Visualizzazione completata")

    # Esportazione
    if not args.skip_export and not args.only_visualization and path:
        esporta_per_cobot(path, z, output_file)
        logger.info(f"Percorso esportato in {output_file}")
        logger.info(f"Numero di punti nel percorso: {len(path)}")

    # Processa con l'extended processor
    logger.info("Inizializzazione Extended Processor")
    processor = ExtendedProcessor(z, gradient, pendenza_max, raggio_cm)
    
    # Trova i bordi lavorabili
    logger.info("Analisi dei bordi degli ostacoli")
    bordi_lavorabili = processor.analizza_bordi_ostacoli()
    
    # Genera il percorso con rotazione della testina
    logger.info("Generazione percorso con gestione ostacoli")
    path_esteso = processor.genera_percorso_con_rotazione(path)
    
    # Statistiche
    n_punti_originali = len(path)
    n_punti_estesi = len(path_esteso)
    n_punti_rotazione = sum(1 for p in path_esteso if abs(p.get("c", 0)) > 0)
    
    logger.info(f"Punti nel percorso originale: {n_punti_originali}")
    logger.info(f"Punti nel percorso esteso: {n_punti_estesi}")
    logger.info(f"Punti con rotazione testina: {n_punti_rotazione}")
    
    # Salva il percorso in formato JSON
    if not args.skip_export:
        logger.info(f"Salvataggio percorso in {output_file}")
        
        # Converti il percorso nel formato richiesto
        routes = []
        current_route = []
        
        for punto in path_esteso:
            pos = punto["posizione"]
            tipo = punto["tipo"]
            
            # Calcola i parametri a, b, c dal vettore normale alla superficie
            normale = punto["normale"]
            a = normale[0]
            b = normale[1]
            c = np.degrees(np.arccos(normale[2]))  # Converti in gradi
            
            # Calcola il parametro g (qualità della superficie)
            g = 1.0 if tipo == "lavorazione" else 0.0
            
            punto_json = {
                "a": float(a),
                "b": float(b),
                "c": float(c),
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
                "g": float(g)
            }
            
            # Aggiungi l'angolo della testina per mantenere compatibilità
            punto_json["angolo_testina"] = float(c)
            
            current_route.append(punto_json)
            
            # Se è un punto di sollevamento, inizia una nuova route
            if tipo == "sollevamento" and len(current_route) > 0:
                routes.append(current_route)
                current_route = []
        
        # Aggiungi l'ultima route se non vuota
        if current_route:
            routes.append(current_route)
        
        # Crea il dizionario finale
        output_data = {
            "measures": {
                "w": float(n_cols),
                "h": float(n_rows)
            },
            "circles_r": float(raggio_cm),
            "circles_r_in": float(raggio_cm * 2/3),  # Raggio interno come 2/3 del raggio esterno
            "routes": routes
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        logger.info(f"Percorso salvato con {len(routes)} route e {sum(len(r) for r in routes)} punti totali")

    # Visualizzazioni
    if not args.skip_visualization:
        logger.info("Generazione visualizzazione 2D")
        visualizza_2d(z, bordi_lavorabili, 
                     [p["posizione"] for p in path_esteso], 
                     [p["c"] for p in path_esteso])
        
        logger.info("Generazione visualizzazione 3D")
        visualizza_3d(z, bordi_lavorabili, 
                     [p["posizione"] for p in path_esteso],
                     [p["c"] for p in path_esteso],
                     raggio_cm)

if __name__ == "__main__":
    main() 