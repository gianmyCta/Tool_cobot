import numpy as np
import json
from collections import deque
import logging

def cerchio_valido(row, col, lavorabile, raggio_cm):
    """Verifica se un cerchio centrato in (row, col) è completamente in area lavorabile."""
    h, w = lavorabile.shape
    
    # Calcola i limiti della finestra considerando i bordi
    row_min = max(0, int(row - raggio_cm))
    row_max = min(h, int(row + raggio_cm + 1))
    col_min = max(0, int(col - raggio_cm))
    col_max = min(w, int(col + raggio_cm + 1))
    
    # Se la finestra è troppo vicina ai bordi, ritorna False
    if row_min >= row_max or col_min >= col_max:
        return False
    
    # Crea una griglia di coordinate relative al centro della finestra effettiva
    window_height = row_max - row_min
    window_width = col_max - col_min
    y, x = np.ogrid[:window_height, :window_width]
    
    # Sposta il centro della griglia
    center_y = row - row_min
    center_x = col - col_min
    
    # Calcola la maschera circolare
    mask = (x - center_x)**2 + (y - center_y)**2 <= raggio_cm**2
    
    # Estrai la finestra di interesse e applica la maschera circolare
    window = lavorabile[row_min:row_max, col_min:col_max]
    
    return np.all(window[mask])

def trova_vicino_valido_vettoriale(row, col, lavorabile, raggio_cm):
    """Trova il punto valido più vicino usando operazioni vettoriali."""
    h, w = lavorabile.shape
    max_search = int(raggio_cm) + 1
    
    # Crea una griglia di offset
    y, x = np.ogrid[-max_search:max_search+1, -max_search:max_search+1]
    distances = np.sqrt(x*x + y*y)
    
    # Ordina i punti per distanza
    valid_points = np.where(distances <= max_search)
    sorted_indices = np.argsort(distances[valid_points])
    y_offsets = valid_points[0][sorted_indices]
    x_offsets = valid_points[1][sorted_indices]
    
    # Verifica ogni punto in ordine di distanza
    for y_off, x_off in zip(y_offsets, x_offsets):
        new_row = row + y_off - max_search
        new_col = col + x_off - max_search
        
        if (0 <= new_row < h and 0 <= new_col < w and 
            cerchio_valido(new_row, new_col, lavorabile, raggio_cm)):
            return new_row
            
    return None

def segmento_sicuro(p1, p2, lavorabile, raggio_cm):
    """Verifica se il segmento tra due punti è completamente in area lavorabile."""
    x1, y1, _ = p1
    x2, y2, _ = p2
    
    # Calcola il numero di punti da verificare
    dx = x2 - x1
    dy = y2 - y1
    steps = max(abs(dx), abs(dy))
    
    if steps == 0:
        return cerchio_valido(int(y1), int(x1), lavorabile, raggio_cm)
    
    # Genera tutti i punti del segmento in modo vettoriale
    t = np.linspace(0, 1, int(steps) + 1)
    x = np.rint(x1 + dx * t).astype(int)
    y = np.rint(y1 + dy * t).astype(int)
    
    # Verifica tutti i punti
    for xi, yi in zip(x, y):
        if not cerchio_valido(yi, xi, lavorabile, raggio_cm):
            return False
    return True

def bfs_deviazione(start, end, lavorabile, raggio_cm, z):
    """Trova un percorso alternativo usando BFS quando il percorso diretto non è sicuro."""
    start = (int(start[0]), int(start[1]))
    end = (int(end[0]), int(end[1]))
    
    h, w = lavorabile.shape
    visited = set()
    queue = deque([(start, [start])])
    
    while queue:
        (x, y), path = queue.popleft()
        if (x, y) == end:
            return [(x, y, z[y, x]) for x, y in path]
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_x, next_y = x + dx, y + dy
            if (0 <= next_y < h and 0 <= next_x < w and 
                (next_x, next_y) not in visited and 
                cerchio_valido(next_y, next_x, lavorabile, raggio_cm)):
                visited.add((next_x, next_y))
                new_path = path + [(next_x, next_y)]
                queue.append(((next_x, next_y), new_path))
    
    return None

def calcola_normale(punto, z):
    """Calcola la normale alla superficie in un punto."""
    x, y = int(punto[0]), int(punto[1])
    h, w = z.shape
    
    if x > 0 and x < w-1 and y > 0 and y < h-1:
        dx = (z[y, x+1] - z[y, x-1]) / 2
        dy = (z[y+1, x] - z[y-1, x]) / 2
        normale = np.array([-dx, -dy, 1])
        return normale / np.linalg.norm(normale)
    
    return np.array([0, 0, 1])

def crea_cerchio_con_normale(punto, raggio_cm, z, n_punti=32):
    """Crea un cerchio orientato secondo la normale alla superficie."""
    x, y, z_val = punto
    normale = calcola_normale(punto, z)
    
    # Crea un cerchio sul piano XY
    theta = np.linspace(0, 2*np.pi, n_punti)
    cerchio = np.zeros((n_punti, 3))
    cerchio[:, 0] = raggio_cm * np.cos(theta)
    cerchio[:, 1] = raggio_cm * np.sin(theta)
    
    # Ruota il cerchio per allinearlo con la normale
    if not np.allclose(normale, [0, 0, 1]):
        v = np.cross([0, 0, 1], normale)
        s = np.linalg.norm(v)
        c = normale[2]  # coseno dell'angolo
        v_x = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + v_x + np.dot(v_x, v_x) * (1 - c) / (s * s)
        cerchio = np.dot(cerchio, R.T)
    
    # Trasla il cerchio alla posizione corretta
    cerchio[:, 0] += x
    cerchio[:, 1] += y
    cerchio[:, 2] += z_val
    
    return cerchio

def genera_punto_sollevamento(punto, altezza_sicurezza=50):
    """Genera un punto di sollevamento sopra il punto dato."""
    x, y, z = punto
    return (x, y, z - altezza_sicurezza)

def trova_aree_contigue(lavorabile, raggio_cm):
    """Trova le aree contigue nella superficie lavorabile."""
    rows, cols = lavorabile.shape
    visited = np.zeros_like(lavorabile, dtype=bool)
    aree_contigue = []
    
    def flood_fill_iterative(start_row, start_col):
        """Versione iterativa del flood fill."""
        if (not (0 <= start_row < rows and 0 <= start_col < cols) or
            visited[start_row, start_col] or 
            not cerchio_valido(start_row, start_col, lavorabile, raggio_cm)):
            return set()
        
        area = set()
        stack = [(start_row, start_col)]
        
        while stack:
            row, col = stack.pop()
            if (not (0 <= row < rows and 0 <= col < cols) or
                visited[row, col] or
                not cerchio_valido(row, col, lavorabile, raggio_cm)):
                continue
            
            visited[row, col] = True
            area.add((row, col))
            
            # Aggiungi i vicini allo stack
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                stack.append((row + dr, col + dc))
        
        return area
    
    # Cerca aree contigue
    for row in range(rows):
        for col in range(cols):
            if not visited[row, col] and cerchio_valido(row, col, lavorabile, raggio_cm):
                area = flood_fill_iterative(row, col)
                if area:
                    aree_contigue.append(area)
    
    return aree_contigue

def esporta_per_cobot(path, z, output_file):
    """Esporta il percorso in formato JSON per il cobot."""
    output_data = {
        "percorso": [],
        "metadata": {
            "numero_punti": len(path),
            "dimensioni_superficie": z.shape,
            "altezza_sicurezza": 50  # mm sopra la superficie
        }
    }
    
    # Trova le aree contigue lavorabili
    aree_contigue = trova_aree_contigue(np.ones_like(z, dtype=bool), 5)  # Usa un raggio appropriato
    punti_per_area = {}
    
    # Raggruppa i punti del percorso per area
    for punto in path:
        x, y, z_val = punto
        for i, area in enumerate(aree_contigue):
            if (int(x), int(y)) in area:
                if i not in punti_per_area:
                    punti_per_area[i] = []
                punti_per_area[i].append(punto)
                break
    
    # Genera il percorso area per area
    for area_idx in sorted(punti_per_area.keys()):
        punti_area = punti_per_area[area_idx]
        
        # Se non è la prima area, aggiungi un punto di sollevamento dall'ultima posizione
        if area_idx > 0 and output_data["percorso"]:
            ultimo_punto = output_data["percorso"][-1]["posizione"]
            punto_up = genera_punto_sollevamento([ultimo_punto[0], ultimo_punto[1], ultimo_punto[2]])
            output_data["percorso"].append({
                "posizione": [float(p) for p in punto_up],
                "normale": [0, 0, 1],
                "tipo": "sollevamento"
            })
        
        # Aggiungi un punto di avvicinamento per la nuova area
        if punti_area:
            primo_punto = punti_area[0]
            punto_down = genera_punto_sollevamento(primo_punto)
            output_data["percorso"].append({
                "posizione": [float(p) for p in punto_down],
                "normale": [0, 0, 1],
                "tipo": "avvicinamento"
            })
        
        # Aggiungi tutti i punti dell'area corrente
        for punto in punti_area:
            x, y, z_val = punto
            normale = calcola_normale(punto, z).tolist()
            punto_data = {
                "posizione": [float(x), float(y), float(z_val)],
                "normale": normale,
                "tipo": "lavorazione"
            }
            output_data["percorso"].append(punto_data)
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logging.info(f"Percorso esportato in {output_file}") 