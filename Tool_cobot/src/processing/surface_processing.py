import numpy as np
import pandas as pd
from config import *

def export_to_ply(surface_df, output_file):
    """
    Esporta la superficie come file PLY per Blender
    
    Parameters:
    -----------
    surface_df : pd.DataFrame
        DataFrame contenente i dati della superficie
    output_file : str
        Percorso del file di output .ply
    """
    print("\nEsportazione della superficie in formato PLY...")
    
    # Ottieni le dimensioni della superficie
    rows, cols = surface_df.shape
    
    # Crea la griglia di coordinate
    x = np.arange(cols) * (SURFACE_WIDTH / cols)  # Scala in mm
    y = np.arange(rows) * (SURFACE_HEIGHT / rows)  # Scala in mm
    X, Y = np.meshgrid(x, y)
    
    # Prepara i vertici
    vertices = []
    for i in range(rows):
        for j in range(cols):
            vertices.append(f"{X[i,j]} {Y[i,j]} {surface_df.iloc[i,j]}")
    
    # Crea le facce (triangoli)
    faces = []
    for i in range(rows-1):
        for j in range(cols-1):
            # Indici dei vertici per i due triangoli che formano un quad
            v1 = i * cols + j
            v2 = v1 + 1
            v3 = (i + 1) * cols + j
            v4 = v3 + 1
            # Primo triangolo
            faces.append(f"3 {v1} {v2} {v4}")
            # Secondo triangolo
            faces.append(f"3 {v1} {v4} {v3}")
    
    # Scrivi il file PLY
    with open(output_file, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        # Vertici
        for vertex in vertices:
            f.write(f"{vertex}\n")
        
        # Facce
        for face in faces:
            f.write(f"{face}\n")
    
    print(f"Superficie esportata in: {output_file}")
    print(f"Numero di vertici: {len(vertices)}")
    print(f"Numero di facce: {len(faces)}")

def clean_surface_data_vectorized(surface_df):
    """Pulisce i dati della superficie rimuovendo i NaN tramite interpolazione"""
    print("Dimensioni originali:", surface_df.shape)
    
    # Crea una copia per non modificare l'originale
    cleaned_df = surface_df.copy()
    
    # Conta i NaN prima della pulizia
    nan_count_before = cleaned_df.isna().sum().sum()
    print(f"Valori NaN presenti: {nan_count_before}")
    
    if nan_count_before > 0:
        print("Interpolazione dei valori mancanti...")
        # Interpolazione lineare per colonne
        cleaned_df = cleaned_df.interpolate(method='linear', axis=0, limit_direction='both')
        # Interpolazione lineare per righe
        cleaned_df = cleaned_df.interpolate(method='linear', axis=1, limit_direction='both')
        
        # Verifica finale
        nan_count_after = cleaned_df.isna().sum().sum()
        if nan_count_after > 0:
            print(f"ATTENZIONE: Rimangono {nan_count_after} valori NaN dopo l'interpolazione")
        else:
            print("Tutti i valori NaN sono stati interpolati con successo")
    else:
        print("Nessun valore NaN presente nei dati")
    
    return cleaned_df

def find_best_alignment_auto(left_surface, right_surface, continuity_threshold=CONTINUITY_THRESHOLD, 
                           max_search_range=MAX_SEARCH_RANGE, min_search_range=MIN_SEARCH_RANGE, 
                           step=SEARCH_STEP):
    """
    Trova automaticamente il miglior punto di allineamento usando l'ostacolo come riferimento
    """
    print("\nRicerca automatica del miglior allineamento...")
    
    best_result = None
    min_total_error = float('inf')
    
    # Calcola il gradiente per identificare le aree con variazioni significative (ostacoli)
    left_gradient = np.gradient(left_surface.values)[0]
    right_gradient = np.gradient(right_surface.values)[0]
    
    # Crea maschere per le aree con variazioni significative
    gradient_threshold = np.percentile(np.abs(left_gradient), 95)  # Usa il 95° percentile come soglia
    left_features = np.abs(left_gradient) > gradient_threshold
    right_features = np.abs(right_gradient) > gradient_threshold
    
    for search_range in range(min_search_range, max_search_range + 1, step):
        print(f"\nProvo con search_range = {search_range}")
        
        min_error = float('inf')
        best_left_idx = -1
        best_right_idx = -1
        
        # Consideriamo una finestra di ricerca su entrambi i lati
        left_range = range(max(0, left_surface.shape[1] - search_range), left_surface.shape[1])
        right_range = range(0, min(search_range, right_surface.shape[1]))
        
        for left_idx in left_range:
            left_column = left_surface.iloc[:, left_idx].values
            left_feature = left_features[:, left_idx]
            
            for right_idx in right_range:
                right_column = right_surface.iloc[:, right_idx].values
                right_feature = right_features[:, right_idx]
                
                # Calcola l'errore dando più peso alle aree con features
                base_error = np.mean((left_column - right_column) ** 2)
                # Usa XOR per confrontare le features booleane
                feature_match = np.mean(left_feature ^ right_feature)
                
                # Combina gli errori dando più peso al match delle features
                error = base_error + feature_match * 5  # Peso maggiore per il match delle features
                
                if error < min_error:
                    min_error = error
                    best_left_idx = left_idx
                    best_right_idx = right_idx
        
        # Calcola gli offset
        left_offset = left_surface.shape[1] - best_left_idx - 1
        right_offset = best_right_idx
        
        # Verifica la continuità
        temp_left = left_surface.iloc[:, :-left_offset] if left_offset > 0 else left_surface
        temp_right = right_surface.iloc[:, right_offset:] if right_offset > 0 else right_surface
        
        left_edge = temp_left.iloc[:, -1].values
        right_edge = temp_right.iloc[:, 0].values
        
        # Calcola la differenza dando più peso alle aree con features
        edge_diff = np.abs(left_edge - right_edge)
        feature_edge_left = left_features[:, -1] if left_offset > 0 else left_features[:, -1]
        feature_edge_right = right_features[:, right_offset] if right_offset > 0 else right_features[:, 0]
        # Usa XOR anche qui per il confronto delle features ai bordi
        feature_diff = np.mean(feature_edge_left ^ feature_edge_right)
        
        max_diff = np.max(edge_diff)
        
        print(f"- Errore quadratico medio: {min_error:.4f}")
        print(f"- Massima differenza nella giunzione: {max_diff:.4f} mm")
        print(f"- Differenza nelle features: {feature_diff:.4f}")
        
        # Aggiorna il miglior risultato considerando sia la continuità che il match delle features
        if max_diff <= continuity_threshold * 1.2 and min_error < min_total_error:  # Permetti una tolleranza leggermente maggiore
            min_total_error = min_error
            best_result = (left_offset, right_offset, search_range, max_diff)
            print(f"✓ Trovata soluzione valida con search_range = {search_range}")
    
    if best_result is None:
        print("\nATTENZIONE: Non è stata trovata una soluzione che soddisfi completamente i criteri di continuità")
        return left_offset, right_offset
    else:
        left_offset, right_offset, best_range, best_diff = best_result
        print(f"\nMiglior allineamento trovato:")
        print(f"- Search range ottimale: {best_range}")
        print(f"- Rimuovere {left_offset} colonne dalla fine della superficie sinistra")
        print(f"- Rimuovere {right_offset} colonne dall'inizio della superficie destra")
        print(f"- Massima differenza nella giunzione: {best_diff:.4f} mm")
        return left_offset, right_offset

def check_continuity(left_surface, right_surface, threshold=CONTINUITY_THRESHOLD):
    """
    Verifica la continuità tra le due fasce
    
    Parameters:
    -----------
    left_surface : pd.DataFrame
        Superficie sinistra
    right_surface : pd.DataFrame
        Superficie destra
    threshold : float, optional
        Differenza massima accettabile tra i punti di giunzione (mm)
    """
    print("\nVerifica continuità tra le fasce...")
    
    left_edge = left_surface.iloc[:, -1].values
    right_edge = right_surface.iloc[:, 0].values
    
    differences = np.abs(left_edge - right_edge)
    max_diff = np.max(differences)
    mean_diff = np.mean(differences)
    
    print(f"Differenza massima tra le fasce: {max_diff:.2f} mm")
    print(f"Differenza media tra le fasce: {mean_diff:.2f} mm")
    
    discontinuities = np.where(differences > threshold)[0]
    if len(discontinuities) > 0:
        print(f"Trovate {len(discontinuities)} discontinuità significative (>{threshold} mm)")
        print(f"Posizioni delle discontinuità: {discontinuities}")
        return False
    else:
        print("Nessuna discontinuità significativa trovata")
        return True

def crop_surface(surface_df, crop_margins, use_mm=True):
    """Ritaglia i bordi della superficie"""
    if not isinstance(crop_margins, dict):
        raise ValueError("crop_margins deve essere un dizionario con chiavi 'top', 'bottom', 'left', 'right'")
    
    required_keys = ['top', 'bottom', 'left', 'right']
    if not all(key in crop_margins for key in required_keys):
        raise ValueError(f"crop_margins deve contenere tutte le chiavi: {required_keys}")
    
    rows, cols = surface_df.shape
    print(f"Dimensioni originali: {rows}x{cols}")
    
    if use_mm:
        # Calcola il fattore di conversione mm -> pixel
        mm_per_pixel_x = SURFACE_WIDTH / cols
        mm_per_pixel_y = SURFACE_HEIGHT / rows
        
        crop_pixels = {
            'top': int(crop_margins['top'] / mm_per_pixel_y),
            'bottom': int(crop_margins['bottom'] / mm_per_pixel_y),
            'left': int(crop_margins['left'] / mm_per_pixel_x),
            'right': int(crop_margins['right'] / mm_per_pixel_x)
        }
        
        print("\nConversione mm -> pixel:")
        print(f"Risoluzione: {mm_per_pixel_x:.2f}mm/pixel (X), {mm_per_pixel_y:.2f}mm/pixel (Y)")
        print(f"Crop in mm: {crop_margins}")
        print(f"Crop in pixel: {crop_pixels}")
    else:
        crop_pixels = crop_margins
    
    cropped_df = surface_df.iloc[
        crop_pixels['top']:rows-crop_pixels['bottom'],
        crop_pixels['left']:cols-crop_pixels['right']
    ]
    
    print(f"Dimensioni dopo il crop: {cropped_df.shape[0]}x{cropped_df.shape[1]}")
    return cropped_df 