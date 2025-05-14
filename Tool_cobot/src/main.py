#unione delle superfici e allineamento automatico in base alla continuità

import numpy as np
import pandas as pd
from time import time
import os
from config import *
from processing.surface_processing import (
    clean_surface_data_vectorized,
    find_best_alignment_auto,
    check_continuity,
    crop_surface,
    export_to_ply
)
from visualization.visualizer import create_point_cloud, visualize_point_clouds, visualizza_2d

def merge_surfaces(top_file, bottom_file, output_file, crop_margins=CROP_MARGINS, 
                  continuity_threshold=CONTINUITY_THRESHOLD, 
                  max_search_range=MAX_SEARCH_RANGE, 
                  show_visualization=True, use_mm=True,
                  export_ply=False):
    
    start_time = time()
    
    print("Lettura del file superficie superiore...")
    left_surface = pd.read_csv(top_file, header=None, skiprows=1)
    
    print("Lettura del file superficie inferiore...")
    right_surface = pd.read_csv(bottom_file, header=None, skiprows=1)
    
    # Pulisce i dati da NaN
    print("\nPulizia superficie sinistra...")
    left_surface = clean_surface_data_vectorized(left_surface)
    
    print("\nPulizia superficie destra...")
    right_surface = clean_surface_data_vectorized(right_surface)
    
    # Applica il crop se specificato
    if crop_margins is not None:
        print("\nApplico il crop alla superficie sinistra...")
        left_surface = crop_surface(left_surface, crop_margins, use_mm)
        
        print("\nApplico il crop alla superficie destra...")
        right_surface = crop_surface(right_surface, crop_margins, use_mm)
    
    # Trova il miglior punto di allineamento automaticamente
    left_offset, right_offset = find_best_alignment_auto(
        left_surface, right_surface,
        continuity_threshold=continuity_threshold,
        max_search_range=max_search_range
    )
    
    # Ritaglia le superfici secondo il miglior allineamento
    left_surface = left_surface.iloc[:, :-left_offset] if left_offset > 0 else left_surface
    right_surface = right_surface.iloc[:, right_offset:] if right_offset > 0 else right_surface
    
    # Verifica finale della continuità
    is_continuous = check_continuity(left_surface, right_surface, continuity_threshold)
    if not is_continuous:
        print("\nATTENZIONE: Rilevate discontinuità significative tra le fasce dopo l'allineamento!")
    
    print("\nUnione delle superfici...")
    # Verifica che le dimensioni delle righe corrispondano
    if left_surface.shape[0] != right_surface.shape[0]:
        raise ValueError(f"Le superfici hanno altezze diverse: {left_surface.shape[0]} vs {right_surface.shape[0]}")
    
    # Concatena le superfici orizzontalmente
    complete_surface = pd.DataFrame(
        np.hstack([left_surface.values, right_surface.values])
    )
    
    print("\nVerifica finale dei NaN...")
    nan_count = complete_surface.isna().sum().sum()
    if nan_count > 0:
        print(f"Attenzione: ci sono ancora {nan_count} valori NaN nella superficie completa")
    else:
        print("Nessun valore NaN presente nella superficie completa")
    
    print("\nSalvataggio della superficie completa...")
    # Trasposizione finale prima del salvataggio per mantenere il formato originale
    complete_surface = complete_surface.T
    complete_surface.to_csv(output_file, index=False, header=False)
    print(f"Superficie completa salvata in {output_file}")
    
    # Esporta in formato PLY per Blender se richiesto
    if export_ply:
        ply_file = os.path.splitext(output_file)[0] + '.ply'
        export_to_ply(complete_surface, ply_file)
    
    if show_visualization:
        print("\nVisualizzazione 2D della superficie...")
        visualizza_2d(complete_surface, None, None, None)
    
    end_time = time()
    print(f"\nTempo di esecuzione: {end_time - start_time:.2f} secondi")
    
    print("\nInformazioni sulle dimensioni:")
    print(f"Dimensioni fascia sinistra originali: {left_surface.shape}")
    print(f"Dimensioni fascia destra originali: {right_surface.shape}")
    print(f"Dimensioni superficie completa: {complete_surface.shape}")
    print(f"Area totale ricostruita: {complete_surface.shape[0]} x {complete_surface.shape[1]} pixels")

if __name__ == "__main__":
    merge_surfaces(INPUT_FILES['top'], 
                  INPUT_FILES['bottom'], 
                  INPUT_FILES['output'],
                  crop_margins=CROP_MARGINS,
                  continuity_threshold=CONTINUITY_THRESHOLD,
                  max_search_range=MAX_SEARCH_RANGE,
                  show_visualization=True,
                  use_mm=True,
                  export_ply=False) 