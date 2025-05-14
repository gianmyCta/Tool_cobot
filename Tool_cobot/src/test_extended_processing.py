import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.ndimage import gaussian_gradient_magnitude, gaussian_filter
from extended_processing import ExtendedProcessor
import logging
import argparse
from config import *

def setup_logging(debug=False):
    """Configura il logger."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def visualizza_2d(superficie, bordi_lavorabili, path_originale, path_esteso):
    """Visualizza i risultati del processing esteso in 2D."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Prima visualizzazione: superficie con bordi lavorabili
    im1 = ax1.imshow(superficie, cmap='viridis')
    plt.colorbar(im1, ax=ax1, label='Altezza')
    
    # Sovrapponi i bordi lavorabili
    mask = np.ma.masked_where(~bordi_lavorabili, np.ones_like(bordi_lavorabili))
    ax1.imshow(mask, cmap='autumn', alpha=0.5)
    
    # Visualizza il percorso originale
    if path_originale:
        x_orig, y_orig, _ = zip(*path_originale)
        ax1.plot(x_orig, y_orig, 'b.-', label='Percorso Originale', linewidth=1, markersize=2)
    
    ax1.set_title('Superficie con Bordi Lavorabili')
    ax1.legend()
    
    # Seconda visualizzazione: confronto percorsi
    im2 = ax2.imshow(superficie, cmap='viridis')
    plt.colorbar(im2, ax=ax2, label='Altezza')
    
    # Visualizza entrambi i percorsi per confronto
    if path_originale:
        x_orig, y_orig, _ = zip(*path_originale)
        ax2.plot(x_orig, y_orig, 'b.-', label='Originale', linewidth=1, markersize=2)
    
    if path_esteso:
        x_ext, y_ext, _ = zip(*path_esteso)
        ax2.plot(x_ext, y_ext, 'r.-', label='Esteso', linewidth=1, markersize=2)
    
    ax2.set_title('Confronto Percorsi')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def calcola_copertura(superficie, path, raggio_cm):
    """Calcola la mappa di copertura della superficie considerando il movimento continuo dell'utensile."""
    h, w = superficie.shape
    copertura = np.zeros((h, w), dtype=int)
    
    if not path:
        return copertura
    
    # Per ogni segmento del percorso
    for i in range(len(path)-1):
        p1 = path[i]
        p2 = path[i+1]
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        
        # Calcola i punti intermedi del movimento
        # Usa più punti per movimenti più lunghi
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        n_points = max(int(dist * 2), 10)  # Almeno 10 punti per segmento
        
        # Genera punti intermedi
        xs = np.linspace(x1, x2, n_points)
        ys = np.linspace(y1, y2, n_points)
        zs = np.linspace(z1, z2, n_points)
        
        # Per ogni punto del movimento
        for x, y, z in zip(xs, ys, zs):
            x_int, y_int = int(round(x)), int(round(y))
            
            # Calcola l'area di contatto dell'utensile
            y_min = max(0, int(y - raggio_cm))
            y_max = min(h, int(y + raggio_cm + 1))
            x_min = max(0, int(x - raggio_cm))
            x_max = min(w, int(x + raggio_cm + 1))
            
            # Per ogni punto nell'area potenziale di contatto
            for yi in range(y_min, y_max):
                for xi in range(x_min, x_max):
                    # Verifica se il punto è all'interno del cerchio dell'utensile
                    if (xi - x)**2 + (yi - y)**2 <= raggio_cm**2:
                        # Calcola la distanza verticale tra utensile e superficie
                        z_superficie = superficie[yi, xi]
                        # L'utensile si adatta alla superficie entro il suo raggio
                        z_utensile = z + np.sqrt(raggio_cm**2 - ((xi-x)**2 + (yi-y)**2))
                        
                        # Se l'utensile può raggiungere questo punto
                        if abs(z_utensile - z_superficie) <= raggio_cm:
                            copertura[yi, xi] = 1
    
    return copertura

def visualizza_3d(superficie, bordi_lavorabili, path_originale, path_esteso, raggio_cm):
    """Visualizza i risultati in 3D usando PyVista."""
    # Crea la griglia strutturata per la superficie
    n_rows, n_cols = superficie.shape
    x = np.arange(n_cols)
    y = np.arange(n_rows)
    x_grid, y_grid = np.meshgrid(x, y)
    
    grid = pv.StructuredGrid()
    grid.points = np.c_[x_grid.ravel(), y_grid.ravel(), superficie.ravel()]
    grid.dimensions = [n_cols, n_rows, 1]
    
    # Calcola le aree di copertura
    copertura_originale = calcola_copertura(superficie, path_originale, raggio_cm)
    copertura_estesa = calcola_copertura(superficie, path_esteso, raggio_cm) if path_esteso else copertura_originale
    
    # Crea una maschera di colori
    colori = np.zeros((n_rows * n_cols, 3))
    colori_piatti = (copertura_estesa - copertura_originale).ravel()
    
    # Grigio per aree non coperte
    colori[copertura_originale.ravel() == 0] = [0.7, 0.7, 0.7]
    # Verde per aree coperte dal percorso originale
    colori[copertura_originale.ravel() == 1] = [0.0, 0.8, 0.0]
    # Blu per aree coperte solo dal percorso esteso
    colori[colori_piatti == 1] = [0.0, 0.4, 0.8]
    
    # Crea il plotter
    plotter = pv.Plotter()
    
    # Aggiungi la superficie base con i colori di copertura
    grid.point_data['colors'] = colori
    plotter.add_mesh(grid, scalars='colors', rgb=True)
    
    # Visualizza il percorso come una superficie continua dell'utensile
    if path_originale:
        # Crea una "scia" dell'utensile
        path_points = []
        path_faces = []
        point_count = 0
        
        for i in range(len(path_originale)-1):
            p1 = path_originale[i]
            p2 = path_originale[i+1]
            x1, y1, z1 = p1
            x2, y2, z2 = p2
            
            # Crea un cilindro tra i punti
            vector = np.array([x2-x1, y2-y1, z2-z1])
            length = np.linalg.norm(vector)
            
            # Crea punti intermedi per il cilindro
            n_circles = max(int(length * 2), 4)
            t = np.linspace(0, 1, n_circles)
            
            for ti in t:
                # Posizione corrente
                x = x1 + ti * (x2-x1)
                y = y1 + ti * (y2-y1)
                z = z1 + ti * (z2-z1)
                
                # Crea cerchio dell'utensile
                theta = np.linspace(0, 2*np.pi, 8)
                for th in theta:
                    dx = raggio_cm * np.cos(th)
                    dy = raggio_cm * np.sin(th)
                    path_points.append([x+dx, y+dy, z])
                
                if ti > 0:
                    # Collega i cerchi consecutivi
                    for j in range(8):
                        j_next = (j + 1) % 8
                        idx1 = point_count + j
                        idx2 = point_count + j_next
                        idx3 = point_count - 8 + j_next
                        idx4 = point_count - 8 + j
                        path_faces.extend([[4, idx1, idx2, idx3, idx4]])
                
                point_count += 8
        
        # Crea la mesh della scia
        path_points = np.array(path_points)
        path_mesh = pv.PolyData(path_points)
        path_mesh.faces = np.hstack(path_faces)
        
        plotter.add_mesh(path_mesh, color='yellow', opacity=0.3, label='Movimento Utensile')
    
    # Configura la visualizzazione
    plotter.add_axes()
    plotter.add_legend()
    plotter.show_grid()
    plotter.set_background('white')
    
    # Aggiungi statistiche
    area_totale = n_rows * n_cols
    area_coperta_orig = np.sum(copertura_originale)
    area_coperta_ext = np.sum(copertura_estesa)
    
    stats_text = [
        f"Area Totale: {area_totale}",
        f"Copertura Originale: {area_coperta_orig} ({area_coperta_orig/area_totale*100:.1f}%)",
        f"Copertura Estesa: {area_coperta_ext} ({area_coperta_ext/area_totale*100:.1f}%)",
        f"Area Aggiuntiva: {area_coperta_ext - area_coperta_orig}"
    ]
    
    plotter.add_text("\n".join(stats_text), position='upper_left', font_size=12)
    
    # Mostra la visualizzazione
    plotter.show()

def main():
    parser = argparse.ArgumentParser(description="Test del processing esteso per la levigatura.")
    parser.add_argument('--input', type=str, default=INPUT_FILES['output'], help='File CSV della superficie')
    parser.add_argument('--diametro', type=float, default=TOOL_PARAMS['diametro_default'], help='Diametro dell\'utensile in cm')
    parser.add_argument('--step', type=int, default=50, help='Passo del percorso in cm (default: 50)')
    parser.add_argument('--pendenza', type=float, default=TOOL_PARAMS['pendenza_max'], help='Pendenza massima accettabile (default: 0.7)')
    parser.add_argument('--debug', action='store_true', help='Attiva log di debug')
    parser.add_argument('--no-2d', action='store_true', help='Salta la visualizzazione 2D')
    parser.add_argument('--no-3d', action='store_true', help='Salta la visualizzazione 3D')
    parser.add_argument('--downsample', type=int, default=4, help='Fattore di riduzione della risoluzione (default: 4)')
    
    args = parser.parse_args()
    logger = setup_logging(args.debug)
    
    # Carica e prepara i dati
    logger.info(f"Caricamento superficie da {args.input}")
    superficie_full = pd.read_csv(args.input, header=None, skiprows=1, delimiter=',').to_numpy()
    
    # Downsampling della superficie
    if args.downsample > 1:
        logger.info(f"Riduzione risoluzione di un fattore {args.downsample}")
        superficie = superficie_full[::args.downsample, ::args.downsample]
    else:
        superficie = superficie_full
    
    logger.info(f"Dimensioni superficie: {superficie.shape}")
    
    # Calcola il gradiente
    z_smooth = gaussian_filter(superficie, sigma=2)
    gradient = gaussian_gradient_magnitude(z_smooth, sigma=1)
    
    # Parametri
    raggio_cm = args.diametro / 2
    pendenza_max = args.pendenza
    
    # Crea un percorso di test semplice (rettilineo)
    n_rows, n_cols = superficie.shape
    step = args.step
    path_originale = []
    
    # Crea un percorso a serpentina semplice
    for col in range(step, n_cols-step, step):
        if (col // step) % 2 == 0:
            # Movimento verso il basso
            for row in range(step, n_rows-step, step):
                path_originale.append((col, row, superficie[row, col]))
        else:
            # Movimento verso l'alto
            for row in range(n_rows-step-1, step-1, -step):
                path_originale.append((col, row, superficie[row, col]))
    
    # Processa con l'extended processor
    logger.info("Inizializzazione Extended Processor")
    processor = ExtendedProcessor(superficie, gradient, pendenza_max, raggio_cm)
    
    # Trova i bordi lavorabili
    logger.info("Analisi dei bordi degli ostacoli")
    bordi_lavorabili = processor.analizza_bordi_ostacoli()
    
    # Genera il percorso esteso
    logger.info("Generazione percorso esteso")
    path_esteso = processor.genera_percorso_esteso(path_originale)
    
    # Statistiche
    n_punti_originali = len(path_originale)
    n_punti_estesi = len(path_esteso)
    n_bordi = np.sum(bordi_lavorabili)
    
    logger.info(f"Punti nel percorso originale: {n_punti_originali}")
    logger.info(f"Punti nel percorso esteso: {n_punti_estesi}")
    logger.info(f"Punti di bordo lavorabili trovati: {n_bordi}")
    logger.info(f"Punti aggiunti al percorso: {n_punti_estesi - n_punti_originali}")
    
    # Visualizzazioni
    if not args.no_2d:
        logger.info("Generazione visualizzazione 2D")
        visualizza_2d(superficie, bordi_lavorabili, path_originale, path_esteso)
    
    if not args.no_3d:
        logger.info("Generazione visualizzazione 3D")
        visualizza_3d(superficie, bordi_lavorabili, path_originale, path_esteso, raggio_cm)

def test_rotazione_ostacoli():
    """Test per visualizzare il comportamento della testina intorno agli ostacoli."""
    # Crea una superficie di test con un ostacolo inclinato
    n_rows, n_cols = 50, 50
    superficie = np.zeros((n_rows, n_cols))
    
    # Crea un ostacolo inclinato di 45 gradi
    for i in range(20, 30):
        for j in range(20, 30):
            superficie[i, j] = (i - 20) * 0.5  # Crea una pendenza di 45 gradi
    
    # Calcola il gradiente
    z_smooth = gaussian_filter(superficie, sigma=2)
    gradient = gaussian_gradient_magnitude(z_smooth, sigma=1)
    
    # Parametri
    pendenza_max = 45  # gradi
    raggio_cm = 3
    
    # Crea il processor
    processor = ExtendedProcessor(superficie, gradient, pendenza_max, raggio_cm)
    
    # Crea un percorso che attraversa l'ostacolo
    path_originale = []
    for x in range(15, 35):
        path_originale.append((x, 25, superficie[25, x]))
    
    # Genera il percorso con rotazione
    path_esteso = processor.genera_percorso_con_rotazione(path_originale)
    
    # Visualizza i risultati in 3D
    plotter = pv.Plotter()
    
    # Crea la griglia strutturata per la superficie
    x = np.arange(n_cols)
    y = np.arange(n_rows)
    x_grid, y_grid = np.meshgrid(x, y)
    
    grid = pv.StructuredGrid()
    grid.points = np.c_[x_grid.ravel(), y_grid.ravel(), superficie.ravel()]
    grid.dimensions = [n_cols, n_rows, 1]
    
    # Colora la superficie in base agli ostacoli
    ostacoli = processor.ostacoli
    colori = np.zeros((n_rows * n_cols, 3))
    colori[~ostacoli.ravel()] = [0.7, 0.7, 0.7]  # Grigio per aree normali
    colori[ostacoli.ravel()] = [0.8, 0.0, 0.0]   # Rosso per ostacoli
    
    # Aggiungi la superficie
    grid.point_data['colors'] = colori
    plotter.add_mesh(grid, scalars='colors', rgb=True, label='Superficie')
    
    # Crea una matrice per tracciare l'area lavorata
    area_lavorata = np.zeros((n_rows, n_cols))
    
    # Per ogni punto del percorso, calcola e visualizza l'area di contatto
    for i, punto in enumerate(path_esteso):
        pos = punto["posizione"]
        normale = punto["normale"]
        punto_contatto = punto["punto_contatto"]
        
        # Crea un cilindro per rappresentare la testina
        center = np.array(pos)
        direction = -normale  # Negativo perché vogliamo che punti verso la superficie
        cylinder = pv.Cylinder(center=center, direction=direction, radius=raggio_cm/2, height=raggio_cm)
        plotter.add_mesh(cylinder, color='blue', opacity=0.3)
        
        # Calcola l'area di contatto sulla superficie
        x_cont, y_cont = int(punto_contatto[0]), int(punto_contatto[1])
        for dy in range(-int(raggio_cm), int(raggio_cm) + 1):
            for dx in range(-int(raggio_cm), int(raggio_cm) + 1):
                x_check = x_cont + dx
                y_check = y_cont + dy
                if (0 <= x_check < n_cols and 0 <= y_check < n_rows and
                    dx*dx + dy*dy <= raggio_cm*raggio_cm):
                    # Marca questo punto come lavorato
                    area_lavorata[y_check, x_check] = 1
                    
                    # Crea un disco piatto sulla superficie per rappresentare l'area di contatto
                    z_check = superficie[y_check, x_check]
                    center = [x_check, y_check, z_check]
                    # Crea un poligono circolare
                    theta = np.linspace(0, 2*np.pi, 20)
                    radius = 0.2
                    points = np.zeros((20, 3))
                    for j, angle in enumerate(theta):
                        # Calcola i punti del cerchio nel piano perpendicolare alla normale
                        # Usa due vettori perpendicolari alla normale per definire il piano
                        if abs(normale[2]) < 0.9:
                            v1 = np.cross(normale, [0, 0, 1])
                        else:
                            v1 = np.cross(normale, [1, 0, 0])
                        v1 = v1 / np.linalg.norm(v1)
                        v2 = np.cross(normale, v1)
                        v2 = v2 / np.linalg.norm(v2)
                        
                        points[j] = center + radius * (v1 * np.cos(angle) + v2 * np.sin(angle))
                    
                    # Crea il poligono
                    poly = pv.PolyData(points)
                    # Aggiungi le celle per formare il poligono
                    poly.lines = np.array([20] + list(range(20)) + [0])
                    plotter.add_mesh(poly, color='green', opacity=0.5)
        
        # Se non è l'ultimo punto, visualizza il percorso fino al prossimo punto
        if i < len(path_esteso) - 1:
            next_punto = path_esteso[i + 1]["punto_contatto"]
            line = pv.Line(punto_contatto, next_punto)
            plotter.add_mesh(line, color='yellow', line_width=2)
    
    # Configura la visualizzazione
    plotter.add_axes()
    plotter.show_grid()
    plotter.set_background('white')
    
    # Mostra statistiche sulla copertura
    area_totale = n_rows * n_cols
    area_lavorata_totale = np.sum(area_lavorata)
    percentuale_copertura = (area_lavorata_totale / area_totale) * 100
    
    stats_text = [
        f"Area Totale: {area_totale}",
        f"Area Lavorata: {area_lavorata_totale:.0f}",
        f"Copertura: {percentuale_copertura:.1f}%"
    ]
    plotter.add_text("\n".join(stats_text), position='upper_left', font_size=12)
    
    # Mostra la visualizzazione
    plotter.show()

if __name__ == "__main__":
    main()  # Esegue il test sulla superficie reale invece della superficie simulata 