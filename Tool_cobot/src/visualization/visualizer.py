import numpy as np
import open3d as o3d
from config import *
import colorsys
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude

def visualizza_2d(superficie, bordi_lavorabili=None, path=None, angoli_testina=None):
    """Visualizza i risultati in 2D."""
    plt.figure(figsize=(12, 8))
    
    # Calcola le dimensioni reali in millimetri
    rows, cols = superficie.shape
    mm_per_pixel_x = SURFACE_WIDTH / cols
    mm_per_pixel_y = SURFACE_HEIGHT / rows
    
    # Crea gli array per gli assi in millimetri
    x_mm = np.arange(cols) * mm_per_pixel_x
    y_mm = np.arange(rows) * mm_per_pixel_y
    
    # Visualizzazione della superficie con dimensioni reali
    im = plt.imshow(superficie, 
                   extent=[0, SURFACE_WIDTH, SURFACE_HEIGHT, 0],  # [left, right, bottom, top]
                   cmap='viridis',
                   aspect='equal')  # Mantiene le proporzioni reali
    
    plt.colorbar(im, label='Altezza (mm)')
    plt.xlabel('Larghezza (mm)')
    plt.ylabel('Altezza (mm)')
    plt.title('Superficie Completa')
    
    plt.tight_layout()
    plt.show()

def create_coordinate_frame(size=100, origin=[0, 0, 0]):
    """Crea un sistema di riferimento con assi colorati"""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)

def create_grid(width=SURFACE_WIDTH, height=SURFACE_HEIGHT, spacing=GRID_SPACING):
    """Crea una griglia di riferimento con linee ogni spacing millimetri"""
    # ... existing code ...

def generate_distinct_colors(n):
    """Genera n colori distinti usando HSV color space"""
    colors = []
    for i in range(n):
        hue = i / n
        # Mantieni saturazione e valore alti per colori vivaci
        color = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
        colors.append(color)
    return colors

def create_point_cloud_by_areas(surface_data, aree_contigue, title="Point Cloud"):
    """Crea una nuvola di punti 3D dalla superficie, colorata per aree"""
    print(f"\nCreazione point cloud con aree distinte: {title}")
    
    # Se surface_data è già un PointCloud, lo usiamo direttamente
    if isinstance(surface_data, o3d.geometry.PointCloud):
        pcd = surface_data
        points = np.asarray(pcd.points)
        rows = int(np.sqrt(len(points)))  # Assumiamo una griglia quadrata
        cols = rows
    else:
        # Calcola il fattore di scala per convertire in mm
        rows, cols = surface_data.shape
        mm_per_pixel_x = SURFACE_WIDTH / cols
        mm_per_pixel_y = SURFACE_HEIGHT / rows
        
        # Crea la griglia in millimetri
        x = np.arange(cols) * mm_per_pixel_x
        y = np.arange(rows) * mm_per_pixel_y
        X, Y = np.meshgrid(x, y)
        
        # Crea i punti 3D
        points = np.zeros((rows * cols, 3))
        points[:, 0] = X.flatten()
        points[:, 1] = Y.flatten()
        points[:, 2] = surface_data.values.flatten()
        
        # Crea la nuvola di punti
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    
    # Calcola le normali
    pcd.estimate_normals()
    pcd.normalize_normals()
    
    # Genera colori distinti per ogni area
    n_aree = len(aree_contigue)
    area_colors = generate_distinct_colors(n_aree)
    
    # Crea un array di colori per tutti i punti
    colors = np.zeros((rows * cols, 3))
    # Colore di default per punti non in aree (grigio chiaro)
    colors[:] = [0.7, 0.7, 0.7]
    
    # Assegna colori diversi per ogni area
    for i, area in enumerate(aree_contigue):
        for x, y in area:
            if 0 <= y < rows and 0 <= x < cols:  # Verifica che gli indici siano validi
                idx = y * cols + x
                if idx < len(colors):  # Verifica che l'indice sia valido
                    colors[idx] = area_colors[i]
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def create_path_lines(path, color=[0, 1, 0]):
    """Crea una rappresentazione visiva del percorso"""
    points = []
    lines = []
    colors = []
    
    for i in range(len(path)):
        punto = path[i]["posizione"]
        points.append(punto)
        
        if i > 0:
            lines.append([i-1, i])
            # Colore in base al tipo di movimento
            if path[i]["tipo"] == "lavorazione":
                colors.append([0, 1, 0])  # Verde per lavorazione
            elif path[i]["tipo"] == "sollevamento":
                colors.append([1, 0, 0])  # Rosso per sollevamento
            elif path[i]["tipo"] == "avvicinamento":
                colors.append([0, 0, 1])  # Blu per avvicinamento
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set

def visualize_surface_and_path(surface_data, path, aree_contigue):
    """Visualizza la superficie con aree colorate e il percorso"""
    # Configura la visualizzazione
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Surface and Path Viewer", 
                     width=WINDOW_WIDTH, height=WINDOW_HEIGHT,
                     left=WINDOW_LEFT, top=WINDOW_TOP)
    
    # Crea e aggiungi la nuvola di punti colorata per aree
    pcd = create_point_cloud_by_areas(surface_data, aree_contigue)
    vis.add_geometry(pcd)
    
    # Crea e aggiungi il percorso
    if path:
        path_lines = create_path_lines(path)
        vis.add_geometry(path_lines)
    
    # Aggiungi la griglia
    grid = create_grid()
    vis.add_geometry(grid)
    
    # Aggiungi il sistema di riferimento
    coordinate_frame = create_coordinate_frame(size=GRID_SPACING)
    vis.add_geometry(coordinate_frame)
    
    # Configura le opzioni di rendering
    render_option = vis.get_render_option()
    render_option.point_size = POINT_SIZE
    render_option.background_color = np.asarray(BACKGROUND_COLOR)
    render_option.show_coordinate_frame = True
    
    # Configura la vista
    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)
    view_control.set_front([0, 0, -1])
    view_control.set_lookat([SURFACE_WIDTH/2, SURFACE_HEIGHT/2, 0])
    view_control.set_up([0, -1, 0])
    
    # Avvia la visualizzazione
    vis.run()
    vis.destroy_window()

def visualize_point_clouds(pcds, titles):
    """Visualizza le nuvole di punti con assi e griglia"""
    # Manteniamo questa funzione per compatibilità
    visualize_surface_and_path(pcds[0], None, [])

def create_point_cloud(surface_data, title="Point Cloud"):
    """Crea una nuvola di punti 3D dalla superficie"""
    print(f"\nCreazione point cloud: {title}")
    
    # Crea la griglia di coordinate
    rows, cols = surface_data.shape
    
    # Calcola il fattore di scala per convertire in mm
    mm_per_pixel_x = SURFACE_WIDTH / cols
    mm_per_pixel_y = SURFACE_HEIGHT / rows
    
    # Crea la griglia in millimetri
    x = np.arange(cols) * mm_per_pixel_x
    y = np.arange(rows) * mm_per_pixel_y
    X, Y = np.meshgrid(x, y)
    
    # Crea i punti 3D
    points = np.zeros((rows * cols, 3))
    points[:, 0] = Y.flatten()  # Scambiamo X e Y
    points[:, 1] = X.flatten()  # Scambiamo X e Y
    points[:, 2] = -surface_data.values.flatten()  # Invertiamo Z
    
    # Crea la nuvola di punti
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Calcola le normali per una migliore visualizzazione
    pcd.estimate_normals()
    pcd.normalize_normals()
    
    # Aggiungi colori basati sulla profondità
    z_values = points[:, 2]
    colors = plt_cm_to_rgb(z_values)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def plt_cm_to_rgb(values):
    """Converte i valori in colori RGB usando una mappa di colori"""
    min_val, max_val = np.min(values), np.max(values)
    normalized = (values - min_val) / (max_val - min_val)
    colors = np.zeros((len(values), 3))
    colors[:, 0] = normalized  # R
    colors[:, 2] = 1 - normalized  # B
    return colors

def visualizza_3d(superficie, bordi_lavorabili, path, angoli_testina, raggio_cm):
    """Visualizza i risultati in 3D usando PyVista."""
    # Crea la griglia strutturata per la superficie
    n_rows, n_cols = superficie.shape
    x = np.arange(n_cols)
    y = np.arange(n_rows)
    x_grid, y_grid = np.meshgrid(x, y)
    
    grid = pv.StructuredGrid()
    grid.points = np.c_[x_grid.ravel(), y_grid.ravel(), superficie.ravel()]
    grid.dimensions = [n_cols, n_rows, 1]
    
    # Calcola il gradiente per identificare gli ostacoli
    z_smooth = gaussian_filter(superficie, sigma=2)
    gradient = gaussian_gradient_magnitude(z_smooth, sigma=1)
    ostacoli = gradient > (45 * 0.9)  # Usa la stessa logica di ExtendedProcessor
    
    # Calcola le aree di copertura considerando l'angolo della testina
    copertura = np.zeros((n_rows, n_cols), dtype=int)
    
    for punto, angolo in zip(path, angoli_testina):
        x, y, z = punto
        x_int, y_int = int(round(x)), int(round(y))
        
        # Calcola il raggio effettivo considerando l'angolo
        raggio_effettivo = raggio_cm * abs(np.cos(angolo))
        
        # Calcola l'area di contatto dell'utensile
        y_min = max(0, int(y - raggio_effettivo))
        y_max = min(n_rows, int(y + raggio_effettivo + 1))
        x_min = max(0, int(x - raggio_effettivo))
        x_max = min(n_cols, int(x + raggio_effettivo + 1))
        
        # Per ogni punto nell'area potenziale di contatto
        for yi in range(y_min, y_max):
            for xi in range(x_min, x_max):
                # Verifica se il punto è all'interno dell'ellisse ruotata e NON è un ostacolo
                if not ostacoli[yi, xi]:  # Solo se non è un ostacolo
                    dx = xi - x
                    dy = yi - y
                    # Applica la rotazione inversa
                    dx_rot = dx * np.cos(-angolo) - dy * np.sin(-angolo)
                    dy_rot = dx * np.sin(-angolo) + dy * np.cos(-angolo)
                    # Verifica se il punto è all'interno dell'ellisse
                    if (dx_rot/raggio_cm)**2 + (dy_rot/(raggio_cm * abs(np.cos(angolo))))**2 <= 1:
                        copertura[yi, xi] = 1
    
    # Crea una maschera di colori
    colori = np.zeros((n_rows * n_cols, 3))
    
    # Grigio per aree non coperte e non ostacoli
    colori[~ostacoli.ravel() & (copertura.ravel() == 0)] = [0.7, 0.7, 0.7]
    # Verde per aree coperte (solo se non sono ostacoli)
    colori[~ostacoli.ravel() & (copertura.ravel() == 1)] = [0.0, 0.8, 0.0]
    # Rosso per gli ostacoli
    colori[ostacoli.ravel()] = [0.8, 0.0, 0.0]
    
    # Crea il plotter
    plotter = pv.Plotter()
    
    # Aggiungi la superficie base con i colori di copertura
    grid.point_data['colors'] = colori
    plotter.add_mesh(grid, scalars='colors', rgb=True)
    
    # Visualizza il percorso come una superficie continua dell'utensile
    if path:
        # Crea una "scia" dell'utensile
        path_points = []
        path_faces = []
        point_count = 0
        
        for i in range(len(path)-1):
            p1 = path[i]
            p2 = path[i+1]
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
        if path_points:
            path_points = np.array(path_points)
            path_mesh = pv.PolyData(path_points)
            if path_faces:
                path_mesh.faces = np.hstack(path_faces)
            
            # Colora in base al tipo di movimento
            plotter.add_mesh(path_mesh, color='yellow', opacity=0.3, label='Movimento Utensile')
    
    # Configura la visualizzazione
    plotter.add_axes()
    plotter.add_legend()
    plotter.show_grid()
    plotter.set_background('white')
    
    # Aggiungi statistiche
    area_totale = n_rows * n_cols
    area_coperta = np.sum(copertura & ~ostacoli)  # Solo aree coperte che non sono ostacoli
    area_ostacoli = np.sum(ostacoli)
    area_lavorabile = area_totale - area_ostacoli
    
    stats_text = [
        f"Area Totale: {area_totale}",
        f"Area Ostacoli: {area_ostacoli} ({area_ostacoli/area_totale*100:.1f}%)",
        f"Area Lavorabile: {area_lavorabile} ({area_lavorabile/area_totale*100:.1f}%)",
        f"Area Coperta: {area_coperta} ({area_coperta/area_lavorabile*100:.1f}% del lavorabile)"
    ]
    
    plotter.add_text("\n".join(stats_text), position='upper_left', font_size=12)
    
    # Mostra la visualizzazione
    plotter.show()