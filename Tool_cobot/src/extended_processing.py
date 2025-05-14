import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, label
from function import cerchio_valido, trova_vicino_valido_vettoriale, segmento_sicuro, calcola_normale
import logging

class ExtendedProcessor:
    def __init__(self, superficie, gradient, pendenza_max, raggio_cm):
        self.superficie = superficie
        self.gradient = gradient
        self.pendenza_max = pendenza_max
        self.raggio_cm = raggio_cm
        self.h, self.w = superficie.shape
        
        # Crea la maschera base delle aree lavorabili
        self.lavorabile_base = gradient < (pendenza_max * 0.9)
        
        # Identifica le aree non lavorabili
        self.ostacoli = ~self.lavorabile_base
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def analizza_bordi_ostacoli(self):
        """Analizza i bordi degli ostacoli per trovare zone potenzialmente lavorabili."""
        # Etichetta gli ostacoli connessi
        ostacoli_labeled, num_ostacoli = label(self.ostacoli)
        
        # Per ogni ostacolo, analizza i suoi bordi
        bordi_lavorabili = np.zeros_like(self.ostacoli, dtype=bool)
        
        for id_ostacolo in range(1, num_ostacoli + 1):
            # Estrai l'ostacolo corrente
            ostacolo = ostacoli_labeled == id_ostacolo
            
            # Trova il bordo dell'ostacolo
            bordo = self._trova_bordo(ostacolo)
            
            # Analizza ogni punto del bordo
            for y, x in zip(*np.where(bordo)):
                if self._bordo_lavorabile(y, x, ostacolo):
                    bordi_lavorabili[y, x] = True
        
        return bordi_lavorabili
    
    def _trova_bordo(self, maschera):
        """Trova i pixel di bordo di una maschera."""
        eroso = binary_erosion(maschera)
        return maschera & ~eroso
    
    def _bordo_lavorabile(self, y, x, ostacolo):
        """Verifica se un punto del bordo è lavorabile in sicurezza."""
        # Crea una maschera circolare centrata nel punto
        y_min = max(0, int(y - self.raggio_cm))
        y_max = min(self.h, int(y + self.raggio_cm + 1))
        x_min = max(0, int(x - self.raggio_cm))
        x_max = min(self.w, int(x + self.raggio_cm + 1))
        
        # Estrai la finestra di interesse
        finestra_ostacoli = ostacolo[y_min:y_max, x_min:x_max]
        finestra_superficie = self.superficie[y_min:y_max, x_min:x_max]
        
        # Calcola il centro relativo alla finestra
        centro_y = y - y_min
        centro_x = x - x_min
        
        # Crea la maschera circolare
        y_grid, x_grid = np.ogrid[:y_max-y_min, :x_max-x_min]
        maschera_circolare = (x_grid - centro_x)**2 + (y_grid - centro_y)**2 <= self.raggio_cm**2
        
        # Verifica se ci sono ostacoli più alti nella zona di sovrapposizione
        z_centro = self.superficie[y, x]
        sovrapposizione = maschera_circolare & finestra_ostacoli
        if sovrapposizione.any():
            z_sovrapposizione = finestra_superficie[sovrapposizione]
            if np.any(z_sovrapposizione > z_centro):
                return False
        
        # Verifica la pendenza nella zona di lavoro
        if maschera_circolare.any():
            z_lavoro = finestra_superficie[maschera_circolare]
            pendenze = np.abs(z_lavoro - z_centro)
            if np.any(pendenze > self.pendenza_max * self.raggio_cm):
                return False
        
        return True
    
    def genera_percorso_esteso(self, path_originale):
        """Genera un percorso che include i bordi lavorabili degli ostacoli."""
        # Trova i bordi lavorabili
        bordi_lavorabili = self.analizza_bordi_ostacoli()
        
        # Converti i punti dei bordi in coordinate
        punti_bordo = []
        for y, x in zip(*np.where(bordi_lavorabili)):
            z = self.superficie[y, x]
            punti_bordo.append((x, y, z))
        
        if not punti_bordo:
            self.logger.info("Nessun bordo lavorabile trovato")
            return path_originale
        
        # Ordina i punti del bordo per vicinanza al percorso originale
        percorso_esteso = self._integra_punti_bordo(path_originale, punti_bordo)
        
        return percorso_esteso
    
    def _integra_punti_bordo(self, path_originale, punti_bordo):
        """Integra i punti del bordo nel percorso originale."""
        percorso_esteso = list(path_originale)
        
        for punto_bordo in punti_bordo:
            # Trova il punto più vicino nel percorso
            distanze = [self._distanza_3d(p, punto_bordo) for p in percorso_esteso]
            idx_vicino = np.argmin(distanze)
            
            # Se il punto è abbastanza vicino e il segmento è sicuro
            if distanze[idx_vicino] < self.raggio_cm * 2:
                punto_vicino = percorso_esteso[idx_vicino]
                if segmento_sicuro(punto_vicino, punto_bordo, self.lavorabile_base, self.raggio_cm):
                    # Inserisci il punto dopo il punto più vicino
                    percorso_esteso.insert(idx_vicino + 1, punto_bordo)
                    self.logger.debug(f"Aggiunto punto bordo: {punto_bordo}")
        
        return percorso_esteso
    
    def _distanza_3d(self, p1, p2):
        """Calcola la distanza 3D tra due punti."""
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

    def _calcola_angolo_ostacolo(self, punto, ostacolo):
        """Calcola l'angolo dell'ostacolo rispetto alla superficie."""
        x, y = int(punto[0]), int(punto[1])
        
        # Calcola il gradiente locale intorno al punto
        if x > 1 and x < self.w-2 and y > 1 and y < self.h-2:
            # Usa una finestra 5x5 per calcolare la pendenza media
            window = self.superficie[y-2:y+3, x-2:x+3]
            
            # Calcola il gradiente nelle direzioni x e y
            dy, dx = np.gradient(window)
            
            # Calcola la pendenza media nella direzione x
            pendenza_x = np.mean(dx)
            # Calcola la pendenza media nella direzione y
            pendenza_y = np.mean(dy)
            
            # Calcola l'angolo di inclinazione rispetto alla verticale
            angolo = -np.arctan2(pendenza_y, 1.0)  # Negativo per far inclinare nella direzione corretta
            
            return angolo
        
        return 0.0
    
    def _calcola_distanza_sicurezza(self, angolo):
        """Calcola la distanza di sicurezza necessaria per la rotazione della testina."""
        # La distanza di sicurezza deve essere sufficiente per permettere la rotazione completa
        angolo_abs = abs(angolo)
        distanza_base = self.raggio_cm * 1.5  # Aumentata la distanza base
        
        # La distanza aumenta con l'angolo in modo non lineare
        # Usiamo una funzione che cresce più rapidamente per angoli maggiori
        fattore_angolo = np.tan(angolo_abs)  # Cresce più rapidamente con l'angolo
        distanza_max = self.raggio_cm * 3.0   # Aumentata la distanza massima
        
        # Calcola la distanza in base all'angolo
        distanza = distanza_base + (distanza_max - distanza_base) * fattore_angolo
        return min(distanza, distanza_max)
    
    def _verifica_sicurezza_rotazione(self, punto, angolo, ostacolo):
        """Verifica se è sicuro ruotare la testina nella posizione data."""
        x, y = int(punto[0]), int(punto[1])
        z = punto[2]
        
        # Calcola l'area che verrebbe coperta dalla testina ruotata
        raggio_effettivo = self.raggio_cm * abs(np.cos(angolo))
        
        # Verifica se ci sono altri ostacoli nell'area di rotazione
        y_min = max(0, int(y - raggio_effettivo))
        y_max = min(self.h, int(y + raggio_effettivo + 1))
        x_min = max(0, int(x - raggio_effettivo))
        x_max = min(self.w, int(x + raggio_effettivo + 1))
        
        # Estrai la finestra di interesse
        finestra_ostacoli = ostacolo[y_min:y_max, x_min:x_max]
        finestra_superficie = self.superficie[y_min:y_max, x_min:x_max]
        
        # Crea una griglia di punti per l'area dell'utensile
        y_coords = np.arange(y_min, y_max)
        x_coords = np.arange(x_min, x_max)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        
        # Calcola la distanza dal centro dell'utensile per ogni punto
        distanza_quadrata = (x_grid - x)**2 + (y_grid - y)**2
        maschera_utensile = distanza_quadrata <= raggio_effettivo**2
        
        # Per ogni punto nell'area dell'utensile
        for yi in range(y_max - y_min):
            for xi in range(x_max - x_min):
                if maschera_utensile[yi, xi]:
                    # Calcola la posizione dell'utensile in questo punto
                    dx = x_grid[yi, xi] - x
                    dy = y_grid[yi, xi] - y
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    # Calcola l'altezza dell'utensile in questo punto
                    # L'utensile si adatta alla superficie mantenendo una distanza costante
                    z_utensile = z + (raggio_effettivo - dist) * np.sin(angolo)
                    
                    # Verifica la distanza dalla superficie
                    z_superficie = finestra_superficie[yi, xi]
                    if abs(z_utensile - z_superficie) < 0.1:  # Tolleranza di 0.1 unità
                        if finestra_ostacoli[yi, xi]:
                            return False
                    elif z_utensile < z_superficie:  # Se l'utensile penetra la superficie
                        return False
        
        return True
    
    def _calcola_punto_avvicinamento(self, punto, angolo, distanza):
        """Calcola il punto di avvicinamento sicuro considerando l'angolo dell'ostacolo."""
        x, y, z = punto
        
        # Calcola gli offset considerando l'angolo dell'ostacolo
        # L'utensile deve posizionarsi lateralmente e indietro rispetto all'ostacolo
        offset_laterale = self.raggio_cm * 1.5 * np.sin(angolo)  # Aumentato l'offset laterale
        offset_indietro = self.raggio_cm * 1.5 * np.cos(angolo)  # Aggiunto offset indietro
        offset_verticale = distanza  # Usa la distanza calcolata per l'offset verticale
        
        # Il punto di avvicinamento deve essere spostato:
        # - Lateralmente per allinearsi con l'angolo dell'ostacolo
        # - Indietro per evitare collisioni durante la rotazione
        # - In alto per garantire spazio per la rotazione
        x_avv = x - offset_laterale - offset_indietro * np.cos(angolo)
        y_avv = y - offset_indietro * np.sin(angolo)
        z_avv = z + offset_verticale
        
        return (x_avv, y_avv, z_avv)

    def _calcola_normale_e_angoli(self, punto):
        """Calcola la normale alla superficie e gli angoli di inclinazione nel punto."""
        x, y = int(punto[0]), int(punto[1])
        
        if x > 0 and x < self.w-1 and y > 0 and y < self.h-1:
            # Calcola il gradiente locale usando una finestra 3x3
            dz_dy = (self.superficie[y+1, x] - self.superficie[y-1, x]) / 2.0
            dz_dx = (self.superficie[y, x+1] - self.superficie[y, x-1]) / 2.0
            
            # La normale è il vettore perpendicolare alla superficie
            normale = np.array([-dz_dx, -dz_dy, 1.0])
            normale = normale / np.linalg.norm(normale)
            
            # Calcola gli angoli di inclinazione
            # L'angolo di inclinazione è l'angolo tra la normale e l'asse z
            angolo_inclinazione = np.arccos(normale[2])
            
            # La direzione è l'angolo nel piano xy
            angolo_direzione = np.arctan2(normale[1], normale[0])
            
            return normale, angolo_inclinazione, angolo_direzione
        
        return np.array([0, 0, 1]), 0, 0

    def _verifica_penetrazione(self, punto_utensile, normale, raggio_cm):
        """Verifica se l'utensile penetra la superficie in qualsiasi punto."""
        x_center, y_center = int(punto_utensile[0]), int(punto_utensile[1])
        z_center = punto_utensile[2]
        
        # Crea due vettori perpendicolari alla normale per definire il piano dell'utensile
        if abs(normale[2]) < 0.9:
            v1 = np.cross(normale, [0, 0, 1])
        else:
            v1 = np.cross(normale, [1, 0, 0])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(normale, v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Controlla una griglia di punti sull'utensile
        n_points = 16  # Numero di punti da controllare sul disco
        penetrazione = False
        min_distanza = float('inf')
        
        for r in np.linspace(0, raggio_cm, 5):  # Controlla a diverse distanze dal centro
            for theta in np.linspace(0, 2*np.pi, n_points):
                # Calcola il punto sul disco dell'utensile
                punto = np.array([
                    x_center + r * (v1[0] * np.cos(theta) + v2[0] * np.sin(theta)),
                    y_center + r * (v1[1] * np.cos(theta) + v2[1] * np.sin(theta)),
                    z_center + r * (v1[2] * np.cos(theta) + v2[2] * np.sin(theta))
                ])
                
                x, y = int(round(punto[0])), int(round(punto[1]))
                if 0 <= x < self.w and 0 <= y < self.h:
                    z_superficie = self.superficie[y, x]
                    distanza = punto[2] - z_superficie
                    min_distanza = min(min_distanza, distanza)
                    if distanza < 0:  # Se il punto è sotto la superficie
                        penetrazione = True
                        break
            if penetrazione:
                break
        
        return penetrazione, min_distanza

    def _calcola_punto_contatto(self, punto, normale):
        """Calcola il punto di contatto dell'utensile con la superficie."""
        x, y = int(punto[0]), int(punto[1])
        z = self.superficie[y, x]
        
        # Il punto di contatto è sulla superficie
        punto_contatto = np.array([x, y, z])
        
        # Inizia posizionando l'utensile a una distanza pari al suo raggio
        punto_utensile = punto_contatto + normale * self.raggio_cm
        
        # Verifica se c'è penetrazione e aggiusta l'altezza se necessario
        penetrazione, min_distanza = self._verifica_penetrazione(punto_utensile, normale, self.raggio_cm)
        
        if penetrazione:
            # Calcola quanto alzare l'utensile per evitare la penetrazione
            # Aggiungi un piccolo offset di sicurezza (0.1)
            offset = -min_distanza + 0.1
            punto_utensile = punto_utensile + normale * offset
            
            # Verifica nuovamente dopo l'aggiustamento
            penetrazione, _ = self._verifica_penetrazione(punto_utensile, normale, self.raggio_cm)
            if penetrazione:
                # Se ancora penetra, usa un approccio più conservativo
                punto_utensile = punto_contatto + normale * (self.raggio_cm * 1.5)
        
        return punto_utensile, punto_contatto

    def genera_percorso_con_rotazione(self, path_originale):
        """Genera un percorso che include la rotazione della testina per seguire la superficie."""
        percorso_esteso = []
        
        for i, punto in enumerate(path_originale):
            x, y = int(punto[0]), int(punto[1])
            
            # Calcola la normale e gli angoli per questo punto
            normale, angolo_inclinazione, angolo_direzione = self._calcola_normale_e_angoli(punto)
            
            # Se l'angolo è troppo ripido, limita l'inclinazione
            if abs(angolo_inclinazione) > np.pi/3:  # Limita a 60 gradi
                # Mantieni la direzione ma limita l'inclinazione
                normale = np.array([
                    np.sin(np.pi/3) * np.cos(angolo_direzione),
                    np.sin(np.pi/3) * np.sin(angolo_direzione),
                    np.cos(np.pi/3)
                ])
                normale = normale / np.linalg.norm(normale)
                angolo_inclinazione = np.pi/3
            
            # Calcola il punto di contatto e la posizione dell'utensile
            punto_utensile, punto_contatto = self._calcola_punto_contatto(punto, normale)
            
            # Aggiungi il punto al percorso
            percorso_esteso.append({
                "posizione": tuple(punto_utensile),
                "punto_contatto": tuple(punto_contatto),
                "normale": normale,
                "angolo_inclinazione": angolo_inclinazione,
                "angolo_direzione": angolo_direzione,
                "tipo": "lavorazione"
            })
            
            # Se il prossimo punto esiste, verifica se è necessario un punto intermedio
            if i < len(path_originale) - 1:
                punto_next = path_originale[i + 1]
                normale_next, _, _ = self._calcola_normale_e_angoli(punto_next)
                
                # Se c'è un cambio significativo nella normale, aggiungi punti intermedi
                if np.dot(normale, normale_next) < 0.95:  # angolo > ~18 gradi
                    n_punti = 5  # Aumentato il numero di punti intermedi
                    for j in range(1, n_punti):
                        t = j / n_punti
                        # Interpola posizione
                        x_int = int(punto[0] + t * (punto_next[0] - punto[0]))
                        y_int = int(punto[1] + t * (punto_next[1] - punto[1]))
                        punto_int = (x_int, y_int, self.superficie[y_int, x_int])
                        
                        # Interpola la normale per una transizione più fluida
                        normale_int = normale * (1-t) + normale_next * t
                        normale_int = normale_int / np.linalg.norm(normale_int)
                        ang_inc_int = np.arccos(normale_int[2])
                        ang_dir_int = np.arctan2(normale_int[1], normale_int[0])
                        
                        # Calcola il punto di contatto interpolato
                        punto_utensile_int, punto_contatto_int = self._calcola_punto_contatto(punto_int, normale_int)
                        
                        percorso_esteso.append({
                            "posizione": tuple(punto_utensile_int),
                            "punto_contatto": tuple(punto_contatto_int),
                            "normale": normale_int,
                            "angolo_inclinazione": ang_inc_int,
                            "angolo_direzione": ang_dir_int,
                            "tipo": "lavorazione"
                        })
        
        return percorso_esteso

# Esempio di utilizzo:
"""
# Nel main:
processor = ExtendedProcessor(z, gradient, pendenza_max, raggio_cm)
path_esteso = processor.genera_percorso_esteso(path)
""" 