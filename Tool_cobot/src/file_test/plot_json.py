#------------------------uso qst file per plottare il json uscente da tool_cobot.py

import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SUPERFICIE = {
    'altezza_mm': 471.375,
    'larghezza_mm': 1609.857
}

# Calcola il rapporto d'aspetto basato sulle dimensioni della superficie
aspect_ratio = SUPERFICIE['larghezza_mm'] / SUPERFICIE['altezza_mm']
# Dimensione base per la figura
base_size = 10
# Calcola l'altezza mantenendo il rapporto d'aspetto
height_size = base_size / aspect_ratio

# Carica il file JSON 
with open('coordinate_cobot.json', 'r') as file:
    data = json.load(file)

# Estrai le routes dal JSON
routes = data['routes']

# Crea liste vuote per x, y, z
x = []
y = []
z = []

# Itera attraverso ogni route
for route in routes:
    for point in route:
        if 'x' in point and 'y' in point and 'z' in point:
            x.append(point['x'])
            y.append(point['y'])
            z.append(point['z'])

# Crea un grafico 3D con le dimensioni corrette
fig = plt.figure(figsize=(base_size, height_size))
ax = fig.add_subplot(111, projection='3d')

# Plotta i punti
ax.scatter(x, y, z, c='b', marker='o')

# Collega i punti con linee
ax.plot(x, y, z, 'r-', alpha=0.5)

# Imposta i limiti degli assi basati sulle dimensioni della superficie
ax.set_xlim([0, SUPERFICIE['larghezza_mm']])
ax.set_ylim([0, SUPERFICIE['altezza_mm']])

# Etichetta gli assi
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')

# Aggiungi un titolo
plt.title('Percorso del Cobot (scala in mm)')

# Mostra il grafico
plt.show()