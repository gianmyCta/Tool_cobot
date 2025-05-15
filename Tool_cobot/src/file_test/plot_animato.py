import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import json

# Carica i dati
with open('coordinate_cobot.json', 'r') as file:
    data = json.load(file)

# Estrai coordinate
x, y, z = [], [], []
for route in data['routes']:
    for point in route:
        if 'x' in point and 'y' in point and 'z' in point:
            x.append(point['x'])
            y.append(point['y'])
            z.append(point['z'])

# Crea figura 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Funzione di aggiornamento per l'animazione
def update(frame):
    ax.cla()
    ax.plot(x[:frame], y[:frame], z[:frame], 'b.-')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Frame {frame}/{len(x)}')

# Crea l'animazione
ani = FuncAnimation(fig, update, frames=len(x), interval=100, repeat=False)
plt.show()