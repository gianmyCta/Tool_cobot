# Dimensioni reali della superficie in millimetri
SURFACE_WIDTH = 1609.875  # larghezza totale in mm
SURFACE_HEIGHT = 471.375  # altezza totale in mm

# Parametri per il crop dei bordi (in millimetri)
CROP_MARGINS = {
    'top': 0,     # rimuove mm dall'alto
    'bottom': 0,  # rimuove mm dal basso
    'left': 0,    # rimuove mm da sinistra
    'right': 0    # rimuove mm da destra
}

# Parametri per l'allineamento automatico
CONTINUITY_THRESHOLD = 1.0  # differenza massima accettabile tra i punti di giunzione (mm)
MAX_SEARCH_RANGE = 300     # massima area di ricerca per l'allineamento
MIN_SEARCH_RANGE = 200      # minima area di ricerca per l'allineamento
SEARCH_STEP = 100          # incremento del search_range ad ogni iterazione

# Parametri di visualizzazione
GRID_SPACING = 100        # spaziatura della griglia in mm
POINT_SIZE = 1           # dimensione dei punti nella visualizzazione
BACKGROUND_COLOR = [1, 1, 1]  # colore dello sfondo (bianco)
GRID_COLOR = [0.7, 0.7, 0.7] # colore della griglia (grigio chiaro)
WINDOW_WIDTH = 1280      # larghezza finestra di visualizzazione
WINDOW_HEIGHT = 720      # altezza finestra di visualizzazione
WINDOW_LEFT = 50         # posizione x della finestra
WINDOW_TOP = 50          # posizione y della finestra

# Percorsi dei file
DATA_FOLDER_OUTPUT = "output"
DATA_FOLDER = "scansioni"
INPUT_FILES = {
    'top': f"{DATA_FOLDER}/surface_top.csv",
    'bottom': f"{DATA_FOLDER}/surface_bot.csv",
    'output': f"{DATA_FOLDER}/complete_surface.csv"
}

# Parametri per la generazione del percorso utensile
TOOL_PARAMS = {
    'diametro_default': 30,  # diametro dell'utensile in cm
    'step_default': 15,        # passo del percorso in cm
    'pendenza_max': 0.5,       # pendenza massima accettabile
    'n_punti_cerchio': 100,    # numero di punti per disegnare il cerchio dell'utensile
    'colore_percorso': 'blue', # colore del percorso principale
    'colore_deviazione': 'red' # colore dei percorsi di deviazione
}

# Parametri per l'esportazione
EXPORT_PARAMS = {
    'output_default': f"{DATA_FOLDER_OUTPUT}/coordinate_cobot.json",  # nome file di output predefinito
    'indent': 2                                 # indentazione del file JSON
} 