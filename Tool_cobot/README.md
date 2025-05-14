# Surface Merger

Script Python per unire e visualizzare due scansioni di una superficie rettangolare (1000mm x 700mm).

## Caratteristiche

- Unione automatica di due file CSV contenenti scansioni di superficie
- Rimozione automatica dei valori NaN tramite interpolazione
- Ricerca automatica del miglior punto di giunzione
- Visualizzazione 3D interattiva con Open3D
- Supporto per il crop dei bordi in millimetri
- Verifica della continuità tra le superfici

## Struttura del Progetto

```
.
└── src/
    ├── config.py              # Configurazione centralizzata
    ├── main.py               # Script principale
    ├── processing/           # Elaborazione dati
    │   └── surface_processing.py
    └── visualization/        # Visualizzazione 3D
        └── visualizer.py
```

## Requisiti

- Python 3.8 o superiore
- Dipendenze elencate in `requirements.txt`

## Installazione

1. Clona il repository
2. Crea un ambiente virtuale (opzionale ma consigliato):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # oppure
   venv\Scripts\activate     # Windows
   ```
3. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```

## Configurazione

Tutti i parametri configurabili si trovano in `src/config.py`:

- Dimensioni della superficie (SURFACE_WIDTH, SURFACE_HEIGHT)
- Parametri di crop (CROP_MARGINS)
- Parametri di allineamento (CONTINUITY_THRESHOLD, MAX_SEARCH_RANGE, etc.)
- Parametri di visualizzazione (GRID_SPACING, WINDOW_WIDTH, etc.)
- Percorsi dei file (DATA_FOLDER, INPUT_FILES)

## Utilizzo

1. Posizionati nella directory `src`:
   ```bash
   cd src
   ```

2. Esegui lo script:
   ```bash
   python main.py
   ```

Lo script:
1. Legge i file CSV delle superfici
2. Rimuove i valori NaN
3. Applica il crop dei bordi se specificato
4. Trova il miglior punto di giunzione
5. Unisce le superfici
6. Mostra la visualizzazione 3D
7. Salva il risultato in un nuovo file CSV

## Formato dei File di Input

I file CSV devono:
- Contenere solo valori numerici (distanze in mm)
- Non avere intestazioni (header)
- Avere la prima riga da ignorare
- Avere le stesse dimensioni in altezza

## Parametri Principali

- `continuity_threshold`: differenza massima accettabile tra i punti di giunzione (mm)
- `max_search_range`: area massima di ricerca per l'allineamento
- `crop_margins`: margini da ritagliare in millimetri
- `use_mm`: se True, interpreta i valori di crop in millimetri

## Visualizzazione

La visualizzazione 3D include:
- Nuvola di punti colorata in base alla profondità
- Griglia di riferimento con spaziatura configurabile
- Sistema di coordinate 3D
- Vista dall'alto predefinita
- Controlli interattivi della telecamera

## Note

- I file di input devono avere le stesse dimensioni in altezza
- Il crop viene applicato prima della ricerca del punto di giunzione
- La visualizzazione può essere disabilitata impostando `show_visualization=False` 