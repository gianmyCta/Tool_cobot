import json
import numpy as np
import matplotlib.pyplot as plt

import json

# Apri il file JSON in modalità lettura
with open('json_prova.json') as f:
    data = json.load(f)
#data = np.array(data)
# Stampa i dati
print(data["persone"][0]["figli"][1]["nome"])
# if "persone" in data and len(data["persone"]) > 1 and "figli" in data["persone"][0]:
#     nomi_figli = [figlio["nome"] for figlio in data["persone"][0]["figli"]]
#     print("Nomi dei figli del secondo elemento:", nomi_figli)
# else:
#     print("Struttura JSON non valida o dati mancanti")
