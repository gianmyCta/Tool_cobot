import numpy as np
import json
import gzip
import os
from datetime import datetime
import hashlib

class CheckpointManager:
    def __init__(self, base_dir='checkpoints'):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
    def save_checkpoint(self, state_dict, metadata=None):
        """Salva lo stato corrente con compressione e validazione."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_data = {
            'state': state_dict,
            'metadata': metadata or {},
            'timestamp': timestamp,
            'version': '1.0'
        }
        
        # Calcola hash dei dati
        data_hash = hashlib.sha256(
            json.dumps(checkpoint_data, sort_keys=True).encode()
        ).hexdigest()[:8]
        
        filename = f"checkpoint_{timestamp}_{data_hash}.gz"
        filepath = os.path.join(self.base_dir, filename)
        
        with gzip.open(filepath, 'wt') as f:
            json.dump(checkpoint_data, f)
        
        return filepath
    
    def load_checkpoint(self, filepath):
        """Carica e valida un checkpoint."""
        try:
            with gzip.open(filepath, 'rt') as f:
                data = json.load(f)
            
            # Validazione base
            required_keys = {'state', 'metadata', 'timestamp', 'version'}
            if not all(k in data for k in required_keys):
                raise ValueError("Checkpoint mancante di campi obbligatori")
            
            # Converti array JSON in NumPy arrays dove necessario
            if 'surface_state' in data['state']:
                data['state']['surface_state'] = np.array(
                    data['state']['surface_state']
                )
            
            return data
        except Exception as e:
            raise ValueError(f"Errore nel caricamento del checkpoint: {e}")
    
    def list_checkpoints(self):
        """Lista tutti i checkpoint disponibili con metadata."""
        checkpoints = []
        for filename in os.listdir(self.base_dir):
            if filename.startswith('checkpoint_') and filename.endswith('.gz'):
                filepath = os.path.join(self.base_dir, filename)
                try:
                    data = self.load_checkpoint(filepath)
                    checkpoints.append({
                        'filepath': filepath,
                        'timestamp': data['timestamp'],
                        'metadata': data['metadata']
                    })
                except ValueError:
                    continue
        return sorted(checkpoints, key=lambda x: x['timestamp'], reverse=True) 