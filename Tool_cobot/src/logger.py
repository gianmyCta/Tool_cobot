import logging
import os
from datetime import datetime
import json
from pathlib import Path

class SurfaceLogger:
    def __init__(self, log_dir='logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup del logger principale
        self.logger = logging.getLogger('surface_processing')
        self.logger.setLevel(logging.DEBUG)
        
        # Log file per debugging
        debug_handler = logging.FileHandler(
            self.log_dir / f'debug_{datetime.now():%Y%m%d_%H%M%S}.log'
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        debug_handler.setFormatter(debug_formatter)
        self.logger.addHandler(debug_handler)
        
        # Log file per statistiche
        self.stats_file = self.log_dir / f'stats_{datetime.now():%Y%m%d_%H%M%S}.json'
        self.stats = {
            'start_time': datetime.now().isoformat(),
            'processed_areas': [],
            'coverage_stats': [],
            'errors': []
        }
    
    def log_processing_step(self, area_info):
        """Logga informazioni su un'area processata."""
        self.logger.info(f"Processing area: {area_info}")
        self.stats['processed_areas'].append({
            'timestamp': datetime.now().isoformat(),
            **area_info
        })
        self._save_stats()
    
    def log_coverage(self, coverage_info):
        """Logga statistiche sulla copertura."""
        self.logger.debug(f"Coverage update: {coverage_info}")
        self.stats['coverage_stats'].append({
            'timestamp': datetime.now().isoformat(),
            **coverage_info
        })
        self._save_stats()
    
    def log_error(self, error_info):
        """Logga errori e problemi."""
        self.logger.error(f"Processing error: {error_info}")
        self.stats['errors'].append({
            'timestamp': datetime.now().isoformat(),
            **error_info
        })
        self._save_stats()
    
    def _save_stats(self):
        """Salva le statistiche su file."""
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def get_coverage_summary(self):
        """Genera un sommario della copertura."""
        if not self.stats['coverage_stats']:
            return None
        
        latest_coverage = self.stats['coverage_stats'][-1]
        total_areas = len(self.stats['processed_areas'])
        error_count = len(self.stats['errors'])
        
        return {
            'total_areas_processed': total_areas,
            'latest_coverage': latest_coverage,
            'error_count': error_count,
            'processing_duration': (
                datetime.now() - datetime.fromisoformat(self.stats['start_time'])
            ).total_seconds()
        }

# Esempio di utilizzo:
# logger = SurfaceLogger()
# logger.log_processing_step({'area_id': 1, 'size': 100, 'position': (10, 20)})
# logger.log_coverage({'total_area': 1000, 'covered_area': 800, 'coverage_percent': 80})
# logger.log_error({'type': 'unreachable_area', 'position': (30, 40), 'details': 'Area non raggiungibile'}) 