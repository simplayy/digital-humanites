"""
Pacchetto per l'analisi statistica dei dati.

Questo pacchetto contiene moduli per:
- Analisi statistica descrittiva
- Test di ipotesi
- Analisi di correlazione
- Modelli di regressione

Questi moduli consentono di esaminare la relazione tra caratteristiche
del sentiment nei commenti e la veridicità delle notizie.
"""

from pathlib import Path

# Definizione percorsi
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Assicurarsi che le directory esistano
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)