# Guida alla Riproducibilità

Questo documento fornisce istruzioni dettagliate per riprodurre l'analisi condotta nello studio sulla relazione tra sentiment nei commenti e veridicità delle notizie utilizzando il dataset PHEME.

## 1. Requisiti di Sistema

### Hardware Raccomandato
- CPU: 4+ core (8+ consigliati per il training dei modelli Random Forest)
- RAM: 8+ GB (16+ GB consigliati per dataset completo)
- Spazio di archiviazione: 5+ GB (per dataset, ambiente virtuale e risultati)

### Software Richiesto
- Python 3.8+ (testato su 3.8, 3.9 e 3.10)
- Git per clonare il repository
- Sistema operativo: Linux, macOS o Windows 10+ (testato principalmente su macOS)

## 2. Configurazione dell'Ambiente

### Creazione dell'Ambiente Virtuale
```bash
# Creare un ambiente virtuale Python
python -m venv hds_env

# Attivare l'ambiente (Linux/macOS)
source hds_env/bin/activate

# Attivare l'ambiente (Windows)
.\hds_env\Scripts\activate
```

### Installazione delle Dipendenze
```bash
# Installare le dipendenze dal file requirements.txt
pip install -r requirements.txt
```

Il file `requirements.txt` contiene le seguenti dipendenze principali:
```
pandas==1.5.3
numpy==1.24.3
scipy==1.10.1
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
textblob==0.17.1
textstat==0.7.3
sentence-transformers==2.2.2
jupyter==1.0.0
pytest==7.3.1
```

## 3. Acquisizione dei Dati

### Download del Dataset PHEME
```bash
# Eseguire lo script di download
python -m src.data.download

# I log del download verranno salvati in download.log
```

Lo script di download recupererà il dataset PHEME dalla fonte originale e lo organizzerà nella struttura di cartelle corretta.

### Struttura del Dataset
Dopo il download, i dati grezzi saranno organizzati nella seguente struttura:
```
data/
  raw/
    pheme/
      pheme-rnr-dataset/
        threads/
          charliehebdo/
            non-rumours/
            rumours/
          ferguson/
            non-rumours/
            rumours/
          ...
```

## 4. Preprocessing dei Dati

### Esecuzione del Preprocessing
```bash
# Eseguire lo script di preprocessing
python -m src.data.preprocess_pheme

# I log del preprocessing verranno salvati in preprocess.log
```

Questo script:
1. Carica i dati grezzi
2. Pulisce e normalizza i testi
3. Organizza i tweet in thread conversazionali
4. Salva il dataset preprocessato in formato CSV

### Output del Preprocessing
Il preprocessing genera i seguenti file:
- `data/processed/pheme_preprocessed.csv`: Dataset principale preprocessato
- `data/processed/pheme_thread_structure.json`: Struttura gerarchica dei thread

## 5. Estrazione delle Feature

### Generazione della Matrice di Feature
```bash
# Generare feature di sentiment
python -m src.features.sentiment

# Generare feature di stance
python -m src.features.stance

# Generare feature di leggibilità
python -m src.features.readability

# Integrare tutte le feature
python -m src.features.integrate_features
```

Questi script generano:
- File CSV intermedi con le feature individuali
- `data/processed/pheme_feature_matrix.csv`: Matrice completa di feature

## 6. Esecuzione dell'Analisi Statistica

### Statistiche Descrittive e Test di Ipotesi
```bash
# Generare statistiche descrittive
python -m src.analysis.descriptive

# Eseguire i test di ipotesi
python -m src.analysis.hypothesis_tests

# Analizzare le correlazioni
python -m src.analysis.correlation
```

### Modelli Predittivi
```bash
# Eseguire la regressione logistica
python -m src.analysis.regression

# Eseguire il Random Forest
python -m src.analysis.random_forest
```

### Output dell'Analisi
L'analisi genera diversi file nei seguenti formati e posizioni:
- CSV con risultati numerici in `results/tables/`
- Grafici in formato PNG in `results/figures/`

## 7. Generazione delle Visualizzazioni

### Visualizzazioni Base
```bash
# Visualizzare statistiche del dataset
python -m src.visualization.plot_pheme_stats

# Visualizzare risultati dell'analisi statistica
python -m src.visualization.plot_stats_results
```

### Visualizzazioni Avanzate
```bash
# Creare visualizzazioni comparative dei modelli
python -m src.visualization.model_comparison

# Generare narrazione visiva dei risultati
python -m src.visualization.narrative_visualization
```

## 8. Riproducibilità Specifica

### Random Seeds
Per garantire la riproducibilità esatta dei risultati, tutti i processi stocastici utilizzano seed fissi:
- `random_state=42` in tutti i modelli scikit-learn
- `np.random.seed(42)` all'inizio degli script principali
- `random.seed(42)` per le operazioni con il modulo random standard

### Gestione delle Dipendenze
Il file `requirements.txt` contiene versioni specifiche di tutte le dipendenze per evitare incompatibilità dovute a aggiornamenti.

### Documentazione del Codice
Ogni modulo e funzione include:
- Intestazione con descrizione dello scopo
- Docstring dettagliate con parametri e valori di ritorno
- Commenti esplicativi per sezioni di codice complesse

## 9. Parametri Personalizzabili

Alcune parti dell'analisi possono essere personalizzate modificando i parametri negli script:

### Preprocessing
In `src/data/preprocess_pheme.py`:
- `MIN_CHARS`: Lunghezza minima del testo dopo la pulizia (default: 10)
- `MIN_REACTIONS`: Numero minimo di reazioni per thread (default: 3)

### Random Forest
In `src/analysis/random_forest.py`:
- `n_estimators`: Numero di alberi nel forest (default: 100)
- `class_weight`: Gestione dello sbilanciamento delle classi (default: 'balanced')

### Divisione Train/Test
In vari script di analisi:
- `test_size`: Proporzione del dataset usata come test set (default: 0.3)

## 10. Verifica della Corretta Esecuzione

### Test Automatizzati
```bash
# Eseguire i test automatizzati
pytest
```

### Output Attesi
Per verificare che l'analisi sia stata eseguita correttamente, controllare:

1. **File di Log**:
   - `download.log` dovrebbe terminare con "Download completato con successo"
   - `preprocess.log` dovrebbe indicare il numero corretto di thread processati

2. **Dimensioni della Matrice di Feature**:
   - `pheme_feature_matrix.csv` dovrebbe contenere circa 105,000 righe

3. **Risultati Chiave**:
   - AUC del Random Forest dovrebbe essere ~0.93
   - AUC della Regressione Logistica dovrebbe essere ~0.54

4. **Visualizzazioni**:
   - Tutti i grafici dovrebbero essere generati nelle rispettive cartelle

## 11. Risoluzioni di Problemi Comuni

### Errore: "ModuleNotFoundError"
Assicurarsi che:
- L'ambiente virtuale sia attivato
- Gli script siano eseguiti dalla directory principale del progetto
- Il package sia installabile con `pip install -e .` (se necessario)

### Errore: "Memory Error"
- Ridurre `n_estimators` nel Random Forest
- Aumentare la RAM disponibile
- Utilizzare un subset del dataset per test iniziali

### Avvisi di ConvergenceWarning nella Regressione Logistica
- Aumentare `max_iter` in `src/analysis/regression.py`
- Verificare la presenza di collinearità tra feature

### Problemi con Grafici o Visualizzazioni
- Assicurarsi che matplotlib e seaborn siano installati correttamente
- Verificare che esistano le directory di output necessarie
- Controllare i permessi di scrittura nelle directory

## 12. Riproducibilità a Lungo Termine

Per garantire la riproducibilità a lungo termine:

1. **Versionamento del Dataset**:
   - Una copia del dataset è archiviata in formato compresso nel repository
   - Gli hash SHA-256 dei file originali sono documentati per verifica

2. **Ambiente Docker**:
   - Un Dockerfile è disponibile per creare un ambiente completamente isolato
   - Istruzioni: `docker build -t pheme-analysis . && docker run pheme-analysis`

3. **Documentazione delle Versioni delle API**:
   - Le API esterne utilizzate sono documentate con versione e data
   - Alternative sono fornite per API potenzialmente non disponibili in futuro

## Contatto e Supporto

Per domande sulla riproducibilità dell'analisi:
- Aprire una issue nel repository GitHub del progetto
- Contattare gli autori via email (indicata nel README principale)

---

Questa guida è progettata per consentire la completa riproducibilità dell'analisi. Se si incontrano difficoltà o discrepanze significative nei risultati, si prega di segnalarlo in modo che possiamo migliorare la documentazione e il codice.
