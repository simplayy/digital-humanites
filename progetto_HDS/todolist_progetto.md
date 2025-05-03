# Todolist del Progetto: Studio sulla Relazione tra Sentiment nei Commenti e Veridicità delle Notizie

## Panoramica
Questa todolist semplificata traccia le attività principali per completare il progetto di Human Data Science, focalizzandosi sul mantenere un solido approccio scientifico.

## Stato di avanzamento
- 🔲 Non iniziato
- 🔄 In corso
- ✅ Completato

## 1. Setup dell'ambiente di lavoro

- [✅] Creare l'ambiente virtuale Python (14/04/2025)
  ```bash
  python -m venv hds_env
  source hds_env/bin/activate  # Per macOS
  ```

- [✅] Installare tutte le dipendenze necessarie (14/04/2025)
  ```bash
  pip install pandas numpy scipy statsmodels scikit-learn
  pip install transformers textstat sentence-transformers
  pip install matplotlib seaborn plotly
  pip install jupyter pytest
  pip freeze > requirements.txt
  ```

- [✅] Configurare Git per il versioning del codice (14/04/2025)
  ```bash
  git init
  git add todolist_progetto.md
  git commit -m "Setup iniziale del progetto"
  ```

- [✅] Creare la struttura delle cartelle del progetto (14/04/2025)
  ```bash
  mkdir -p data/{raw,processed}
  mkdir -p src/{data,features,analysis,visualization}
  mkdir -p tests
  mkdir -p results/{figures,tables}
  mkdir -p reports/figures
  touch src/__init__.py
  touch src/data/__init__.py
  touch src/features/__init__.py
  touch src/analysis/__init__.py
  touch src/visualization/__init__.py
  ```

## 2. Acquisizione e preparazione dei dati

- [✅] Sostituire FakeNewsNet con il dataset PHEME (14/04/2025)
  - [✅] Identificare un dataset più adatto con commenti (14/04/2025)
  - [✅] Selezionare PHEME dataset con thread di conversazioni Twitter (14/04/2025)

- [✅] Scaricare il dataset PHEME (14/04/2025)
  - [✅] Aggiornare lo script `src/data/download.py` per il dataset PHEME (14/04/2025)
  - [✅] Scaricare il dataset PHEME completo con conversazioni Twitter (14/04/2025)
  - [✅] Verificare l'integrità del dataset scaricato (14/04/2025)
  - [✅] Estrarre correttamente il formato .tar.bz2 (14/04/2025)
  
- [✅] Implementare il preprocessing del dataset PHEME
  - [✅] Creare lo script dedicato `src/data/preprocess_pheme.py` (14/04/2025)
  - [✅] Adattare lo script per la struttura specifica del dataset PHEME (14/04/2025)
  - [✅] Eseguire il preprocessing sui dati scaricati (29/04/2025)

- [✅] Creare lo script `src/data/download.py` (14/04/2025)
  - [✅] Implementare funzioni per scaricare i dati (14/04/2025)
  - [✅] Implementare funzioni per organizzare i dati scaricati (14/04/2025)

- [✅] Implementare il preprocessing 
  - [✅] Funzioni per pulizia dei dati (rimozione duplicati, gestione valori mancanti) (29/04/2025)
  - [✅] Normalizzazione dei testi dei commenti (29/04/2025)
  - [✅] Preparazione per analisi del sentiment (29/04/2025)

- [✅] Visualizzazione esplorativa dei dati
  - [✅] Implementazione dello script `src/visualization/plot_pheme_stats.py` (29/04/2025)
  - [✅] Generazione statistiche descrittive iniziali (29/04/2025)
  - [✅] Creazione visualizzazioni della distribuzione dei dati (29/04/2025)
  - [✅] Identificazione pattern principali nella distribuzione degli eventi e veridicità (29/04/2025)

- [✅] Documentare la struttura e le caratteristiche del dataset (29/04/2025)
  - [✅] Generare statistiche descrittive complete sul dataset processato (29/04/2025)

## 3. Estrazione delle feature

- [✅] Implementare `src/features/sentiment.py` (29/04/2025)
  - [✅] Analisi del sentiment (positività/negatività) usando TextBlob (29/04/2025)
  - [✅] Calcolo della polarità e soggettività del sentiment (29/04/2025)

- [✅] Implementare `src/features/stance.py` (29/04/2025)
  - [✅] Identificare se i commenti sono favorevoli o contrari alla notizia principale (29/04/2025)
  - [✅] Classificare le reazioni per atteggiamento verso la veridicità della notizia (29/04/2025)

- [✅] Implementare `src/features/readability.py` (29/04/2025)
  - [✅] Calcolo di indici di leggibilità (Flesch, Gunning Fog, SMOG) (29/04/2025)
  - [✅] Misura del livello di acculturazione nei commenti (29/04/2025)
  - [✅] Analisi della complessità lessicale e ricchezza del vocabolario (29/04/2025)
  - [✅] Rilevamento del linguaggio formale vs informale (29/04/2025)

- [✅] Creare una matrice di feature integrata (30/04/2025)
  - [✅] Combinare le feature di sentiment, stance e leggibilità (30/04/2025)
  - [✅] Standardizzare i valori numerici per l'analisi (30/04/2025)
  - [✅] Generare matrice di correlazione tra feature (30/04/2025)

- [✅] Documentare le feature in `feature_description.md` (30/04/2025)
  - [✅] Spiegare il significato di ogni feature estratta (30/04/2025)
  - [✅] Descrivere gli algoritmi utilizzati (30/04/2025)

## 4. Analisi statistica

- [✅] Implementare `src/analysis/descriptive.py` (03/05/2025)
  - [✅] Calcolare statistiche descrittive (media, mediana, deviazione standard) (03/05/2025)
  - [✅] Visualizzare distribuzioni principali (03/05/2025)
  - [✅] Identificare e gestire outlier (03/05/2025)

- [✅] Implementare test statistici essenziali in `src/analysis/hypothesis_tests.py` (03/05/2025)
  - [✅] Test per confrontare il sentiment tra notizie vere e false (03/05/2025)
  - [✅] Test per analizzare la correlazione tra sentiment e veridicità (03/05/2025)
  - [✅] Test per valutare la relazione tra livello di acculturazione e propensione alla disinformazione (03/05/2025)

- [✅] Implementare analisi di correlazione in `src/analysis/correlation.py` (03/05/2025)
  - [✅] Calcolare coefficienti di correlazione tra le variabili principali (03/05/2025)
  - [✅] Creare matrice di correlazione (03/05/2025)

- [✅] Implementare modelli predittivi (03/05/2025)
  - [✅] Modello lineare: Regressione logistica (`src/analysis/regression.py`) (03/05/2025)
  - [✅] Modello non lineare: Random Forest (`src/analysis/random_forest.py`) (03/05/2025)
  - [✅] Confrontare le performance dei modelli (03/05/2025)
  - [✅] Analizzare l'importanza delle feature nei diversi modelli (03/05/2025)

- [✅] Calcolare la significatività statistica (03/05/2025)
  - [✅] Documentare p-value e dimensione dell'effetto (03/05/2025)
  - [✅] Interpretare i risultati nel contesto del progetto (03/05/2025)

- [✅] Validare i risultati (03/05/2025)
  - [✅] Utilizzare tecniche di validazione incrociata (03/05/2025)
  - [✅] Verificare la robustezza dei risultati (03/05/2025)
  - [✅] Testare diversi set di feature per valutarne il contributo (03/05/2025)

## 5. Visualizzazione e interpretazione dei risultati

- [✅] Implementare `src/visualization/plot_pheme_stats.py` (29/04/2025)
  - [✅] Funzioni per visualizzare distribuzioni degli eventi nel dataset (29/04/2025)
  - [✅] Funzioni per visualizzare distribuzione della veridicità (29/04/2025)
  - [✅] Funzioni per visualizzare distribuzioni delle reazioni (29/04/2025)
  - [✅] Ottimizzazione delle visualizzazioni per chiarezza e leggibilità (29/04/2025)
  
- [✅] Visualizzare risultati dell'analisi statistica (03/05/2025)
  - [✅] Distribuzioni del sentiment per notizie vere vs false (03/05/2025)
  - [✅] Correlazioni principali con heatmap (03/05/2025)
  - [✅] Coefficienti della regressione logistica (03/05/2025)
  - [✅] Curve ROC e metriche di valutazione (03/05/2025)
  - [✅] Grafici di scatter plot per visualizzare la separabilità tra classi (03/05/2025)
  - [✅] Diagrammi di dispersione per outlier e pattern anomali (03/05/2025)

- [✅] Creare grafici essenziali (03/05/2025)
  - [✅] Grafici comparativi per le variabili chiave (03/05/2025)
  - [✅] Heatmap della matrice di correlazione (03/05/2025)
  - [✅] Rete di correlazioni tra feature (03/05/2025)
  - [✅] Boxplot per confrontare distribuzioni tra gruppi (03/05/2025)
  - [✅] Violin plot per visualizzare densità delle distribuzioni (03/05/2025)
  - [✅] Dimensionality reduction (PCA/t-SNE) per visualizzazione 2D dei dati (03/05/2025)

- [✅] Visualizzare confronto tra modelli (03/05/2025)
  - [✅] Implementare `src/visualization/model_comparison.py` (03/05/2025)
  - [✅] Confronto di performance tra modelli lineari e non lineari (03/05/2025)
  - [✅] Visualizzazione dell'importanza delle feature nei diversi modelli (03/05/2025)
  - [✅] Grafico di riepilogo dei risultati principali (03/05/2025)
  - [✅] Curve di apprendimento per valutare overfitting (03/05/2025)
  - [✅] Visualizzazioni interattive per esplorare le relazioni tra variabili (03/05/2025)
  - [✅] Dashboard di confronto con principali metriche e grafici (03/05/2025)

- [✅] Preparare tabelle riassuntive (03/05/2025)
  - [✅] Tabella con risultati statistici principali (03/05/2025)
  - [✅] Tabella di confronto tra notizie vere e false (03/05/2025)
  - [✅] Tabella di confronto delle performance dei modelli (03/05/2025)
  - [✅] Matrice di confusione per valutare errori di classificazione (03/05/2025)
  - [✅] Tabella di significatività statistica con p-value ed effect size (03/05/2025)
  - [✅] Report dettagliato delle metriche per ciascun set di feature testato (03/05/2025)

- [✅] Ottimizzare l'esperienza di visualizzazione (03/05/2025)
  - [✅] Applicare palette di colori coerenti e percettivamente efficaci (03/05/2025)
  - [✅] Implementare visualizzazioni accessibili per daltonici (03/05/2025)
  - [✅] Aggiungere annotazioni esplicative ai grafici principali (03/05/2025)
  - [✅] Creare versioni ad alta risoluzione per pubblicazioni (03/05/2025)
  - [✅] Standardizzare stile e formattazione tra tutte le visualizzazioni (03/05/2025)
  - [✅] Verificare l'efficacia comunicativa delle visualizzazioni (03/05/2025)

- [✅] Sviluppare narrazione visiva dei risultati (03/05/2025)
  - [✅] Creare un flusso logico di visualizzazioni che raccontino la storia dei dati (03/05/2025)
  - [✅] Progettare infografiche riassuntive dei risultati principali (03/05/2025)
  - [✅] Integrare visualizzazioni con testo esplicativo per facilitare l'interpretazione (03/05/2025)
  - [✅] Evidenziare visivamente i risultati più significativi e sorprendenti (03/05/2025)
  - [✅] Sviluppare visualizzazioni per pubblici con diversi livelli di competenza tecnica (03/05/2025)

- [✅] Interpretare i risultati (03/05/2025)
  - [✅] Documentare le interpretazioni principali in reports/conclusioni.md (03/05/2025)
  - [✅] Analizzare criticamente i risultati dei test di ipotesi (03/05/2025)
  - [✅] Interpretare il significato dell'effect size limitato nelle correlazioni (03/05/2025)
  - [✅] Spiegare le implicazioni del miglioramento di performance con Random Forest (03/05/2025)
  - [✅] Valutare criticamente il possibile overfitting sui dati specifici (03/05/2025)
  - [✅] Contestualizzare i risultati nel panorama più ampio della ricerca sulla disinformazione (03/05/2025)
  - [✅] Analizzare le limitazioni metodologiche emerse dai risultati (03/05/2025)
  - [✅] Identificare pattern e anomalie non evidenti nelle analisi iniziali (03/05/2025)
  - [✅] Proporre interpretazioni alternative dei risultati (03/05/2025)

- [✅] Validare le interpretazioni (03/05/2025)
  - [✅] Confrontare i risultati con la letteratura scientifica esistente (03/05/2025)
  - [✅] Triangolare le interpretazioni utilizzando diversi metodi di analisi (03/05/2025)
  - [✅] Verificare la consistenza interna tra diversi risultati (03/05/2025)
  - [✅] Identificare e spiegare eventuali apparenti contraddizioni (03/05/2025)
  - [✅] Valutare la generalizzabilità delle interpretazioni oltre il dataset specifico (03/05/2025)

- [✅] Integrare risultati quantitativi e qualitativi (03/05/2025)
  - [✅] Complementare le analisi statistiche con esempi qualitativi specifici (03/05/2025)
  - [✅] Illustrare i pattern identificati con casi studio rappresentativi (03/05/2025)
  - [✅] Connettere metriche quantitative con fenomeni sociali qualitativi (03/05/2025)
  - [✅] Contestualizzare i risultati numerici nel framework teorico del progetto (03/05/2025)
  - [✅] Sviluppare una comprensione olistica integrando diversi tipi di evidenze (03/05/2025)

## 6. Documentazione e reporting

- [✅] Preparare un report scientifico (03/05/2025)
  - [✅] Riassumere la metodologia utilizzata (03/05/2025)
  - [✅] Presentare i risultati principali con interpretazioni (03/05/2025)
  - [✅] Discutere la significatività statistica (03/05/2025)

- [✅] Documentare il protocollo di analisi (03/05/2025)
  - [✅] Descrivere chiaramente ipotesi e metodi (03/05/2025)
  - [✅] Specificare criteri di accettazione/rifiuto delle ipotesi (03/05/2025)

- [✅] Discutere limitazioni e implicazioni (03/05/2025)
  - [✅] Identificare limiti metodologici (03/05/2025)
  - [✅] Considerare le possibili generalizzazioni dei risultati (03/05/2025)

- [✅] Preparare materiale per la riproducibilità (03/05/2025)
  - [✅] Documentare le procedure di analisi (03/05/2025)
  - [✅] Assicurare che il codice sia commentato e comprensibile (03/05/2025)

## 7. Validazione e verifica

- [✅] Verificare i risultati (03/05/2025)
  - [✅] Controllare la correttezza dell'implementazione (03/05/2025)
  - [✅] Validare le conclusioni statistiche (03/05/2025)

- [✅] Valutare la validità dello studio (03/05/2025)
  - [✅] Esaminare la validità interna (03/05/2025)
  - [✅] Considerare la validità esterna (03/05/2025)

## 8. Conclusione

- [✅] Finalizzare il progetto (03/05/2025)
  - [✅] Controllare la coerenza complessiva (03/05/2025)
  - [✅] Verificare che le conclusioni siano supportate dai dati (03/05/2025)

- [✅] Preparare la presentazione (03/05/2025)
  - [✅] Creare slide focalizzate sui risultati chiave (03/05/2025)
  - [✅] Evidenziare le implicazioni principali (03/05/2025)

- [✅] Riflettere sulle direzioni future (03/05/2025)
  - [✅] Identificare possibili estensioni della ricerca (03/05/2025)

## Note e osservazioni

*Utilizza questa sezione per annotare idee, problemi incontrati o decisioni importanti prese durante il progetto.*
