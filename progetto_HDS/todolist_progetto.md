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

- [ ] Creare una matrice di feature integrata
  - [ ] Combinare le feature di sentiment, stance e leggibilità
  - [ ] Standardizzare i valori numerici per l'analisi

- [ ] Documentare le feature in `feature_description.md`
  - [ ] Spiegare il significato di ogni feature estratta
  - [ ] Descrivere gli algoritmi utilizzati

## 4. Analisi statistica

- [ ] Implementare `src/analysis/descriptive.py`
  - [ ] Calcolare statistiche descrittive (media, mediana, deviazione standard)
  - [ ] Visualizzare distribuzioni principali
  - [ ] Identificare e gestire outlier

- [ ] Implementare test statistici essenziali in `src/analysis/hypothesis_tests.py`
  - [ ] Test per confrontare il sentiment tra notizie vere e false
  - [ ] Test per analizzare la correlazione tra sentiment e veridicità
  - [ ] Test per valutare la relazione tra livello di acculturazione e propensione alla disinformazione

- [ ] Implementare analisi di correlazione
  - [ ] Calcolare coefficienti di correlazione tra le variabili principali
  - [ ] Creare matrice di correlazione

- [ ] Calcolare la significatività statistica
  - [ ] Documentare p-value e dimensione dell'effetto
  - [ ] Interpretare i risultati nel contesto del progetto

- [ ] Validare i risultati
  - [ ] Utilizzare tecniche di validazione incrociata
  - [ ] Verificare la robustezza dei risultati

## 5. Visualizzazione e interpretazione dei risultati

- [✅] Implementare `src/visualization/plot_pheme_stats.py` (29/04/2025)
  - [✅] Funzioni per visualizzare distribuzioni degli eventi nel dataset (29/04/2025)
  - [✅] Funzioni per visualizzare distribuzione della veridicità (29/04/2025)
  - [✅] Funzioni per visualizzare distribuzioni delle reazioni (29/04/2025)
  
- [ ] Implementare `src/visualization/sentiment_plots.py`  
  - [ ] Visualizzare distribuzioni del sentiment per notizie vere vs false
  - [ ] Visualizzare correlazioni principali

- [ ] Creare grafici essenziali
  - [ ] Grafici comparativi per le variabili chiave
  - [ ] Heatmap della matrice di correlazione

- [ ] Preparare tabelle riassuntive
  - [ ] Tabella con risultati statistici principali
  - [ ] Tabella di confronto tra notizie vere e false

- [ ] Interpretare i risultati
  - [ ] Documentare le interpretazioni in reports/analysis_results.md

## 6. Documentazione e reporting

- [ ] Preparare un report scientifico
  - [ ] Riassumere la metodologia utilizzata
  - [ ] Presentare i risultati principali con interpretazioni
  - [ ] Discutere la significatività statistica

- [ ] Documentare il protocollo di analisi
  - [ ] Descrivere chiaramente ipotesi e metodi
  - [ ] Specificare criteri di accettazione/rifiuto delle ipotesi

- [ ] Discutere limitazioni e implicazioni
  - [ ] Identificare limiti metodologici
  - [ ] Considerare le possibili generalizzazioni dei risultati

- [ ] Preparare materiale per la riproducibilità
  - [ ] Documentare le procedure di analisi
  - [ ] Assicurare che il codice sia commentato e comprensibile

## 7. Validazione e verifica

- [ ] Verificare i risultati
  - [ ] Controllare la correttezza dell'implementazione
  - [ ] Validare le conclusioni statistiche

- [ ] Valutare la validità dello studio
  - [ ] Esaminare la validità interna
  - [ ] Considerare la validità esterna

## 8. Conclusione

- [ ] Finalizzare il progetto
  - [ ] Controllare la coerenza complessiva
  - [ ] Verificare che le conclusioni siano supportate dai dati

- [ ] Preparare la presentazione
  - [ ] Creare slide focalizzate sui risultati chiave
  - [ ] Evidenziare le implicazioni principali

- [ ] Riflettere sulle direzioni future
  - [ ] Identificare possibili estensioni della ricerca

## Note e osservazioni

*Utilizza questa sezione per annotare idee, problemi incontrati o decisioni importanti prese durante il progetto.*
