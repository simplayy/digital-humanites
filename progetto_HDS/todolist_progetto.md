# Todolist del Progetto: Studio sulla Relazione tra Sentiment Analysis dei Commenti e Fake News

## Panoramica
Questa todolist traccia tutte le attività necessarie per completare il progetto di Human Data Science sull'analisi statistica della relazione tra sentiment nei commenti e veridicità delle notizie.

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
  mkdir -p notebooks
  mkdir -p src/{data,features,analysis,visualization}
  mkdir -p tests
  mkdir -p results/{figures,tables}
  touch src/__init__.py
  touch src/data/__init__.py
  touch src/features/__init__.py
  touch src/analysis/__init__.py
  touch src/visualization/__init__.py
  ```

## 2. Acquisizione e preparazione dei dati

- [ ] Scaricare il dataset FakeNewsNet
  - [ ] Ricercare la fonte più aggiornata del dataset
  - [ ] Implementare lo script per il download automatizzato

- [ ] Creare lo script `src/data/download.py`
  - [ ] Implementare funzioni per scaricare i dati
  - [ ] Implementare funzioni per organizzare i dati scaricati

- [ ] Implementare `src/data/preprocess.py`
  - [ ] Funzioni per pulizia dei dati (rimozione duplicati, gestione valori mancanti)
  - [ ] Normalizzazione dei testi dei commenti
  - [ ] Tokenizzazione e preparazione per analisi NLP

- [ ] Creare e completare il notebook `notebooks/1_exploratory.ipynb`
  - [ ] Caricamento dati
  - [ ] Statistiche descrittive iniziali
  - [ ] Visualizzazione della distribuzione dei dati
  - [ ] Identificazione di anomalie o pattern interessanti

- [ ] Documentare la struttura e le caratteristiche del dataset
  - [ ] Creare un file `data_dictionary.md` con dettagli su tutte le variabili

## 3. Estrazione delle feature

- [ ] Creare il notebook `notebooks/2_feature_extraction.ipynb`

- [ ] Implementare `src/features/sentiment.py`
  - [ ] Funzione per analisi del sentiment base (positività/negatività)
  - [ ] Calcolo della polarità e dell'intensità del sentiment

- [ ] Implementare `src/features/emotions.py`
  - [ ] Analisi delle emozioni fondamentali (rabbia, paura, gioia, tristezza, ecc.)
  - [ ] Calcolo dell'intensità delle emozioni
  - [ ] Analisi della variabilità emotiva nei commenti

- [ ] Implementare il rilevamento del sarcasmo
  - [ ] Funzione per rilevare la presenza di sarcasmo
  - [ ] Calcolo del grado di certezza nella detection del sarcasmo

- [ ] Implementare funzioni per la stance detection
  - [ ] Identificazione della posizione rispetto alla notizia (favorevole/contrario)
  - [ ] Calcolo del grado di certezza nella stance

- [ ] Implementare `src/features/readability.py`
  - [ ] Calcolo degli indici di Flesch Reading Ease
  - [ ] Calcolo di Flesch-Kincaid Grade Level
  - [ ] Calcolo di SMOG Index e Gunning Fog

- [ ] Implementare funzioni per misurare la diversità lessicale
  - [ ] Calcolo del Type-Token Ratio
  - [ ] Analisi della ricchezza del vocabolario

- [ ] Implementare `src/features/metadata.py`
  - [ ] Estrazione del volume di interazione (numero di commenti)
  - [ ] Analisi dell'engagement (like, risposte, profondità delle discussioni)
  - [ ] Analisi dei pattern temporali nei commenti
  - [ ] Calcolo delle statistiche utente

- [ ] Creare una matrice di feature completa
  - [ ] Aggregazione di tutte le feature estratte
  - [ ] Normalizzazione/standardizzazione delle feature numeriche

- [ ] Documentare tutte le feature estratte in `feature_description.md`
  - [ ] Descrizione di ogni feature
  - [ ] Interpretazione e significato nel contesto del progetto

## 4. Analisi statistica

- [ ] Creare il notebook `notebooks/3_statistical_analysis.ipynb`

- [ ] Implementare `src/analysis/descriptive.py`
  - [ ] Calcolo delle statistiche descrittive per ogni feature
  - [ ] Visualizzazione delle distribuzioni
  - [ ] Identificazione di outlier

- [ ] Implementare `src/analysis/hypothesis_tests.py`
  - [ ] Test t di Student per confrontare medie di sentiment
  - [ ] Test ANOVA per confronti multipli
  - [ ] Test chi-quadro per variabili categoriche
  - [ ] Test non parametrici (Mann-Whitney U, Kruskal-Wallis) quando necessario

- [ ] Implementare `src/analysis/correlation.py`
  - [ ] Calcolo del coefficiente di correlazione di Pearson
  - [ ] Calcolo del coefficiente di correlazione di Spearman
  - [ ] Creazione della matrice di correlazione

- [ ] Implementare `src/analysis/regression.py`
  - [ ] Modelli di regressione logistica
  - [ ] Calcolo dell'R² e analisi della varianza spiegata
  - [ ] Calcolo degli odds ratio e intervalli di confidenza

- [ ] Applicare correzioni per test multipli
  - [ ] Implementare la correzione di Bonferroni
  - [ ] Implementare la procedura di Benjamini-Hochberg (FDR)
  - [ ] Implementare la correzione di Holm-Bonferroni

- [ ] Calcolare e interpretare p-value ed effect size
  - [ ] Documentare ogni test con i relativi risultati
  - [ ] Interpretare la significatività statistica in contesto

- [ ] Eseguire validazione incrociata
  - [ ] Implementare k-fold cross validation
  - [ ] Verificare la stabilità dei risultati

- [ ] Condurre analisi di sensibilità
  - [ ] Variare i parametri dei test
  - [ ] Rimuovere outlier e ripetere le analisi
  - [ ] Eseguire subsampling per verificare la consistenza

## 5. Visualizzazione e interpretazione dei risultati

- [ ] Creare il notebook `notebooks/4_visualization.ipynb`

- [ ] Implementare `src/visualization/plots.py`
  - [ ] Funzioni per visualizzare distribuzioni del sentiment
  - [ ] Funzioni per visualizzare correlazioni
  - [ ] Funzioni per visualizzare effect size e intervalli di confidenza

- [ ] Creare visualizzazioni delle distribuzioni del sentiment
  - [ ] Boxplot e violinplot per confrontare gruppi
  - [ ] Istogrammi e density plot per visualizzare distribuzioni

- [ ] Generare grafici per le correlazioni significative
  - [ ] Heatmap della matrice di correlazione
  - [ ] Scatter plot per relazioni bivariate importanti

- [ ] Visualizzare gli effect size e intervalli di confidenza
  - [ ] Forest plot per odds ratio
  - [ ] Grafici ad errore per intervalli di confidenza

- [ ] Preparare tabelle riassuntive dei risultati statistici
  - [ ] Tabella con p-value, effect size e significatività
  - [ ] Tabella di confronto tra feature diverse

- [ ] Interpretare i risultati in relazione alle ipotesi iniziali
  - [ ] Documentare le interpretazioni nel notebook

## 6. Documentazione e reporting

- [ ] Preparare un report metodologico dettagliato
  - [ ] Descrivere tutti i test statistici eseguiti
  - [ ] Giustificare la scelta di ciascun test
  - [ ] Presentare i risultati con interpretazioni

- [ ] Documentare il protocollo di analisi completo
  - [ ] Procedure di estrazione delle feature
  - [ ] Pipeline di analisi statistica
  - [ ] Criteri di decisione per l'accettazione/rifiuto delle ipotesi

- [ ] Creare una presentazione dei risultati principali
  - [ ] Slide con risultati chiave
  - [ ] Visualizzazioni efficaci
  - [ ] Conclusioni principali

- [ ] Documentare limitazioni e considerazioni etiche
  - [ ] Limitazioni metodologiche
  - [ ] Considerazioni sulla privacy e sui bias
  - [ ] Riflessioni sulle implicazioni sociali

- [ ] Preparare materiale per la riproducibilità
  - [ ] README dettagliato
  - [ ] Istruzioni per riprodurre lo studio
  - [ ] Codice ben commentato

## 7. Test e validazione

- [ ] Implementare `tests/test_features.py`
  - [ ] Test unitari per le funzioni di estrazione delle feature
  - [ ] Verifiche di output per input noti

- [ ] Implementare `tests/test_analysis.py`
  - [ ] Test per le procedure di analisi statistica
  - [ ] Controlli di correttezza dei calcoli

- [ ] Verificare la correttezza dei risultati statistici
  - [ ] Confrontare con implementazioni alternative
  - [ ] Verificare i risultati con casi di test noti

- [ ] Eseguire controlli sulla validità
  - [ ] Valutare la validità interna dello studio
  - [ ] Considerare la validità esterna e la generalizzabilità

## 8. Passi finali

- [ ] Revisionare l'intero progetto
  - [ ] Controllare la coerenza tra le diverse parti
  - [ ] Verificare la completezza dell'analisi

- [ ] Preparare una presentazione finale
  - [ ] Creare slide riassuntive
  - [ ] Preparare demo o esempi significativi

- [ ] Condividere i risultati in formato accessibile
  - [ ] Generare report PDF
  - [ ] Preparare abstract e punti chiave

- [ ] Considerare possibili estensioni future
  - [ ] Documentare idee per miglioramenti
  - [ ] Identificare direzioni per ricerche future

## Note e osservazioni

*Utilizza questa sezione per annotare idee, problemi incontrati o decisioni importanti prese durante il progetto.*
