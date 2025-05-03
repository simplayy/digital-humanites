# Protocollo di Analisi

Questo documento descrive in dettaglio il protocollo di analisi seguito nello studio sulla relazione tra pattern di sentiment nei commenti e veridicità delle notizie utilizzando il dataset PHEME.

## 1. Ipotesi di Ricerca

### Ipotesi Principale
**H0**: Non esistono differenze statisticamente significative nei pattern di sentiment tra i commenti alle notizie vere e quelli alle notizie false.

**H1**: Esistono differenze statisticamente significative nei pattern di sentiment tra i commenti alle notizie vere e quelli alle notizie false.

### Ipotesi Secondarie

1. **Sulla soggettività**
   - **H0**: Non c'è differenza significativa nel livello di soggettività tra commenti a notizie vere e false.
   - **H1**: I commenti alle notizie false mostrano livelli di soggettività significativamente diversi rispetto ai commenti alle notizie vere.

2. **Sulla polarità**
   - **H0**: Non c'è differenza significativa nella polarità del sentiment tra commenti a notizie vere e false.
   - **H1**: I commenti alle notizie false mostrano polarità significativamente diversa rispetto ai commenti alle notizie vere.

3. **Sull'atteggiamento (stance)**
   - **H0**: Non c'è differenza significativa nell'atteggiamento (stance) tra commenti a notizie vere e false.
   - **H1**: I commenti alle notizie false mostrano atteggiamenti significativamente diversi rispetto ai commenti alle notizie vere.

4. **Sulla leggibilità e acculturazione**
   - **H0**: Non c'è differenza significativa nelle metriche di leggibilità e acculturazione tra commenti a notizie vere e false.
   - **H1**: I commenti alle notizie false mostrano livelli di leggibilità e acculturazione significativamente diversi rispetto ai commenti alle notizie vere.

5. **Sul potere predittivo**
   - **H0**: Le feature di sentiment non hanno potere predittivo significativo sulla veridicità delle notizie.
   - **H1**: Le feature di sentiment hanno potere predittivo significativo sulla veridicità delle notizie.

## 2. Criteri di Accettazione/Rifiuto delle Ipotesi

### Soglia di Significatività
- Livello di significatività α = 0.05
- Correzione di Bonferroni per test multipli: α' = 0.05 / n (dove n è il numero di test)

### Criteri per i Test di Ipotesi
1. **Rifiuto dell'ipotesi nulla**:
   - p-value < α' (dopo correzione di Bonferroni)
   - Effect size non trascurabile (Cohen's d ≥ 0.2 o |r| ≥ 0.1)

2. **Accettazione dell'ipotesi nulla**:
   - p-value ≥ α' (dopo correzione di Bonferroni) OPPURE
   - Effect size trascurabile (Cohen's d < 0.2 o |r| < 0.1) anche con p-value significativo

### Criteri per i Modelli Predittivi
1. **Capacità predittiva significativa**:
   - AUC significativamente superiore a 0.5 (test bootstrapping con p < 0.05)
   - Performance cross-validation stabile (deviazione standard della metrica < 10% della media)

2. **Differenze tra modelli significative**:
   - Differenza di AUC > 0.05 tra modelli comparati
   - Test statistico di confronto delle curve ROC significativo (p < 0.05)

### Valutazione della Rilevanza Pratica
Anche in presenza di risultati statisticamente significativi, la rilevanza pratica è valutata considerando:
- Dimensione dell'effetto (effect size)
- Miglioramento incrementale delle performance predittive
- Complessità computazionale dei modelli rispetto al miglioramento ottenuto

## 3. Metodologia Dettagliata

### 3.1 Preparazione del Dataset

#### Fonte dei Dati
- **Dataset**: PHEME
- **Contenuto**: Thread di conversazione Twitter relativi a eventi di attualità
- **Acquisizione**: Download da repository pubblici o API autorizzate

#### Criteri di Inclusione ed Esclusione
- **Inclusione**:
  - Thread con etichetta di veridicità confermata (vero/falso)
  - Thread con almeno 3 reazioni (commenti)
  - Testi in lingua inglese

- **Esclusione**:
  - Thread con veridicità non verificata o ambigua
  - Thread senza reazioni
  - Testi troppo brevi (<10 caratteri)

#### Preprocessing dei Dati
1. **Pulizia del testo**:
   - Rimozione di URL e link
   - Rimozione di emoji e caratteri speciali
   - Normalizzazione di hashtag e menzioni

2. **Normalizzazione linguistica**:
   - Conversione a minuscolo
   - Rimozione di stop words
   - Lemmatizzazione o stemming
   - Gestione di abbreviazioni comuni

3. **Gestione dei valori mancanti**:
   - Esclusione di record con valori mancanti critici
   - Imputazione per valori mancanti non critici (quando possibile)
   - Documentazione di tutte le decisioni di imputazione

4. **Strutturazione gerarchica**:
   - Organizzazione dei tweet in thread
   - Assegnazione di indici di posizione nella conversazione
   - Tracciamento di relazioni di risposta

### 3.2 Estrazione delle Feature

#### Feature di Sentiment
- **sentiment_polarity**: Calcolata con TextBlob, normalizzata nell'intervallo [-1, 1]
- **sentiment_subjectivity**: Calcolata con TextBlob, normalizzata nell'intervallo [0, 1]

#### Feature di Stance
- **stance_score**: Combinazione di similarità coseno tra testo sorgente e risposta, ponderata con sentiment

#### Feature di Leggibilità e Acculturazione
- **flesch_reading_ease**: Indice di Flesch calcolato con textstat
- **type_token_ratio**: Rapporto tra numero di parole uniche e totali
- **formal_language_score**: Basato su dizionario di espressioni formali e informali
- **vocabulary_richness**: Basato su hapax legomena e diversità lessicale
- **avg_word_length**: Media della lunghezza delle parole
- **long_words_ratio**: Proporzione di parole con >6 caratteri
- **culture_score**: Punteggio composito basato su diversità lessicale, complessità sintattica e formalità

### 3.3 Analisi Statistica

#### Statistiche Descrittive
- Medie, mediane, deviazioni standard di tutte le feature
- Visualizzazioni delle distribuzioni (istogrammi, box plot)
- Analisi di outlier con metodo IQR (Interquartile Range)

#### Test di Ipotesi
1. **Test di normalità**:
   - Shapiro-Wilk per valutare la normalità delle distribuzioni
   - Q-Q plot per visualizzazione

2. **Test per confrontare gruppi**:
   - Per distribuzioni normali: t-test indipendente
   - Per distribuzioni non normali: test di Mann-Whitney U

3. **Correzione per test multipli**:
   - Metodo di Bonferroni per controllare il family-wise error rate
   - Calcolo di p-value corretti: p' = p * n

4. **Calcolo dell'effect size**:
   - Cohen's d per t-test
   - r per Mann-Whitney U

#### Analisi di Correlazione
1. **Correlazioni bivariate**:
   - Correlazione di Pearson per variabili normali
   - Correlazione di Spearman per variabili non normali

2. **Matrice di correlazione completa**:
   - Visualizzazione con heatmap
   - Identificazione di multicollinearità

3. **Significatività delle correlazioni**:
   - Test di significatività con correzione per test multipli
   - Calcolo della forza della correlazione (trascurabile, debole, moderata, forte)

### 3.4 Modellazione Predittiva

#### Preparazione dei Dati per i Modelli
1. **Suddivisione del dataset**:
   - Training set (70%)
   - Test set (30%)
   - Stratificazione per preservare la distribuzione della variabile target

2. **Standardizzazione delle feature**:
   - Applicazione di StandardScaler (media=0, deviazione standard=1)
   - Scaling applicato separatamente a training e test set

3. **Gestione dello sbilanciamento**:
   - Utilizzo di class_weight='balanced' nei modelli
   - Valutazione dell'impatto con metriche appropriate

#### Modello Lineare: Regressione Logistica
1. **Configurazione**:
   - Regolarizzazione L2
   - Ottimizzazione con saga solver
   - Massimo 1000 iterazioni

2. **Valutazione**:
   - Accuracy, precision, recall, F1-score
   - Curva ROC e AUC
   - Valutazione dei coefficienti e loro significatività

#### Modello Non Lineare: Random Forest
1. **Configurazione**:
   - 100 estimatori (alberi)
   - class_weight='balanced'
   - Criteri di split: gini impurity

2. **Valutazione**:
   - Accuracy, precision, recall, F1-score
   - Curva ROC e AUC
   - Importanza delle feature

3. **Analisi dell'importanza delle feature**:
   - Feature importance standard
   - Permutation importance per robustezza
   - Visualizzazione delle top N feature

#### Validazione Incrociata
1. **Strategia**:
   - 5-fold stratified cross-validation
   - Mantenimento della proporzione delle classi in ogni fold

2. **Metriche**:
   - Media e deviazione standard delle metriche
   - Valutazione della stabilità delle performance

#### Confronto tra Set di Feature
1. **Definizione dei set**:
   - sentiment_only: solo feature di sentiment
   - stance_only: solo feature di stance
   - readability_only: solo feature di leggibilità
   - sentiment_stance: combinazione di sentiment e stance
   - sentiment_readability: combinazione di sentiment e leggibilità
   - all_features: tutte le feature

2. **Valutazione comparativa**:
   - Performance con gli stessi parametri di modello
   - Confronto di metriche (AUC, F1-score)
   - Valutazione del trade-off complessità/performance

### 3.5 Interpretazione dei Risultati

#### Approccio sistematico
1. **Triangolazione dei metodi**:
   - Confronto tra risultati di test statistici e modelli predittivi
   - Verifica di coerenza tra diversi approcci analitici

2. **Contestualizzazione dei risultati**:
   - Confronto con studi simili nella letteratura
   - Considerazione del contesto specifico del dataset

3. **Valutazione critica**:
   - Identificazione di potenziali bias nei dati o nell'analisi
   - Discussione di interpretazioni alternative

#### Framework interpretativo
1. **Significatività statistica vs. rilevanza pratica**:
   - Distinzione esplicita tra risultati statisticamente significativi e praticamente rilevanti
   - Enfasi sull'effect size e sulla sua interpretazione

2. **Causalità vs. correlazione**:
   - Chiara distinzione tra associazioni e relazioni causali
   - Discussione di potenziali meccanismi causali, quando appropriato

3. **Generalizzabilità**:
   - Valutazione esplicita dei limiti di generalizzabilità
   - Discussione dei domini di applicabilità dei risultati

## 4. Procedure di Controllo Qualità

### Validità Interna
- Controlli di coerenza tra diversi metodi analitici
- Verifica di assunzioni statistiche per ogni test utilizzato
- Analisi di sensibilità per decisioni metodologiche chiave

### Riproducibilità
- Utilizzo di seed per processi stocastici (random_state=42)
- Documentazione dettagliata di ogni passaggio analitico
- Script annotati con commenti esplicativi
- Versioning del codice e dei dati

### Revisione Peer
- Revisione del codice per identificare errori logici o implementativi
- Verifica indipendente dei risultati principali
- Feedback sulla validità delle interpretazioni

## 5. Considerazioni Etiche

### Privacy dei Dati
- Utilizzo di dati pubblicamente disponibili
- Nessuna identificazione di utenti individuali
- Aggregazione appropriata per proteggere la privacy

### Bias e Rappresentatività
- Riconoscimento esplicito dei limiti di rappresentatività del dataset
- Discussione di potenziali bias (linguistici, culturali, temporali)
- Considerazioni sul contesto culturale e sociale dei dati

### Implicazioni dell'Interpretazione
- Cautela nell'esprimere conclusioni con potenziali implicazioni normative
- Distinzione tra risultati descrittivi e prescrittivi
- Riconoscimento delle possibili conseguenze sociali delle interpretazioni

## 6. Criteri di Documentazione

### Reportistica
- Documentazione completa di tutti i passaggi analitici
- Inclusione di risultati negativi o non significativi
- Visualizzazioni chiare ed esplicative

### Gestione del Codice
- Organizzazione modulare del codice
- Documentazione delle funzioni con docstring
- Controllo di versione con Git

### Archivio dei Risultati
- Salvataggio strutturato di tutti gli output analitici
- Conservazione dei parametri di ogni modello
- Esportazione di tabelle e figure in formati standard

---

Questo protocollo è stato sviluppato per garantire rigore metodologico, trasparenza e riproducibilità nell'analisi della relazione tra sentiment nei commenti e veridicità delle notizie.
