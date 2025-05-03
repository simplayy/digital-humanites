# 2. Dataset e Metodologia

## Dataset PHEME

### Origine e Struttura

Il PHEME dataset (Zubiaga et al., 2016) è una risorsa sviluppata specificamente per lo studio della diffusione di rumour sui social media. Contiene conversazioni Twitter relative a diversi eventi di attualità, con annotazioni sulla veridicità delle affermazioni e sulla struttura conversazionale dei thread.

Il dataset prende il nome dal personaggio mitologico greco Pheme, personificazione delle voci e dei rumour, una scelta simbolica che riflette l'obiettivo del progetto di studiare la diffusione di informazioni non verificate online.

**Caratteristiche principali**:
- 6.425 thread di conversazione
- 105.354 tweet totali
- Copertura di eventi diversi (Charlie Hebdo, Ferguson, Germanwings crash, ecc.)
- Annotazioni manuali di veridicità (true, false, unverified)
- Metadati conversazionali (relazioni di risposta, posizione nel thread)

### Distribuzione del Target

Una caratteristica importante del dataset è lo sbilanciamento significativo tra le classi di veridicità:

```
Vere: 93% (5.973 thread)
False: 7% (452 thread)
```

Questo sbilanciamento riflette la realtà dei social media, dove le notizie verificate tendono ad essere più numerose, ma rappresenta anche una sfida metodologica che richiede tecniche specifiche per garantire risultati affidabili.

### Eventi Coperti

Il dataset include tweet relativi a cinque eventi principali:

1. **Charlie Hebdo**: l'attacco terroristico alla sede del settimanale satirico francese nel gennaio 2015
2. **Sydney Siege**: la crisi degli ostaggi a Sydney nel dicembre 2014
3. **Ferguson**: le proteste seguite all'uccisione di Michael Brown in Missouri nel 2014
4. **Ottawa Shooting**: la sparatoria al Parlamento canadese nell'ottobre 2014
5. **Germanwings Crash**: l'incidente aereo sulle Alpi francesi nel marzo 2015

Questa diversità di eventi permette di analizzare le reazioni a diverse tipologie di notizie (terrorismo, proteste sociali, incidenti) in contesti culturali e geografici diversi.

### Caratteristiche dei Thread Conversazionali

I thread nel dataset PHEME sono strutturati gerarchicamente:
- **Tweet sorgente**: il post originale che contiene la notizia
- **Reazioni dirette**: risposte immediate al tweet sorgente
- **Reazioni indirette**: risposte alle risposte, che formano conversazioni ramificate

Questa struttura conversazionale è stata preservata nell'analisi attraverso feature come `reaction_index`, che indica la posizione del commento nella catena di risposte.

## Metodologia

### Approccio Generale

La nostra metodologia segue un approccio rigoroso basato sui principi del metodo scientifico:

1. **Formulazione delle ipotesi**: definizione chiara delle domande di ricerca e delle ipotesi da testare
2. **Acquisizione e preparazione dei dati**: download, pulizia e strutturazione del dataset PHEME
3. **Estrazione delle feature**: calcolo di metriche linguistiche, di sentiment e di leggibilità
4. **Analisi esplorativa**: esame delle distribuzioni e identificazione di pattern preliminari
5. **Test statistici**: verifica formale delle ipotesi tramite test appropriati
6. **Modellazione predittiva**: sviluppo e confronto di modelli lineari e non lineari
7. **Interpretazione**: analisi critica dei risultati nel contesto della letteratura esistente
8. **Validazione**: verifica della robustezza dei risultati attraverso tecniche di cross-validation

Tutte le analisi sono state condotte utilizzando Python 3.10 e librerie specializzate per l'analisi dei dati e il natural language processing (pandas, scikit-learn, nltk, TextBlob, textstat).

### Preprocessing dei Dati

La preparazione dei dati è stata una fase critica per garantire risultati affidabili. Il processo ha incluso:

#### 1. Pulizia dei testi

```python
def clean_text(text):
    """Pulisce e normalizza il testo del tweet."""
    if not text or pd.isna(text):
        return ""
    
    # Rimozione URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Rimozione menzioni e hashtag
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Normalizzazione
    text = text.lower()
    
    # Rimozione caratteri speciali
    text = re.sub(r'[^\w\s]', '', text)
    
    # Rimozione spazi multipli
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
```

Le operazioni di pulizia hanno incluso:
- Rimozione di URL e link tramite espressioni regolari
- Rimozione di menzioni (@username) e hashtag
- Normalizzazione di emoji e caratteri speciali
- Correzione di errori comuni di ortografia

#### 2. Normalizzazione linguistica

```python
def normalize_text(text, remove_stopwords=True):
    """Normalizza il testo: tokenizzazione, rimozione stopwords, lemmatizzazione."""
    if not text or len(text) < 3:
        return ""
    
    # Tokenizzazione
    tokens = word_tokenize(text)
    
    # Rimozione stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatizzazione
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)
```

Il processo di normalizzazione ha compreso:
- Conversione a minuscolo
- Rimozione di stop words
- Lemmatizzazione per ridurre le parole alla forma base
- Tokenizzazione per analisi a livello di parola

#### 3. Gestione dei valori mancanti

Abbiamo adottato criteri rigorosi per la gestione dei dati incompleti o inadeguati:
- Esclusione di tweet con testi insufficienti (< 10 caratteri)
- Esclusione di thread con meno di 3 reazioni
- Documentazione di tutti i criteri di esclusione per trasparenza

#### 4. Strutturazione gerarchica

Per preservare il contesto conversazionale, abbiamo:
- Ricostruito la struttura dei thread
- Assegnato indici posizionali a ciascun tweet
- Associato ciascun commento al proprio tweet sorgente

### Estrazione delle Feature

Per catturare diverse dimensioni delle reazioni linguistiche, abbiamo estratto tre categorie principali di feature:

#### 1. Feature di Sentiment Analysis

Utilizzando la libreria TextBlob, abbiamo estratto:

- **sentiment_polarity** [-1.0, 1.0]: misura quanto positivo o negativo è il testo
  - Valori negativi indicano sentiment negativo
  - Valori positivi indicano sentiment positivo
  - Zero indica neutralità

- **sentiment_subjectivity** [0.0, 1.0]: misura quanto soggettivo od oggettivo è il testo
  - Valori vicini a 0 indicano linguaggio oggettivo
  - Valori vicini a 1 indicano linguaggio soggettivo

```python
def extract_sentiment_features(text):
    """Estrae feature di sentiment dal testo."""
    if not text or len(text) < 5:
        return None, None
    
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity
```

#### 2. Feature di Stance Analysis

La stance misura l'atteggiamento di un commento rispetto al tweet principale:

- **stance_score** [-1.0, 1.0]: combinazione di similarità tematica e sentiment
  - Valori negativi indicano atteggiamento critico/oppositivo
  - Valori positivi indicano atteggiamento supportivo
  - Valori vicini a 0 indicano neutralità o non pertinenza

La stance è stata calcolata combinando:
- Similarità del coseno tra vettori TF-IDF di tweet sorgente e commento
- Polarità del sentiment del commento
- Posizione del commento nel thread

```python
def calculate_stance_score(source_text, comment_text, comment_sentiment):
    """Calcola lo stance score come combinazione di similarità e sentiment."""
    if not source_text or not comment_text:
        return 0.0
    
    # Calcolo similarità del coseno
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([source_text, comment_text])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        cosine_sim = 0.0
    
    # Combinazione ponderata
    stance_score = (0.7 * cosine_sim) + (0.3 * comment_sentiment)
    
    return max(min(stance_score, 1.0), -1.0)  # Limitato a [-1.0, 1.0]
```

#### 3. Feature di Leggibilità e Acculturazione

Queste feature misurano la complessità linguistica e il livello di acculturazione:

- **flesch_reading_ease** [0-100]: indice di leggibilità (più alto = più leggibile)
- **type_token_ratio** [0-1]: rapporto tra parole uniche e totali
- **formal_language_score** [0-1]: livello di formalità del linguaggio
- **vocabulary_richness** [0-1]: basata su hapax legomena
- **avg_word_length**: lunghezza media delle parole
- **long_words_ratio** [0-1]: proporzione di parole lunghe (>6 caratteri)
- **culture_score** [0-1]: punteggio composito di acculturazione

Il **culture_score** è una feature composita calcolata come:

```python
def calculate_culture_score(text_features):
    """Calcola il punteggio di acculturazione come combinazione ponderata di feature linguistiche."""
    if text_features is None:
        return None
    
    # Normalizzazione dell'indice di Flesch (inverso, poiché valori alti = bassa complessità)
    norm_flesch = 1 - (text_features['flesch_reading_ease'] / 100)
    
    # Composizione ponderata
    culture_score = (
        (0.4 * text_features['vocabulary_richness']) + 
        (0.3 * text_features['formal_language_score']) + 
        (0.2 * text_features['type_token_ratio']) + 
        (0.1 * norm_flesch)
    )
    
    return min(max(culture_score, 0.0), 1.0)  # Garantisce intervallo [0,1]
```

Questa misura composita riflette:
- La ricchezza del vocabolario utilizzato
- Il livello di formalità del linguaggio
- La diversità lessicale
- La complessità sintattica

### Metodi di Analisi

La nostra strategia analitica ha combinato diverse tecniche complementari:

#### Analisi Esplorativa

Prima di testare formalmente le ipotesi, abbiamo condotto un'analisi esplorativa per:
- Esaminare le distribuzioni delle feature
- Identificare outlier e valori anomali
- Visualizzare relazioni preliminari tra feature
- Generare statistiche descrittive per gruppi di thread

#### Test Statistici

Per verificare le differenze nei pattern linguistici tra commenti a notizie vere e false, abbiamo utilizzato:

1. **Test di Shapiro-Wilk**: per verificare la normalità delle distribuzioni
2. **Test di Mann-Whitney U**: test non parametrico per confrontare le distribuzioni tra i due gruppi (vero/falso)
3. **Correzione di Bonferroni**: per controllare l'errore di tipo I nei test multipli
4. **Calcolo dell'Effect Size**: per valutare la rilevanza pratica delle differenze statisticamente significative

La soglia di significatività è stata fissata a α = 0.05, con correzione per test multipli.

#### Analisi delle Correlazioni

Abbiamo analizzato le correlazioni tra le feature linguistiche e la veridicità delle notizie:

1. **Correlazione di Pearson**: per relazioni lineari
2. **Correlazione di Spearman**: per relazioni monotoniche non necessariamente lineari
3. **Test di significatività**: p-value per determinare la significatività statistica delle correlazioni
4. **Interpretazione della forza**: categorizzazione delle correlazioni secondo i criteri standard (trascurabile < 0.1, debole 0.1-0.3, moderata 0.3-0.5, forte > 0.5)

#### Modelli Predittivi

Per esplorare le relazioni tra feature linguistiche e veridicità, abbiamo implementato due approcci:

1. **Regressione Logistica**: un modello lineare che stabilisce una relazione diretta tra feature e probabilità di classe

```python
def fit_logistic_regression(X_train, y_train, X_test, y_test, feature_names):
    """Addestra e valuta un modello di regressione logistica."""
    # Configurazione del modello
    lr = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        solver='saga',
        max_iter=1000,
        random_state=42
    )
    
    # Addestramento e valutazione
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_pred_proba = lr.predict_proba(X_test)[:,1]
    
    # Calcolo metriche
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Coefficienti
    coefficients = pd.DataFrame({
        'feature': feature_names,
        'coefficient': lr.coef_[0]
    }).sort_values('coefficient', ascending=False)
    
    return lr, metrics, coefficients, y_pred, y_pred_proba
```

2. **Random Forest**: un modello non lineare basato su ensemble di alberi decisionali

```python
def train_random_forest(X_train, y_train, X_test, y_test, feature_names):
    """Addestra e valuta un modello Random Forest."""
    # Configurazione del modello
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Addestramento e valutazione
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:,1]
    
    # Calcolo metriche
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Importanza feature
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return rf, metrics, feature_importance, y_pred, y_pred_proba
```

#### Confronto tra Set di Feature

Per valutare il contributo delle diverse categorie di feature, abbiamo testato i seguenti set:

1. **sentiment_only**: solo feature di sentiment (`sentiment_polarity`, `sentiment_subjectivity`)
2. **stance_only**: solo feature di stance (`stance_score`)
3. **readability_only**: solo feature di leggibilità e acculturazione
4. **sentiment_stance**: combinazione di sentiment e stance
5. **sentiment_readability**: combinazione di sentiment e leggibilità
6. **all_features**: tutte le feature linguistiche

Per ciascun set, abbiamo addestrato modelli Random Forest con parametri identici e confrontato le performance, utilizzando metriche come ROC AUC e F1 Score.

### Riproducibilità e Trasparenza

Per garantire la riproducibilità dello studio, abbiamo adottato diverse misure:

1. **Seed fisso**: tutti i processi randomizzati utilizzano `random_state=42`
2. **Versioni documentate**: file `requirements.txt` con versioni specifiche delle librerie
3. **Open source**: codice completo disponibile e commentato
4. **Logging**: registrazione dettagliata di ogni fase del processo

Nel capitolo successivo, presenteremo i risultati dell'analisi esplorativa condotta su questo dataset.
