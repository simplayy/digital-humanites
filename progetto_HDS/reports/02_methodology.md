# 2. Metodologia

## 2.1 Dataset PHEME

### 2.1.1 Descrizione del Dataset

Il dataset PHEME contiene thread di conversazione Twitter relativi a diversi eventi di attualità, con annotazioni sulla veridicità delle notizie. Le caratteristiche principali del dataset includono:

- **Dimensione**: 6,425 thread di conversazione, contenenti 105,354 tweet
- **Distribuzione delle classi**: 93% notizie vere, 7% notizie false
- **Eventi coperti**: Charlie Hebdo, Germanwings crash, Ferguson, altri eventi significativi
- **Periodo temporale**: 2014-2016
- **Lingua**: Prevalentemente inglese

### 2.1.2 Struttura dei Dati

Ogni thread nel dataset è strutturato come segue:

```plaintext
thread/
├── source-tweet/
│   ├── tweet.json
│   └── annotations.json
└── reactions/
    ├── 1.json
    ├── 2.json
    └── ...
```

### 2.1.3 Criteri di Inclusione/Esclusione

Per garantire la qualità dell'analisi, abbiamo applicato i seguenti criteri:

**Inclusione**:
- Thread con etichetta di veridicità confermata (vero/falso)
- Thread con almeno 3 reazioni (commenti)
- Testi in lingua inglese

**Esclusione**:
- Thread con veridicità non verificata o ambigua
- Thread senza reazioni
- Testi troppo brevi (<10 caratteri)

## 2.2 Preprocessing dei Dati

### 2.2.1 Pulizia del Testo

1. **Rimozione di Elementi Non Testuali**:
   ```python
   def clean_text(text):
       # Rimozione URL
       text = re.sub(r'http\S+|www\S+', '', text)
       # Rimozione menzioni
       text = re.sub(r'@\w+', '', text)
       # Rimozione hashtag
       text = re.sub(r'#\w+', '', text)
       return text
   ```

2. **Normalizzazione**:
   - Conversione a minuscolo
   - Rimozione di caratteri speciali
   - Gestione delle emoji
   - Standardizzazione della punteggiatura

3. **Tokenizzazione e Lemmatizzazione**:
   ```python
   def preprocess_text(text):
       # Tokenizzazione
       tokens = word_tokenize(text)
       # Lemmatizzazione
       lemmatizer = WordNetLemmatizer()
       lemmas = [lemmatizer.lemmatize(token) for token in tokens]
       return lemmas
   ```

### 2.2.2 Gestione dei Valori Mancanti

1. **Strategia per Valori Mancanti**:
   - Rimozione di tweet con testo mancante
   - Imputazione per metadati non critici
   - Documentazione delle decisioni di imputazione

2. **Validazione della Completezza**:
   ```python
   def validate_completeness(thread):
       return (
           thread['source']['text'] is not None and
           len(thread['reactions']) >= 3 and
           all(r['text'] for r in thread['reactions'])
       )
   ```

### 2.2.3 Strutturazione Gerarchica

1. **Organizzazione dei Thread**:
   ```python
   def build_thread_structure(source, reactions):
       return {
           'thread_id': source['id'],
           'source': source,
           'reactions': sorted(reactions, key=lambda x: x['timestamp']),
           'reaction_count': len(reactions)
       }
   ```

2. **Tracciamento delle Relazioni**:
   - Identificazione delle risposte dirette
   - Costruzione dell'albero conversazionale
   - Assegnazione di indici temporali

## 2.3 Estrazione delle Feature

### 2.3.1 Feature di Sentiment

1. **Polarità e Soggettività**:
   ```python
   def extract_sentiment_features(text):
       blob = TextBlob(text)
       return {
           'sentiment_polarity': blob.sentiment.polarity,
           'sentiment_subjectivity': blob.sentiment.subjectivity
       }
   ```

2. **Categorizzazione del Sentiment**:
   ```python
   def categorize_sentiment(polarity):
       if polarity > 0.1:
           return 'positive'
       elif polarity < -0.1:
           return 'negative'
       return 'neutral'
   ```

### 2.3.2 Feature di Stance

1. **Calcolo dello Stance Score**:
   ```python
   def compute_stance_score(source_text, reaction_text):
       # Similarità coseno
       sim = cosine_similarity(
           vectorize(source_text),
           vectorize(reaction_text)
       )
       # Combinazione con sentiment
       sentiment = get_sentiment(reaction_text)
       return sim * sentiment
   ```

2. **Categorizzazione della Stance**:
   - supportive
   - critical
   - neutral
   - opposing
   - indirect_supportive

### 2.3.3 Feature di Leggibilità e Acculturazione

1. **Indici di Leggibilità**:
   ```python
   def compute_readability_features(text):
       return {
           'flesch_reading_ease': textstat.flesch_reading_ease(text),
           'type_token_ratio': calculate_ttr(text),
           'formal_language_score': compute_formality(text)
       }
   ```

2. **Culture Score**:
   ```python
   def compute_culture_score(text):
       return (
           0.4 * vocabulary_richness(text) +
           0.3 * formal_language_score(text) +
           0.2 * type_token_ratio(text) +
           0.1 * normalized_flesch(text)
       )
   ```

## 2.4 Framework Analitico

### 2.4.1 Pipeline di Analisi

![Pipeline di Analisi](../figures/analysis_pipeline.png)

*La pipeline illustra il flusso completo dall'acquisizione dei dati alla validazione dei risultati.*

### 2.4.2 Approcci Statistici

1. **Test di Ipotesi**:
   - Test di Mann-Whitney U per confronti tra gruppi
   - Correzione di Bonferroni per test multipli
   - Calcolo dell'effect size

2. **Analisi delle Correlazioni**:
   - Correlazione di Pearson per variabili normali
   - Correlazione di Spearman per variabili non normali
   - Matrice di correlazione con heatmap

### 2.4.3 Modelli Predittivi

1. **Regressione Logistica**:
   ```python
   def train_logistic_regression(X, y):
       model = LogisticRegression(
           class_weight='balanced',
           random_state=42
       )
       return model.fit(X, y)
   ```

2. **Random Forest**:
   ```python
   def train_random_forest(X, y):
       model = RandomForestClassifier(
           n_estimators=100,
           class_weight='balanced',
           random_state=42
       )
       return model.fit(X, y)
   ```

### 2.4.4 Metriche di Valutazione

1. **Metriche di Performance**:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC AUC

2. **Validazione Incrociata**:
   ```python
   def perform_cross_validation(model, X, y):
       return cross_val_score(
           model, X, y,
           cv=5,
           scoring='roc_auc'
       )
   ```

## 2.5 Strumenti e Tecnologie

### 2.5.1 Stack Tecnologico

- **Linguaggio**: Python 3.8+
- **Librerie Principali**:
  - pandas per manipolazione dati
  - scikit-learn per machine learning
  - TextBlob per sentiment analysis
  - NLTK per NLP
  - matplotlib e seaborn per visualizzazioni

### 2.5.2 Gestione del Codice

- **Controllo Versione**: Git
- **Documentazione**: Docstring e Markdown
- **Testing**: pytest
- **Logging**: logging standard Python

### 2.5.3 Ambiente di Sviluppo

```bash
# Creazione ambiente virtuale
python -m venv hds_env

# Attivazione
source hds_env/bin/activate

# Installazione dipendenze
pip install -r requirements.txt
```

## 2.6 Considerazioni Etiche

### 2.6.1 Privacy dei Dati

- Utilizzo di dati pubblicamente disponibili
- Anonimizzazione degli identificatori degli utenti
- Conformità con le linee guida di ricerca su social media

### 2.6.2 Bias e Rappresentatività

- Riconoscimento di bias potenziali nel dataset
- Documentazione delle limitazioni di generalizzabilità
- Considerazione di contesti culturali diversi

## 2.7 Riproducibilità

### 2.7.1 Documentazione

- Repository GitHub con codice completo
- Guide dettagliate per setup e esecuzione
- File di configurazione e parametri

### 2.7.2 Controllo di Qualità

- Test automatizzati per funzioni chiave
- Validazione dei risultati intermedi
- Logging dettagliato delle operazioni

---

*Continua nella prossima sezione: [3. Analisi Esplorativa](03_exploratory_analysis.md)*
