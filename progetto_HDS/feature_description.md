# Descrizione delle Feature Estratte

Questo documento descrive tutte le feature estratte dal dataset PHEME per l'analisi della relazione tra caratteristiche linguistiche e veridicità delle notizie.

## Caratteristiche di Base

| Feature | Descrizione |
|---------|-------------|
| `thread_id` | Identificatore del thread di conversazione |
| `event` | Eventi principali nel dataset (es. charliehebdo, germanwings) |
| `tweet_id` | Identificatore unico del tweet |
| `veracity` | Veridicità dell'informazione (true, false, unverified) |
| `is_rumour` | Indica se il tweet è parte di una voce/rumour |
| `is_source` | Indica se il tweet è il tweet originale che ha iniziato il thread |
| `reaction_index` | Posizione del tweet nella catena di reazioni |

## Feature di Sentiment Analysis

### Implementate in `src/features/sentiment.py`

Queste feature sono state estratte utilizzando TextBlob, una libreria Python per l'elaborazione del linguaggio naturale.

| Feature | Descrizione | Intervallo |
|---------|-------------|------------|
| `sentiment_polarity` | Misura quanto positivo o negativo è il testo | [-1.0, 1.0] |
| `sentiment_subjectivity` | Misura quanto soggettivo o oggettivo è il testo | [0.0, 1.0] |
| `sentiment_category` | Categorizzazione del sentiment | ["positive", "neutral", "negative"] |

La polarità è normalizzata utilizzando uno StandardScaler per ottenere una media di 0 e una deviazione standard di 1.

## Feature di Stance Analysis

### Implementate in `src/features/stance.py`

Queste feature misurano l'atteggiamento del commento rispetto al tweet principale del thread.

| Feature | Descrizione | Intervallo/Valori |
|---------|-------------|-------------------|
| `stance_score` | Misura numerica della stance (combinazione di similarità e sentiment) | [-1.0, 1.0] |
| `stance_category` | Categorizzazione della stance | ["supportive", "critical", "neutral", "opposing", "indirect_supportive", "unrelated", "source"] |

La stance è calcolata utilizzando la similarità del coseno tra il testo del tweet sorgente e la reazione, combinata con il sentiment.

## Feature di Leggibilità e Acculturazione

### Implementate in `src/features/readability.py`

Queste feature misurano la complessità linguistica e il livello di acculturazione nei tweet.

| Feature | Descrizione | Intervallo |
|---------|-------------|------------|
| `flesch_reading_ease` | Indice di leggibilità Flesch (più alto = più leggibile) | [0.0, 100.0] |
| `type_token_ratio` | Rapporto tra parole uniche e totale parole (misura la diversità lessicale) | [0.0, 1.0] |
| `formal_language_score` | Misura quanto formale è il linguaggio | [0.0, 1.0] |
| `vocabulary_richness` | Misura la ricchezza del vocabolario basata su hapax legomena | [0.0, 1.0] |
| `avg_word_length` | Lunghezza media delle parole | [0.0, ∞) |
| `long_words_ratio` | Proporzione di parole lunghe (>6 caratteri) | [0.0, 1.0] |
| `culture_score` | Punteggio composito di acculturazione | [0.0, 1.0] |
| `readability_category` | Categorizzazione della leggibilità | ["very_easy", "easy", "moderate", "difficult", "very_difficult"] |

## Algoritmi Utilizzati

### Sentiment Analysis
Per l'analisi del sentiment è stata utilizzata la libreria TextBlob, che implementa un approccio basato sul dizionario e sul machine learning per determinare la polarità e la soggettività dei testi.

### Stance Analysis
L'analisi della stance utilizza una combinazione di:
1. **TF-IDF vectorization**: per convertire i testi in vettori numerici
2. **Similarità del coseno**: per misurare la vicinanza tematica tra tweet sorgente e reazioni
3. **Combinazione con sentiment**: per determinare se la reazione è supportiva o critica

### Leggibilità e Acculturazione
Per queste metriche sono stati utilizzati i seguenti algoritmi:
1. **Indici di Flesch**: formule basate su lunghezza delle frasi e delle parole
2. **Type-Token Ratio**: misura della diversità lessicale
3. **Rilevamento di linguaggio formale/informale**: basato su dizionari e pattern di espressioni informali
4. **Punteggio di acculturazione**: formula composita che pesa diverse metriche linguistiche

## Note sulla Normalizzazione

Nella matrice di feature finale (`pheme_feature_matrix.csv`), le seguenti normalizzazioni sono state applicate:

- Feature con distribuzione normale (sentiment, stance): **StandardScaler** (media=0, deviazione standard=1)
- Feature con distribuzione non normale e intervallo limitato: **MinMaxScaler** (intervallo [0,1])
- Feature categoriche: **One-Hot Encoding** (colonne separate per ogni categoria)
