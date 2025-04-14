# Piano di Implementazione Tecnica

## 1. Setup dell'Ambiente di Sviluppo

### 1.1 Strumenti e Tecnologie

Per questo studio utilizzeremo un ambiente basato su Python con le seguenti librerie e framework:

#### 1.1.1 Librerie Principali
- **pandas**: Per la manipolazione e l'analisi dei dati
- **numpy**: Per le operazioni numeriche
- **scipy**: Per i test statistici avanzati
- **statsmodels**: Per modelli statistici e test di ipotesi
- **scikit-learn**: Per funzionalità di machine learning e preprocessing

#### 1.1.2 Librerie NLP e Sentiment Analysis
- **transformers** (Hugging Face): Per modelli pre-addestrati di sentiment analysis, stance detection ed emotion detection
- **textstat**: Per le metriche di leggibilità
- **sentence-transformers**: Per analisi di similarità semantica

#### 1.1.3 Visualizzazione
- **matplotlib** e **seaborn**: Per grafici statistici
- **plotly**: Per visualizzazioni interattive
- **networkx**: Per visualizzazione di relazioni (opzionale)

#### 1.1.4 Gestione del Progetto
- **Git**: Per il versioning del codice
- **Jupyter notebooks**: Per analisi esplorative e documentazione
- **pytest**: Per testing delle funzioni di analisi

### 1.2 Setup dell'Ambiente

```bash
# Creazione dell'ambiente virtuale
python -m venv hds_env
source hds_env/bin/activate  # Per macOS/Linux

# Installazione delle dipendenze principali
pip install pandas numpy scipy statsmodels scikit-learn

# Installazione delle librerie NLP
pip install transformers textstat sentence-transformers

# Librerie di visualizzazione
pip install matplotlib seaborn plotly

# Gestione e testing
pip install jupyter pytest

# Salvataggio delle dipendenze
pip freeze > requirements.txt
```

### 1.3 Struttura del Progetto

```
progetto_HDS/
│
├── data/                      # Dati grezzi e preprocessati
│   ├── raw/                   # Dataset originale scaricato
│   └── processed/             # Dataset dopo il preprocessing
│
├── notebooks/                 # Jupyter notebooks
│   ├── 1_exploratory.ipynb    # Analisi esplorativa
│   ├── 2_feature_extraction.ipynb
│   ├── 3_statistical_analysis.ipynb
│   └── 4_visualization.ipynb
│
├── src/                       # Codice sorgente
│   ├── __init__.py
│   ├── data/                  # Script per la gestione dei dati
│   │   ├── __init__.py
│   │   ├── download.py        # Scarica il dataset
│   │   └── preprocess.py      # Funzioni di preprocessing
│   │
│   ├── features/              # Estrazione feature
│   │   ├── __init__.py
│   │   ├── sentiment.py       # Funzioni per sentiment analysis
│   │   ├── readability.py     # Metriche di leggibilità
│   │   ├── emotions.py        # Analisi delle emozioni
│   │   └── metadata.py        # Estrazione di metadata
│   │
│   ├── analysis/              # Analisi statistica
│   │   ├── __init__.py
│   │   ├── descriptive.py     # Statistiche descrittive
│   │   ├── hypothesis_tests.py # Test di ipotesi
│   │   ├── correlation.py     # Analisi di correlazione
│   │   └── regression.py      # Modelli di regressione
│   │
│   └── visualization/         # Visualizzazione
│       ├── __init__.py
│       └── plots.py           # Funzioni per grafici
│
├── tests/                     # Test unitari
│   ├── __init__.py
│   ├── test_features.py
│   └── test_analysis.py
│
├── results/                   # Risultati delle analisi
│   ├── figures/               # Grafici generati
│   └── tables/                # Tabelle di risultati
│
├── docs/                      # Documentazione
│   ├── metodologia_dettagliata.md
│   └── report_finale.md
│
├── requirements.txt           # Dipendenze del progetto
└── README.md                  # Guida al progetto
```

## 2. Acquisizione e Preprocessing dei Dati

### 2.1 Download del Dataset

```python
# src/data/download.py

import os
import urllib.request
import zipfile

def download_fakeddit(destination_folder="data/raw"):
    """
    Download del dataset Fakeddit da GitHub o altra fonte.
    
    Parameters:
    -----------
    destination_folder : str
        Cartella dove salvare i dati scaricati
    """
    # Crea la cartella se non esiste
    os.makedirs(destination_folder, exist_ok=True)
    
    # URL del dataset (questo è un esempio, l'URL reale potrebbe essere diverso)
    dataset_url = "https://github.com/entitize/Fakeddit/releases/download/v1.0/all_submissions.csv"
    comments_url = "https://github.com/entitize/Fakeddit/releases/download/v1.0/all_comments.csv"
    
    # Download dei file
    urllib.request.urlretrieve(dataset_url, os.path.join(destination_folder, "all_submissions.csv"))
    urllib.request.urlretrieve(comments_url, os.path.join(destination_folder, "all_comments.csv"))
    
    print("Dataset scaricato con successo in", destination_folder)
    
    return os.path.join(destination_folder, "all_submissions.csv"), os.path.join(destination_folder, "all_comments.csv")
```

### 2.2 Preprocessing dei Dati

```python
# src/data/preprocess.py

import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def clean_text(text):
    """
    Pulisce il testo dai caratteri speciali e normalizza.
    
    Parameters:
    -----------
    text : str
        Testo da pulire
        
    Returns:
    --------
    str
        Testo pulito
    """
    if isinstance(text, str):
        # Converti a minuscolo
        text = text.lower()
        # Rimuovi HTML
        text = re.sub(r'<.*?>', '', text)
        # Rimuovi URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Rimuovi emoji e caratteri non ASCII
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        # Rimuovi punteggiatura
        text = re.sub(r'[^\w\s]', '', text)
        # Rimuovi spazi multipli
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

def preprocess_comments(comments_file, submissions_file, output_file="data/processed/processed_data.csv"):
    """
    Preprocessa i commenti e li unisce alle informazioni sulle notizie.
    
    Parameters:
    -----------
    comments_file : str
        Path al file CSV dei commenti
    submissions_file : str
        Path al file CSV delle notizie
    output_file : str
        Path dove salvare i dati preprocessati
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con i dati preprocessati
    """
    # Carica i dati
    comments_df = pd.read_csv(comments_file)
    submissions_df = pd.read_csv(submissions_file)
    
    # Preprocessing base
    comments_df['body_clean'] = comments_df['body'].apply(clean_text)
    submissions_df['title_clean'] = submissions_df['title'].apply(clean_text)
    
    # Rimuovi commenti vuoti o troppo corti (< 5 caratteri)
    comments_df = comments_df[comments_df['body_clean'].str.len() > 5]
    
    # Join tra commenti e notizie
    merged_df = pd.merge(
        comments_df,
        submissions_df[['id', 'title_clean', 'label', 'created_utc']],
        left_on='submission_id',
        right_on='id',
        how='inner'
    )
    
    # Crea cartella output se non esiste
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Salva il dataset preprocessato
    merged_df.to_csv(output_file, index=False)
    
    print(f"Preprocessing completato. Dataset salvato in {output_file}")
    
    return merged_df
```

## 3. Estrazione delle Feature

### 3.1 Sentiment Analysis di Base

```python
# src/features/sentiment.py

from transformers import pipeline

# Inizializza il modello una sola volta e riutilizzalo
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def extract_sentiment_features(text):
    """
    Estrae le feature di sentiment da un testo.
    
    Parameters:
    -----------
    text : str
        Il testo da analizzare
        
    Returns:
    --------
    dict
        Dizionario con le feature di sentiment
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {
            "sentiment_label": "NEUTRAL",
            "sentiment_score": 0.0,
            "sentiment_positive": 0.0,
            "sentiment_negative": 0.0,
            "sentiment_neutral": 1.0
        }
    
    try:
        result = sentiment_analyzer(text)[0]
        
        # Converte il risultato in un formato numerico standard
        # Assumendo che il modello restituisca label come "1 star" fino a "5 stars"
        # dove 1 è molto negativo, 3 è neutrale, 5 è molto positivo
        label = result["label"]
        score = result["score"]
        
        # Normalizza il punteggio da 1-5 a -1...+1
        # Esempio: "3 stars" → 0.0, "5 stars" → 1.0, "1 star" → -1.0
        stars = int(label.split()[0])
        normalized_score = (stars - 3) / 2
        
        # Calcola positivity e negativity
        if normalized_score > 0:
            positivity = normalized_score
            negativity = 0.0
            neutrality = 1.0 - positivity
        elif normalized_score < 0:
            positivity = 0.0
            negativity = -normalized_score
            neutrality = 1.0 - negativity
        else:
            positivity = 0.0
            negativity = 0.0
            neutrality = 1.0
        
        return {
            "sentiment_label": label,
            "sentiment_score": normalized_score,
            "sentiment_positive": positivity,
            "sentiment_negative": negativity,
            "sentiment_neutral": neutrality
        }
    except Exception as e:
        print(f"Errore nell'analisi del sentiment: {e}")
        return {
            "sentiment_label": "ERROR",
            "sentiment_score": 0.0,
            "sentiment_positive": 0.0,
            "sentiment_negative": 0.0,
            "sentiment_neutral": 1.0
        }
```

### 3.2 Analisi delle Emozioni

```python
# src/features/emotions.py

from transformers import pipeline

# Inizializza il modello
emotion_classifier = pipeline("text-classification", 
                             model="j-hartmann/emotion-english-distilroberta-base", 
                             return_all_scores=True)

def extract_emotion_features(text):
    """
    Estrae le distribuzioni delle emozioni da un testo.
    
    Parameters:
    -----------
    text : str
        Il testo da analizzare
        
    Returns:
    --------
    dict
        Dizionario con le feature delle emozioni
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {
            "emotion_anger": 0.0,
            "emotion_disgust": 0.0,
            "emotion_fear": 0.0,
            "emotion_joy": 0.0,
            "emotion_neutral": 1.0,
            "emotion_sadness": 0.0,
            "emotion_surprise": 0.0,
            "emotion_dominant": "neutral"
        }
    
    try:
        # Analizza le emozioni
        result = emotion_classifier(text)[0]
        
        # Estrai i punteggi per ogni emozione
        emotions = {f"emotion_{item['label']}": item['score'] for item in result}
        
        # Identifica l'emozione dominante
        dominant_emotion = max(result, key=lambda x: x['score'])['label']
        emotions["emotion_dominant"] = dominant_emotion
        
        return emotions
    except Exception as e:
        print(f"Errore nell'analisi delle emozioni: {e}")
        return {
            "emotion_anger": 0.0,
            "emotion_disgust": 0.0,
            "emotion_fear": 0.0,
            "emotion_joy": 0.0,
            "emotion_neutral": 1.0,
            "emotion_sadness": 0.0,
            "emotion_surprise": 0.0,
            "emotion_dominant": "error"
        }
```

### 3.3 Metriche di Leggibilità

```python
# src/features/readability.py

import textstat

def extract_readability_metrics(text):
    """
    Calcola varie metriche di leggibilità per un testo.
    
    Parameters:
    -----------
    text : str
        Il testo da analizzare
        
    Returns:
    --------
    dict
        Dizionario con le metriche di leggibilità
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        # Valori di default per testi vuoti
        return {
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "smog_index": 0.0,
            "coleman_liau_index": 0.0,
            "automated_readability_index": 0.0,
            "dale_chall_readability_score": 0.0,
            "difficult_words": 0,
            "gunning_fog": 0.0,
            "text_standard": "N/A",
            "fernandez_huerta": 0.0,
            "szigriszt_pazos": 0.0,
            "gutierrez_polini": 0.0,
            "crawford": 0.0
        }
    
    try:
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "smog_index": textstat.smog_index(text),
            "coleman_liau_index": textstat.coleman_liau_index(text),
            "automated_readability_index": textstat.automated_readability_index(text),
            "dale_chall_readability_score": textstat.dale_chall_readability_score(text),
            "difficult_words": textstat.difficult_words(text),
            "gunning_fog": textstat.gunning_fog(text),
            "text_standard": textstat.text_standard(text),
            "fernandez_huerta": textstat.fernandez_huerta(text) if hasattr(textstat, 'fernandez_huerta') else 0.0,
            "szigriszt_pazos": textstat.szigriszt_pazos(text) if hasattr(textstat, 'szigriszt_pazos') else 0.0,
            "gutierrez_polini": textstat.gutierrez_polini(text) if hasattr(textstat, 'gutierrez_polini') else 0.0,
            "crawford": textstat.crawford(text) if hasattr(textstat, 'crawford') else 0.0
        }
    except Exception as e:
        print(f"Errore nel calcolo delle metriche di leggibilità: {e}")
        return {
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "smog_index": 0.0,
            "coleman_liau_index": 0.0,
            "automated_readability_index": 0.0,
            "dale_chall_readability_score": 0.0,
            "difficult_words": 0,
            "gunning_fog": 0.0,
            "text_standard": "ERROR",
            "fernandez_huerta": 0.0,
            "szigriszt_pazos": 0.0,
            "gutierrez_polini": 0.0,
            "crawford": 0.0
        }
```

### 3.4 Rilevamento del Sarcasmo

```python
# src/features/sentiment.py (aggiunta alla fine del file)

from transformers import pipeline

# Inizializzazione del modello di rilevamento del sarcasmo
sarcasm_detector = pipeline("text-classification", model="mrm8488/t5-base-finetuned-sarcasm-detection")

def detect_sarcasm(text):
    """
    Rileva la presenza di sarcasmo in un testo.
    
    Parameters:
    -----------
    text : str
        Il testo da analizzare
        
    Returns:
    --------
    dict
        Dizionario con l'indicazione di sarcasmo e il relativo punteggio
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {
            "is_sarcastic": False,
            "sarcasm_score": 0.0
        }
    
    try:
        result = sarcasm_detector(text)[0]
        
        # Assumendo che "LABEL_1" indichi sarcasmo
        is_sarcastic = result["label"] == "LABEL_1"
        score = result["score"]
        
        return {
            "is_sarcastic": is_sarcastic,
            "sarcasm_score": score if is_sarcastic else 1.0 - score
        }
    except Exception as e:
        print(f"Errore nel rilevamento del sarcasmo: {e}")
        return {
            "is_sarcastic": False,
            "sarcasm_score": 0.0
        }
```

## 4. Analisi Statistica

### 4.1 Statistiche Descrittive

```python
# src/analysis/descriptive.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_descriptive_statistics(df, feature_columns, group_column='is_fake'):
    """
    Calcola statistiche descrittive raggruppate per notizie vere/fake.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con i dati
    feature_columns : list
        Lista di colonne per cui calcolare le statistiche
    group_column : str
        Colonna da usare per il raggruppamento (default: 'is_fake')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con le statistiche descrittive
    """
    # Verifica che le colonne esistano
    existing_cols = [col for col in feature_columns if col in df.columns]
    
    if not existing_cols:
        return pd.DataFrame()
    
    # Calcola statistiche descrittive per ogni gruppo
    stats = []
    for group_value, group_data in df.groupby(group_column):
        group_stats = group_data[existing_cols].describe().T
        group_stats['group'] = group_value
        stats.append(group_stats)
    
    all_stats = pd.concat(stats).reset_index()
    
    # Pivot per avere un formato più leggibile
    pivot_stats = all_stats.pivot(index='index', columns='group')
    
    return pivot_stats

def visualize_distributions(df, feature_column, group_column='is_fake', bins=30):
    """
    Visualizza la distribuzione di una feature per notizie vere vs fake.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con i dati
    feature_column : str
        Colonna da visualizzare
    group_column : str
        Colonna da usare per il raggruppamento (default: 'is_fake')
    bins : int
        Numero di bin per l'istogramma
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figura con il grafico
    """
    if feature_column not in df.columns or group_column not in df.columns:
        return None
    
    # Crea la figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ottieni valori unici per il gruppo e assegna colori
    groups = df[group_column].unique()
    colors = sns.color_palette("Set1", len(groups))
    
    # Crea istogrammi sovrapposti
    for i, group in enumerate(groups):
        group_data = df[df[group_column] == group][feature_column].dropna()
        ax.hist(group_data, bins=bins, alpha=0.5, color=colors[i], 
                label=f"{group_column}={group} (n={len(group_data)})")
    
    # Aggiungi dettagli al grafico
    ax.set_xlabel(feature_column)
    ax.set_ylabel('Frequenza')
    ax.set_title(f'Distribuzione di {feature_column} per {group_column}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    return fig
```

### 4.2 Test di Ipotesi

```python
# src/analysis/hypothesis_tests.py

import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

def ttest_by_group(df, feature_column, group_column='is_fake', equal_var=False):
    """
    Esegue un test t per confrontare la media di una feature tra due gruppi.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con i dati
    feature_column : str
        Colonna da analizzare
    group_column : str
        Colonna da usare per il raggruppamento (default: 'is_fake')
    equal_var : bool
        Indica se assumere varianze uguali (default: False)
        
    Returns:
    --------
    dict
        Risultati del test t
    """
    if feature_column not in df.columns or group_column not in df.columns:
        return {
            'error': f"Colonne {feature_column} o {group_column} non trovate nel DataFrame"
        }
    
    # Ottieni i gruppi unici
    groups = df[group_column].unique()
    if len(groups) != 2:
        return {
            'error': f"Il test t richiede esattamente 2 gruppi, trovati {len(groups)}"
        }
    
    # Estrai i dati per ciascun gruppo
    group_data = [df[df[group_column] == g][feature_column].dropna() for g in groups]
    
    # Calcola statistiche di base
    means = [data.mean() for data in group_data]
    stds = [data.std() for data in group_data]
    
    # Esegui il test t
    t_stat, p_val = stats.ttest_ind(group_data[0], group_data[1], equal_var=equal_var)
    
    # Calcola dimensione dell'effetto (Cohen's d)
    n1, n2 = len(group_data[0]), len(group_data[1])
    if n1 > 0 and n2 > 0:
        pooled_std = np.sqrt(((n1 - 1) * stds[0]**2 + (n2 - 1) * stds[1]**2) / (n1 + n2 - 2))
        cohens_d = abs(means[0] - means[1]) / pooled_std
    else:
        cohens_d = None
    
    # Interpreta la dimensione dell'effetto
    if cohens_d is not None:
        if cohens_d < 0.2:
            effect_size_interpretation = "Trascurabile"
        elif cohens_d < 0.5:
            effect_size_interpretation = "Piccolo"
        elif cohens_d < 0.8:
            effect_size_interpretation = "Medio"
        else:
            effect_size_interpretation = "Grande"
    else:
        effect_size_interpretation = "Non calcolabile"
    
    return {
        'feature': feature_column,
        'group_column': group_column,
        'groups': groups.tolist(),
        'means': means,
        'stds': stds,
        'sample_sizes': [len(data) for data in group_data],
        't_statistic': t_stat,
        'p_value': p_val,
        'significant_05': p_val < 0.05,
        'cohens_d': cohens_d,
        'effect_size': effect_size_interpretation
    }

def run_multiple_t_tests(df, feature_columns, group_column='is_fake', equal_var=False, correction_method='fdr_bh'):
    """
    Esegue test t per confrontare le medie di più feature tra due gruppi, 
    applicando correzione per test multipli.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con i dati
    feature_columns : list
        Lista di colonne da analizzare
    group_column : str
        Colonna da usare per il raggruppamento (default: 'is_fake')
    equal_var : bool
        Indica se assumere varianze uguali (default: False)
    correction_method : str
        Metodo di correzione per test multipli (default: 'fdr_bh')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con i risultati di tutti i test
    """
    # Filtra le colonne esistenti
    existing_cols = [col for col in feature_columns if col in df.columns]
    
    if not existing_cols:
        return pd.DataFrame()
    
    # Esegui test t per ogni feature
    results = []
    for col in existing_cols:
        result = ttest_by_group(df, col, group_column, equal_var)
        if 'error' not in result:
            results.append(result)
    
    if not results:
        return pd.DataFrame()
    
    # Converti risultati in DataFrame
    results_df = pd.DataFrame(results)
    
    # Applica correzione per test multipli
    if len(results_df) > 1:
        _, corrected_pvals, _, _ = multipletests(
            results_df['p_value'], 
            alpha=0.05, 
            method=correction_method
        )
        results_df['corrected_p_value'] = corrected_pvals
        results_df['significant_corrected'] = corrected_pvals < 0.05
    else:
        results_df['corrected_p_value'] = results_df['p_value']
        results_df['significant_corrected'] = results_df['significant_05']
    
    return results_df
```

### 4.3 Analisi di Correlazione

```python
# src/analysis/correlation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def calculate_correlation_matrix(df, feature_columns, method='pearson'):
    """
    Calcola la matrice di correlazione tra le feature selezionate.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con i dati
    feature_columns : list
        Lista di colonne per cui calcolare le correlazioni
    method : str
        Metodo di correlazione ('pearson', 'spearman', o 'kendall')
        
    Returns:
    --------
    pd.DataFrame
        Matrice di correlazione
    """
    # Filtra le colonne esistenti
    existing_cols = [col for col in feature_columns if col in df.columns]
    
    if not existing_cols:
        return pd.DataFrame()
    
    # Calcola la matrice di correlazione
    corr_matrix = df[existing_cols].corr(method=method)
    
    return corr_matrix

def calculate_correlation_significance(df, feature_columns, method='pearson'):
    """
    Calcola la significatività delle correlazioni tra le feature selezionate.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con i dati
    feature_columns : list
        Lista di colonne per cui calcolare le correlazioni
    method : str
        Metodo di correlazione ('pearson' o 'spearman')
        
    Returns:
    --------
    tuple
        (matrice di correlazione, matrice di p-values)
    """
    # Filtra le colonne esistenti
    existing_cols = [col for col in feature_columns if col in df.columns]
    
    if not existing_cols:
        return pd.DataFrame(), pd.DataFrame()
    
    # Inizializza matrici vuote
    n = len(existing_cols)
    corr_matrix = pd.DataFrame(np.zeros((n, n)), index=existing_cols, columns=existing_cols)
    p_values = pd.DataFrame(np.zeros((n, n)), index=existing_cols, columns=existing_cols)
    
    # Calcola correlazioni e p-values
    for i, col1 in enumerate(existing_cols):
        for j, col2 in enumerate(existing_cols):
            if i == j:
                corr_matrix.iloc[i, j] = 1.0
                p_values.iloc[i, j] = 0.0
            else:
                if method == 'pearson':
                    corr, p = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
                elif method == 'spearman':
                    corr, p = stats.spearmanr(df[col1].dropna(), df[col2].dropna())
                else:
                    raise ValueError(f"Metodo {method} non supportato")
                
                corr_matrix.iloc[i, j] = corr
                p_values.iloc[i, j] = p
    
    return corr_matrix, p_values

def visualize_correlation_heatmap(corr_matrix, p_values=None, significance_level=0.05):
    """
    Visualizza la matrice di correlazione come heatmap.
    
    Parameters:
    -----------
    corr_matrix : pd.DataFrame
        Matrice di correlazione
    p_values : pd.DataFrame, optional
        Matrice di p-values
    significance_level : float, optional
        Soglia di significatività (default: 0.05)
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figura con l'heatmap
    """
    # Crea la figura
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Crea la maschera per valori non significativi
    mask = None
    if p_values is not None:
        mask = p_values > significance_level
    
    # Visualizza la heatmap
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                mask=mask, vmin=-1, vmax=1, center=0, ax=ax)
    
    # Aggiungi dettagli al grafico
    ax.set_title('Matrice di Correlazione')
    if p_values is not None:
        ax.set_title(f'Matrice di Correlazione (solo correlazioni significative con p < {significance_level})')
    
    plt.tight_layout()
    
    return fig
```

## 5. Pipeline Completo dell'Analisi

```python
# src/main.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import dei moduli del progetto
from data.download import download_fakeddit
from data.preprocess import preprocess_comments
from features.sentiment import extract_sentiment_features, detect_sarcasm
from features.emotions import extract_emotion_features
from features.readability import extract_readability_metrics
from analysis.descriptive import calculate_descriptive_statistics, visualize_distributions
from analysis.hypothesis_tests import run_multiple_t_tests
from analysis.correlation import calculate_correlation_matrix, calculate_correlation_significance, visualize_correlation_heatmap

def run_analysis_pipeline(base_path="."):
    """
    Esegue la pipeline completa di analisi.
    
    Parameters:
    -----------
    base_path : str
        Path di base del progetto
    """
    # Crea struttura delle directory
    data_dir = Path(base_path) / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    results_dir = Path(base_path) / "results"
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"
    
    # Crea directory se non esistono
    for dir_path in [raw_dir, processed_dir, results_dir, figures_dir, tables_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("1. Download del dataset...")
    submissions_file, comments_file = download_fakeddit(destination_folder=str(raw_dir))
    
    print("2. Preprocessing dei dati...")
    processed_file = str(processed_dir / "processed_data.csv")
    data = preprocess_comments(comments_file, submissions_file, output_file=processed_file)
    
    print("3. Estrazione delle feature...")
    # Campiona un subset per debugging/test rapidi (rimuovere per analisi completa)
    sample_size = min(10000, len(data))
    data_sample = data.sample(sample_size, random_state=42)
    
    # Estrai feature per ogni commento
    print(f"   Analizzando {sample_size} commenti...")
    
    # Inizializza liste per le feature
    sentiment_features = []
    emotion_features = []
    readability_metrics = []
    sarcasm_features = []
    
    # Elabora ogni commento
    for i, comment in enumerate(data_sample['body_clean']):
        if i % 100 == 0:
            print(f"   Progresso: {i}/{sample_size} commenti analizzati")
        
        sentiment = extract_sentiment_features(comment)
        emotions = extract_emotion_features(comment)
        readability = extract_readability_metrics(comment)
        sarcasm = detect_sarcasm(comment)
        
        sentiment_features.append(sentiment)
        emotion_features.append(emotions)
        readability_metrics.append(readability)
        sarcasm_features.append(sarcasm)
    
    # Converti liste in DataFrame
    sentiment_df = pd.DataFrame(sentiment_features)
    emotion_df = pd.DataFrame(emotion_features)
    readability_df = pd.DataFrame(readability_metrics)
    sarcasm_df = pd.DataFrame(sarcasm_features)
    
    # Unisci tutti i DataFrame
    all_features = pd.concat([
        data_sample.reset_index(drop=True),
        sentiment_df,
        emotion_df,
        readability_df,
        sarcasm_df
    ], axis=1)
    
    # Converti la colonna label in binaria (is_fake)
    if 'label' in all_features.columns:
        all_features['is_fake'] = all_features['label'].apply(lambda x: 1 if x == 'fake' else 0)
    
    # Salva il dataset con le feature estratte
    features_file = processed_dir / "features_extracted.csv"
    all_features.to_csv(features_file, index=False)
    print(f"   Feature estratte salvate in {features_file}")
    
    print("4. Analisi statistica...")
    # Lista di feature da analizzare
    sentiment_cols = [col for col in all_features.columns if col.startswith('sentiment_')]
    emotion_cols = [col for col in all_features.columns if col.startswith('emotion_')]
    readability_cols = [col for col in all_features.columns if col in readability_df.columns]
    sarcasm_cols = [col for col in all_features.columns if col in sarcasm_df.columns]
    
    all_feature_cols = sentiment_cols + emotion_cols + readability_cols + sarcasm_cols
    
    # Statistiche descrittive
    print("   Calcolando statistiche descrittive...")
    desc_stats = calculate_descriptive_statistics(all_features, all_feature_cols, 'is_fake')
    desc_stats.to_csv(tables_dir / "descriptive_statistics.csv")
    
    # Visualizzazione delle distribuzioni per alcune feature chiave
    key_features = ['sentiment_score', 'emotion_anger', 'emotion_joy', 'sarcasm_score', 'flesch_reading_ease']
    for feature in key_features:
        if feature in all_features.columns:
            fig = visualize_distributions(all_features, feature, 'is_fake')
            if fig:
                fig.savefig(figures_dir / f"distribution_{feature}.png")
                plt.close(fig)
    
    # Test di ipotesi
    print("   Eseguendo test t per confrontare feature tra notizie vere e fake...")
    ttest_results = run_multiple_t_tests(all_features, all_feature_cols, 'is_fake', correction_method='fdr_bh')
    ttest_results.to_csv(tables_dir / "t_test_results.csv")
    
    # Analisi di correlazione
    print("   Calcolando matrici di correlazione...")
    corr_matrix = calculate_correlation_matrix(all_features, all_feature_cols, 'pearson')
    corr_matrix.to_csv(tables_dir / "correlation_matrix.csv")
    
    corr_matrix_with_label = calculate_correlation_matrix(
        all_features, all_feature_cols + ['is_fake'], 'pearson'
    )
    
    # Ordina feature per correlazione con is_fake
    if 'is_fake' in corr_matrix_with_label.columns:
        top_corr = corr_matrix_with_label['is_fake'].sort_values(ascending=False)
        top_corr.to_csv(tables_dir / "feature_correlations_with_fakeness.csv")
    
    # Visualizzazione correlazione
    corr_matrix, p_values = calculate_correlation_significance(
        all_features, key_features + ['is_fake'], 'pearson'
    )
    
    fig = visualize_correlation_heatmap(corr_matrix, p_values)
    if fig:
        fig.savefig(figures_dir / "correlation_heatmap.png")
        plt.close(fig)
    
    print("Analisi completata! Risultati salvati nella directory 'results'.")
    
    return {
        'features_file': features_file,
        'desc_stats_file': tables_dir / "descriptive_statistics.csv",
        'ttest_file': tables_dir / "t_test_results.csv",
        'corr_matrix_file': tables_dir / "correlation_matrix.csv",
        'top_corr_file': tables_dir / "feature_correlations_with_fakeness.csv"
    }

if __name__ == "__main__":
    run_analysis_pipeline()
```

## 6. Guida all'Esecuzione del Progetto

Per avviare il progetto, seguire questi passaggi:

1. **Setup dell'ambiente**:
```bash
# Clona il repository (se applicabile)
git clone https://github.com/username/progetto_hds.git
cd progetto_hds

# Crea e attiva l'ambiente virtuale
python -m venv hds_env
source hds_env/bin/activate  # Per macOS/Linux

# Installa le dipendenze
pip install -r requirements.txt
```

2. **Download e preprocessing dei dati**:
```bash
# Esegui lo script di download
python -c "from src.data.download import download_fakeddit; download_fakeddit()"

# Processa i dati
python -c "from src.data.preprocess import preprocess_comments; preprocess_comments('data/raw/all_comments.csv', 'data/raw/all_submissions.csv')"
```

3. **Esecuzione dell'analisi completa**:
```bash
# Esegui la pipeline completa
python src/main.py
```

4. **Esplorazione tramite notebook** (opzionale):
```bash
# Avvia Jupyter Notebook
jupyter notebook notebooks/
```

## 7. Timeline del Progetto

### Settimana 1: Setup e Acquisizione Dati
- Setup dell'ambiente di sviluppo
- Download del dataset Fakeddit
- Preprocessing iniziale e pulizia dei dati
- Analisi esplorativa preliminare

### Settimana 2-3: Estrazione Feature e Analisi Preliminare
- Implementazione dei moduli di estrazione delle feature
- Estrazione delle feature di sentiment
- Estrazione delle metriche di leggibilità
- Estrazione dei metadata dei commenti
- Validazione della qualità delle feature estratte

### Settimana 3-4: Analisi Statistica
- Implementazione dei test di ipotesi
- Calcolo delle statistiche descrittive
- Analisi di correlazione
- Correzione per test multipli
- Documentazione dei risultati preliminari

### Settimana 4-5: Analisi Avanzata e Reporting
- Analisi di regressione
- Valutazione della robustezza dei risultati
- Creazione di visualizzazioni
- Stesura del report finale
- Preparazione della presentazione

## 8. Output Attesi del Progetto

1. **Dataset arricchito**: Dataset Fakeddit con feature di sentiment estratte
2. **Risultati statistici**:
   - Tabelle con statistiche descrittive
   - Risultati dei test di ipotesi
   - Matrici di correlazione
   - P-value e significatività delle relazioni
3. **Visualizzazioni**:
   - Distribuzioni delle feature per notizie vere vs. fake
   - Heatmap di correlazione
   - Grafici di regressione
4. **Report finale**: Documento dettagliato con metodologia, risultati e conclusioni
5. **Codice riutilizzabile**: Pipeline e moduli Python ben documentati e riutilizzabili

---

**Nota**: Questa implementazione tecnica è focalizzata sull'analisi statistica della relazione tra sentiment dei commenti e veridicità delle notizie, come richiesto dal professore. La parte di modellazione predittiva è stata mantenuta minima, concentrandosi invece sull'aspetto statistico e metodologico del problema.
