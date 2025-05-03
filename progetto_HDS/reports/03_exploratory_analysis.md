# 3. Analisi Esplorativa

## 3.1 Statistiche Descrittive

### 3.1.1 Panoramica del Dataset

```python
dataset_stats = {
    'n_threads': 6425,
    'n_tweets': 105354,
    'n_true': 5975,  # 93%
    'n_false': 450,  # 7%
    'avg_reactions': 16.4,
    'median_reactions': 12
}
```

![Distribuzione delle Classi](../figures/class_distribution.png)

*Figura 3.1: Distribuzione delle classi di veridicità nel dataset*

### 3.1.2 Caratteristiche dei Tweet

| Metrica | Media | Mediana | Std Dev | Min | Max |
|---------|-------|---------|---------|-----|-----|
| Lunghezza (caratteri) | 112.3 | 98.5 | 45.7 | 10 | 280 |
| Parole per tweet | 18.6 | 16.0 | 8.4 | 2 | 57 |
| URL per tweet | 0.45 | 0 | 0.62 | 0 | 3 |
| Menzioni per tweet | 1.2 | 1 | 1.4 | 0 | 8 |
| Hashtag per tweet | 0.8 | 0 | 1.1 | 0 | 6 |

### 3.1.3 Distribuzione Temporale

![Distribuzione Temporale](../figures/temporal_distribution.png)

*Figura 3.2: Distribuzione temporale dei thread nel dataset*

## 3.2 Analisi delle Feature

### 3.2.1 Feature di Sentiment

**Distribuzione della Polarità**

```python
sentiment_stats = {
    'polarity': {
        'mean': 0.042,
        'median': 0.000,
        'std': 0.285,
        'skew': 0.156
    },
    'subjectivity': {
        'mean': 0.384,
        'median': 0.400,
        'std': 0.318,
        'skew': 0.124
    }
}
```

![Distribuzione del Sentiment](../figures/sentiment_distribution.png)

*Figura 3.3: Distribuzione della polarità e soggettività del sentiment*

**Categorizzazione del Sentiment**

| Categoria | % Totale | % Notizie Vere | % Notizie False |
|-----------|----------|----------------|-----------------|
| Positivo | 32.4% | 33.1% | 24.6% |
| Neutro | 41.2% | 40.8% | 45.2% |
| Negativo | 26.4% | 26.1% | 30.2% |

### 3.2.2 Feature di Stance

**Distribuzione della Stance**

```python
stance_distribution = {
    'supportive': 28.5%,
    'critical': 24.3%,
    'neutral': 31.2%,
    'opposing': 12.1%,
    'indirect_supportive': 3.9%
}
```

![Distribuzione della Stance](../figures/stance_distribution.png)

*Figura 3.4: Distribuzione delle categorie di stance nei commenti*

### 3.2.3 Feature di Leggibilità

**Statistiche di Leggibilità**

| Metrica | Media | Mediana | Std Dev |
|---------|-------|---------|---------|
| Flesch Reading Ease | 68.4 | 70.2 | 15.6 |
| Type-Token Ratio | 0.82 | 0.84 | 0.11 |
| Formal Language Score | 0.56 | 0.54 | 0.18 |
| Culture Score | 0.63 | 0.65 | 0.14 |

![Distribuzione delle Metriche di Leggibilità](../figures/readability_metrics.png)

*Figura 3.5: Distribuzione delle principali metriche di leggibilità*

## 3.3 Pattern Emergenti

### 3.3.1 Correlazioni tra Feature

![Matrice di Correlazione](../figures/correlation_matrix.png)

*Figura 3.6: Matrice di correlazione tra le principali feature*

### 3.3.2 Pattern per Classe di Veridicità

**Sentiment per Classe**

![Sentiment per Veridicità](../figures/sentiment_by_veracity.png)

*Figura 3.7: Distribuzione del sentiment per classe di veridicità*

**Leggibilità per Classe**

![Leggibilità per Veridicità](../figures/readability_by_veracity.png)

*Figura 3.8: Metriche di leggibilità per classe di veridicità*

### 3.3.3 Pattern Temporali

**Evoluzione del Sentiment nei Thread**

![Evoluzione del Sentiment](../figures/sentiment_evolution.png)

*Figura 3.9: Evoluzione del sentiment all'interno dei thread*

## 3.4 Analisi per Evento

### 3.4.1 Distribuzione degli Eventi

| Evento | % Thread | % Notizie False | Avg Reactions |
|--------|----------|-----------------|---------------|
| Charlie Hebdo | 35.2% | 8.4% | 18.3 |
| Ferguson | 28.7% | 6.2% | 15.8 |
| Germanwings | 22.4% | 5.8% | 14.2 |
| Altri | 13.7% | 7.1% | 16.9 |

### 3.4.2 Pattern Specifici per Evento

![Pattern per Evento](../figures/patterns_by_event.png)

*Figura 3.10: Confronto dei pattern principali tra diversi eventi*

## 3.5 Outlier e Casi Particolari

### 3.5.1 Identificazione degli Outlier

```python
def identify_outliers(df, columns):
    outliers = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers[col] = df[
            (df[col] < Q1 - 1.5*IQR) |
            (df[col] > Q3 + 1.5*IQR)
        ]
    return outliers
```

### 3.5.2 Analisi dei Casi Estremi

**Thread con Sentiment Estremo**

| Tipo | % del Totale | Caratteristiche Comuni |
|------|--------------|------------------------|
| Molto Positivi (>0.8) | 2.3% | Alto engagement, tema controverso |
| Molto Negativi (<-0.8) | 1.8% | Breaking news, eventi critici |

**Thread con Pattern Inusuali**

| Pattern | Frequenza | Note |
|---------|-----------|------|
| Inversione Sentiment | 4.2% | Cambi drastici dopo nuove info |
| Alta Dispersione | 3.7% | Temi polarizzanti |
| Consensus Rapido | 2.9% | Breaking news verificate |

## 3.6 Implicazioni per l'Analisi

### 3.6.1 Considerazioni Metodologiche

1. **Sbilanciamento delle Classi**
   - Necessità di tecniche di bilanciamento
   - Importanza di metriche appropriate
   - Stratificazione nel sampling

2. **Non-Normalità delle Distribuzioni**
   - Preferenza per test non parametrici
   - Trasformazioni dei dati necessarie
   - Robustezza alla non-normalità

3. **Dipendenze Temporali**
   - Considerazione dell'ordine temporale
   - Analisi delle sequenze
   - Pattern evolutivi

### 3.6.2 Direzioni per l'Analisi Dettagliata

1. **Feature Engineering**
   - Combinazione di feature esistenti
   - Creazione di feature temporali
   - Interazioni tra feature

2. **Stratificazione**
   - Analisi per evento
   - Considerazione del contesto
   - Pattern specifici per sottogruppi

3. **Modellazione**
   - Approcci non lineari
   - Considerazione di interazioni
   - Feature selection informata

## 3.7 Visualizzazioni Supplementari

### 3.7.1 Distribuzione delle Feature Chiave

![Feature Distributions](../figures/feature_distributions.png)

*Figura 3.11: Distribuzioni delle feature principali con densità kernel*

### 3.7.2 Analisi Bivariate

![Bivariate Analysis](../figures/bivariate_analysis.png)

*Figura 3.12: Plot di dispersione tra coppie di feature rilevanti*

### 3.7.3 Pattern Temporali Dettagliati

![Temporal Patterns](../figures/temporal_patterns.png)

*Figura 3.13: Analisi dettagliata dei pattern temporali per diverse metriche*

---

*Continua nella prossima sezione: [4. Analisi Statistica](04_statistical_analysis.md)*
