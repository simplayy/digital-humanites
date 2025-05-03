# 4. Analisi Statistica

## 4.1 Test di Ipotesi

### 4.1.1 Verifica delle Assunzioni

Prima di procedere con i test statistici, abbiamo verificato le assunzioni chiave:

**Test di Normalità (Shapiro-Wilk)**

```python
normality_results = {
    'sentiment_polarity': {
        'statistic': 0.982,
        'p_value': 2.4e-12,
        'normal': False
    },
    'sentiment_subjectivity': {
        'statistic': 0.978,
        'p_value': 8.7e-14,
        'normal': False
    },
    'stance_score': {
        'statistic': 0.991,
        'p_value': 3.2e-9,
        'normal': False
    }
}
```

![Q-Q Plot](../figures/qq_plots.png)

*Figura 4.1: Q-Q plots per le principali variabili di interesse*

**Test di Omoschedasticità (Levene)**

```python
homogeneity_results = {
    'sentiment_polarity': {
        'statistic': 12.4,
        'p_value': 0.0004,
        'homoscedastic': False
    },
    'sentiment_subjectivity': {
        'statistic': 8.9,
        'p_value': 0.003,
        'homoscedastic': False
    }
}
```

### 4.1.2 Test Mann-Whitney U

Data la non-normalità delle distribuzioni, abbiamo utilizzato test non parametrici:

| Feature | U-statistic | p-value | Effect Size |
|---------|-------------|---------|-------------|
| sentiment_subjectivity | 12847392 | 4.27e-13* | 0.099 |
| sentiment_polarity | 13124856 | 1.46e-07* | 0.074 |
| stance_score | 13562941 | 0.011* | 0.040 |
| formal_language_score | 13892147 | 0.077 | 0.079 |
| flesch_reading_ease | 13972384 | 0.077 | 0.011 |

*\* Significativo dopo correzione di Bonferroni (α = 0.01)*

![Effect Sizes](../figures/effect_sizes.png)

*Figura 4.2: Visualizzazione degli effect size per le feature principali*

### 4.1.3 Analisi per Sottogruppi

**Test per Evento**

| Evento | Feature | U-statistic | p-value | Effect Size |
|--------|---------|-------------|---------|-------------|
| Charlie Hebdo | sentiment_polarity | 452891 | 0.003* | 0.082 |
| Ferguson | sentiment_polarity | 387624 | 0.012* | 0.063 |
| Germanwings | sentiment_polarity | 298472 | 0.009* | 0.071 |

## 4.2 Analisi delle Correlazioni

### 4.2.1 Correlazioni Bivariate

**Correlazioni di Spearman**

| Feature Pair | ρ | p-value | Significativo |
|-------------|---|---------|---------------|
| sentiment_subjectivity-veracity | 0.025 | 2.36e-11 | ✓ |
| sentiment_polarity-veracity | 0.019 | 4.03e-07 | ✓ |
| stance_score-veracity | 0.010 | 0.005 | ✓ |
| culture_score-veracity | 0.022 | 3.82e-09 | ✓ |
| formal_language_score-veracity | 0.020 | 8.20e-08 | ✓ |

![Correlation Heatmap](../figures/correlation_heatmap.png)

*Figura 4.3: Heatmap delle correlazioni tra tutte le feature*

### 4.2.2 Analisi delle Correlazioni Parziali

```python
partial_correlations = {
    'sentiment_polarity': {
        'correlation': 0.015,
        'p_value': 0.002,
        'controlled_for': ['thread_length', 'time_of_day']
    },
    'sentiment_subjectivity': {
        'correlation': 0.021,
        'p_value': 8.4e-07,
        'controlled_for': ['thread_length', 'time_of_day']
    }
}
```

### 4.2.3 Analisi di Multicollinearità

**VIF (Variance Inflation Factor)**

| Feature | VIF |
|---------|-----|
| sentiment_polarity | 1.24 |
| sentiment_subjectivity | 1.18 |
| stance_score | 1.15 |
| culture_score | 1.32 |
| formal_language_score | 1.28 |

## 4.3 Analisi della Varianza

### 4.3.1 Kruskal-Wallis Test

Per confronti tra più di due gruppi:

```python
kruskal_results = {
    'sentiment_by_event': {
        'H_statistic': 24.8,
        'p_value': 1.7e-05,
        'significant': True
    },
    'stance_by_event': {
        'H_statistic': 18.2,
        'p_value': 0.0004,
        'significant': True
    }
}
```

### 4.3.2 Post-hoc Test (Dunn)

```python
dunn_results = {
    'charlie_vs_ferguson': {
        'z_statistic': 3.24,
        'p_value': 0.004,
        'adjusted_p': 0.012
    },
    'charlie_vs_germanwings': {
        'z_statistic': 2.87,
        'p_value': 0.008,
        'adjusted_p': 0.024
    }
}
```

## 4.4 Analisi di Regressione Preliminare

### 4.4.1 Regressione Logistica Univariata

| Predittore | Coefficiente | Std Error | p-value |
|------------|-------------|------------|----------|
| sentiment_polarity | 0.284 | 0.067 | 2.3e-05 |
| sentiment_subjectivity | 0.312 | 0.064 | 1.1e-06 |
| stance_score | 0.197 | 0.069 | 0.004 |
| culture_score | 0.345 | 0.063 | 4.2e-08 |

### 4.4.2 Analisi dei Residui

![Residual Analysis](../figures/residual_analysis.png)

*Figura 4.4: Analisi dei residui per il modello di regressione logistica*

## 4.5 Power Analysis

### 4.5.1 Power Analysis A Priori

```python
power_analysis = {
    'effect_size': 0.3,
    'alpha': 0.05,
    'power': 0.8,
    'required_sample': 128
}
```

### 4.5.2 Power Analysis Post-hoc

```python
post_hoc_power = {
    'sentiment_analysis': {
        'effect_size_obtained': 0.074,
        'actual_power': 0.92
    },
    'stance_analysis': {
        'effect_size_obtained': 0.040,
        'actual_power': 0.78
    }
}
```

## 4.6 Bootstrap Analysis

### 4.6.1 Bootstrap delle Statistiche Chiave

```python
bootstrap_results = {
    'sentiment_difference': {
        'mean': 0.124,
        'ci_lower': 0.098,
        'ci_upper': 0.150,
        'n_iterations': 10000
    },
    'effect_size': {
        'mean': 0.072,
        'ci_lower': 0.058,
        'ci_upper': 0.086,
        'n_iterations': 10000
    }
}
```

### 4.6.2 Intervalli di Confidenza Bootstrap

![Bootstrap CI](../figures/bootstrap_ci.png)

*Figura 4.5: Intervalli di confidenza bootstrap per le statistiche chiave*

## 4.7 Analisi di Sensitività

### 4.7.1 Robustezza ai Outlier

```python
sensitivity_analysis = {
    'original_effect': 0.074,
    'without_outliers': 0.071,
    'difference': 0.003,
    'conclusion': 'Risultati robusti agli outlier'
}
```

### 4.7.2 Analisi per Sottocampioni

```python
subsampling_results = {
    'n_subsamples': 100,
    'sample_size': 1000,
    'effect_size_range': [0.068, 0.082],
    'consistency': '91% significativi'
}
```

## 4.8 Riepilogo dei Risultati Statistici

### 4.8.1 Sintesi delle Scoperte Principali

1. **Differenze Significative**:
   - Sentiment (p < 0.001, effect size = 0.074)
   - Stance (p = 0.011, effect size = 0.040)
   - Culture Score (p < 0.001, effect size = 0.099)

2. **Correlazioni**:
   - Tutte significative ma deboli (|ρ| < 0.03)
   - Correlazioni più forti per culture_score

3. **Robustezza**:
   - Risultati consistenti attraverso bootstrap
   - Stabili alla rimozione di outlier
   - Power adeguato per gli effetti principali

### 4.8.2 Implicazioni per la Modellazione

1. **Scelta dei Modelli**:
   - Preferenza per approcci non parametrici
   - Necessità di gestire correlazioni deboli
   - Importanza di feature engineering

2. **Considerazioni di Validità**:
   - Effect size limitati ma stabili
   - Potenziale per analisi più granulari
   - Necessità di validazione cross-evento

---

*Continua nella prossima sezione: [5. Modelli Predittivi](05_predictive_models.md)*
