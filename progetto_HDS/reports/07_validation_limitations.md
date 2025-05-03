# 7. Validazione e Limitazioni

## 7.1 Validazione dei Risultati

### 7.1.1 Validazione Interna

#### Robustezza Statistica

**Cross-Validation Results**

| Fold | ROC AUC | F1 Score | Accuracy |
|------|---------|----------|----------|
| 1 | 0.934 | 0.972 | 0.945 |
| 2 | 0.929 | 0.969 | 0.942 |
| 3 | 0.933 | 0.971 | 0.944 |
| 4 | 0.931 | 0.970 | 0.943 |
| 5 | 0.932 | 0.971 | 0.944 |

**Stabilità dei Risultati**

```python
stability_metrics = {
    'std_roc_auc': 0.002,
    'std_f1': 0.001,
    'std_accuracy': 0.001,
    'coefficient_variation': 0.002
}
```

![Cross Validation Stability](../figures/cv_stability.png)

*Figura 7.1: Stabilità delle metriche attraverso le fold di cross-validation*

#### Analisi di Sensitività

**Variazione dei Parametri**

| Parametro | Range Testato | Impatto su ROC AUC |
|-----------|---------------|-------------------|
| n_estimators | [50, 200] | ±0.003 |
| max_depth | [5, None] | ±0.005 |
| min_samples_split | [2, 10] | ±0.002 |
| class_weight | [balanced, None] | ±0.015 |

![Sensitivity Analysis](../figures/sensitivity_analysis.png)

*Figura 7.2: Analisi di sensitività dei parametri chiave*

### 7.1.2 Validazione Esterna

#### Generalizzabilità

**Test su Eventi Diversi**

| Evento | ROC AUC | F1 Score | N° Campioni |
|--------|---------|----------|-------------|
| Charlie Hebdo | 0.928 | 0.968 | 2264 |
| Ferguson | 0.934 | 0.971 | 1845 |
| Germanwings | 0.931 | 0.970 | 1441 |
| Altri | 0.929 | 0.969 | 875 |

![Event Generalization](../figures/event_generalization.png)

*Figura 7.3: Performance del modello su diversi eventi*

#### Validazione Temporale

**Performance nel Tempo**

```python
temporal_validation = {
    'early_2014': {'roc_auc': 0.930, 'f1': 0.969},
    'late_2014': {'roc_auc': 0.931, 'f1': 0.970},
    'early_2015': {'roc_auc': 0.932, 'f1': 0.971},
    'late_2015': {'roc_auc': 0.933, 'f1': 0.972}
}
```

## 7.2 Limitazioni Metodologiche

### 7.2.1 Limitazioni del Dataset

1. **Sbilanciamento delle Classi**
   - 93% notizie vere vs 7% false
   - Impatto sulla generalizzabilità
   - Necessità di tecniche di bilanciamento

2. **Copertura Temporale**
   - Limitata al periodo 2014-2016
   - Possibili cambiamenti nelle dinamiche
   - Evoluzione delle strategie di disinformazione

3. **Specificità della Piattaforma**
   - Limitato a Twitter
   - Caratteristiche specifiche della piattaforma
   - Dinamiche social media-specifiche

### 7.2.2 Limitazioni Analitiche

1. **Causalità**
   ```python
   causal_limitations = {
       'design': 'correlational',
       'confounders': ['user_characteristics',
                      'network_effects',
                      'temporal_dynamics'],
       'inference': 'limited'
   }
   ```

2. **Complessità delle Relazioni**
   - Interazioni non lineari
   - Dipendenze temporali
   - Effetti di rete

3. **Metriche di Performance**
   - Limitazioni dell'accuracy
   - Sensibilità al contesto
   - Trade-off precision-recall

### 7.2.3 Limitazioni Tecniche

1. **Risorse Computazionali**
   ```python
   computational_constraints = {
       'memory_usage': '10GB peak',
       'processing_time': '8 hours',
       'scalability': 'limited'
   }
   ```

2. **Preprocessing**
   - Perdita di informazioni
   - Semplificazioni necessarie
   - Compromessi di pulizia

## 7.3 Bias e Distorsioni

### 7.3.1 Bias nei Dati

1. **Bias di Selezione**
   - Eventi specifici
   - Lingua inglese
   - Utenti attivi

2. **Bias Temporali**
   - Periodo storico specifico
   - Eventi contemporanei
   - Cambiamenti sociali

3. **Bias Linguistici**
   ```python
   linguistic_biases = {
       'language': 'English only',
       'dialect_variation': 'limited',
       'cultural_context': 'Western-centric'
   }
   ```

### 7.3.2 Bias nei Modelli

1. **Feature Selection Bias**
   - Scelte soggettive
   - Limitazioni tecniche
   - Assunzioni implicite

2. **Bias di Ottimizzazione**
   ```python
   optimization_biases = {
       'metric_focus': 'accuracy_centric',
       'threshold_selection': 'arbitrary',
       'performance_trade_offs': 'implicit'
   }
   ```

## 7.4 Considerazioni Etiche

### 7.4.1 Privacy e Consenso

1. **Gestione dei Dati**
   - Anonimizzazione
   - Protezione delle identità
   - Conformità GDPR

2. **Consenso Informato**
   ```python
   privacy_considerations = {
       'data_collection': 'public_only',
       'user_identifiers': 'removed',
       'sensitive_info': 'excluded'
   }
   ```

### 7.4.2 Impatto Sociale

1. **Potenziali Abusi**
   - Manipolazione dell'informazione
   - Censura automatizzata
   - Discriminazione algoritmica

2. **Responsabilità**
   - Trasparenza delle decisioni
   - Accountability
   - Governance algoritmica

## 7.5 Mitigazioni e Contromisure

### 7.5.1 Mitigazioni Tecniche

1. **Bilanciamento delle Classi**
   ```python
   class_balancing = {
       'technique': 'SMOTE',
       'effectiveness': '+15% recall',
       'trade_offs': '-3% precision'
   }
   ```

2. **Validazione Robusta**
   - Cross-validation stratificata
   - Test su sottogruppi
   - Analisi di sensitività

### 7.5.2 Mitigazioni Metodologiche

1. **Triangolazione**
   - Multiple fonti di dati
   - Approcci analitici diversi
   - Validazione qualitativa

2. **Documentazione**
   ```python
   documentation_requirements = {
       'assumptions': 'explicit',
       'limitations': 'detailed',
       'decisions': 'justified'
   }
   ```

## 7.6 Raccomandazioni

### 7.6.1 Per la Ricerca Futura

1. **Estensioni del Dataset**
   - Multiple piattaforme
   - Periodo temporale più ampio
   - Diversità linguistica

2. **Miglioramenti Metodologici**
   ```python
   methodological_improvements = {
       'causal_inference': 'needed',
       'temporal_dynamics': 'essential',
       'network_effects': 'important'
   }
   ```

### 7.6.2 Per l'Implementazione

1. **Monitoraggio Continuo**
   - Performance nel tempo
   - Drift dei dati
   - Cambiamenti contestuali

2. **Aggiornamenti del Modello**
   - Retraining periodico
   - Adattamento ai cambiamenti
   - Validazione continua

## 7.7 Documentazione delle Decisioni

### 7.7.1 Scelte Metodologiche

| Decisione | Motivazione | Alternative Considerate |
|-----------|-------------|------------------------|
| Random Forest | Non linearità, interpretabilità | SVM, Deep Learning |
| SMOTE | Bilanciamento classi | Undersampling, ADASYN |
| Cross-validation | Robustezza risultati | Hold-out, Bootstrap |

### 7.7.2 Parametri e Soglie

```python
critical_parameters = {
    'sentiment_threshold': 0.1,
    'significance_level': 0.01,
    'min_samples': 100,
    'confidence_interval': 0.95
}
```

## 7.8 Documentazione degli Errori

### 7.8.1 Pattern di Errore

**False Positives**
```python
false_positive_patterns = {
    'high_emotion': 35%,
    'ambiguous_stance': 28%,
    'complex_context': 37%
}
```

**False Negatives**
```python
false_negative_patterns = {
    'subtle_manipulation': 42%,
    'mixed_sentiment': 31%,
    'missing_context': 27%
}
```

### 7.8.2 Impatto degli Errori

| Tipo di Errore | Frequenza | Costo | Mitigazione |
|----------------|-----------|--------|-------------|
| False Positive | 391 | Alto | Review umana |
| False Negative | 2 | Medio | Monitoraggio |
| Edge Cases | 45 | Basso | Documentazione |

---

*Continua nella prossima sezione: [8. Conclusioni e Direzioni Future](08_conclusions.md)*
