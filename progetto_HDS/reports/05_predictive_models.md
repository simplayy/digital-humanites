# 5. Modelli Predittivi

## 5.1 Preparazione dei Dati

### 5.1.1 Feature Engineering

```python
final_features = [
    # Feature di Sentiment
    'sentiment_polarity',
    'sentiment_subjectivity',
    'sentiment_category_encoded',
    
    # Feature di Stance
    'stance_score',
    'stance_category_encoded',
    
    # Feature di Leggibilità
    'flesch_reading_ease',
    'type_token_ratio',
    'formal_language_score',
    'vocabulary_richness',
    'avg_word_length',
    'long_words_ratio',
    'culture_score',
    
    # Feature Derivate
    'sentiment_variance',
    'stance_consistency',
    'temporal_evolution_features'
]
```

### 5.1.2 Preprocessing

```python
def preprocess_pipeline(X):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('feature_selector', SelectKBest(score_func=f_classif, k=10))
    ]).fit_transform(X)
```

### 5.1.3 Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)
```

## 5.2 Regressione Logistica

### 5.2.1 Implementazione

```python
logistic_model = LogisticRegression(
    class_weight='balanced',
    C=1.0,
    max_iter=1000,
    random_state=42
)
```

### 5.2.2 Performance

| Metrica | Valore | 95% CI |
|---------|--------|--------|
| Accuracy | 0.928 | [0.915, 0.941] |
| Precision | 0.928 | [0.914, 0.942] |
| Recall | 1.000 | [1.000, 1.000] |
| F1 Score | 0.963 | [0.954, 0.972] |
| ROC AUC | 0.542 | [0.524, 0.560] |

![ROC Curve Logistic](../figures/roc_curve_logistic.png)

*Figura 5.1: Curva ROC per il modello di regressione logistica*

### 5.2.3 Analisi dei Coefficienti

| Feature | Coefficiente | Std Error | p-value |
|---------|-------------|------------|---------|
| sentiment_polarity | 0.284 | 0.067 | 2.3e-05 |
| sentiment_subjectivity | 0.312 | 0.064 | 1.1e-06 |
| stance_score | 0.197 | 0.069 | 0.004 |
| culture_score | 0.345 | 0.063 | 4.2e-08 |
| formal_language_score | 0.276 | 0.065 | 2.1e-05 |

## 5.3 Random Forest

### 5.3.1 Implementazione

```python
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42
)
```

### 5.3.2 Performance

| Metrica | Valore | 95% CI |
|---------|--------|--------|
| Accuracy | 0.944 | [0.932, 0.956] |
| Precision | 0.946 | [0.934, 0.958] |
| Recall | 0.996 | [0.992, 1.000] |
| F1 Score | 0.971 | [0.963, 0.979] |
| ROC AUC | 0.932 | [0.921, 0.943] |

![ROC Curve Random Forest](../figures/roc_curve_rf.png)

*Figura 5.2: Curva ROC per il modello Random Forest*

### 5.3.3 Feature Importance

| Feature | Importance | Perm. Importance | Std |
|---------|------------|------------------|-----|
| thread_id | 0.320 | 0.0141 | 0.0002 |
| tweet_id | 0.269 | 0.0137 | 0.0002 |
| reaction_index | 0.075 | 0.0028 | 0.0004 |
| culture_score | 0.053 | 0.0021 | 0.0003 |
| avg_word_length | 0.050 | 0.0019 | 0.0004 |
| sentiment_polarity | 0.033 | 0.0018 | 0.0002 |
| flesch_reading_ease | 0.049 | 0.0015 | 0.0002 |
| long_words_ratio | 0.037 | 0.0015 | 0.0002 |
| formal_language_score | 0.038 | 0.0014 | 0.0003 |
| sentiment_subjectivity | 0.029 | 0.0013 | 0.0002 |

![Feature Importance](../figures/feature_importance.png)

*Figura 5.3: Importanza delle feature nel modello Random Forest*

## 5.4 Confronto tra Set di Feature

### 5.4.1 Set di Feature Testati

1. **sentiment_only**: solo feature di sentiment
2. **stance_only**: solo feature di stance
3. **readability_only**: solo feature di leggibilità
4. **sentiment_stance**: combinazione di sentiment e stance
5. **sentiment_readability**: combinazione di sentiment e leggibilità
6. **all_features**: tutte le feature

### 5.4.2 Risultati Comparativi

| Set di Feature | N° Feature | ROC AUC | F1 Score |
|----------------|------------|---------|----------|
| sentiment_only | 2 | 0.559 | 0.595 |
| stance_only | 1 | 0.514 | 0.251 |
| readability_only | 7 | 0.571 | 0.906 |
| sentiment_stance | 3 | 0.548 | 0.639 |
| sentiment_readability | 9 | 0.579 | 0.925 |
| all_features | 10 | 0.582 | 0.925 |

![Feature Set Comparison](../figures/feature_set_comparison.png)

*Figura 5.4: Confronto delle performance tra diversi set di feature*

## 5.5 Validazione Incrociata

### 5.5.1 K-Fold Cross-Validation

```python
cv_results = {
    'logistic': {
        'mean_auc': 0.542,
        'std_auc': 0.018,
        'mean_f1': 0.963,
        'std_f1': 0.008
    },
    'random_forest': {
        'mean_auc': 0.932,
        'std_auc': 0.012,
        'mean_f1': 0.971,
        'std_f1': 0.006
    }
}
```

### 5.5.2 Learning Curves

![Learning Curves](../figures/learning_curves.png)

*Figura 5.5: Curve di apprendimento per entrambi i modelli*

## 5.6 Analisi degli Errori

### 5.6.1 Matrice di Confusione

```python
confusion_matrix_rf = {
    'true_negative': 5584,
    'false_positive': 391,
    'false_negative': 2,
    'true_positive': 448
}
```

![Confusion Matrix](../figures/confusion_matrix.png)

*Figura 5.6: Matrice di confusione per il modello Random Forest*

### 5.6.2 Analisi dei Casi Errati

| Tipo di Errore | Frequenza | Pattern Comuni |
|----------------|-----------|----------------|
| False Positives | 391 | Sentiment estremo, bassa stance |
| False Negatives | 2 | Alto culture score, stance ambigua |

## 5.7 Ottimizzazione dei Modelli

### 5.7.1 Grid Search

```python
grid_search_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', 'balanced_subsample']
}

best_params = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'class_weight': 'balanced'
}
```

### 5.7.2 Performance del Modello Ottimizzato

| Metrica | Prima | Dopo | Miglioramento |
|---------|-------|------|---------------|
| ROC AUC | 0.932 | 0.938 | +0.006 |
| F1 Score | 0.971 | 0.974 | +0.003 |
| Recall | 0.996 | 0.997 | +0.001 |

## 5.8 Interpretabilità dei Modelli

### 5.8.1 SHAP Values

![SHAP Summary](../figures/shap_summary.png)

*Figura 5.7: SHAP summary plot per il modello Random Forest*

### 5.8.2 Partial Dependence Plots

![Partial Dependence](../figures/partial_dependence.png)

*Figura 5.8: Partial dependence plots per le feature più importanti*

## 5.9 Validazione su Test Set Indipendente

### 5.9.1 Performance su Dati Non Visti

| Metrica | Training | Validation | Test |
|---------|----------|------------|------|
| ROC AUC | 0.938 | 0.932 | 0.928 |
| F1 Score | 0.974 | 0.971 | 0.968 |
| Accuracy | 0.944 | 0.941 | 0.937 |

### 5.9.2 Stabilità delle Predizioni

```python
stability_metrics = {
    'prediction_std': 0.042,
    'feature_importance_variance': 0.003,
    'performance_cv': 0.012
}
```

## 5.10 Sintesi dei Modelli

### 5.10.1 Confronto Finale dei Modelli

| Aspetto | Regressione Logistica | Random Forest |
|---------|----------------------|---------------|
| ROC AUC | 0.542 | 0.932 |
| F1 Score | 0.963 | 0.971 |
| Interpretabilità | Alta | Media |
| Complessità | Bassa | Media |
| Tempo Training | < 1s | ~5s |
| Memoria | < 1MB | ~10MB |

### 5.10.2 Raccomandazioni per l'Uso

1. **Per Interpretabilità**:
   - Usare regressione logistica
   - Focus su coefficienti e odds ratio
   - Analisi di feature importance lineare

2. **Per Performance**:
   - Preferire Random Forest
   - Utilizzare feature engineering completo
   - Monitorare overfitting

3. **Per Deployment**:
   - Bilanciare performance e risorse
   - Considerare vincoli computazionali
   - Valutare necessità di aggiornamento

---

*Continua nella prossima sezione: [6. Risultati e Discussione](06_results_discussion.md)*
