# Relazione tra Sentiment nei Commenti e Veridicità delle Notizie
## Analisi del Dataset PHEME

---

## Agenda

1. Introduzione e obiettivi
2. Dataset e metodologia
3. Principali risultati
4. Discussione e implicazioni
5. Limitazioni e direzioni future
6. Conclusioni
7. Domande

---

## Introduzione

### Contesto
- Disinformazione online: sfida significativa per la società contemporanea
- Focus tradizionale: caratteristiche intrinseche delle fake news
- Gap: pattern di risposta che le notizie false generano negli utenti

### Obiettivi dello Studio
- Esplorare la relazione tra sentiment nei commenti e veridicità delle notizie
- Identificare pattern linguistici distintivi
- Valutare il potere predittivo delle caratteristiche linguistiche

---

## Ipotesi di Ricerca

### Ipotesi Principale
**H1**: Esistono differenze statisticamente significative nei pattern di sentiment tra i commenti alle notizie vere e quelli alle notizie false

### Ipotesi Secondarie
1. Differenze nella soggettività dei commenti
2. Differenze nella polarità del sentiment
3. Differenze nell'atteggiamento (stance)
4. Differenze nelle metriche di leggibilità e acculturazione
5. Potere predittivo significativo delle feature linguistiche

---

## Dataset PHEME

### Caratteristiche
- 6,425 thread di conversazione Twitter
- 105,354 tweet totali
- Distribuzione: 93% notizie vere, 7% notizie false
- Eventi coperti: Charlie Hebdo, Germanwings, Ferguson, ecc.

### Vantaggi
- Thread conversazionali completi
- Annotazioni di veridicità verificate
- Copertura di eventi diversi

---

## Metodologia

### Preprocessing
- Pulizia dei testi (URL, menzioni, hashtag)
- Normalizzazione (minuscolo, stop words, lemmatizzazione)
- Organizzazione in thread conversazionali

### Feature Extraction
1. **Sentiment Analysis**
   - Polarità e soggettività (TextBlob)

2. **Stance Analysis**
   - Atteggiamento rispetto al tweet principale

3. **Leggibilità e Acculturazione**
   - Indice di Flesch, type-token ratio
   - Culture score, formal language score
   - Metriche di complessità linguistica

---

## Analisi Statistica

### Approccio Multi-metodologico
1. Test di ipotesi (Mann-Whitney U)
2. Analisi delle correlazioni
3. Modelli predittivi
   - Regressione logistica (modello lineare)
   - Random Forest (modello non lineare)
4. Confronto tra set di feature

### Valutazione
- Significatività statistica (p-value)
- Effect size (rilevanza pratica)
- Metriche di performance predittiva (AUC, F1)

---

## Risultati: Test di Ipotesi

| Feature | p-value corretto | Significativo | Effect Size |
|---------|------------------|---------------|-------------|
| sentiment_subjectivity | 4.27e-13 | ✅ | 0.099 |
| sentiment_polarity | 1.46e-07 | ✅ | 0.074 |
| stance_score | 0.011 | ✅ | 0.040 |
| formal_language_score | 0.077 | ❌ | 0.079 |
| flesch_reading_ease | 0.077 | ❌ | 0.011 |

**Interpretazione**: Differenze statisticamente significative ma con effect size trascurabile

---

## Risultati: Modelli Predittivi

| Metrica | Regressione Logistica | Random Forest | Differenza |
|---------|----------------------|---------------|------------|
| Accuracy | 0.928 | 0.944 | +0.016 |
| Precision | 0.928 | 0.946 | +0.018 |
| Recall | 1.000 | 0.996 | -0.004 |
| F1 Score | 0.963 | 0.971 | +0.008 |
| ROC AUC | 0.542 | 0.932 | +0.390 |

**Interpretazione**: Significativa superiorità del modello non lineare

---

## Risultati: Importanza delle Feature

Top 10 feature più importanti nel Random Forest:

1. thread_id (0.0141)
2. tweet_id (0.0137)
3. reaction_index (0.0028)
4. culture_score (0.0021) 👈
5. avg_word_length (0.0019)
6. sentiment_polarity (0.0018) 👈
7. flesch_reading_ease (0.0015)
8. long_words_ratio (0.0015)
9. formal_language_score (0.0014)
10. sentiment_subjectivity (0.0013) 👈

**Culture score** e complessità linguistica più importanti del sentiment!

---

## Confronto tra Set di Feature

| Set di Feature | N° Feature | ROC AUC | F1 Score |
|----------------|------------|---------|----------|
| sentiment_only | 2 | 0.559 | 0.595 |
| stance_only | 1 | 0.514 | 0.251 |
| readability_only | 7 | 0.571 | 0.906 |
| sentiment_stance | 3 | 0.548 | 0.639 |
| sentiment_readability | 9 | 0.579 | 0.925 |
| all_features | 10 | 0.582 | 0.925 |

**Feature di leggibilità** più predittive del sentiment!

---

## Discussione: Principali Contributi

### 1. Relazioni Non Lineari
- Significativo miglioramento del Random Forest (+39% AUC)
- Le relazioni tra sentiment e veridicità sono complesse e non lineari

### 2. Limitato Ruolo del Sentiment
- Effect size trascurabile nelle differenze di sentiment
- Performance limitata dei modelli basati solo sul sentiment

### 3. Importanza dell'Acculturazione
- Culture score tra le feature più importanti
- Set di feature di leggibilità superiori alle feature di sentiment

---

## Implicazioni Pratiche

### Per Strumenti di Fact-checking
- Integrare analisi di sentiment con metriche di leggibilità e acculturazione
- Utilizzare modelli non lineari per catturare relazioni complesse

### Per Educazione ai Media
- Enfatizzare l'importanza di valutare la qualità argomentativa
- Sviluppare sensibilità alla complessità linguistica come indicatore

### Per Piattaforme Social
- Implementare sistemi di allerta basati anche su pattern linguistici
- Considerare la dinamica conversazionale oltre al sentiment

---

## Limitazioni dello Studio

### 1. Dataset Specifico
- Limitato a eventi particolari e alla piattaforma Twitter
- Sbilanciamento significativo (93% notizie vere, 7% false)

### 2. Rischio di Overfitting
- Alta importanza di feature specifiche del dataset (thread_id, tweet_id)
- Possibile memorizzazione di pattern non generalizzabili

### 3. Analisi Statica
- Mancata considerazione dell'evoluzione temporale del sentiment nei thread
- Perdita potenziale di informazioni sulla struttura conversazionale

---

## Direzioni Future

### 1. Analisi Temporali
- Studiare l'evoluzione del sentiment nei thread
- Analizzare velocità e pattern di propagazione

### 2. Stratificazione Contestuale
- Analisi separate per tema o evento
- Incorporare metadati contestuali nei modelli

### 3. Feature Engineering Avanzato
- Sviluppare feature per dinamiche conversazionali
- Approfondire l'analisi del culture score

### 4. Validazione Cross-dataset
- Testare i modelli su dataset diversi
- Confrontare risultati tra piattaforme diverse

---

## Conclusioni

### 1. Evidenze Empiriche
- Relazioni statisticamente significative ma con effect size limitato
- Superiorità dei modelli non lineari
- Importanza delle feature di leggibilità e acculturazione

### 2. Ridimensionamento del Ruolo del Sentiment
- L'analisi del sentiment da sola è insufficiente
- Le dinamiche cognitive potrebbero essere più rilevanti delle dinamiche affettive

### 3. Contributo alla Comprensione della Disinformazione
- Necessità di approcci multidimensionali
- Importanza della complessità linguistica e dell'acculturazione

---

## Grazie per l'attenzione

### Domande?

---

## Slide di Backup

---

### Dettagli sul Culture Score

Il **culture score** è un punteggio composito che misura il livello di acculturazione del testo, calcolato come:

```
culture_score = (0.4 * vocabulary_richness) + 
                (0.3 * formal_language_score) + 
                (0.2 * type_token_ratio) + 
                (0.1 * normalized_flesch)
```

Dove:
- `vocabulary_richness`: Misura basata sugli hapax legomena
- `formal_language_score`: Grado di formalità del linguaggio
- `type_token_ratio`: Diversità lessicale
- `normalized_flesch`: Versione normalizzata dell'indice di leggibilità di Flesch

---

### Validazione Incrociata

Performance del Random Forest in 5-fold cross-validation:

- **Media AUC**: 0.582 ± 0.028
- **Media F1**: 0.925 ± 0.015
- **Media Accuracy**: 0.862 ± 0.023

Coefficiente di variazione: 4.8% (stabilità accettabile)

---

### Differenze di Sentiment per Evento

| Evento | Diff. Polarità | p-value | Diff. Soggettività | p-value |
|--------|---------------|---------|-------------------|---------|
| Charlie Hebdo | 0.082 | 0.003 | 0.114 | 5.2e-9 |
| Ferguson | 0.063 | 0.012 | 0.087 | 0.008 |
| Germanwings | 0.071 | 0.009 | 0.095 | 0.001 |

**Interpretazione**: Pattern consistenti ma con variazioni nell'effect size
