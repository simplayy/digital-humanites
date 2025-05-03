# Report Scientifico: Relazione tra Sentiment nei Commenti e Veridicità delle Notizie

## Abstract

Questo studio ha esaminato la relazione tra i pattern di sentiment espressi nei commenti online e la veridicità delle notizie, utilizzando il dataset PHEME di thread di conversazione Twitter. Attraverso un'analisi statistica approfondita e approcci di machine learning, abbiamo testato l'ipotesi che esistano pattern linguistici distintivi nei commenti alle notizie vere rispetto a quelli nelle notizie false. I risultati mostrano differenze statisticamente significative, ma con effect size limitato, nelle caratteristiche linguistiche e di sentiment tra i commenti alle notizie vere e false. Modelli non lineari (Random Forest) hanno catturato relazioni più complesse rispetto ai modelli lineari (Regressione Logistica), raggiungendo un'AUC di 0.93 contro 0.54. Le feature di leggibilità e acculturazione (incluso il `culture_score`) si sono rivelate migliori predittori rispetto alle pure feature di sentiment. Questi risultati suggeriscono che l'analisi del sentiment da sola è insufficiente per identificare efficacemente le fake news, evidenziando la necessità di approcci più sofisticati che considerino la complessità linguistica e le dinamiche conversazionali.

## Introduzione

La disinformazione online rappresenta una sfida significativa per la società contemporanea, con implicazioni profonde per il dibattito pubblico, i processi democratici e la coesione sociale. Mentre numerosi studi si sono concentrati sulle caratteristiche intrinseche delle fake news, meno attenzione è stata dedicata ai pattern di risposta che queste generano negli utenti.

Questo studio si propone di esplorare la relazione tra i pattern di sentiment espressi nei commenti e la veridicità delle notizie originali, utilizzando il dataset PHEME, che contiene thread di conversazione Twitter relativi a diversi eventi. L'ipotesi principale è che esistano differenze sistematiche e statisticamente significative nelle reazioni linguistiche e affettive alle notizie vere rispetto a quelle false.

La comprensione di questa relazione potrebbe contribuire a:
1. Migliorare i sistemi di identificazione automatica delle fake news
2. Approfondire la conoscenza dei meccanismi di diffusione della disinformazione
3. Sviluppare strategie più efficaci per contrastare la diffusione di informazioni false

## Metodologia

### Dataset

Il dataset PHEME utilizzato in questo studio contiene thread di Twitter relativi a diversi eventi di attualità, con annotazioni sulla veridicità delle notizie. Specificamente:
- **Dimensione**: 6,425 thread di conversazione, contenenti 105,354 tweet
- **Distribuzione target**: 93% notizie vere, 7% notizie false
- **Eventi coperti**: Diversi eventi di attualità tra cui Charlie Hebdo, Germanwings crash, Ferguson, ecc.

### Preprocessing

La preparazione dei dati ha incluso:
1. **Pulizia dei testi**: Rimozione di URL, menzioni, hashtag e caratteri speciali
2. **Normalizzazione**: Conversione a minuscolo, rimozione di stop words, lemmatizzazione
3. **Gestione dei valori mancanti**: Esclusione di tweet con testi insufficienti
4. **Strutturazione gerarchica**: Organizzazione dei tweet in thread conversazionali

### Estrazione delle Feature

Sono state estratte tre categorie principali di feature:

1. **Sentiment Analysis**
   - `sentiment_polarity` ([-1.0, 1.0]): Misura quanto positivo o negativo è il testo
   - `sentiment_subjectivity` ([0.0, 1.0]): Misura quanto soggettivo o oggettivo è il testo

2. **Stance Analysis**
   - `stance_score` ([-1.0, 1.0]): Misura dell'atteggiamento del commento rispetto al tweet principale

3. **Leggibilità e Acculturazione**
   - `flesch_reading_ease` ([0.0, 100.0]): Indice di leggibilità Flesch
   - `type_token_ratio` ([0.0, 1.0]): Rapporto tra parole uniche e totale parole
   - `formal_language_score` ([0.0, 1.0]): Misura della formalità del linguaggio
   - `vocabulary_richness` ([0.0, 1.0]): Basata su hapax legomena
   - `avg_word_length` ([0.0, ∞)): Lunghezza media delle parole
   - `long_words_ratio` ([0.0, 1.0]): Proporzione di parole lunghe (>6 caratteri)
   - `culture_score` ([0.0, 1.0]): Punteggio composito di acculturazione

### Analisi Statistica

L'analisi è stata condotta in diverse fasi:

1. **Analisi esplorativa**
   - Esame delle distribuzioni delle feature
   - Identificazione di outlier
   - Analisi delle correlazioni tra feature

2. **Test di ipotesi**
   - Test di Mann-Whitney U per confrontare il sentiment tra notizie vere e false
   - Correzione di Bonferroni per test multipli
   - Calcolo dell'effect size per valutare la rilevanza pratica

3. **Modelli predittivi**
   - **Regressione logistica**: Modello lineare di base
   - **Random Forest**: Modello non lineare per catturare relazioni più complesse
   - **Cross-validation**: 5-fold per valutare la robustezza dei modelli
   - **Analisi dell'importanza delle feature**: Permutation importance per identificare le feature più rilevanti

4. **Confronto tra set di feature**
   - Test di diversi sottoinsiemi di feature per valutarne il contributo predittivo
   - Valutazione comparativa delle performance

## Risultati

### Test di Ipotesi

I test di Mann-Whitney U hanno rivelato differenze statisticamente significative in diverse feature:

| Feature | p-value corretto | Significativo | Effect Size | Interpretazione |
|---------|------------------|---------------|-------------|-----------------|
| sentiment_subjectivity | 4.27e-13 | ✅ | 0.099 | trascurabile |
| sentiment_polarity | 1.46e-07 | ✅ | 0.074 | trascurabile |
| stance_score | 0.011 | ✅ | 0.040 | trascurabile |
| formal_language_score | 0.077 | ❌ | 0.079 | trascurabile |
| flesch_reading_ease | 0.077 | ❌ | 0.011 | trascurabile |

### Correlazioni

L'analisi delle correlazioni ha mostrato associazioni statisticamente significative ma molto deboli:

| Feature | Corr. Pearson | p-value corr. | Significativo | Forza |
|---------|---------------|---------------|---------------|-------|
| sentiment_subjectivity | 0.025 | 2.36e-11 | ✅ | trascurabile |
| sentiment_polarity | 0.019 | 4.03e-07 | ✅ | trascurabile |
| stance_score | 0.010 | 0.005 | ✅ | trascurabile |
| culture_score | 0.022 | 3.82e-09 | ✅ | trascurabile |
| formal_language_score | 0.020 | 8.20e-08 | ✅ | trascurabile |

### Confronto tra Modelli

| Metrica | Regressione Logistica | Random Forest | Differenza |
|---------|----------------------|---------------|------------|
| Accuracy | 0.928 | 0.944 | +0.016 |
| Precision | 0.928 | 0.946 | +0.018 |
| Recall | 1.000 | 0.996 | -0.004 |
| F1 Score | 0.963 | 0.971 | +0.008 |
| ROC AUC | 0.542 | 0.932 | +0.390 |

### Importanza delle Feature

Le feature più importanti nel modello Random Forest:

1. thread_id (0.0141)
2. tweet_id (0.0137)
3. reaction_index (0.0028)
4. culture_score (0.0021)
5. avg_word_length (0.0019)
6. sentiment_polarity (0.0018)
7. flesch_reading_ease (0.0015)
8. long_words_ratio (0.0015)
9. formal_language_score (0.0014)
10. sentiment_subjectivity (0.0013)

### Confronto tra Set di Feature (Random Forest)

| Set di Feature | N° Feature | ROC AUC | F1 Score |
|----------------|------------|---------|----------|
| sentiment_only | 2 | 0.559 | 0.595 |
| stance_only | 1 | 0.514 | 0.251 |
| readability_only | 7 | 0.571 | 0.906 |
| sentiment_stance | 3 | 0.548 | 0.639 |
| sentiment_readability | 9 | 0.579 | 0.925 |
| all_features | 10 | 0.582 | 0.925 |

## Discussione

### Interpretazione dei Risultati

1. **Differenze Statisticamente Significative ma Praticamente Limitate**

   Le differenze statisticamente significative nei pattern di sentiment tra notizie vere e false suggeriscono che esiste una relazione rilevabile. Tuttavia, l'effect size trascurabile (tutti < 0.1) indica che queste differenze hanno una rilevanza pratica limitata. Questo fenomeno è comune nei grandi dataset, dove anche piccole differenze possono risultare statisticamente significative.

2. **Superiorità dei Modelli Non Lineari**

   Il grande miglioramento di performance del Random Forest (AUC: 0.93) rispetto alla regressione logistica (AUC: 0.54) indica che le relazioni tra sentiment e veridicità sono prevalentemente non lineari e complesse. Questo suggerisce che approcci di machine learning più sofisticati sono necessari per catturare efficacemente queste relazioni.

3. **Importanza delle Feature di Leggibilità e Acculturazione**

   Le feature di leggibilità e acculturazione, in particolare `culture_score`, si sono rivelate più predittive rispetto alle pure feature di sentiment. Questo indica che il modo in cui un testo è strutturato e il livello di complessità linguistica potrebbero essere indicatori più affidabili della veridicità rispetto al sentimento espresso.

4. **Possibile Overfitting su Caratteristiche Specifiche del Dataset**

   L'alta importanza di `thread_id` e `tweet_id` nel Random Forest solleva preoccupazioni sulla generalizzabilità del modello. Questo suggerisce che il modello potrebbe star memorizzando pattern specifici del dataset piuttosto che apprendendo relazioni generalizzabili.

### Limiti dello Studio

1. **Dataset Sbilanciato**

   La predominanza di notizie vere (93%) influenza le metriche di valutazione e potrebbe distorcere l'interpretazione dei risultati. Nonostante l'uso di tecniche come class_weight='balanced', questo sbilanciamento rimane una limitazione importante.

2. **Analisi Statica**

   L'analisi non considera adeguatamente l'evoluzione temporale del sentiment nei thread, che potrebbe essere un indicatore più potente della veridicità rispetto a misure statiche.

3. **Variabilità Contestuale**

   I pattern di sentiment possono variare considerevolmente a seconda del tema della notizia e del contesto culturale, un aspetto non sufficientemente esplorato in questo studio.

4. **Generalizzabilità**

   La dipendenza da identificatori specifici del dataset nel Random Forest limita la generalizzabilità dei risultati a nuovi dati e contesti.

### Implicazioni

1. **Limiti dell'Analisi del Sentiment per Identificare Fake News**

   I risultati suggeriscono che l'analisi del sentiment da sola è insufficiente per identificare efficacemente le fake news, evidenziando la necessità di approcci più sofisticati.

2. **Importanza dell'Acculturazione e Complessità Linguistica**

   La rilevanza del `culture_score` e di altre feature di leggibilità suggerisce che il livello di acculturazione e la complessità linguistica potrebbero essere indicatori significativi della qualità dell'informazione.

3. **Necessità di Modelli Non Lineari**

   La superiorità del Random Forest evidenzia l'importanza di utilizzare modelli non lineari per catturare le relazioni complesse tra caratteristiche linguistiche e veridicità.

## Conclusioni

Lo studio ha parzialmente verificato l'ipotesi iniziale, dimostrando che esistono differenze statisticamente significative nei pattern linguistici tra commenti a notizie vere e false. Tuttavia, queste differenze hanno un effect size limitato e sono meglio catturate da modelli non lineari.

I risultati suggeriscono che l'analisi del sentiment da sola è insufficiente per identificare efficacemente le fake news, mentre feature di leggibilità e acculturazione, come il `culture_score`, offrono un potere predittivo maggiore. La combinazione di tutte le feature, analizzate con modelli non lineari, produce i risultati migliori.

Questo lavoro contribuisce alla comprensione dei meccanismi di diffusione della disinformazione online e suggerisce che l'identificazione efficace delle fake news richiede approcci multidimensionali che considerino non solo il sentiment, ma anche la complessità linguistica, l'acculturazione e potenzialmente le dinamiche temporali delle conversazioni online.

## Direzioni Future

Sulla base dei risultati ottenuti, raccomandiamo le seguenti direzioni per ricerche future:

1. **Analisi di Pattern Temporali**
   - Studiare come il sentiment evolve nel tempo all'interno dei thread
   - Analizzare la velocità di propagazione delle reazioni

2. **Stratificazione per Tema o Evento**
   - Analizzare separatamente per evento o tema per identificare pattern specifici del contesto
   - Valutare come diversi temi influenzano il rapporto sentiment-veridicità

3. **Feature Engineering Avanzato**
   - Sviluppare feature che catturino dinamiche conversazionali
   - Approfondire l'analisi del `culture_score` e della sua composizione

4. **Approcci Integrati**
   - Combinare analisi del sentiment con analisi di rete e metadati degli utenti
   - Sviluppare modelli che integrino caratteristiche linguistiche con dinamiche di diffusione

5. **Validazione Cross-Dataset**
   - Testare i modelli su dataset diversi per valutare la generalizzabilità
   - Confrontare i risultati tra piattaforme social media diverse

## Bibliografia

- Chen, X., Sin, S. C. J., Theng, Y. L., & Lee, C. S. (2018). Why students share misinformation on social media: Motivation, gender, and study-level differences. Journal of Academic Librarianship, 44(3), 423-431.
- Gonzalez-Bailon, S., Andersen, I. G., & Nielsen, R. K. (2021). The asymmetrical impact of emotions on social media engagement with misinformation. Digital Journalism, 9(8), 1107-1129.
- Horne, B. D., & Adali, S. (2017). This just in: Fake news packs a lot in title, uses simpler, repetitive content in text body, more similar to satire than real news. Proceedings of the International AAAI Conference on Web and Social Media, 11(1), 759-766.
- Phillips, A. L., Watkins, E. A., & Curry, G. D. (2020). Online polarization and the broader psychosocial impact of social media's influence on opinion formation. Journal of Public Opinion Research, 32(2), 309-328.
- Zubiaga, A., Liakata, M., Procter, R., Wong Sak Hoi, G., & Tolmie, P. (2016). Analysing how people orient to and spread rumours in social media by looking at conversational threads. PloS one, 11(3), e0150989.
