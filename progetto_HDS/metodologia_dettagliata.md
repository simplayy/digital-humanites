# Metodologia di Ricerca

## 1. Design dello Studio

### 1.1 Tipo di Studio
Questo progetto adotta un design di **studio osservazionale analitico** con approccio quantitativo. La scelta di questo design è appropriata perché:
- Non interveniamo sulla generazione dei dati ma osserviamo fenomeni già esistenti (commenti e notizie)
- Cerchiamo relazioni statistiche tra variabili (sentiment e veridicità)
- Applichiamo test di ipotesi formali per valutare la significatività delle relazioni

### 1.2 Framework Metodologico

Il framework metodologico si basa sui principi del **metodo scientifico**, seguendo questi passaggi:

1. **Formulazione del problema**: Identificazione della domanda di ricerca sulla relazione tra sentiment e veridicità.
2. **Definizione delle ipotesi**: 
   - **Ipotesi nulla (H₀)**: Non esiste una differenza statisticamente significativa nei pattern di sentiment dei commenti tra notizie vere e fake.
   - **Ipotesi alternativa (H₁)**: Esiste una differenza statisticamente significativa nei pattern di sentiment dei commenti tra notizie vere e fake.
3. **Raccolta dati**: Acquisizione del dataset Fakeddit con commenti annotati.
4. **Analisi dei dati**: Applicazione di test statistici e analisi multivariata.
5. **Valutazione delle ipotesi**: Determinazione se rifiutare o meno l'ipotesi nulla.
6. **Interpretazione dei risultati**: Discussione delle implicazioni delle analisi statistiche.

## 2. Dataset e Campionamento

### 2.1 Scelta del Dataset
Il dataset principale sarà **Fakeddit**, selezionato per le seguenti caratteristiche:
- Focus specifico sui commenti degli utenti
- Classificazione delle notizie come vere o false
- Disponibilità di metadata sui commenti (engagement, timestamp, etc.)
- Volume di dati sufficiente per analisi statistiche robuste

### 2.2 Strategie di Campionamento
Per garantire la validità statistica dello studio, adotteremo le seguenti strategie:

1. **Campionamento stratificato**: Per assicurare una rappresentazione bilanciata di notizie vere e false
2. **Dimensionamento del campione**: Calcolo a priori della dimensione campionaria necessaria per ottenere potenza statistica adeguata (β = 0.8) con livello di significatività α = 0.05
3. **Gestione dei bias**: Identificazione e controllo di potenziali fonti di bias nel dataset

### 2.3 Validità e Affidabilità
Per massimizzare la validità e l'affidabilità dei risultati:
- **Validità interna**: Controllo di variabili confondenti (es. lunghezza dei commenti, popolarità delle notizie)
- **Validità esterna**: Discussione della generalizzabilità dei risultati
- **Validità di costrutto**: Giustificazione delle metriche di sentiment come misure valide del fenomeno
- **Affidabilità**: Test di consistenza interna delle misure

## 3. Estrazione delle Feature

### 3.1 Feature di Sentiment
Estrarremo sistematicamente le seguenti feature dai commenti:

1. **Sentiment di base**:
   - Positività/negatività complessiva
   - Neutralità
   - Polarità (intensità del sentiment)

2. **Analisi delle emozioni**:
   - Distribuzione delle emozioni fondamentali (rabbia, paura, gioia, tristezza, disgusto, sorpresa)
   - Intensità di ciascuna emozione
   - Variabilità emotiva nei commenti

3. **Rilevamento del sarcasmo**:
   - Probabilità di presenza di sarcasmo
   - Intensità del sarcasmo

4. **Stance detection**:
   - Posizione rispetto alla notizia (favorevole, contrario, neutrale)
   - Grado di certezza nella stance

5. **Metriche di leggibilità**:
   - Indici di Flesch Reading Ease
   - Flesch-Kincaid Grade Level
   - SMOG Index
   - Gunning Fog

6. **Diversità lessicale**:
   - Type-Token Ratio
   - Ricchezza del vocabolario

### 3.2 Metadati dei Commenti
Oltre alle feature di sentiment, analizzeremo:

1. **Volume di interazione**:
   - Numero di commenti per notizia
   - Velocità di accumulo dei commenti

2. **Engagement**:
   - Media dei like per commento
   - Numero di risposte per commento
   - Profondità delle thread di discussione

3. **Pattern temporali**:
   - Distribuzione temporale dei commenti
   - Picchi di attività commentaria

4. **Statistiche utente**:
   - Numero di utenti unici
   - Diversità degli autori

## 4. Analisi Statistica

### 4.1 Analisi Descrittiva
Prima fase di analisi per comprendere la distribuzione dei dati:
- Statistiche descrittive (media, mediana, deviazione standard, quartili)
- Visualizzazione delle distribuzioni
- Identificazione di outlier e pattern preliminari

### 4.2 Test di Ipotesi

Per ciascuna feature estratta, applicheremo specifici test statistici:

1. **Test per confronto di medie**:
   - **Test t di Student** per confrontare le medie di sentiment tra notizie vere e false
   - **ANOVA** per confrontare più gruppi (es. diverse categorie di notizie)

2. **Test per variabili categoriche**:
   - **Test chi-quadro** per relazioni tra variabili categoriche
   - **Test esatto di Fisher** per tabelle di contingenza con conteggi bassi

3. **Test non parametrici** (quando appropriato):
   - **Test di Mann-Whitney U** per confrontare distribuzioni non normali
   - **Test di Kruskal-Wallis** come alternativa non parametrica all'ANOVA

### 4.3 Analisi di Correlazione
Esplorazione delle relazioni tra feature di sentiment:
- **Coefficiente di correlazione di Pearson** per relazioni lineari
- **Coefficiente di correlazione di Spearman** per relazioni monotoniche non necessariamente lineari
- **Matrice di correlazione** per visualizzare e quantificare le relazioni tra tutte le feature

### 4.4 Regressione e Modellazione
Per quantificare le relazioni identificate:
- **Regressione logistica** per modellare la probabilità che una notizia sia fake basandosi sulle feature di sentiment
- **Analisi della varianza spiegata** (R²)
- **Calcolo degli odds ratio** e intervalli di confidenza

### 4.5 Correzione per Test Multipli
Per controllare l'errore di tipo I dovuto a test multipli:
- **Correzione di Bonferroni** per controllo conservativo
- **Procedura di Benjamini-Hochberg** per controllare il False Discovery Rate (FDR)
- **Correzione di Holm-Bonferroni** come approccio intermedio

### 4.6 Interpretazione della Significatività
Per ogni test, riporteremo:
- **P-value** calcolato
- **Soglia di significatività** (α=0.05 come standard, con discussione di altre soglie)
- **Effect size** per quantificare l'importanza pratica oltre alla significatività statistica
- **Intervalli di confidenza** al 95% per le stime

## 5. Validazione e Robustezza

### 5.1 Validazione Incrociata
Per garantire la robustezza dei risultati:
- **Cross-validation** (k-fold) per verificare la stabilità delle relazioni identificate
- **Split del dataset** in training e test set per validazione

### 5.2 Analisi di Sensibilità
Per valutare quanto siano robusti i risultati:
- **Variazione dei parametri** dei test statistici
- **Rimozione di outlier** e ripetizione delle analisi
- **Subsampling** per verificare la consistenza su sottoinsiemi del dataset

### 5.3 Gestione di Confondenti
Per controllare variabili che potrebbero confondere la relazione studiata:
- **Analisi stratificata** per controllare l'effetto di variabili confondenti note
- **Propensity score matching** per bilanciare gruppi
- **Analisi multivariata** per controllare simultaneamente più confondenti

## 6. Etica e Limitazioni

### 6.1 Considerazioni Etiche
- **Privacy dei dati**: Utilizzo di dati in forma aggregata e anonima
- **Bias nei dataset**: Discussione e mitigazione di potenziali bias
- **Implicazioni sociali**: Riflessione sulle implicazioni dell'utilizzo di questi metodi

### 6.2 Limitazioni Metodologiche
Riconoscimento esplicito delle limitazioni:
- **Generalizzabilità**: Limiti nella generalizzazione dei risultati a piattaforme diverse da Reddit
- **Causalità**: Chiarire che le correlazioni identificate non implicano necessariamente causalità
- **Temporalità**: Discussione dei possibili cambiamenti nei pattern nel tempo

## 7. Documentazione e Riproducibilità

### 7.1 Documentazione del Processo
Per garantire trasparenza e riproducibilità:
- **Protocollo dettagliato** di tutte le procedure
- **Versioning dei codici** e degli script di analisi
- **Pre-registrazione** dello studio e delle ipotesi (quando possibile)

### 7.2 Open Science
Promozione della riproducibilità:
- **Pubblicazione di codice** e pipeline di analisi
- **Condivisione dei dati** (nel rispetto delle limitazioni legali)
- **Reporting completo** di tutti i test eseguiti, compresi quelli con risultati non significativi

## 8. Output e Deliverable Metodologici

### 8.1 Report Metodologico
Documento che dettagli:
- Tutti i test statistici eseguiti
- Giustificazione per la scelta di ciascun test
- Risultati completi con p-value ed effect size
- Interpretazione critica dei risultati

### 8.2 Protocollo di Analisi
Documento formale che specifichi:
- Procedure di estrazione delle feature
- Pipeline di analisi statistica
- Criteri di decisione per l'accettazione o il rifiuto delle ipotesi
- Procedure di validazione

### 8.3 Visualizzazione dei Risultati
Set di visualizzazioni che illustrino chiaramente:
- Distribuzioni del sentiment nei diversi gruppi
- Correlazioni significative
- Effect size visualizzati
- Confidence intervals per le stime chiave
