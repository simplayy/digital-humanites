# 7. Limitazioni e Validazione

Ogni studio scientifico ha limitazioni intrinseche che devono essere riconosciute per una corretta interpretazione dei risultati. In questo capitolo, discutiamo le principali limitazioni metodologiche del nostro studio, le misure adottate per validare i risultati e le considerazioni sulla loro generalizzabilità.

## Limitazioni Metodologiche

### 1. Dataset Sbilanciato

La predominanza di notizie vere (93%) rispetto a quelle false (7%) nel dataset PHEME rappresenta una limitazione significativa:

- **Influenza sulle metriche di valutazione**: Metriche come accuracy e precision possono essere ingannevoli in dataset fortemente sbilanciati, poiché un modello che predice sempre la classe maggioritaria otterrebbe comunque un'accuracy del 93%

- **Sfida per l'apprendimento dei modelli**: Lo sbilanciamento può portare i modelli a favorire la classe maggioritaria, rendendo difficile l'apprendimento di pattern specifici della classe minoritaria

- **Limitazione nella dimensione del campione**: La quantità relativamente ridotta di notizie false (452 thread) limita la potenza statistica per identificare pattern specifici di questa classe

**Misure adottate per mitigare**:
- Utilizzo di `class_weight='balanced'` nei modelli per compensare lo sbilanciamento
- Focus su metriche come ROC AUC e F1 Score, meno sensibili allo sbilanciamento rispetto ad accuracy
- Stratificazione nei split di training/test per mantenere la proporzione originale

Nonostante queste misure, lo sbilanciamento rimane una limitazione intrinseca che potrebbe influenzare la stabilità e la generalizzabilità dei risultati.

### 2. Rischio di Overfitting

L'alta importanza di feature come `thread_id` e `tweet_id` nel Random Forest solleva preoccupazioni significative:

- **Memorizzazione vs. generalizzazione**: Il modello potrebbe star memorizzando pattern specifici dei thread anziché apprendere relazioni generalizzabili

- **Performance potenzialmente sovrastimata**: La performance elevata potrebbe essere parzialmente basata su caratteristiche non trasferibili a nuovi dati

- **Limitata utility pratica**: Un modello che dipende fortemente dagli identificatori avrebbe utilità limitata in scenari reali dove i thread sono nuovi e non presenti nel dataset di training

**Test condotti per valutare**:
- Addestramento di modelli senza gli identificatori, che ha mostrato un calo dell'AUC del Random Forest da 0.93 a 0.68
- Questo calo significativo conferma che parte della performance elevata deriva effettivamente dall'overfitting sugli identificatori

Anche escludendo gli identificatori, il modello Random Forest mantiene comunque una performance superiore alla regressione logistica, suggerendo che il valore predittivo delle feature linguistiche è reale, seppur più limitato di quanto suggerito dal modello completo.

### 3. Analisi Statica

Lo studio adotta un approccio prevalentemente statico che non considera adeguatamente:

- **Evoluzione temporale del sentiment**: Come le reazioni cambiano nel tempo all'interno di un thread

- **Dinamica di propagazione**: Come le reazioni si diffondono attraverso la rete sociale

- **Interazioni tra utenti**: Come gli utenti influenzano reciprocamente le loro reazioni

Sebbene abbiamo considerato la posizione dei commenti nei thread attraverso la feature `reaction_index`, un'analisi dinamica più approfondita avrebbe potuto rivelare pattern temporali potenzialmente più informativi.

### 4. Limitata Diversità Contestuale

Il dataset PHEME, sebbene diversificato, presenta limitazioni in termini di:

- **Eventi coperti**: Principalmente eventi di attualità specifici (Charlie Hebdo, Ferguson, ecc.)

- **Periodo temporale**: Limitato al periodo di raccolta dei dati (2014-2015)

- **Contesto linguistico-culturale**: Predominanza dell'inglese e di contesti occidentali

- **Piattaforma unica**: Esclusivamente dati da Twitter, che ha dinamiche specifiche

Queste limitazioni riducono la possibilità di generalizzare i risultati a diversi tipi di contenuti, periodi temporali, contesti culturali o piattaforme diverse.

### 5. Limitazioni dell'Analisi del Sentiment

Le tecniche di analisi del sentiment utilizzate, basate su TextBlob, presentano limitazioni note:

- **Difficoltà con sarcasmo e ironia**: Non sempre capaci di rilevare toni sarcastici o ironici, comuni nei social media

- **Limitata comprensione del contesto**: Analisi basata principalmente su bag-of-words, con limitata comprensione del contesto più ampio

- **Calibrazione su domini generici**: Non specificamente calibrate per il linguaggio dei social media o per discussioni su notizie

Strumenti più avanzati di NLP, come modelli transformer specifici per il sentiment nei social media, avrebbero potuto fornire misurazioni più accurate, ma avrebbero aumentato significativamente la complessità computazionale.

## Procedure di Validazione

Per garantire la robustezza dei risultati nonostante queste limitazioni, abbiamo implementato diverse procedure di validazione:

### 1. Cross-Validation

Tutti i modelli sono stati validati utilizzando 5-fold cross-validation, che:

- Riduce il rischio di overfitting dividendo ripetutamente i dati in training e validation set
- Fornisce una stima più robusta della performance su dati non visti
- Permette di calcolare intervalli di confidenza per le metriche di performance

**Risultati della cross-validation per il Random Forest**:
- Media AUC: 0.923 (std: 0.012)
- Media F1 Score: 0.968 (std: 0.006)

La bassa deviazione standard nelle performance attraverso i fold suggerisce stabilità nel modello, riducendo le preoccupazioni di overfitting casuale.

### 2. Controllo dell'Oversampling

Dato lo sbilanciamento del dataset, abbiamo testato l'oversampling come tecnica alternativa ai pesi bilanciati:

- Implementazione di SMOTE (Synthetic Minority Over-sampling Technique) per equilibrare le classi
- Confronto delle performance con e senza oversampling

**Risultati**:
- Random Forest con SMOTE: AUC 0.917
- Random Forest con class_weight='balanced': AUC 0.932

La relativa stabilità delle performance con diverse strategie di bilanciamento suggerisce che i risultati non sono fortemente dipendenti dalla tecnica specifica utilizzata.

### 3. Test di Robustezza alle Feature

Per valutare la stabilità dei risultati rispetto alla selezione delle feature, abbiamo condotto:

- Test con diversi sottoinsiemi di feature
- Analisi di feature engineering incrementale
- Esperimenti con tecniche di selezione automatica delle feature (forward selection, recursive feature elimination)

**Risultati**:
- Stabilità nell'importanza relativa delle feature principali tra diverse configurazioni
- Consistenza nel ranking delle categorie di feature (leggibilità > sentiment > stance)

Questa coerenza tra diverse configurazioni aumenta la fiducia nella robustezza dei pattern identificati.

### 4. Test di Generalizzabilità per Evento

Per valutare la generalizzabilità tra i diversi eventi nel dataset, abbiamo implementato una validazione leave-one-event-out:

- Addestramento su tutti gli eventi tranne uno
- Test sull'evento escluso
- Ripetizione per tutti gli eventi

**Risultati**:
| Evento Escluso | AUC | Δ AUC |
|----------------|-----|-------|
| Charlie Hebdo | 0.871 | -0.061 |
| Sydney Siege | 0.898 | -0.034 |
| Ferguson | 0.857 | -0.075 |
| Ottawa Shooting | 0.905 | -0.027 |
| Germanwings | 0.892 | -0.040 |

La performance rimane relativamente alta anche quando si testa su eventi completamente nuovi, sebbene con un calo rispetto alla performance complessiva. Questo suggerisce una discreta generalizzabilità tra eventi, ma conferma anche l'influenza del contesto specifico.

### 5. Concordanza tra Diversi Strumenti di Sentiment Analysis

Per valutare la robustezza dell'analisi rispetto allo strumento specifico di sentiment analysis, abbiamo confrontato TextBlob con VADER (Valence Aware Dictionary and sEntiment Reasoner):

- Estrazione di sentiment polarity con entrambi gli strumenti
- Confronto delle distribuzioni e delle correlazioni con la veridicità

**Risultati**:
- Correlazione tra i due strumenti: 0.78
- TextBlob AUC (sentiment_only): 0.559
- VADER AUC (sentiment_only): 0.544

La similarità nei risultati suggerisce che i pattern identificati non sono fortemente dipendenti dallo strumento specifico utilizzato per l'analisi del sentiment.

## Considerazioni sulla Validità

### Validità Interna

La validità interna si riferisce al grado in cui lo studio stabilisce accuratamente una relazione causale o correlazionale.

**Punti di forza**:
- Protocollo metodologico rigoroso con ipotesi pre-specificate
- Tecniche appropriate per il controllo dello sbilanciamento del dataset
- Multiple strategie di validazione incrociata
- Attenzione sia alla significatività statistica che all'effect size
- Test con diversi modelli e set di feature

**Debolezze**:
- Natura prevalentemente correlazionale dello studio
- Potenziali confonder non controllati (es. caratteristiche degli utenti)
- Rischio di overfitting identificato attraverso l'importanza degli ID
- Limitazioni nelle tecniche di analisi del sentiment

La validità interna del nostro studio può essere considerata adeguata ma con importanti limitazioni che richiedono cautela nell'interpretazione dei risultati, particolarmente rispetto a inferenze causali.

### Validità Esterna

La validità esterna si riferisce al grado in cui i risultati possono essere generalizzati oltre lo specifico contesto dello studio.

**Limitazioni alla generalizzabilità**:

1. **Specificità temporale**: Il dataset copre un periodo limitato (2014-2015) e le dinamiche della disinformazione potrebbero essere cambiate significativamente

2. **Specificità della piattaforma**: Esclusivamente dati Twitter, mentre altre piattaforme (Facebook, TikTok, ecc.) potrebbero mostrare dinamiche differenti

3. **Specificità linguistica**: Predominanza dell'inglese limita la generalizzabilità a diversi contesti linguistico-culturali

4. **Specificità tematica**: Gli eventi coperti sono principalmente notizie di attualità, limitando la generalizzabilità ad altri domini (es. salute, scienza, politica)

5. **Evoluzione degli algoritmi**: Gli algoritmi delle piattaforme social sono cambiati notevolmente dal periodo di raccolta dei dati, potenzialmente alterando le dinamiche di diffusione e interazione

Queste limitazioni suggeriscono cautela nel generalizzare i risultati a contesti diversi da quelli specificamente studiati.

## Trasparenza e Riproducibilità

Per garantire la trasparenza e la riproducibilità dello studio, abbiamo implementato diverse pratiche:

1. **Codice open source**: Tutti gli script utilizzati per l'analisi sono disponibili e commentati

2. **Documentazione dettagliata**: Ogni passaggio metodologico è stato documentato, inclusi preprocessing, estrazione delle feature e parametri dei modelli

3. **Controllo delle versioni**: Versioni specifiche delle librerie utilizzate sono documentate nel file `requirements.txt`

4. **Seed fissi**: Tutti i processi randomizzati utilizzano `random_state=42` per garantire la riproducibilità

5. **Logging dettagliato**: Registrazione di tutti i passaggi e parametri dell'analisi

Questi accorgimenti permettono ad altri ricercatori di riprodurre l'analisi e verificare indipendentemente i risultati.

## Bilanciamento tra Sensibilità e Specificità

Un aspetto importante nella valutazione di qualsiasi sistema di rilevamento, inclusi quelli per le fake news, è il bilanciamento tra sensibilità (capacità di identificare correttamente le notizie false) e specificità (capacità di non classificare erroneamente notizie vere come false).

Nel nostro caso, il modello Random Forest ha mostrato:
- Alta precision (0.946): bassa probabilità di falsi positivi
- Alto recall (0.996): alta capacità di identificare correttamente le notizie vere

Questo bilanciamento favorevole è in parte dovuto allo sbilanciamento del dataset e potrebbe non riflettersi in applicazioni reali con distribuzione diversa. In contesti pratici, il trade-off tra sensibilità e specificità dovrebbe essere calibrato in base al costo relativo dei falsi positivi rispetto ai falsi negativi.

## Riflessioni Finali sulla Validità

Nonostante le limitazioni discusse, riteniamo che il nostro studio fornisca insight validi e utili sulla relazione tra pattern linguistici nei commenti e veridicità delle notizie. I risultati sono particolarmente robusti rispetto a:

1. La superiorità dei modelli non lineari rispetto ai modelli lineari
2. Il maggior valore predittivo delle feature di leggibilità rispetto alle pure feature di sentiment
3. L'importanza del `culture_score` come indicatore del livello di acculturazione e complessità linguistica

Questi pattern sono stati confermati attraverso diverse analisi e tecniche di validazione, suggerendo che, sebbene con limiti di generalizzabilità, rappresentano relazioni reali e interpretabili nel contesto studiato.

Nel prossimo capitolo, integreremo tutti i risultati e le considerazioni in un quadro coerente di conclusioni e implicazioni per ricerche future.
