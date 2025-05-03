# 1. Introduzione e Background

## 1.1 Contesto della Ricerca

La disinformazione online rappresenta una delle sfide più significative per la società contemporanea, con implicazioni profonde per il discorso pubblico, i processi democratici e la coesione sociale. Nell'era dei social media, la velocità e la scala con cui le informazioni false possono diffondersi hanno raggiunto livelli senza precedenti, rendendo cruciale lo sviluppo di strumenti e metodologie efficaci per l'identificazione e il contrasto della disinformazione.

Mentre numerosi studi si sono concentrati sulle caratteristiche intrinseche delle fake news o sui pattern di diffusione, relativamente poca attenzione è stata dedicata all'analisi sistematica delle reazioni che queste generano negli utenti. In particolare, il ruolo del sentiment espresso nei commenti come potenziale indicatore della veridicità delle notizie rimane un'area di indagine ancora largamente inesplorata.

### Rilevanza del Problema

La diffusione di fake news ha dimostrato di avere impatti significativi su:

1. **Processi Democratici**
   - Influenza sulle elezioni
   - Polarizzazione del dibattito pubblico
   - Erosione della fiducia nelle istituzioni

2. **Salute Pubblica**
   - Disinformazione durante la pandemia COVID-19
   - Scetticismo verso vaccini e trattamenti medici
   - Diffusione di cure non scientifiche

3. **Coesione Sociale**
   - Amplificazione di divisioni sociali
   - Creazione di "camere d'eco" informative
   - Radicalizzazione di posizioni estreme

## 1.2 Obiettivi dello Studio

Questo studio si propone di esplorare la relazione tra i pattern di sentiment espressi nei commenti e la veridicità delle notizie originali, utilizzando il dataset PHEME, una collezione di thread di conversazione Twitter relativi a diversi eventi di attualità.

### Obiettivi Specifici

1. **Identificazione di Pattern Linguistici**
   - Analizzare le caratteristiche linguistiche dei commenti
   - Identificare pattern distintivi tra reazioni a notizie vere e false
   - Quantificare la forza di queste differenze

2. **Valutazione del Potere Predittivo**
   - Sviluppare modelli predittivi basati su feature linguistiche
   - Confrontare l'efficacia di diversi approcci analitici
   - Valutare il contributo specifico di diverse categorie di feature

3. **Comprensione dei Meccanismi**
   - Esplorare i meccanismi attraverso cui il sentiment si manifesta
   - Analizzare il ruolo della complessità linguistica
   - Studiare l'interazione tra sentiment e altre caratteristiche testuali

## 1.3 Stato dell'Arte

### Ricerca sulla Disinformazione Online

La letteratura esistente sulla disinformazione online si è concentrata su diverse aree:

1. **Caratteristiche delle Fake News**
   - Analisi stilistiche del testo
   - Pattern di diffusione
   - Caratteristiche degli autori

2. **Meccanismi di Propagazione**
   - Reti sociali e viralità
   - Ruolo dei bot e account automatizzati
   - Dinamiche di condivisione

3. **Strategie di Contrasto**
   - Fact-checking automatico
   - Educazione ai media
   - Interventi delle piattaforme

### Gap nella Letteratura

Nonostante l'ampia ricerca esistente, diverse aree rimangono poco esplorate:

1. **Analisi Sistematica delle Reazioni**
   - Pochi studi sui pattern di risposta
   - Limitata attenzione al sentiment dei commenti
   - Mancanza di analisi quantitative su larga scala

2. **Integrazione di Multiple Dimensioni**
   - Analisi separate di sentiment e veridicità
   - Limitata considerazione della complessità linguistica
   - Scarsa attenzione alle dinamiche conversazionali

3. **Validazione Cross-contestuale**
   - Studi limitati a singoli eventi o piattaforme
   - Mancanza di validazione su dataset diversi
   - Limitata generalizzabilità dei risultati

## 1.4 Ipotesi di Ricerca

### Ipotesi Principale

**H1**: Esistono differenze statisticamente significative nei pattern di sentiment tra i commenti alle notizie vere e quelli alle notizie false.

### Ipotesi Secondarie

1. **Sulla Soggettività**
   - H0: Non c'è differenza significativa nel livello di soggettività tra commenti a notizie vere e false
   - H1: I commenti alle notizie false mostrano livelli di soggettività significativamente diversi

2. **Sulla Polarità**
   - H0: Non c'è differenza significativa nella polarità del sentiment tra commenti a notizie vere e false
   - H1: I commenti alle notizie false mostrano polarità significativamente diversa

3. **Sull'Atteggiamento (Stance)**
   - H0: Non c'è differenza significativa nell'atteggiamento tra commenti a notizie vere e false
   - H1: I commenti alle notizie false mostrano atteggiamenti significativamente diversi

4. **Sulla Leggibilità e Acculturazione**
   - H0: Non c'è differenza significativa nelle metriche di leggibilità tra commenti a notizie vere e false
   - H1: I commenti alle notizie false mostrano livelli di leggibilità significativamente diversi

5. **Sul Potere Predittivo**
   - H0: Le feature di sentiment non hanno potere predittivo significativo sulla veridicità
   - H1: Le feature di sentiment hanno potere predittivo significativo sulla veridicità

## 1.5 Framework Teorico

### Teorie di Riferimento

1. **Teoria della Dissonanza Cognitiva**
   - Ruolo delle reazioni emotive nella gestione dell'informazione contrastante
   - Meccanismi di riduzione della dissonanza attraverso il commento

2. **Elaborazione dell'Informazione Sociale**
   - Processi di validazione sociale dell'informazione
   - Influenza del gruppo sulle reazioni individuali

3. **Teoria della Polarizzazione dell'Opinione**
   - Dinamiche di estremizzazione delle posizioni
   - Ruolo delle camere d'eco nella formazione del sentiment

### Modello Concettuale

![Modello Concettuale](../figures/conceptual_model.png)

*Il modello concettuale illustra le relazioni ipotizzate tra le variabili chiave dello studio.*

## 1.6 Contributo Atteso

Questo studio mira a contribuire alla letteratura esistente in diversi modi:

1. **Contributo Teorico**
   - Sviluppo di un framework integrato per l'analisi delle reazioni alla disinformazione
   - Comprensione più profonda dei meccanismi di risposta alle fake news
   - Identificazione di pattern linguistici diagnostici

2. **Contributo Metodologico**
   - Sviluppo di metriche avanzate per l'analisi del sentiment
   - Integrazione di approcci statistici e di machine learning
   - Validazione di metodologie su un dataset significativo

3. **Contributo Pratico**
   - Implicazioni per lo sviluppo di strumenti di fact-checking
   - Linee guida per l'educazione ai media
   - Raccomandazioni per le piattaforme social

## 1.7 Struttura del Report

Questo report è organizzato come segue:

- La **Sezione 2** descrive in dettaglio la metodologia utilizzata
- La **Sezione 3** presenta l'analisi esplorativa dei dati
- La **Sezione 4** riporta i risultati dell'analisi statistica
- La **Sezione 5** descrive i modelli predittivi sviluppati
- La **Sezione 6** discute i risultati e le loro implicazioni
- La **Sezione 7** esamina la validità e le limitazioni dello studio
- La **Sezione 8** conclude con sintesi e direzioni future

---

*Continua nella prossima sezione: [2. Metodologia](02_methodology.md)*
