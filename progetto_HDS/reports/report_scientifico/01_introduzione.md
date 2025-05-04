# 1. Introduzione

## Contesto e Motivazione

Nell'era della comunicazione digitale, la disinformazione rappresenta una delle sfide più significative per la società contemporanea. La rapidità e l'ampiezza con cui le informazioni false possono diffondersi attraverso le piattaforme social hanno implicazioni profonde sul dibattito pubblico, sui processi democratici e sulla coesione sociale. Comprendere i meccanismi che facilitano o ostacolano la propagazione della disinformazione è dunque diventato un obiettivo di ricerca prioritario in diverse discipline, dalle scienze dell'informazione alla psicologia sociale, dalle scienze politiche all'informatica.

Il presente studio si inserisce in questo filone di ricerca, adottando un approccio innovativo che sposta il focus dalle caratteristiche intrinseche delle fake news ai pattern di risposta che queste generano negli utenti. L'ipotesi di partenza è che le notizie false possano suscitare reazioni linguistiche e affettive diverse rispetto alle notizie vere, e che queste differenze possano essere rilevate attraverso l'analisi computazionale del linguaggio.

## Il Problema della Disinformazione Online

La disinformazione online si è evoluta significativamente negli ultimi anni, diventando sempre più sofisticata nei contenuti e nelle strategie di diffusione. Come evidenziato da Vosoughi et al. (2018) in uno studio pubblicato su Science, le notizie false si diffondono più velocemente, più ampiamente e più in profondità rispetto alle notizie vere, grazie a meccanismi di engagement che sfruttano bias cognitivi e trigger emotivi.

Tradizionalmente, l'identificazione delle fake news si è basata su:

1. **Approcci basati sul contenuto**: analisi del testo della notizia, del suo stile linguistico, della sua struttura
2. **Approcci basati sulla fonte**: valutazione dell'affidabilità della fonte, storia di pubblicazioni precedenti
3. **Approcci basati sulla diffusione**: pattern di propagazione attraverso le reti sociali

Tuttavia, questi approcci presentano limitazioni significative. I metodi basati sul contenuto possono essere aggirabili attraverso tecniche di scrittura sempre più sofisticate; quelli basati sulla fonte possono fallire di fronte a nuovi siti o account creati ad hoc; quelli basati sulla diffusione richiedono dati longitudinali non sempre disponibili.

### Gap nella Letteratura

Una dimensione meno esplorata, ma potenzialmente rivelatrice, riguarda i pattern di risposta che le notizie false generano negli utenti dei social media. Mentre diversi studi hanno analizzato come le fake news si diffondono attraverso le reti sociali (Vosoughi et al., 2018), meno attenzione è stata dedicata alle caratteristiche linguistiche delle reazioni degli utenti.

Gonzalez-Bailon et al. (2021) hanno suggerito che le reazioni emotive alle fake news tendono ad essere più intense e polarizzate, ma queste osservazioni non sono state sistematicamente testate su dataset diversificati. Inoltre, pochi studi hanno analizzato come la complessità linguistica e il livello di acculturazione nei commenti possano correlarsi con la veridicità dell'informazione originale.

Zubiaga et al. (2016), nello studio che ha portato alla creazione del dataset PHEME, hanno analizzato le conversazioni su Twitter relative a diversi eventi di attualità, focalizzandosi però principalmente sui pattern di diffusione piuttosto che sulle caratteristiche linguistiche delle reazioni. Il nostro studio intende colmare questa lacuna, esplorando sistematicamente come il sentiment, la stance e le caratteristiche di leggibilità dei commenti possano differire tra notizie vere e false.

## Obiettivi dello Studio

Il presente studio si propone di:

1. **Verificare l'esistenza di differenze significative** nei pattern di sentiment tra i commenti alle notizie vere e quelli alle notizie false
2. **Quantificare la forza delle associazioni** tra caratteristiche linguistiche dei commenti e veridicità delle notizie
3. **Confrontare modelli predittivi lineari e non lineari** per determinare la natura delle relazioni tra feature linguistiche e veridicità
4. **Identificare le feature linguistiche più rilevanti** per distinguere tra reazioni a notizie vere e false
5. **Valutare il potenziale predittivo** di diversi set di feature, con particolare attenzione al confronto tra feature di sentiment e di leggibilità/acculturazione

Questi obiettivi non sono solo di interesse teorico, ma hanno anche importanti implicazioni pratiche. Comprendere come gli utenti reagiscono linguisticamente a notizie vere e false potrebbe contribuire a:

- Sviluppare sistemi più accurati per l'identificazione automatica delle fake news
- Creare strumenti di fact-checking che considerino non solo il contenuto originale ma anche le reazioni che genera
- Progettare interventi educativi mirati per aumentare la resilienza degli utenti alla disinformazione
- Approfondire la conoscenza dei meccanismi cognitivi e sociali coinvolti nella diffusione della disinformazione

## Ipotesi di Ricerca


Il nostro studio è guidato da cinque ipotesi principali:

1. **Ipotesi 1**: Esistono differenze statisticamente significative nel sentiment (polarità e soggettività) dei commenti alle notizie vere rispetto a quelli alle notizie false.
   - **H0**: Non c'è differenza significativa nel sentiment tra i due gruppi
   - **H1**: Il sentiment differisce significativamente tra i due gruppi

2. **Ipotesi 2**: Esistono differenze statisticamente significative nella stance (atteggiamento) dei commenti rispetto alla notizia originale tra thread di notizie vere e false.
   - **H0**: Non c'è differenza significativa nella stance tra i due gruppi
   - **H1**: La stance differisce significativamente tra i due gruppi

3. **Ipotesi 3**: Esistono differenze statisticamente significative nelle misure di leggibilità e acculturazione (in particolare nel `culture_score`) dei commenti tra notizie vere e false.
   - **H0**: Non c'è differenza significativa nelle misure di leggibilità tra i due gruppi
   - **H1**: Le misure di leggibilità differiscono significativamente tra i due gruppi

4. **Ipotesi 4**: Le feature di leggibilità e acculturazione hanno un maggior potere predittivo sulla veridicità rispetto alle pure feature di sentiment.
   - **H0**: Le feature di leggibilità non sono più predittive delle feature di sentiment
   - **H1**: Le feature di leggibilità sono più predittive delle feature di sentiment

5. **Ipotesi 5**: Modelli non lineari catturano relazioni significativamente più forti tra feature linguistiche e veridicità rispetto ai modelli lineari.
   - **H0**: Non c'è differenza significativa nella performance tra modelli lineari e non lineari
   - **H1**: I modelli non lineari hanno una performance significativamente migliore

Queste ipotesi sono state formulate sulla base di studi precedenti che suggeriscono che le reazioni alle fake news possono essere caratterizzate da maggiore emotività, minor complessità linguistica e minor riflessione critica (Pennycook & Rand, 2019). Tuttavia, queste osservazioni non sono state sistematicamente testate in contesti di social media e conversazioni online.

Nel capitolo successivo, descriveremo in dettaglio il dataset utilizzato e la metodologia adottata per testare queste ipotesi.
