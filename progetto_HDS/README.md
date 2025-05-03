# Studio sulla Relazione tra Sentiment nei Commenti e Veridicità delle Notizie

![Copertina del progetto](./results/narrative/01_title_page.png)

## Panoramica del Progetto

Questo progetto analizza la relazione tra i pattern linguistici presenti nei commenti sui social media e la veridicità delle notizie originali, utilizzando il dataset PHEME di conversazioni Twitter. L'obiettivo è determinare se esistano differenze sistematiche nelle reazioni linguistiche e affettive alle notizie vere rispetto a quelle false, e se queste differenze possano essere sfruttate per identificare la disinformazione.

## Principali Risultati

I risultati principali dello studio includono:

1. **Differenze statisticamente significative ma con effect size limitato** nei pattern linguistici tra commenti a notizie vere e false.

2. **Netta superiorità dei modelli non lineari** (Random Forest: AUC 0.932) rispetto ai modelli lineari (Regressione logistica: AUC 0.542), suggerendo che le relazioni tra caratteristiche linguistiche e veridicità sono prevalentemente non lineari e complesse.

3. **Maggiore importanza delle feature di leggibilità e acculturazione** rispetto alle pure feature di sentiment, con il `culture_score` che emerge come la feature linguistica più rilevante.

4. **Necessità di approcci multidimensionali** che integrino diverse caratteristiche linguistiche per catturare efficacemente i pattern di risposta associati alla disinformazione.

## Struttura del Repository

- **[reports/](./reports/)**: Report scientifico completo e altri documenti esplicativi
  - **[report_scientifico/](./reports/report_scientifico/)**: Documentazione dettagliata dello studio
  - **[indice_generale.md](./reports/indice_generale.md)**: Indice di navigazione per tutti i documenti del progetto

- **[src/](./src/)**: Codice sorgente del progetto
  - **[analysis/](./src/analysis/)**: Script per l'analisi statistica e i modelli predittivi
  - **[preprocessing/](./src/preprocessing/)**: Script per il preprocessing dei dati
  - **[visualization/](./src/visualization/)**: Script per la generazione di grafici e visualizzazioni

- **[results/](./results/)**: Risultati delle analisi e figure generate
  - **[figures/](./results/figures/)**: Grafici e visualizzazioni principali
  - **[narrative/](./results/narrative/)**: Figure narrative per presentazioni
  - **[tables/](./results/tables/)**: Tabelle con risultati numerici dettagliati

- **[data/](./data/)**: Dataset PHEME e dati derivati

- **[notebooks/](./notebooks/)**: Jupyter notebook per analisi interattive

- **[tests/](./tests/)**: Test automatizzati per verificare la correttezza del codice

## Documentazione

- **[todolist_progetto.md](./todolist_progetto.md)**: Elenco completo e stato delle attività del progetto
- **[descrizione_progetto.md](./descrizione_progetto.md)**: Panoramica generale e obiettivi
- **[metodologia_dettagliata.md](./metodologia_dettagliata.md)**: Approccio metodologico completo
- **[implementazione_tecnica.md](./implementazione_tecnica.md)**: Dettagli implementativi del progetto
- **[feature_description.md](./feature_description.md)**: Descrizione dettagliata delle feature utilizzate
- **[requirements.txt](./requirements.txt)**: Dipendenze software del progetto

## Come Navigare il Progetto

Per una panoramica completa del progetto e dei risultati ottenuti, si consiglia di:

1. Consultare l'[indice generale](./reports/indice_generale.md) che contiene collegamenti a tutti i documenti principali.

2. Leggere il [sommario esecutivo](./reports/report_scientifico/sommario_esecutivo.md) per una sintesi concisa ma completa dei principali risultati.

3. Approfondire con il report scientifico completo, partendo dall'[introduzione](./reports/report_scientifico/01_introduzione.md) e proseguendo con le sezioni successive.

## Come Generare il Report Completo in PDF

È disponibile uno script per generare un PDF completo del report scientifico:

```bash
cd reports
./genera_report_pdf.sh
```

Il file PDF risultante sarà disponibile nella directory `reports` con il nome `report_completo.pdf`.

## Requisiti Software

Per eseguire il codice di questo progetto sono necessari:

- Python 3.10 o superiore
- Le librerie elencate in [requirements.txt](./requirements.txt)

È possibile installare tutte le dipendenze con:

```bash
pip install -r requirements.txt
```

## Note sulla Riproducibilità

Per garantire la riproducibilità dell'analisi:

- Tutti i processi randomizzati utilizzano `random_state=42`
- I parametri dei modelli sono documentati nel codice
- Le procedure di preprocessing sono dettagliatamente documentate

## Autori

- Simone - Corso di Digital Humanities - Maggio 2025
