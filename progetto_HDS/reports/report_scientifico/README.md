# README: Report Scientifico sul Sentiment nei Commenti e Veridicità delle Notizie

## Struttura del Report

Questo report scientifico è organizzato in diversi documenti per facilitare la navigazione e la consultazione. Di seguito trovi una guida alla struttura dei documenti e suggerimenti su come navigarli in base ai tuoi interessi.

### Documenti Principali

1. **[00_copertina.md](00_copertina.md)**  
   Pagina di copertina con indice generale e sommario esecutivo del progetto.

2. **[01_introduzione.md](01_introduzione.md)**  
   Contesto, motivazione, obiettivi dello studio e ipotesi di ricerca.

3. **[02_dataset_metodologia.md](02_dataset_metodologia.md)**  
   Descrizione del dataset PHEME e della metodologia utilizzata per l'analisi.

4. **[03_analisi_esplorativa.md](03_analisi_esplorativa.md)**  
   Statistiche descrittive e analisi delle distribuzioni delle feature.

5. **[04_analisi_statistica.md](04_analisi_statistica.md)**  
   Test di ipotesi, analisi delle correlazioni e loro interpretazione.

6. **[05_modelli_predittivi.md](05_modelli_predittivi.md)**  
   Implementazione e confronto di modelli lineari e non lineari.

7. **[06_risultati_discussione.md](06_risultati_discussione.md)**  
   Interpretazione integrata dei risultati e discussione delle implicazioni.

8. **[07_limitazioni_validazione.md](07_limitazioni_validazione.md)**  
   Limitazioni metodologiche e procedure di validazione adottate.

9. **[08_conclusioni.md](08_conclusioni.md)**  
   Sintesi finale, contributo alla letteratura e direzioni future.

10. **[09_bibliografia.md](09_bibliografia.md)**  
    Riferimenti bibliografici utilizzati nello studio.

### Documenti di Supporto

- **[sommario_esecutivo.md](sommario_esecutivo.md)**  
  Una sintesi concisa dei principali risultati e conclusioni (5 pagine).

- **[guida_visualizzazioni.md](guida_visualizzazioni.md)**  
  Guida dettagliata all'interpretazione dei grafici e delle visualizzazioni.

## Suggerimenti di Lettura

### Per una panoramica rapida:
1. **[sommario_esecutivo.md](sommario_esecutivo.md)** - Fornisce una visione d'insieme completa ma concisa

### Per approfondire aspetti metodologici:
1. **[01_introduzione.md](01_introduzione.md)** - Per comprendere gli obiettivi e le ipotesi
2. **[02_dataset_metodologia.md](02_dataset_metodologia.md)** - Per i dettagli sul dataset e l'approccio metodologico
3. **[07_limitazioni_validazione.md](07_limitazioni_validazione.md)** - Per una riflessione critica sui limiti dello studio

### Per concentrarsi sui risultati:
1. **[03_analisi_esplorativa.md](03_analisi_esplorativa.md)** - Per un'analisi descrittiva iniziale
2. **[04_analisi_statistica.md](04_analisi_statistica.md)** - Per i risultati dei test statistici
3. **[05_modelli_predittivi.md](05_modelli_predittivi.md)** - Per i risultati dei modelli predittivi
4. **[06_risultati_discussione.md](06_risultati_discussione.md)** - Per un'interpretazione integrata

### Per le implicazioni teoriche e pratiche:
1. **[06_risultati_discussione.md](06_risultati_discussione.md)** - Per la discussione delle implicazioni
2. **[08_conclusioni.md](08_conclusioni.md)** - Per il contributo alla letteratura e le direzioni future

### Per comprendere i grafici e le visualizzazioni:
- **[guida_visualizzazioni.md](guida_visualizzazioni.md)** - Riferimento essenziale per interpretare correttamente i dati visivi

## Note sulla Consultazione

- I documenti sono progettati per essere letti in sequenza, ma possono anche essere consultati individualmente a seconda degli interessi specifici.
- Tutti i grafici e le tabelle sono numerati e referenziati nel testo per facilitare la navigazione.
- I riferimenti bibliografici sono citati nel testo e raccolti nel documento [09_bibliografia.md](09_bibliografia.md).
- Per un'interpretazione corretta dei grafici, si consiglia di consultare la [guida_visualizzazioni.md](guida_visualizzazioni.md) durante la lettura.

## Nota sul Report Completo

Se desideri consultare il report come un documento unico, è possibile combinare i file markdown in un unico documento utilizzando uno strumento come Pandoc con il seguente comando:

```bash
pandoc -s 00_copertina.md 01_introduzione.md 02_dataset_metodologia.md 03_analisi_esplorativa.md 04_analisi_statistica.md 05_modelli_predittivi.md 06_risultati_discussione.md 07_limitazioni_validazione.md 08_conclusioni.md 09_bibliografia.md -o report_completo.pdf
```

Questo genererà un PDF contenente il report scientifico completo, conservando la struttura e i riferimenti tra le sezioni.
