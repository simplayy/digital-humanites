#!/bin/bash

# Script per generare un report scientifico semplificato in formato HTML
# Questo script è una versione semplificata che genera solo HTML senza immagini
# Utile come backup se lo script principale fallisce

echo "Generazione del report scientifico in formato HTML..."

# Directory del report scientifico
REPORT_DIR="./report_scientifico"
OUTPUT_DIR="."

# Verifica se pandoc è installato
if ! command -v pandoc &> /dev/null; then
    echo "Errore: pandoc non è installato. È necessario installarlo per generare l'HTML."
    echo "Puoi installarlo con: brew install pandoc"
    exit 1
fi

# Ordine dei file markdown
FILES=(
    "sommario_esecutivo.md"
    "01_introduzione.md"
    "02_dataset_metodologia.md"
    "03_analisi_esplorativa.md"
    "04_analisi_statistica.md"
    "05_modelli_predittivi.md"
    "06_risultati_discussione.md"
    "07_limitazioni_validazione.md"
    "08_conclusioni.md"
    "09_bibliografia.md"
)

# Creare un file temporaneo che contiene tutti i markdown combinati
echo "Combinazione dei file markdown..."
TEMP_FILE=$(mktemp)

# Aggiungiamo un titolo e un'intestazione al file HTML
echo "# Relazione tra Sentiment nei Commenti e Veridicità delle Notizie: Analisi del Dataset PHEME" > "$TEMP_FILE"
echo "" >> "$TEMP_FILE"
echo "Autore: Simone" >> "$TEMP_FILE"
echo "" >> "$TEMP_FILE"
echo "Data: 3 maggio 2025" >> "$TEMP_FILE"
echo "" >> "$TEMP_FILE"

# Concatenazione dei file
for file in "${FILES[@]}"; do
    echo "" >> "$TEMP_FILE"
    echo "---" >> "$TEMP_FILE"
    echo "" >> "$TEMP_FILE"
    cat "$REPORT_DIR/$file" >> "$TEMP_FILE"
done

# Generare l'HTML
echo "Generazione del file HTML..."
pandoc "$TEMP_FILE" \
    -o "$OUTPUT_DIR/report_semplificato.html" \
    --from markdown \
    --to html \
    --standalone \
    --toc \
    --toc-depth=3 \
    --highlight-style=tango \
    --metadata title="Report Scientifico: Sentiment e Veridicità"

# Verifica se la generazione è andata a buon fine
if [ -f "$OUTPUT_DIR/report_semplificato.html" ]; then
    echo "Il report semplificato è stato generato con successo: $OUTPUT_DIR/report_semplificato.html"
else
    echo "Si è verificato un errore durante la generazione del report semplificato."
    exit 1
fi

# Pulizia file temporanei
rm -f "$TEMP_FILE"

echo "Processo completato."
