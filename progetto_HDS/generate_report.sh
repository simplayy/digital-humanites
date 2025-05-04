#!/bin/bash

# Definizione delle directory
REPORT_DIR="reports/report_scientifico"
OUTPUT_DIR="reports"
OUTPUT_FILE="$OUTPUT_DIR/report_completo.docx"

# Crea un array con i file nell'ordine corretto
FILES=(
  "$REPORT_DIR/00_copertina.md"
  "$REPORT_DIR/sommario_esecutivo.md"
  "$REPORT_DIR/01_introduzione.md"
  "$REPORT_DIR/02_dataset_metodologia.md"
  "$REPORT_DIR/03_analisi_esplorativa.md"
  "$REPORT_DIR/04_analisi_statistica.md"
  "$REPORT_DIR/05_modelli_predittivi.md"
  "$REPORT_DIR/06_risultati_discussione.md"
  "$REPORT_DIR/07_limitazioni_validazione.md"
  "$REPORT_DIR/08_conclusioni.md"
  "$REPORT_DIR/09_bibliografia.md"
)

# Converti tutti i file markdown in un unico documento DOCX
pandoc "${FILES[@]}" \
  --from markdown \
  --to docx \
  --output "$OUTPUT_FILE" \
  --standalone \
  --toc \
  --toc-depth=3 \
  --highlight-style=tango \
  --resource-path=.:results:results/figures:results/narrative \
  --wrap=preserve \
  --extract-media=reports/media

echo "Documento Word generato: $OUTPUT_FILE"
