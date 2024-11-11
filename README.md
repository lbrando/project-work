# PxDS Project Work 2024-2025

Il progetto consiste nella creazione di una pipeline di arricchimento dati per l'analisi di forme di propaganda nei discorsi di personaggi politici.

## Obiettivi

L'obiettivo principale è progettare e implementare una pipeline in Python che trasformi il dataset `speech-a.tsv` in `speech-b.csv`, arricchendolo con metadati e informazioni rilevanti per la detection di propaganda. Il concetto di propaganda viene inteso come la disseminazione di idee con lo scopo di influenzare attivamente il comportamento e le opinioni del pubblico.

## Dataset

- **File di Input**: `speech-a.tsv`
  - **Colonne**: Autore del discorso, codice numerico sconosciuto, testo del discorso.

## Requisiti di Arricchimento

La pipeline dovrebbe aggiungere le seguenti informazioni:

1. **Informazioni sull'autore del discorso**: nome completo, data e luogo di nascita, eventuale data di morte, nazionalità.
2. **Metadati del discorso**: luogo, data e occasione del discorso.
3. **Feature text-based**: leggibilità, sentiment, emozioni, caratteristiche linguistiche, conteggio di parole specifiche.
4. **Abstract e Keyword**: riassunto del discorso, parole chiave, argomenti principali, concetti rilevanti (utilizzando tecniche come summarization, keyword extraction e topic modeling).
5. **Informazioni utili per la detection di propaganda**: come l’offset di inizio e fine delle parti contenenti propaganda, tecniche di propaganda identificate.
6. **Narrativa di fondo**: schema narrativo o narrativa utilizzata nel discorso.

## Requisiti Tecnici

Il progetto richiede:
- **Script Python** per implementare la pipeline.
- **Metodologia di studio** sulle tecniche di propaganda e narrazione.

## Output Attesi

1. **File di output**: `speech-b.csv` contenente il dataset arricchito.
2. **Documentazione**: relazione tecnica e metodologica, codice sorgente funzionante, presentazione per la discussione.

## Valutazione

- **Sufficienza**: pipeline implementata con aggiunta delle informazioni dai punti 1, 2, e 3.
- **Massimo voto**: pipeline completa con arricchimenti dai punti 1-5, comprensiva di analisi narrativa e detection di tecniche di propaganda (punti 6 e 7).

## Organizzazione

- **Gruppi di lavoro**: massimo 3 persone.
- **Interazioni con il docente**: via Teams o in presenza previo appuntamento.
