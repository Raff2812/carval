# Carval

Carval è un sistema di Machine Learning sviluppato per supportare gli utenti 
nella valutazione del prezzo delle vetture automobilistiche di seconda mano.

L'obiettivo principale del progetto è fornire una valutazione data-driven 
del prezzo delle auto usate, aiutando sia i privati che le concessionarie a 
prendere decisioni informate.

## Autore
- Raffaele Coppola - [Raff2812](https://github.com/Raff2812)

## Struttura del progetto  

La repository di Carval è organizzata nelle seguenti directory:  

- **`datasets/`**. Contiene il dataset originale reperito da Kaggle ([qui](https://www.kaggle.com/datasets/tunguz/used-car-auction-prices)) 
e il dataset pre-elaborato utilizzato per l’addestramento dei modelli.  
- **`diagrams/`**. Raccoglie tutti i grafici generati durante l'analisi esplorativa 
dei dati e la fase di sviluppo della pipeline di Machine Learning.  
- **`docs/`**. Include la documentazione e presentazione relativa al progetto.  
- **`notebooks/`**: Contiene tutti i Jupyter Notebook sviluppati.  
- **`pipeline/`**: Contiene tutti i file Python sviluppati per la pipeline di ml.  


## Come utilizzare il progetto  

Per eseguire Carval, segui questi semplici passaggi:  

### 1. Installazione delle dipendenze  

Prima di avviare il sistema, è necessario installare tutte le dipendenze richieste. 
Assicurati di avere Python installato, quindi esegui il seguente comando dalla root del progetto:  

```bash
pip install -r requirements.txt
```

### 2. Esecuzione della pipeline

Una volta installate le dipendenze, esegui la pipeline di pre-elaborazione dei dati e addestramento del modello. 
Vai nella cartella pipeline e lancia lo script.

```bash
python pipeline/pipeline.py
```

### 3. Avvio dell'interfaccia utente

Dopo aver completato la fase di preprocessing e addestramento, puoi eseguire l'interfaccia web per interagire con il modello.
Spostati nella directory pipeline con il comando `cd pipeline` ed esegui il seguente comando:

```bash
streamlit run app.py
```

Questo avvierà l'applicazione Streamlit, permettendoti di testare il sistema direttamente dal browser.


