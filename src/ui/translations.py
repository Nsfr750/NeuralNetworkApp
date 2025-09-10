"""
Translations for the Benchmark application.
Contains all text strings in all supported languages.
"""

TRANSLATIONS = {
    'en': {
        "app": {
            "title": "Neural Network Creator",
            "loading": "Loading...",
            "error": "Error",
            "success": "Success",
            "warning": "Warning",
            "info": "Information",
            "confirm": "Confirm",
            "yes": "Yes",
            "no": "No",
            "cancel": "Cancel",
            "ok": "OK",
            "save": "Save",
            "open": "Open",
            "close": "Close"
        },
        "menu": {
            "file": "File",
            "new": "New",
            "open": "Open",
            "save": "Save",
            "save_as": "Save As...",
            "exit": "Exit",
            "edit": "Edit",
            "cut": "Cut",
            "copy": "Copy",
            "paste": "Paste",
            "delete": "Delete",
            "select_all": "Select All",
            "view": "View",
            "zoom_in": "Zoom In",
            "zoom_out": "Zoom Out",
            "reset_zoom": "Reset Zoom",
            "help": "Help",
            "about": "About",
            "language": "Language",
            "english": "English",
            "italian": "Italiano",
            "theme": "Theme",
            "dark": "Dark",
            "light": "Light",
            "system": "System"
        },
        "model": {
            "title": "Model",
            "create": "Create Model",
            "load": "Load Model",
            "save": "Save Model",
            "visualize": "Visualize Model",
            "configuration": "Model Configuration",
            "input_size": "Input Size:",
            "hidden_layers": "Hidden Layers (comma-separated):",
            "output_size": "Output Size:",
            "activation": "Activation:",
            "dropout": "Dropout (0 to disable):",
            "batch_norm": "Batch Normalization:",
            "summary": "Model Summary:",
            "created_success": "Model created successfully!",
            "save_success": "Model saved successfully!",
            "load_success": "Model loaded successfully!",
            "save_error": "Error saving model!",
            "load_error": "Error loading model!"
        },
        "training": {
            "title": "Training",
            "start": "Start Training",
            "stop": "Stop Training",
            "epochs": "Epochs:",
            "batch_size": "Batch Size:",
            "learning_rate": "Learning Rate:",
            "optimizer": "Optimizer:",
            "loss": "Loss Function:",
            "metrics": "Metrics:",
            "progress": "Training Progress",
            "epoch": "Epoch {current}/{total}",
            "loss_value": "Loss: {loss:.4f}",
            "accuracy": "Accuracy: {accuracy:.2f}%",
            "completed": "Training completed!",
            "stopped": "Training stopped by user.",
            "error": "Error during training!"
        },
        "data": {
            "title": "Data",
            "load": "Load Dataset",
            "preprocess": "Preprocess Data",
            "split": "Train/Test Split:",
            "train_size": "Training Size: {size}%",
            "val_size": "Validation Size: {size}%",
            "test_size": "Test Size: {size}%",
            "shuffle": "Shuffle Data",
            "normalize": "Normalize Data",
            "augment": "Data Augmentation",
            "samples": "Samples: {count}",
            "classes": "Classes: {count}",
            "features": "Features: {count}",
            "load_success": "Dataset loaded successfully!",
            "load_error": "Error loading dataset!"
        },
        "log_viewer": {
            "title": "Log Viewer",
            "refresh": "&Refresh",
            "clear": "&Clear Log",
            "save_as": "&Save As...",
            "no_logs": "No log files found.",
            "select_file": "Select log file to view:",
            "clear_confirm": "Are you sure you want to clear the log?",
            "save_success": "Log saved successfully!",
            "save_error": "Error saving log!"
        },
        "about": {
            "title": "About Neural Network Creator",
            "description": "A PySide6 application for creating and training neural networks with a modern GUI interface.",
            "version": "Version: {version}",
            "author": "© {year} Nsfr750 - All rights reserved",
            "license": "License: GPLv3",
            "github": "GitHub Repository",
            "close": "Close"
        },
        "help": {
            "title": "Help",
            "error_loading": "Could not load help documentation\n\nPlease visit the GitHub repository for documentation",
            "getting_started": "Getting Started",
            "welcome": "Welcome to Neural Network Creator! This application allows you to create, train, and evaluate neural networks with an intuitive graphical interface."
        }
    },
    'it': {
        "app": {
            "title": "Creatore di Reti Neurali",
            "loading": "Caricamento...",
            "error": "Errore",
            "success": "Operazione completata",
            "warning": "Avviso",
            "info": "Informazione",
            "confirm": "Conferma",
            "yes": "Sì",
            "no": "No",
            "cancel": "Annulla",
            "ok": "OK",
            "save": "Salva",
            "open": "Apri",
            "close": "Chiudi"
        },
        "menu": {
            "file": "File",
            "new": "Nuovo",
            "open": "Apri",
            "save": "Salva",
            "save_as": "Salva con nome...",
            "exit": "Esci",
            "edit": "Modifica",
            "cut": "Taglia",
            "copy": "Copia",
            "paste": "Incolla",
            "delete": "Elimina",
            "select_all": "Seleziona tutto",
            "view": "Visualizza",
            "zoom_in": "Ingrandisci",
            "zoom_out": "Riduci",
            "reset_zoom": "Reimposta zoom",
            "help": "Aiuto",
            "about": "Informazioni",
            "language": "Lingua",
            "english": "Inglese",
            "italian": "Italiano",
            "theme": "Tema",
            "dark": "Scuro",
            "light": "Chiaro",
            "system": "Sistema"
        },
        "model": {
            "title": "Modello",
            "create": "Crea Modello",
            "load": "Carica Modello",
            "save": "Salva Modello",
            "visualize": "Visualizza Modello",
            "configuration": "Configurazione Modello",
            "input_size": "Dimensione Input:",
            "hidden_layers": "Livelli Nascosti (separati da virgola):",
            "output_size": "Dimensione Output:",
            "activation": "Funzione di Attivazione:",
            "dropout": "Dropout (0 per disabilitare):",
            "batch_norm": "Normalizzazione Batch:",
            "summary": "Riepilogo Modello:",
            "created_success": "Modello creato con successo!",
            "save_success": "Modello salvato con successo!",
            "load_success": "Modello caricato con successo!",
            "save_error": "Errore nel salvataggio del modello!",
            "load_error": "Errore nel caricamento del modello!"
        },
        "training": {
            "title": "Addestramento",
            "start": "Avvia Addestramento",
            "stop": "Ferma Addestramento",
            "epochs": "Epoche:",
            "batch_size": "Dimensione Batch:",
            "learning_rate": "Tasso di Apprendimento:",
            "optimizer": "Ottimizzatore:",
            "loss": "Funzione di Perdita:",
            "metrics": "Metriche:",
            "progress": "Progresso Addestramento",
            "epoch": "Epoca {current}/{total}",
            "loss_value": "Perdita: {loss:.4f}",
            "accuracy": "Accuratezza: {accuracy:.2f}%",
            "completed": "Addestramento completato!",
            "stopped": "Addestramento interrotto dall'utente.",
            "error": "Errore durante l'addestramento!"
        },
        "data": {
            "title": "Dati",
            "load": "Carica Dataset",
            "preprocess": "Preelabora Dati",
            "split": "Divisione Train/Test:",
            "train_size": "Dimensione Addestramento: {size}%",
            "val_size": "Dimensione Validazione: {size}%",
            "test_size": "Dimensione Test: {size}%",
            "shuffle": "Mescola Dati",
            "normalize": "Normalizza Dati",
            "augment": "Aumento Dati",
            "samples": "Campioni: {count}",
            "classes": "Classi: {count}",
            "features": "Caratteristiche: {count}",
            "load_success": "Dataset caricato con successo!",
            "load_error": "Errore nel caricamento del dataset!"
        },
        "log_viewer": {
            "title": "Visualizzatore Log",
            "refresh": "&Aggiorna",
            "clear": "&Pulisci Log",
            "save_as": "&Salva con nome...",
            "no_logs": "Nessun file di log trovato.",
            "select_file": "Seleziona il file di log da visualizzare:",
            "clear_confirm": "Sei sicuro di voler cancellare il log?",
            "save_success": "Log salvato con successo!",
            "save_error": "Errore nel salvataggio del log!"
        },
        "about": {
            "title": "Informazioni su Neural Network Creator",
            "description": "Un'applicazione PySide6 per creare e addestrare reti neurali con un'interfaccia grafica moderna.",
            "version": "Versione: {version}",
            "author": "© {year} Nsfr750 - Tutti i diritti riservati",
            "license": "Licenza: GPLv3",
            "github": "Repository GitHub",
            "close": "Chiudi"
        },
        "help": {
            "title": "Aiuto",
            "error_loading": "Impossibile caricare la documentazione\n\nVisita il repository GitHub per la documentazione",
            "getting_started": "Per Iniziare",
            "welcome": "Benvenuto in Neural Network Creator! Questa applicazione ti permette di creare, addestrare e valutare reti neurali con un'interfaccia grafica intuitiva."
        }
    }
}

def get_translation(lang_code, key, default=None):
    """
    Get a translation for the given language code and key.
    
    Args:
        lang_code: Language code (e.g., 'en', 'it')
        key: Dot-separated key path (e.g., 'app.title')
        default: Default value to return if key not found
    
    Returns:
        The translated string or the default value if not found
    """
    if lang_code not in TRANSLATIONS:
        lang_code = 'en'  # Fallback to English
    
    parts = key.split('.')
    result = TRANSLATIONS[lang_code]
    
    try:
        for part in parts:
            result = result[part]
        return result
    except (KeyError, TypeError):
        return default if default is not None else key
