# Galaxy Classifier MBV2

Aplikacja webowa do klasyfikacji galaktyk przy użyciu sieci neuronowej MobileNetV2.
Prześlij zdjęcie galaktyki, a model oceni czy jest spiralna czy eliptyczna - wraz z mapą ciepła Grad-CAM pokazującą na co sieć zwróciła uwagę.

## Jak to działa

Model bazuje na architekturze MobileNetV2 wstępnie wytrenowanej na ImageNet.
Ostatnia warstwa decyzyjna została zastąpiona warstwą z dwoma wyjściami (spiralna / eliptyczna)
i douczana na 6000 zdjęciach galaktyk pobranych z katalogu SDSS (Sloan Digital Sky Survey) —
3000 spiral i 3000 eliptycznych, z etykietami pochodzącymi z projektu Galaxy Zoo.

## Struktura projektu

- backend/
   `main.py - API FastAPI, klasyfikacja + Grad-CAM
  - najlepszy_MobileNetV2.pth — wagi wytrenowanego modelu
- frontend/
  - app.py - interfejs użytkownika (Gradio)
- notebooks/
  - SDSS_Pipeline.ipynb - pobieranie danych, augmentacja, trening
- requirements.txt
- Dockerfile
​

## Dataset

- **Źródło:** SDSS DR18 + Galaxy Zoo (zooSpec)
- **Liczba zdjęć:** 6000 (3000 spiral, 3000 eliptycznych)
- **Format:** JPG 224×224 px, skala 0.2 arcsec/px
- **Podział:** 80% trening / 20% walidacja

## Architektura modelu

​
MobileNetV2 (pretrained ImageNet)
    -classifier[1]: Linear(1280 → 2)
​

## Trening

- **Optimizer:** Adam, lr=0.001
- **Loss:** CrossEntropyLoss
- **Epoki:** 10
- **Batch size:** 32
- **Augmentacja:** obroty 0–360°, odbicia, przesunięcia, zmiany jasności/kontrastu
- **Normalizacja:** mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

## Grad-CAM

Po klasyfikacji generowana jest mapa ciepła nakładana na oryginalne zdjęcie.
Pokazuje które obszary galaktyki miały największy wpływ na decyzję modelu.

## Technologie

- Python 3.11
- PyTorch + TorchVision
- FastAPI + Uvicorn
- Gradio
- pytorch-grad-cam
- astroquery
- Pillow, NumPy
