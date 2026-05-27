from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import io
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import base64

app = FastAPI(title="Galaxy_Classifier_MBV2")

#konfiguracja
urzadzenie = torch.device("cpu")

print("Uruchamiam silnik i ładuje wagi modelu")

mobilenet = models.mobilenet_v2(weights=None)
mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, 2)

mobilenet.load_state_dict(torch.load("notebooks/najlepszy_MobileNetV2.pth", map_location=urzadzenie))
mobilenet.eval() #blokada nauki

# grad_cam
print("Podłączam system Grad-CAM...")

# wskazujemy warstwe docelowa
docelowa_warstwa = [mobilenet.features[-1]]

# tworzymy obiekt kamery
kamera_cam = GradCAM(model=mobilenet, target_layers=docelowa_warstwa)

print("System Grad-CAM gotowy!")
print("model gotowy do pracy")

#soczewki
transformacje = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

etykiety = {0: "Eliptyczna", 1: "Spiralna"}

#brama api
@app.post("/skanuj")
async def skanuj_galaktyke(plik: UploadFile = File(...)):
    zawartosc = await plik.read()

    #zlozenie spowrotem na obraz
    img = Image.open(io.BytesIO(zawartosc)).convert("RGB")

    #przepuszczenie przez soczewki i wdrozenie do modelu
    img_tensor = transformacje(img).unsqueeze(0).to(urzadzenie)

    with torch.no_grad():
        wyniki = mobilenet(img_tensor)
        procenty = torch.nn.functional.softmax(wyniki, dim=1)[0] * 100
    zwyciezca = etykiety[torch.argmax(procenty).item()]

    #grad cam

    # heatmap dla czarnobialego zdj
    grayscale_cam = kamera_cam(input_tensor=img_tensor)[0, :]
    
    
    rgb_img = np.float32(img.resize((224, 224))) / 255
    
    # nakladamy mape na zdjeice
    wizualizacja = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    #zamieniamy gotowa macierz na obrazek, a potem na dlugi tekst Base64
    img_cam = Image.fromarray(wizualizacja)
    bufor = io.BytesIO()
    img_cam.save(bufor, format="JPEG")
    kod_obrazka = base64.b64encode(bufor.getvalue()).decode("utf-8")


    #zwrocenie json do uzytkownika
    return {
        "nazwa_pliku": plik.filename,
        "werdykt_modelu": zwyciezca,
        "pewnosc_eliptyczna": f"{procenty[0].item():.2f}%",
        "pewnosc_spiralna": f"{procenty[1].item():.2f}%",
        "mapa_ciepla": kod_obrazka 
    }