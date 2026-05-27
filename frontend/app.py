import gradio as gr
import requests
import base64
import io
from PIL import Image

#adres 
API_URL = "http://127.0.0.1:8000/skanuj"

def wyslij_do_serwera(sciezka_zdjecia):
    with open(sciezka_zdjecia, "rb") as plik:
        paczka = {"plik": plik}
        odpowiedz = requests.post(API_URL, files=paczka)
        
   
    dane = odpowiedz.json()
    
    
    prawd_eliptyczna = float(dane["pewnosc_eliptyczna"].replace("%", "")) / 100
    prawd_spiralna = float(dane["pewnosc_spiralna"].replace("%", "")) / 100
    
   # wyciagamy z json tekst 
    kod_obrazka = dane["mapa_ciepla"]
    
    #odkodowujemy tekst z powrotem na surowe bajty
    bajty_obrazka = base64.b64decode(kod_obrazka)
    
    
    obrazek_cam = Image.open(io.BytesIO(bajty_obrazka))
    
    
    
    return {"Eliptyczna": prawd_eliptyczna, "Spiralna": prawd_spiralna}, obrazek_cam
    


aplikacja = gr.Interface(
    fn=wyslij_do_serwera,
    inputs=gr.Image(type="filepath", label="Wgraj zdjęcie galaktyki"), #okienko wejscia
    outputs=[
        gr.Label(num_top_classes=2, label="Werdykt sztucznej inteligencji"),
        gr.Image(label="Mapa ciepła (Grad-CAM)")
    ], #okienko wyjscia
    title="Klasyfikator Galaktyk - MobileNetV2",
    description="Przeciągnij zdjęcie aby sprawdzić czy galaktyka jest spiralna czy eliptyczna" 
)

aplikacja.launch(server_name="0.0.0.0", server_port=7860)