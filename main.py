import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

# Chargement de la clé API depuis le fichier .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Erreur : La clé GEMINI_API_KEY n'est pas définie dans le fichier .env")

client = genai.Client(api_key=api_key)

# Modèles actualisés selon la réponse de l'API
MODELS = ["gemini-3.6-flash", "gemini-2.5-flash"]

def appeler_gemini(prompt, max_retries=3):
    """Appelle l'API via l'interface Chat recommandée pour éviter le warning AFC."""
    last_exception = None
    
    for model_name in MODELS:
        for attempt in range(max_retries):
            try:
                # Utilisation de l'API Chat recommandée
                chat = client.chats.create(model=model_name)
                response = chat.send_message(prompt)
                return response
            except APIError as e:
                last_exception = e
                if e.code == 503:
                    time.sleep(2)
                elif e.code in (404, 403):
                    break  # Passer au modèle suivant
                else:
                    time.sleep(1)
            except Exception as e:
                last_exception = e
                break

    raise RuntimeError(f"Échec de l'appel API. Dernier message d'erreur : {last_exception}")


# Donnée d'entrée
avis_client = "L'application est fluide mais le paiement par carte Visa échoue une fois sur deux."


# ==================================================
# QUESTION 3.1 — ANALYSE INITIALE DU PROBLÈME
# ==================================================
print("==================================================")
print("   QUESTION 3.1 : ANALYSE INITIALE                ")
print("==================================================\n")

prompt_q3_1 = f"""### TÂCHE
 Analyse ces avis clients et donne-moi les problèmes les plus importants ainsi que les recommandations. .

### AVIS CLIENT
"{avis_client}"

### CONTRAINTE
Rédige un résumé du problème sous forme de tableau."""

res_analyse = appeler_gemini(prompt_q3_1)
analyse_initiale = res_analyse.text.strip()

print(f"Avis original    : {avis_client}")
print(f"Analyse initiale : {analyse_initiale}\n")