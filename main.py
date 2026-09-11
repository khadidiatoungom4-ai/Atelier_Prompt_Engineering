import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Erreur : La clé GEMINI_API_KEY n'est pas définie dans le fichier .env")

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-3.6-flash"

print("==================================================")
print("   PARTIE 2 — TECHNIQUE 2 : ONE-SHOT              ")
print("==================================================\n")

commentaire = "Le service est rapide mais l'application plante régulièrement."

# Un seul exemple est fourni dans le prompt
prompt_oneshot = f"""Classer le commentaire suivant selon les classes (positif, négatif, neutre).

Exemple :
Commentaire : "La livraison a pris un mois, très déçu du délai."
Classe : négatif

Commentaire : "{commentaire}"
Classe :"""

response = client.models.generate_content(
    model=MODEL_NAME,
    contents=prompt_oneshot
)

print(f"Résultat One-shot :\n{response.text.strip()}")
print("\n==================================================")