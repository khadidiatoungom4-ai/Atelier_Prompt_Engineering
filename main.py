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
print("   PARTIE 2 — TECHNIQUE 1 : ZERO-SHOT             ")
print("==================================================\n")

commentaire = "Le service est rapide mais l'application plante régulièrement."

prompt_zeroshot = f"""Classer le commentaire suivant : "{commentaire}"
Classes possibles : positif, négatif, neutre."""

response = client.models.generate_content(
    model=MODEL_NAME,
    contents=prompt_zeroshot
)

print(f"Résultat Zero-shot :\n{response.text.strip()}")
print("\n==================================================")