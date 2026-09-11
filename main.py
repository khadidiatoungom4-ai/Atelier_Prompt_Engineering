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
print("   PARTIE 2 — TECHNIQUE 3 : FEW-SHOT              ")
print("==================================================\n")

commentaire = "Le service est rapide mais l'application plante régulièrement."

# Plusieurs exemples couvrant chaque classe
prompt_fewshot = f"""Classer le commentaire suivant selon les classes (positif, négatif, neutre).

Exemple 1 :
Commentaire : "J'adore cette application, elle est extrêmement rapide et fluide !"
Classe : positif

Exemple 2 :
Commentaire : "L'interface a changé de couleur, c'est différent d'avant."
Classe : neutre

Exemple 3 :
Commentaire : "Le service client ne répond jamais et le paiement échoue systématiquement."
Classe : négatif

Commentaire : "{commentaire}"
Classe :"""

response = client.models.generate_content(
    model=MODEL_NAME,
    contents=prompt_fewshot
)

print(f"Résultat Few-shot :\n{response.text.strip()}")
print("\n==================================================")