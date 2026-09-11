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
print("   PARTIE 2 — TECHNIQUE 4 : PROMPT STRUCTURÉ      ")
print("==================================================\n")

commentaire = "Le service est rapide mais l'application plante régulièrement."

# Prompt structuré par blocs avec règles de gestion des avis mitigés
prompt_structure = f"""### RÔLE
Tu es un système automatisé de classification de sentiments pour une application mobile.

### CONTEXTE
Analyse des retours utilisateurs pour prioriser les corrections techniques et l'amélioration de l'expérience client.

### TÂCHE
Classer le commentaire fourni ci-dessous dans l'une des trois catégories : positif, négatif ou neutre.

### DONNÉE D'ENTRÉE
Commentaire : "{commentaire}"

### CONSIGNES ET REGLES D'ARBITRAGE
- Si l'avis contient à la fois des éléments positifs et négatifs :
  1. Si le problème technique bloque l'usage principal (ex: plantage), privilégie la classe **négatif**.
  2. Si les deux aspects s'équilibrent parfaitement sans blocage majeur, classe en **neutre**.
- Ne génère aucun texte d'explication, ni d'introduction, ni de ponctuation inutile.

### FORMAT DE SORTIE ATTENDU
Réponds uniquement par un seul mot en minuscules : positif, négatif ou neutre."""

response = client.models.generate_content(
    model=MODEL_NAME,
    contents=prompt_structure
)

print(f"Résultat Prompt Structuré :\n{response.text.strip()}")
print("\n==================================================")