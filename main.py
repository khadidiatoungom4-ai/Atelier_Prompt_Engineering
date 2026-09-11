import os
from dotenv import load_dotenv
from google import genai
# 1. Initialisation
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Erreur : La clé GEMINI_API_KEY n'est pas définie dans le fichier .env")

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-3.6-flash"

print("==================================================")
print("   PARTIE 1 — ANATOMIE D'UN PROMPT (MULTIPLE DATA)")
print("==================================================\n")

# Prompt structuré avec plusieurs données d'entrée
prompt_partie1_multi = """
### RÔLE
Tu es un Expert en Analyse de l'Expérience Client (CX Analyst).
### CONTEXTE
Une entreprise souhaite analyser un lot de retours clients afin de classifier automatiquement les problèmes et mesurer le sentiment global.
### TÂCHE
Pour CHAQUE avis client fourni dans la liste ci-dessous :
1. Détermine le sentiment général.
2. Identifie la catégorie principale.
3. Rédige un résumé du problème ou du point clé en une seule phrase.
### CONTRAINTES
- Reste strictement factuel et neutre.
- N'invente aucune information non mentionnée.
- Conserve l'identifiant de chaque avis dans la réponse.
### FORMAT DE SORTIE
Pour chaque avis, utilise le format suivant :
---
**Avis [ID]**
- **Sentiment** : [Positif / Négatif / Neutre]
- **Catégorie** : [Livraison / Service Client / Produit / Facturation / Application / Autre]
- **Résumé** : [Résumé en 1 phrase]
### DONNÉES D'ENTRÉE (AVIS CLIENTS)
Avis 1 : "La commande est arrivée avec trois jours de retard et l'emballage était complètement déchiré, mais le produit à l'intérieur n'est pas endommagé."
Avis 2 : "Le service client au téléphone a été parfait, l'agent a résolu mon problème de remboursement en moins de cinq minutes."
Avis 3 : "L'application mobile plante systématiquement dès que je valide le panier avec une carte Mastercard."
Avis 4 : "Le produit livré est conforme à la description, mais les frais de livraison sont beaucoup trop élevés par rapport aux concurrents."
"""

# Exécution de la requête
response = client.models.generate_content(
    model=MODEL_NAME,
    contents=prompt_partie1_multi
)

print(response.text.strip())
print("\n==================================================")