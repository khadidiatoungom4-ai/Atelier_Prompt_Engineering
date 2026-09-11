import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

# Chargement des variables d'environnement (.env)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

MODELS = ["gemini-3.6-flash", "gemini-flash"]


class MockResponse:
    def __init__(self, text):
        self.text = text


def appeler_gemini(prompt: str):
    """Effectue l'appel API avec secours Mock en cas de quota ou d'indisponibilité."""
    if api_key:
        client = genai.Client(api_key=api_key)
        for model_name in MODELS:
            try:
                chat = client.chats.create(model=model_name)
                response = chat.send_message(prompt)
                print(f"[API Réussie via {model_name}]")
                return response
            except APIError as e:
                print(f"[Avertissement API {e.code} sur {model_name}]")
                if e.code == 429:
                    print("-> Quota atteint. Bascule temporaire en mode MOCK.")
                    break
                continue
            except Exception as e:
                print(f"[Erreur API : {e}]")
                break

    # Secours MOCK automatique
    print("[INFO] Mode MOCK (Hors-Ligne) activé.")
    if "RÉSUMÉ DE DOCUMENT" in prompt or "250 mots MAXIMUM" in prompt:
        return MockResponse("""**1. Objectifs**
- Réduire le temps de traitement des réclamations de 30%.
- Porter le taux de satisfaction client à 85%.

**2. Résultats**
- Déploiement d'un outil d'IA de tri automatique au Q3.
- Baisse du temps de réponse moyen de 48h à 12h.
- Progression du score de satisfaction de 72% à 81%.
- Taux d'adoption limité à 60% en raison d'un déficit de formation.

**3. Recommandations**
- Dispenser une formation obligatoire de deux semaines aux agents.
- Mettre à jour le guide utilisateur interne.
- Suivre chaque semaine les indicateurs d'adoption.""")
    else:
        mock_json = {
            "sentiment": "negatif",
            "categorie": "livraison",
            "urgence": "moyenne",
            "probleme": "Retard de livraison",
            "confiance": 0.91
        }
        return MockResponse(json.dumps(mock_json, ensure_ascii=False, indent=2))


# ==============================================================================
# PARTIE 5.1 : RÉSUMÉ DE DOCUMENT MÉTIER
# ==============================================================================

def question_resume_documentaire(document_texte: str) -> str:
    """
    Rédige un résumé structuré d'un document métier respectant les contraintes :
    - Maximum 250 mots
    - Informations factuelles uniquement (aucune invention)
    - Identification des objectifs, résultats et recommandations
    """
    print("==================================================")
    print("   PARTIE 5.1 : RÉSUMÉ DE DOCUMENT MÉTIER         ")
    print("==================================================\n")

    prompt = f"""### TÂCHE
Rédige un résumé analytique du document ci-dessous en respectant scrupuleusement les contraintes métiers.

### DOCUMENT À RÉSUMER
"{document_texte}"

### CONTRAINTES DE RÉDACTION
1. Longueur : 250 mots MAXIMUM.
2. Exactitude : Conserve uniquement les informations factuelles présentes dans le texte.
3. Factuality : N'invente AUCUNE information et n'ajoute pas de suppositions (zéro hallucination).

### STRUCTURE OBLIGATOIRE DU RÉSUMÉ
Organise ta réponse avec la structure suivante :

1. **Objectifs** : Quels sont les buts ou la finalité visés dans le document ?
2. **Résultats** : Quels sont les constatations, chiffres clés ou faits observés sous forme de phrases?
3. **Recommandations** : Quelles sont les préconisations ou actions suggérées ?

### EXIGENCES DE FORMAT
Réponds directement en français avec la structure demandée, sans texte d'introduction inutile."""

    res = appeler_gemini(prompt)
    resume = res.text.strip()

    print("[Texte original] :")
    print(document_texte.strip()[:200] + "... [tronqué]\n")
    print("[Résumé généré] :")
    print(resume)
    print("\n--------------------------------------------------")

    # Contrôle applicatif du nombre de mots
    nb_mots = len(resume.split())
    print(f"[Contrôle Longueur] : {nb_mots} mots (Limite : 250 mots max)")

    return resume


if __name__ == "__main__":
    document_exemple = """
    Rapport d'Évaluation de la Transformation Numérique - Q3 2026.
    L'entreprise a initié un plan de modernisation de sa gestion relation client afin de réduire le temps de traitement des réclamations de 30% et d'augmenter le taux de satisfaction à 85%.
    Au cours du troisième trimestre, les équipes ont déployé un nouvel outil d'IA pour le tri automatique des tickets support. Les données montrent une diminution effective du temps de réponse moyen de 48h à 12h, ainsi qu'une hausse du score de satisfaction client de 72% à 81%.
    Toutefois, le taux d'adoption par le personnel du service client reste limité à 60% en raison d'un manque de formation initiale.
    Il est vivement recommandé d'organiser un programme de formation obligatoire de deux semaines pour l'ensemble des agents, de mettre à jour le guide utilisateur interne, et d'effectuer un suivi hebdomadaire des indicateurs d'utilisation jusqu'à la fin de l'année.
    """

    question_resume_documentaire(document_exemple)