import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

# Chargement des variables d'environnement
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

MODELS = ["gemini-3.6-flash", "gemini-flash"]


class MockResponse:
    def __init__(self, text):
        self.text = text


def appeler_gemini(prompt: str):
    """Effectue l'appel API avec secours Mock en cas de quota ou indisponibilité."""
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
    if "MÉTRIQUES DE CLASSIFICATION" in prompt or "Accuracy" in prompt:
        return MockResponse("""### GUIDE COMPLET DES MÉTRIQUES DE CLASSIFICATION

1. **Accuracy (Exactitude)**
   - **Définition** : Proportion de prédictions correctes (vrais positifs et vrais négatifs) parmi le total des prédictions.
   - **Interprétation** : Indique à quel point le modèle a raison globalement.
   - **Exemple concret** : Un modèle classant des e-mails (spam/non-spam) obtient 95 bonnes prédictions sur 100 ; son accuracy est de 95%.
   - **Contexte utile** : Recommandé uniquement lorsque les classes sont bien équilibrées (ex: 50% classe A / 50% classe B).

2. **Precision (Précision)**
   - **Définition** : Proportion de vrais positifs parmi l'ensemble des éléments prédits comme positifs (VP / (VP + FP)).
   - **Interprétation** : Mesure la fiabilité des alarmes ou prédictions positives déclenchées par le modèle.
   - **Exemple concret** : Si le filtre détecte 10 spams et que 9 d'entre eux en sont réellement, la précision est de 90%.
   - **Contexte utile** : Crucial lorsque le coût d'un **Faux Positif (fausse alerte)** est très élevé (ex: filtre anti-spam d'e-mails importants, blocage automatique de comptes bancaires légitimes).

3. **Recall (Rappel / Sensibilité)**
   - **Définition** : Proportion de vrais positifs détectés parmi l'ensemble des cas réellement positifs (VP / (VP + FN)).
   - **Interprétation** : Mesure la capacité du modèle à ne pas rater les cas critiques.
   - **Exemple concret** : Si sur 100 patients malades, le test en repère 98, le rappel est de 98%.
   - **Contexte utile** : Indispensable lorsque le coût d'un **Faux Négatif (cas raté)** est critique (ex: diagnostic de maladies graves, détection de fraudes, détection de pannes critiques).

4. **F1-score**
   - **Définition** : Moyenne harmonique entre la Précision et le Rappel : 2 * (Precision * Recall) / (Precision + Recall).
   - **Interprétation** : Donne une vue d'ensemble équilibrée entre le contrôle des fausses alarmes et la détection globale.
   - **Exemple concret** : Un modèle avec Précision = 0.80 et Rappel = 0.90 obtient un F1-score de 0.84, évitant de surévaluer un modèle déséquilibré.
   - **Contexte utile** : Très utile en présence de jeux de données déséquilibrés lorsqu'un arbitrage équitable entre Précision et Rappel est recherché.

5. **ROC-AUC (Area Under the ROC Curve)**
   - **Définition** : Aire sous la courbe représentant le taux de vrais positifs en fonction du taux de faux positifs pour tous les seuils de décision possibles.
   - **Interprétation** : Mesure la capacité de discrimination globale du modèle (sa capacité à classer une instance positive au-dessus d'une instance négative).
   - **Exemple concret** : Une AUC de 0.92 signifie qu'il y a 92% de chances que le modèle attribue un score plus élevé à un individu réellement à risque qu'à un individu sain.
   - **Contexte utile** : Idéal pour comparer plusieurs modèles indépendamment du seuil de classification choisi et pour évaluer les performances globales sur des données déséquilibrées.""")
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
# PARTIE 5.9 : EXPLICATION DES MÉTRIQUES DE CLASSIFICATION
# ==============================================================================

def question_explication_metriques_classification() -> str:
    """
    Génère un guide explicatif structuré des 5 principales métriques de classification :
    Accuracy, Precision, Recall, F1-score et ROC-AUC.
    """
    print("==================================================")
    print("   PARTIE 5.9 : MÉTRIQUES DE CLASSIFICATION       ")
    print("==================================================\n")

    prompt = """### TÂCHE : EXPLICATION PÉDAGOGIQUE DES MÉTRIQUES DE CLASSIFICATION
Rédige un guide explicatif clair et structuré pour présenter les 5 métriques d'évaluation de classification suivantes :
1. Accuracy (Exactitude)
2. Precision (Précision)
3. Recall (Rappel / Sensibilité)
4. F1-score
5. ROC-AUC

### EXIGENCES DE CONTENU
Pour CHAQUE métrique mentionnée ci-dessus, tu dois obligatoirement détailler les 4 points suivants :
- **Définition** : Formule mathématique ou explication conceptuelle simple.
- **Interprétation** : Ce que la métrique mesure concrètement en langage clair.
- **Exemple concret** : Un cas d'usage illustratif avec des chiffres simples (ex: médical, spam, finance).
- **Contexte utile** : Dans quelle situation/problématique métier cette métrique doit être privilégiée (ex: données déséquilibrées, impact des faux positifs vs faux négatifs).

### FORMAT DE SORTIE
Structure la réponse avec un titre numéroté par métrique et des puces d'explication bien alignées."""

    res = appeler_gemini(prompt)
    explications = res.text.strip()

    print("[Guide des métriques généré] :")
    print(explications)
    print("\n--------------------------------------------------")

    return explications


if __name__ == "__main__":
    question_explication_metriques_classification()