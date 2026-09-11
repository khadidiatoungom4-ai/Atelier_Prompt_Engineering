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
    if "MÉTRIQUES DE RÉGRESSION" in prompt or "MAE" in prompt:
        return MockResponse("""### GUIDE COMPLET DES MÉTRIQUES DE RÉGRESSION

1. **MAE (Mean Absolute Error - Erreur Absolue Moyenne)**
   - **Définition** : Moyenne des écarts absolus entre les valeurs prédites et les valeurs réelles : MAE = (1/n) * Σ |y_i - ŷ_i|.
   - **Interprétation** : Représente l'erreur moyenne du modèle exprimée directement dans la même unité que la variable cible.
   - **Exemple concret** : Si la MAE d'un modèle de prédiction de consommation électrique est de 15 kWh, cela signifie qu'en moyenne, les prédictions s'écartent de 15 kWh de la réalité.
   - **Contexte utile** : Idéal lorsque toutes les erreurs doivent être traitées de manière linéaire et que le jeu de données contient des valeurs aberrantes (outliers) qu'on ne veut pas sur-pénaliser.

2. **MSE (Mean Squared Error - Erreur Quadratique Moyenne)**
   - **Définition** : Moyenne des carrés des écarts entre les valeurs prédites et les valeurs réelles : MSE = (1/n) * Σ (y_i - ŷ_i)².
   - **Interprétation** : Mesure la variance de l'erreur en pénalisant de manière quadratique (au carré) les grands écarts.
   - **Exemple concret** : Une erreur de 2 unités produit une pénalité de 4, tandis qu'une erreur de 10 unités produit une pénalité de 100.
   - **Contexte utile** : Particulièrement utile lors de l'entraînement d'algorithmes (fonction de perte) car elle est dérivable partout et pénalise sévèrement les erreurs importantes.

3. **RMSE (Root Mean Squared Error - Racine de l'Erreur Quadratique Moyenne)**
   - **Définition** : Racine carrée de l'erreur quadratique moyenne : RMSE = √MSE.
   - **Interprétation** : Mesure l'écart-type des résidus, exprimée dans la même unité que la variable cible tout en conservant la pénalisation forte des grands écarts.
   - **Exemple concret** : Pour la prédiction de la consommation d'un bâtiment, une RMSE de 22 kWh indique que les grosses erreurs de prédiction ont tiré la moyenne des écarts vers le haut par rapport à la MAE (15 kWh).
   - **Contexte utile** : Indispensable lorsqu'une grande erreur de prédiction a des conséquences beaucoup plus graves ou coûteuses qu'une petite erreur (ex: gestion du réseau électrique, prévision des pics de charge).""")
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
# PARTIE 5.10 : EXPLICATION DES MÉTRIQUES DE RÉGRESSION
# ==============================================================================

def question_explication_metriques_regression() -> str:
    """
    Génère un guide explicatif structuré des 3 principales métriques de régression :
    MAE, MSE et RMSE.
    """
    print("==================================================")
    print("   PARTIE 5.10 : MÉTRIQUES DE RÉGRESSION          ")
    print("==================================================\n")

    prompt = """### TÂCHE : EXPLICATION PÉDAGOGIQUE DES MÉTRIQUES DE RÉGRESSION
Rédige un guide explicatif clair et structuré pour présenter les 3 métriques clés d'évaluation des modèles de régression suivantes :
1. MAE (Mean Absolute Error)
2. MSE (Mean Squared Error)
3. RMSE (Root Mean Squared Error)

### EXIGENCES DE CONTENU
Pour CHAQUE métrique mentionnée ci-dessus, tu dois obligatoirement détailler les 4 points suivants :
- **Définition** : Formule mathématique ou explication conceptuelle simple.
- **Interprétation** : Ce que la métrique mesure concrètement et son unité de mesure.
- **Exemple concret** : Un cas d'usage illustratif avec des chiffres simples (ex: prédiction de prix d'immobilier, consommation énergétique, température).
- **Contexte utile** : Dans quelle situation métier cette métrique doit être privilégiée (ex: présence d'outliers, sensibilité aux grandes erreurs, interprétabilité).

### FORMAT DE SORTIE
Structure la réponse avec un titre numéroté par métrique et des puces d'explication bien alignées."""

    res = appeler_gemini(prompt)
    explications = res.text.strip()

    print("[Guide des métriques de régression généré] :")
    print(explications)
    print("\n--------------------------------------------------")

    return explications


if __name__ == "__main__":
    question_explication_metriques_regression()