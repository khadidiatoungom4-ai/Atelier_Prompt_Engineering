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
    if "MODÈLES DE PRÉDICTION ÉNERGÉTIQUE" in prompt or "prédiction de la consommation" in prompt:
        return MockResponse("""### RECOMMANDATION DE MODÈLES DE PRÉDICTION ÉNERGÉTIQUE

1. **XGBoost / LightGBM (Gradient Boosted Decision Trees)**
   - **Type de problème** : Régression supervisée sur données tabulaires / caractéristiques temporelles créées (lags, variables calendaires).
   - **Principe** : Enchaînement séquentiel d'arbres de décision faibles où chaque nouvel arbre corrige les erreurs d'invalidation des précédents.
   - **Avantages** : Performances de pointe sur données tabulaires, gestion native des non-linéarités et interactions complexes entre variables (ex: météo vs occupation).
   - **Limites** : Sensible au surapprentissage si mal réglé ; nécessite de créer manuellement les variables de décalage temporel (lags).
   - **Métriques pertinentes** : RMSE, MAE, R², MAPE.

2. **Random Forest Regressor**
   - **Type de problème** : Régression supervisée (Ensemble Learning).
   - **Principe** : Agrégation en parallèle (bagging) de multiples arbres de décision entraînés sur des sous-échantillons bootstrap du jeu de données.
   - **Avantages** : Robuste au surapprentissage et aux bruits de mesure des capteurs ; fournit une mesure explicite de l'importance des variables.
   - **Limites** : Incapable d'extrapoler des tendances au-delà des valeurs observées dans l'ensemble d'entraînement ; temps de prédiction plus élevé que les modèles linéaires.
   - **Métriques pertinentes** : RMSE, MAE, R².

3. **LSTM / GRU (Réseaux de Nerf Récurrents - Deep Learning)**
   - **Type de problème** : Régression sur séries temporelles séquentielles (Time Series Forecasting).
   - **Principe** : Utilisation de mécanismes de portes mémoire pour capturer les dépendances temporelles à long terme et les récurrences cycliques.
   - **Avantages** : Modélise directement la dynamicité temporelle sans ingénierie complexe de variables décalées ; excelle sur les jeux de données volumineux.
   - **Limites** : Nécessite un grand volume de données et des ressources de calcul importantes (GPU) ; boîte noire difficile à interpréter.
   - **Métriques pertinentes** : RMSE, MAE, MAPE.

4. **Régression Ridge / Lasso (Modèles Linéaires Régularisés)**
   - **Type de problème** : Régression linéaire régularisée.
   - **Principe** : Modèle linéaire pénalisant les grands coefficients (L2 pour Ridge, L1 pour Lasso) pour éviter le surapprentissage.
   - **Avantages** : Très rapide à entraîner, fortement interprétable (coefficients explicites), parfait comme baseline de comparaison.
   - **Limites** : Ne capture pas naturellement les relations non linéaires complexes sans transformation de variables.
   - **Métriques pertinentes** : MAE, R².""")
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
# PARTIE 5.8 : PROPOSITION DE MODÈLES DE PRÉDICTION ÉNERGÉTIQUE
# ==============================================================================

def question_modeles_prediction_energetique(description_dataset: str) -> str:
    """
    Propose des modèles de Machine Learning / Deep Learning adaptés à la prédiction
    de la consommation énergétique en détaillant principe, avantages, limites,
    type de problème et métriques.
    """
    print("==================================================")
    print("   PARTIE 5.8 : MODÈLES DE PRÉDICTION ÉNERGÉTIQUE  ")
    print("==================================================\n")

    prompt = f"""### TÂCHE : MODÈLES DE PRÉDICTION ÉNERGÉTIQUE
En te basant sur la description du dataset de capteurs ci-dessous, propose 3 à 5 modèles d'apprentissage automatique (Machine Learning / Deep Learning) adaptés à la prédiction de la consommation énergétique d'un bâtiment.

### DESCRIPTION DU DATASET CAPTEURS
"{description_dataset}"

### EXIGENCES DE CONTENU
Pour CHAQUE modèle proposé, tu dois obligatoirement détailler les 5 éléments suivants :
1. **Type de problème** : Régression, séries temporelles, apprentissage supervisé, etc.
2. **Principe** : Explication concise du fonctionnement algorithmique du modèle.
3. **Avantages** : Points forts pour la prédiction énergétique de bâtiment.
4. **Limites** : Inconvénients, contraintes de données ou risques de surapprentissage.
5. **Métriques pertinentes** : Métriques d'évaluation de performance recommandées (ex: RMSE, MAE, R², MAPE).

### FORMAT DE SORTIE
Structure la réponse de manière claire avec une section numérotée pour chaque modèle et des puces bien définies."""

    res = appeler_gemini(prompt)
    modeles = res.text.strip()

    print(f"[Description du dataset] :\n{description_dataset}\n")
    print("[Recommandations de modèles] :")
    print(modeles)
    print("\n--------------------------------------------------")

    return modeles


if __name__ == "__main__":
    dataset_energie_info = """
    Dataset de télémesure énergétique d'un bâtiment tertiaire de 5 étages :
    - timestamp (datetime : pas de 15 minutes)
    - batiment_id / etage (string)
    - consommation_kwh (float : consommation électrique instantanée)
    - puissance_kw (float : puissance appelée)
    - temperature_exterieure_C (float : météo extérieure)
    - temperature_interieure_C (float : consigne d'ambiance)
    - taux_occupation (float : pourcentage de présence via capteurs de mouvement)
    - type_usage (string : "CVC", "Éclairage", "Prises", "Informatique")
    """

    question_modeles_prediction_energetique(dataset_energie_info)