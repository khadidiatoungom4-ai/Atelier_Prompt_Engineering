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
    if "STRATÉGIES DE PRÉTRAITEMENT DATASET CAPTEURS" in prompt:
        return MockResponse("""### STRATÉGIES DE PRÉTRAITEMENT DES DONNÉES CAPTEURS

1. **Valeurs manquantes**
   - **Détection** : Contrôle de nullité (`isna()`), détection d'interruption dans les séries temporelles.
   - **Traitement** : Interpolation temporelle linéaire ou spline pour courtes coupures ; imputation par filtre de Kalman pour les longues séries.
   - **Risques** : Biais d'interpolation si la panne du capteur correspond à un incident critique.

2. **Doublons**
   - **Détection** : Reconstitution de clés primaires composite `(sensor_id, timestamp)`.
   - **Traitement** : Déduplication en conservant la première mesure ou agrégation par moyenne si décalage inférieur à 10ms.
   - **Risques** : Suppression de renvois légitimes en cas d'horodatage imprécis.

3. **Valeurs aberrantes (Outliers)**
   - **Détection** : Écart-type mobiles (Z-Score > 3), test d'Isolation Forest ou seuils physiques d'équipement.
   - **Traitement** : Écrestage (winsorisation), lissage par moyenne glissante ou remplacement par la médiane locale.
   - **Risques** : Masquage d'anomalies réelles ou de pannes imminentes masquées en tant qu'aberrations.

4. **Variables catégorielles**
   - **Détection** : Vérification des statuts de capteurs (`OK`, `FAULT`, `CALIBRATION`) et identifiants matériels.
   - **Traitement** : Encodage One-Hot pour catégories nominales ; Target/Ordinal Encoding pour statuts d'erreur hiérarchisés.
   - **Risques** : Explosion de la dimensionnalité si le nombre d'identifiants de capteurs est trop élevé.""")
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
# PARTIE 5.6 : STRATÉGIES DE PRÉTRAITEMENT DE DATASET CAPTEURS
# ==============================================================================

def question_strategie_nettoyage_capteurs(description_dataset: str) -> str:
    """
    Génère un plan stratégique complet pour le traitement des données capteurs :
    - Valeurs manquantes
    - Doublons
    - Valeurs aberrantes
    - Variables catégorielles
    """
    print("==================================================")
    print("   PARTIE 5.6 : STRATÉGIES PRÉTRAITEMENT CAPTEURS ")
    print("==================================================\n")

    prompt = f"""### TÂCHE : STRATÉGIES DE PRÉTRAITEMENT DATASET CAPTEURS
En vous basant sur la description du dataset de capteurs ci-dessous, propose un plan stratégique détaillé pour la préparation et le nettoyage des données avant modélisation.

### DESCRIPTION DU DATASET CAPTEURS
"{description_dataset}"

### EXIGENCES DE CONTENU
Pour CHAQUENNE des 4 problématiques suivantes, tu dois détailler la stratégie en 3 axes précis :
1. **Valeurs manquantes**
   - Détection : Méthodes de repérage adaptées aux séries temporelles/capteurs.
   - Traitement : Techniques recommandées (imputation, interpolation, suppression).
   - Risques associés : Effets de la stratégie choisie sur le modèle ou les analyses.

2. **Doublons**
   - Détection : Identification des répétitions d'horodatage ou de mesures identiques.
   - Traitement : Stratégie de déduplication ou d'agrégation.
   - Risques associés : Perte d'information critique ou distorsion des fréquences d'échantillonnage.

3. **Valeurs aberrantes (Outliers)**
   - Détection : Méthodes statistiques ou algorithmiques (Z-score, IQR, Isolation Forest, seuils physiques).
   - Traitement : Stratégies de filtrage, écrestage ou remplacement.
   - Risques associés : Masquage de vrais signaux d'alerte ou conservation de bruits de mesure.

4. **Variables catégorielles**
   - Détection : Repérage des variables qualitatives (ex: état du capteur, localisation, modèle).
   - Traitement : Encodage adapté (One-Hot Encoding, Ordinal Encoding, Target Encoding).
   - Risques associés : Explosion de la dimensionnalité ou création de relations d'ordre artificielles.

### FORMAT DE SORTIE
Structure la réponse de manière claire avec des titres explicites et des points à puces pour chaque section."""

    res = appeler_gemini(prompt)
    strategie = res.text.strip()

    print(f"[Description du dataset fourni] :\n{description_dataset}\n")
    print("[Stratégie proposée] :")
    print(strategie)
    print("\n--------------------------------------------------")

    return strategie


if __name__ == "__main__":
    dataset_capteurs_info = """
    Dataset télémetrique de 50 capteurs d'une chaîne d'assemblage industrielle.
    Variables enregistrées toutes les secondes :
    - timestamp (datetime)
    - sensor_id (string : ex: "SENS_001")
    - temperature (float : °C)
    - pression (float : bar)
    - vitesse_rotation (float : RPM)
    - status_capteur (string : "OK", "WARNING", "ERROR", "CALIBRATION")
    - emplacement (string : "Zone_A", "Zone_B", "Zone_C")
    """

    question_strategie_nettoyage_capteurs(dataset_capteurs_info)