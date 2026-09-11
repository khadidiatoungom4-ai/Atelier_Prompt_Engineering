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
    if "VISUALISATIONS ENERGÉTIQUES" in prompt or "consommation énergétique" in prompt:
        return MockResponse("""### PROPOSITION DE VISUALISATIONS ÉNERGÉTIQUES

1. **Courbe de charge temporelle (TimeSeries Line Chart)**
   - **Type de graphique** : Graphique linéaire temporel.
   - **Variables utilisées** : X = `timestamp` (heure/jour), Y = `puissance_kw` ou `consommation_kwh`.
   - **Objectif** : Analyser le profil de consommation au cours du temps et identifier les pics de charge.
   - **Interprétation attendue** : Visualiser les périodes de pointe (ex: 09h-18h), vérifier le talon de consommation nocturne et détecter les anomalies de fonctionnement hors heures ouvrées.

2. **Carte thermique temporelle (Heatmap Heure x Jour)**
   - **Type de graphique** : Heatmap (Carte de chaleur 2D).
   - **Variables utilisées** : X = `heure_de_la_journée` (0 à 23h), Y = `jour_de_la_semaine` (Lundi au Dimanche), Couleur = `consommation_kwh`.
   - **Objectif** : Identifier les motifs récurrents de surconsommation selon l'heure et le jour.
   - **Interprétation attendue** : Repérer les dérives de consommation durant le week-end ou la nuit et cibler les plages horaires nécessitant une régulation CVC (chauffage/climatisation).

3. **Nuage de points de régression (Scatter Plot avec ligne de tendance)**
   - **Type de graphique** : Nuage de points (Scatter Plot).
   - **Variables utilisées** : X = `temperature_exterieure_C`, Y = `consommation_kwh`, Couleur/Taille = `humidite_pct` ou `zone_batiment`.
   - **Objectif** : Évaluer la sensibilité de la consommation aux conditions météorologiques extérieures (Courbe en V / Signature énergétique).
   - **Interprétation attendue** : Déterminer la température de neutralité thermique du bâtiment et mesurer la performance de l'isolation thermique.

4. **Diagramme en barres empilées par zone/usage (Stacked Bar Chart)**
   - **Type de graphique** : Diagramme en barres empilées.
   - **Variables utilisées** : X = `mois` ou `étage`, Y = `consommation_kwh`, Empilement = `type_usage` (Éclairage, CVC, Prises, Ascenseurs).
   - **Objectif** : Décomposer la consommation globale par poste de dépense et par zone.
   - **Interprétation attendue** : Identifier les équipements ou zones les plus budgétivores pour prioriser les actions d'efficacité énergétique.""")
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
# PARTIE 5.7 : VISUALISATIONS DE LA CONSOMMATION ÉNERGÉTIQUE
# ==============================================================================

def question_visualisations_energetiques(description_dataset: str) -> str:
    """
    Propose un ensemble de visualisations clés pour analyser la consommation
    énergétique d'un bâtiment à partir des données de capteurs.
    """
    print("==================================================")
    print("   PARTIE 5.7 : VISUALISATIONS CONSOMMATION ÉNERGÉTIQUE ")
    print("==================================================\n")

    prompt = f"""### TÂCHE : VISUALISATIONS ÉNERGÉTIQUES BÂTIMENT
En te basant sur la description du dataset de capteurs ci-dessous, propose les visualisations les plus pertinentes pour comprendre et analyser la consommation énergétique du bâtiment.

### DESCRIPTION DU DATASET CAPTEURS
"{description_dataset}"

### EXIGENCES DE CONTENU
Pour CHAQUE visualisation proposée (propose 3 à 5 visualisations clés), tu dois systématiquement fournir :
1. **Type de graphique** : Le type exact de diagramme/chart recommandé (ex: Heatmap, Scatter plot, Line chart, etc.).
2. **Variables utilisées** : Les variables du dataset associées aux axes (X, Y, couleurs, filtres).
3. **Objectif** : Le but analytique ou la question métier à laquelle répond la visualisation.
4. **Interprétation attendue** : Ce que la visualisation permet d'observer, de déduire ou de diagnostiquer concrètement pour l'efficacité énergétique.

### FORMAT DE SORTIE
Structure la réponse de manière claire avec une section numérotée pour chaque visualisation et des puces d'explication."""

    res = appeler_gemini(prompt)
    recommandations = res.text.strip()

    print(f"[Description du dataset] :\n{description_dataset}\n")
    print("[Recommandations de visualisations] :")
    print(recommandations)
    print("\n--------------------------------------------------")

    return recommandations


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

    question_visualisations_energetiques(dataset_energie_info)