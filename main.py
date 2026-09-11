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
    if "EXTRACTION DE DONNÉES DE FACTURE" in prompt:
        mock_facture = {
            "numéro_facture": "FAC-2026-0042",
            "date": "2026-09-10",
            "client": "Société West Africa Retail",
            "montant_ht": 1500000.0,
            "tva": 270000.0,
            "montant_ttc": 1770000.0
        }
        return MockResponse(json.dumps(mock_facture, ensure_ascii=False, indent=2))
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
# PARTIE 5.4 : EXTRACTION DE DONNÉES DE FACTURE (JSON)
# ==============================================================================

def question_extraction_facture(texte_facture: str) -> dict:
    """
    Extrait les données clés d'une facture sous forme de JSON strict.
    Utilise 'null' pour toute valeur manquante.
    """
    print("==================================================")
    print("   PARTIE 5.4 : EXTRACTION DE FACTURE (JSON)      ")
    print("==================================================\n")

    prompt = f"""### TÂCHE : EXTRACTION DE DONNÉES DE FACTURE
Analyse le texte de la facture ci-dessous et extrais les informations clés sous forme de JSON strict.

### TEXTE DE LA FACTURE
"{texte_facture}"

### CHAMPS À EXTRAIRE
Tu dois extraire EXACTEMENT les clés suivantes :
1. "numéro_facture" : Le numéro ou la référence de la facture (string ou null).
2. "date"           : La date d'émission de la facture (string ou null).
3. "client"         : Le nom ou la raison sociale du client (string ou null).
4. "montant_ht"     : Le montant hors taxes (nombre ou null).
5. "tva"            : Le montant de la TVA (nombre ou null).
6. "montant_ttc"    : Le montant toutes taxes comprises (nombre ou null).

### RÈGLES STRICTES
- Si une information n'est pas présente ou incertaine dans la facture, attribue-lui la valeur JSON `null`.
- Conserve le format numérique (float/int) pour `montant_ht`, `tva` et `montant_ttc` si disponibles.
- Renvoie UNIQUEMENT l'objet JSON valide, sans aucune phrase d'introduction ni de conclusion.
- N'utilise pas de balises Markdown (pas de ```json)."""

    res = appeler_gemini(prompt)
    json_brut = res.text.strip()

    print(f"[Texte Facture Analysé] :\n{texte_facture}\n")
    print("[Réponse brute du modèle] :")
    print(json_brut)
    print("\n--------------------------------------------------")

    # Parsing et validation
    try:
        cleaned_str = json_brut.replace("```json", "").replace("```", "").strip()
        donnees = json.loads(cleaned_str)

        champs_attendus = {"numéro_facture", "date", "client", "montant_ht", "tva", "montant_ttc"}
        if set(donnees.keys()) == champs_attendus:
            print("[SUCCÈS] JSON extrait avec la structure exacte requise :\n")
            for k, v in donnees.items():
                print(f" - {k} : {v}")
        else:
            print(f"[ATTENTION] Clés différentes de la structure attendue : {list(donnees.keys())}")

        return donnees

    except json.JSONDecodeError as e:
        print(f"[ERREUR] Parsing JSON impossible : {e}")
        return {}


if __name__ == "__main__":
    facture_exemple = """
    FACTURE N° FAC-2026-0042
    Date : 10/09/2026
    Client : Société West Africa Retail
    
    Désignation : Prestation d'analyse de données financières
    Montant HT : 1 500 000 XOF
    TVA (18%) : 270 000 XOF
    Montant TTC : 1 770 000 XOF
    """

    question_extraction_facture(facture_exemple)