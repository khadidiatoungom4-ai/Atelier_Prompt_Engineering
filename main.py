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
    if "CLASSIFICATION TICKET INFORMATIQUE" in prompt:
        mock_it = {
            "categorie": "accès",
            "justification": "Le ticket mentionne l'impossibilité de se connecter au VPN avec un message de mot de passe expiré."
        }
        return MockResponse(json.dumps(mock_it, ensure_ascii=False, indent=2))
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
# PARTIE 5.3 : CLASSIFICATION DE TICKETS INFORMATIQUES (JSON)
# ==============================================================================

def question_classification_ticket_it(ticket_texte: str) -> dict:
    """
    Classe un ticket informatique dans une catégorie prédéfinie et fournit
    une justification sous forme de JSON strict.
    """
    print("==================================================")
    print("   PARTIE 5.3 : CLASSIFICATION DE TICKET IT (JSON)")
    print("==================================================\n")

    prompt = f"""### TÂCHE : CLASSIFICATION TICKET INFORMATIQUE
Analyse le ticket support informatique ci-dessous, attribue-lui la catégorie adéquate et fournit une brève justification.

### TICKET INFORMATIQUE
"{ticket_texte}"

### RÈGLES DE CATÉGORISATION
Champ "categorie" : Choisis STRICTEMENT une seule valeur parmi la liste autorisée :
- "réseau"     : Problèmes Wi-Fi, VPN, lenteurs internet, connexions serveurs.
- "logiciel"   : Bugs d'applications, plantages d'outils, mises à jour, erreurs système.
- "matériel"   : Écran cassé, PC qui ne s'allume pas, imprimante, périphérique en panne.
- "sécurité"   : Attaque de phishing, virus, comportement suspect, fuite de données.
- "accès"      : Réinitialisation de mot de passe, création de compte, droits d'accès.
- "autre"      : Demandes non couvertes par les catégories ci-dessus.

### FORMAT DE SORTIE EXIGÉ
Tu dois répondre UNIQUEMENT avec un objet JSON valide contenant EXACTEMENT les deux champs suivants :
- "categorie"     : (string) Une des 6 valeurs autorisées ci-dessus.
- "justification" : (string) Explication concise (1 à 2 phrases max) du choix de la catégorie.

### CONTRAINTES STRICTES
- Aucune propriété supplémentaire dans le JSON.
- Aucun texte d'introduction ou de conclusion.
- Aucune balise Markdown (ne pas mettre de ```json)."""

    res = appeler_gemini(prompt)
    json_brut = res.text.strip()

    print(f"[Ticket à analyser] : {ticket_texte}\n")
    print("[Réponse brute du modèle] :")
    print(json_brut)
    print("\n--------------------------------------------------")

    # Parsing et validation en Python
    try:
        cleaned_str = json_brut.replace("```json", "").replace("```", "").strip()
        donnees = json.loads(cleaned_str)

        categories_autorisees = {"réseau", "logiciel", "matériel", "sécurité", "accès", "autre"}
        cat_obtenue = donnees.get("categorie")

        if cat_obtenue in categories_autorisees:
            print("[SUCCÈS] JSON valide et catégorie conforme :")
            print(f" - Catégorie    : {cat_obtenue}")
            print(f" - Justification: {donnees.get('justification')}")
        else:
            print(f"[ERREUR] Catégorie non autorisée reçue : '{cat_obtenue}'")

        return donnees

    except json.JSONDecodeError as e:
        print(f"[ERREUR] Échec du parsing JSON : {e}")
        return {}


if __name__ == "__main__":
    ticket_exemple = "Je n'arrive plus à accéder au serveur de l'entreprise depuis ce matin, ma session affiche 'mot de passe expiré'."
    question_classification_ticket_it(ticket_exemple)