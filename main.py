import os
import time
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
    if "15 mots maximum" in prompt:
        return MockResponse("Échec intermittent du paiement par carte Visa lors de la transaction.")
    elif "GRILLE D'ÉVALUATION OBLIGATOIRE" in prompt:
        return MockResponse("""- Statut global : VALIDE
- Détail du contrôle :
  - Informations non justifiées : SANS ANOMALIE
  - Contradictions : SANS ANOMALIE
  - Informations absentes : SANS ANOMALIE
  - Hallucinations : AUCUNE
  - Respect des contraintes : RESPECTÉ
- Conclusion : La réponse est conforme.""")
    else:
        mock_json = {
            "sentiment": "negatif",
            "categorie": "livraison",
            "urgence": "moyenne",
            "probleme": "Retard de livraison",
            "confiance": 0.91
        }
        return MockResponse(json.dumps(mock_json, ensure_ascii=False, indent=2))


def valider_reponse_json(donnees: dict) -> tuple[bool, list[str]]:
    """Vérifie le respect strict des règles métiers sur le dictionnaire JSON."""
    erreurs = []
    champs_autorises = {"sentiment", "categorie", "urgence", "probleme", "confiance"}

    # 1. Propriétés supplémentaires
    champs_extra = set(donnees.keys()) - champs_autorises
    if champs_extra:
        erreurs.append(f"Propriétés non autorisées : {champs_extra}")

    # 2. Validation du sentiment
    sentiments_valides = {"positif", "negatif", "neutre"}
    if donnees.get("sentiment") not in sentiments_valides:
        erreurs.append(f"Sentiment invalide : '{donnees.get('sentiment')}'")

    # 3. Validation de l'urgence
    urgences_valides = {"faible", "moyenne", "élevée"}
    if donnees.get("urgence") not in urgences_valides:
        erreurs.append(f"Urgence invalide : '{donnees.get('urgence')}'")

    # 4. Validation de la confiance
    confiance = donnees.get("confiance")
    if not isinstance(confiance, (int, float)) or not (0.0 <= confiance <= 1.0):
        erreurs.append(f"Confiance invalide : '{confiance}' (doit être un float entre 0 et 1)")

    return len(erreurs) == 0, erreurs


def question_structuration_avec_validation(commentaire_client: str):
    """Analyse un commentaire et valide le JSON selon des règles strictes."""
    print("==================================================")
    print("   QUESTION : STRUCTURATION ET VALIDATION JSON   ")
    print("==================================================\n")

    prompt = f"""### TÂCHE
Analyse le commentaire client ci-dessous et extrait les informations au format JSON STRICT.

### COMMENTAIRE CLIENT
"{commentaire_client}"

### SPÉCIFICATION DU FORMAT ET RÈGLES DE SORTIE
Tu dois répondre UNIQUEMENT avec un objet JSON respectant STRICTEMENT les règles suivantes :

1. Format JSON valide.
2. AUCUNE propriété supplémentaire que les 5 clés spécifiées ci-dessous.
3. "sentiment" : Valeurs autorisées uniquement : "positif", "negatif", "neutre".
4. "urgence"   : Valeurs autorisées uniquement : "faible", "moyenne", "élevée".
5. "confiance" : Nombre flottant obligatoirement compris entre 0.0 et 1.0.
6. "categorie" : Exemples ("livraison", "produit", "service_client", "paiement").
7. "probleme"  : Description succincte du problème (chaîne de caractères).

### CONTRAINTES STRICTES
- Pas de texte explicatif avant ou après.
- Pas de balises Markdown (ne pas utiliser ```json)."""

    res = appeler_gemini(prompt)
    json_brut = res.text.strip()

    print(f"Commentaire analysé : {commentaire_client}\n")
    print("[Réponse brute du modèle] :")
    print(json_brut)
    print("\n--------------------------------------------------")

    try:
        cleaned_str = json_brut.replace("```json", "").replace("```", "").strip()
        donnees = json.loads(cleaned_str)
        print("[Étape 1] Format JSON valide : OK")
    except json.JSONDecodeError as e:
        print(f"[Étape 1 ERREUR] Parsing JSON impossible : {e}")
        return {}

    est_valide, erreurs = valider_reponse_json(donnees)
    if est_valide:
        print("[Étape 2] Validation des règles métier : VALIDE\n")
        for k, v in donnees.items():
            print(f" - {k} : {v}")
    else:
        print("[Étape 2 ERREUR] Règles non respectées :")
        for err in erreurs:
            print(f"   ❌ {err}")

    return donnees


if __name__ == "__main__":
    commentaire = "Le commentaire semble plutôt négatif. Le client est mécontent du délai de livraison..."
    question_structuration_avec_validation(commentaire)