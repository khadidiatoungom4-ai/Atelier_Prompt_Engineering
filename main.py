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
    if "RÉDACTION D'EMAIL CLIENT" in prompt or "retard de livraison" in prompt:
        return MockResponse("""Objet : Information concernant la livraison de votre commande

Bonjour,

Nous tenons à vous informer que la livraison de votre commande subit un retard par rapport au délai initialement prévu. Nous vous présentons nos plus sincères excuses pour cette gêne occasionnée.

Ce contretemps est lié à un ralentissement imprévu dans l'acheminement logistique de votre colis. Nos équipes suivent la situation de très près afin de débloquer l'envoi dans les meilleurs délais.

Afin de vous assurer un suivi optimal, nous vous offrons les frais de livraison sur cette commande et vous transmettrons un nouveau lien de suivi dès demain matin.

Nous vous remercions pour votre compréhension et restons à votre entière disposition.

Cordialement,
Le Service Client""")
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
# PARTIE 5.5 : RÉDACTION D'EMAIL RETARD DE LIVRAISON
# ==============================================================================

def question_email_retard_livraison(nom_client: str, numero_commande: str) -> str:
    """
    Génère un email professionnel pour informer un client d'un retard de livraison
    en respectant les contraintes de contenu, de ton et de longueur.
    """
    print("==================================================")
    print("   PARTIE 5.5 : EMAIL CLIENT - RETARD LIVRAISON   ")
    print("==================================================\n")

    prompt = f"""### TÂCHE : RÉDACTION D'EMAIL CLIENT
Rédige un courriel professionnel destiné au client "{nom_client}" concernant le retard de livraison de sa commande n° "{numero_commande}".

### OBJECTIFS OBLIGATOIRES À REMPLIR
1. Reconnaître explicitement le retard de la livraison.
2. Présenter des excuses sincères au client.
3. Expliquer brièvement la situation SANS inventer de cause fictive ou de détails non vérifiables (mentionner un retard d'acheminement logistique).
4. Proposer une solution concrète (suivi prioritaire, geste commercial ou assistance dédiée).

### TON ET STYLE EXIGÉS
- Ton : Professionnel, courtois et rassurant.
- Style : Empathique et orienté solution.

### CONTRAINTES DE LONGUEUR ET FORMAT
- Longueur : 150 mots MAXIMUM pour l'ensemble de l'email.
- Structure : Un objet de mail, une salutation, le corps du texte et une formule de politesse.
- Réponds directement en français avec le texte de l'email."""

    res = appeler_gemini(prompt)
    email_redige = res.text.strip()

    print(f"[Informations Client] : {nom_client} | Commande n° {numero_commande}\n")
    print("[Email Généré] :")
    print(email_redige)
    print("\n--------------------------------------------------")

    # Contrôle applicatif du nombre de mots
    nb_mots = len(email_redige.split())
    print(f"[Contrôle Longueur] : {nb_mots} mots (Limite : 150 mots max)")

    return email_redige


if __name__ == "__main__":
    question_email_retard_livraison("Mme Diallo", "CMD-2026-8891")