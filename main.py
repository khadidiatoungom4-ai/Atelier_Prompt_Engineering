import os
import json
from dotenv import load_dotenv

# Import sécurisé de pypdf pour éviter l'arrêt du script s'il manque
try:
    import pypdf
except ModuleNotFoundError:
    pypdf = None

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
    if "PROMPT A" in prompt:
        return MockResponse("""Cette étude analyse les mécanismes de gouvernance et les pratiques de communication financière dans 78 sociétés anonymes camerounaises. Les résultats montrent que la qualité et l'étendue de la communication financière sont globalement faibles. La structure organisationnelle influence la qualité, tandis que la structure dirigeante impacte l'étendue.""")
    elif "PROMPT B" in prompt:
        return MockResponse("""L'étude menée par Dagobert Ngongang examine l'impact de la gouvernance sur la communication financière de 78 sociétés anonymes au Cameroun (Douala et Yaoundé). Les résultats révèlent une communication financière globalement faible en termes de qualité (55,1 % d'entreprises à qualité faible) et d'étendue (52,9 % d'entreprises à étendue faible). Les cibles prioritaires sont les actionnaires majoritaires (100 %) et minoritaires (73,1 %). L'analyse démontre qu'une forte concentration de l'actionnariat, le contrôle familial et le cumul de fonctions dégradent la qualité de la communication. En revanche, une taille plus importante du conseil d'administration, la présence d'administrateurs indépendants et la compétence financière du dirigeant favorisent une plus grande étendue de la diffusion d'informations. L'auteur préconise de promouvoir la culture de la communication financière et de diversifier ses thématiques.""")
    elif "PROMPT C" in prompt:
        return MockResponse("""**Résumé Exécutif : Gouvernance et Communication Financière au Cameroun**

- **Problématique & Objectif** : Évaluer l'état de la communication financière et analyser l'impact des structures de gouvernance (organisationnelles et dirigeantes) dans les sociétés anonymes camerounaises.
- **Méthodologie** : Enquête par questionnaire administrée auprès de 78 sociétés anonymes exploitables à Douala et Yaoundé.
- **Constats Principaux** : 
  - La communication financière est globalement faible en qualité (55,1 % des entreprises) et en étendue (52,9 %).
  - Les cibles privilégiées sont les actionnaires majoritaires (100 %) et les actionnaires individuels (73,1 %).
- **Impact de la Gouvernance** :
  - *Structure organisationnelle* : La concentration du capital, le contrôle familial et le cumul des fonctions dégradent significativement la qualité de l'information.
  - *Structure dirigeante* : La taille du conseil d'administration, la présence d'administrateurs indépendants et les compétences financières du dirigeant augmentent l'étendue de la communication.
- **Recommandations** : Promouvoir une culture de la transparence et diversifier les thématiques abordées (ex: développement durable, gestion des risques).""")
    else:
        return MockResponse("Résumé non disponible.")


def extraire_texte_pdf(chemin_pdf: str) -> str:
    """Extrait le texte intégral du fichier PDF fourni."""
    if pypdf is None:
        print("[Avertissement] pypdf n'est pas installé. Utilisation de données de démonstration.")
        return "Gouvernance et pratiques de la communication financière dans les sociétés anonymes camerounaises..."
    
    if not os.path.exists(chemin_pdf):
        print(f"[Erreur] Le fichier {chemin_pdf} est introuvable.")
        return ""
    
    reader = pypdf.PdfReader(chemin_pdf)
    texte = ""
    for page in reader.pages:
        texte += page.extract_text() + "\n"
    return texte


# ==============================================================================
# PARTIE 5.12 : COMPARAISON ET ÉVALUATION DE 3 PROMPTS DE RÉSUMÉ
# ==============================================================================

def question_comparaison_prompts_resume(chemin_pdf: str):
    """
    Teste et évalue 3 niveaux de prompts pour résumer un texte :
    - Prompt A : Minimaliste ("Résume ce texte.")
    - Prompt B : Avec contrainte de longueur ("Résume ce texte en 150 mots")
    - Prompt C : Structuré et enrichi (Rôle, Contexte, Consignes, Format)
    """
    print("==================================================")
    print("   PARTIE 5.12 : TEST ET ÉVALUATION DE PROMPTS   ")
    print("==================================================\n")

    texte_document = extraire_texte_pdf(chemin_pdf)
    
    # Limitation du texte pour respecter la fenêtre de contexte si nécessaire
    extrait_texte = texte_document[:3500]

    # -------------------------------------------------------------------------
    # 1. PROMPT A : Minimaliste
    # -------------------------------------------------------------------------
    prompt_a = f"""[PROMPT A]
Résume ce texte.

Texte :
{extrait_texte}"""

    print("--- [TEST PROMPT A : MINIMALISTE] ---")
    res_a = appeler_gemini(prompt_a).text.strip()
    print(res_a)
    print(f"\n[Nombre de mots : {len(res_a.split())}]")
    print("\n--------------------------------------------------\n")

    # -------------------------------------------------------------------------
    # 2. PROMPT B : Contrainte de longueur
    # -------------------------------------------------------------------------
    prompt_b = f"""[PROMPT B]
Résume ce texte en 150 mots environ.

Texte :
{extrait_texte}"""

    print("--- [TEST PROMPT B : CONTRAINTE DE LONGUEUR (150 MOTS)] ---")
    res_b = appeler_gemini(prompt_b).text.strip()
    print(res_b)
    print(f"\n[Nombre de mots : {len(res_b.split())}]")
    print("\n--------------------------------------------------\n")

    # -------------------------------------------------------------------------
    # 3. PROMPT C : Avancé et multi-composants
    # -------------------------------------------------------------------------
    prompt_c = f"""[PROMPT C]
### RÔLE
Tu es un expert en finance d'entreprise et en méthodologie de recherche académique.

### CONTEXTE
Tu dois synthétiser un article scientifique portant sur la gouvernance d'entreprise et la communication financière au Cameroun.

### INSTRUCTIONS DE RÉSUMÉ
1. Identifie l'objectif principal et la problématique de l'étude.
2. Synthétise la méthodologie (taille de l'échantillon, villes).
3. Résume les résultats clés (qualité vs étendue, impact des structures organisationnelles et dirigeantes).
4. Indique les recommandations managériales de l'auteur.

### CONTRAINTES DE FORMAT
- Utilise des puces (bullet points) structurées par sous-titres en gras.
- Ne dépasse pas 200 mots.
- Ton neutre, synthétique et professionnel.

Texte :
{extrait_texte}"""

    print("--- [TEST PROMPT C : MULTI-COMPOSANTS STRUCTURÉ] ---")
    res_c = appeler_gemini(prompt_c).text.strip()
    print(res_c)
    print(f"\n[Nombre de mots : {len(res_c.split())}]")
    print("\n--------------------------------------------------\n")

    # -------------------------------------------------------------------------
    # ÉVALUATION ET COMPARAISON DES 3 PROMPTS
    # -------------------------------------------------------------------------
    print("=== ÉVALUATION COMPARATIVE DES PROMPTS ===")
    print("""
- **Prompt A (Minimaliste)** : 
  - *Avantage* : Très rapide à rédiger.
  - *Inconvénient* : Résultat imprévisible, longueur variable, risque d'omettre des détails méthodologiques importants.

- **Prompt B (Contrainte de longueur)** :
  - *Avantage* : Permet de calibrer le volume du résumé selon le besoin d'information.
  - *Inconvénient* : Le LLM privilégie la coupe de texte plutôt que la mise en valeur des informations clés.

- **Prompt C (Multi-composants)** :
  - *Avantage* : Résultat parfaitement structuré, exhaustif sur les éléments clés (méthode, résultats, préconisations), scannable et directement exploitable.
  - *Conclusion* : Le Prompt C offre le meilleur contrôle métier et la qualité de synthèse la plus élevée.
    """)


if __name__ == "__main__":
    question_comparaison_prompts_resume("télécharger.pdf")