from flask import Flask, request, jsonify
from flask_cors import CORS
import unicodedata
import traceback
import re
import os
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

MODEL_NAME = "HuggingFace Inference API - paraphrase-multilingual-MiniLM-L12-v2"
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/pipeline/feature-extraction"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

MAX_RESULTS = 8


def get_embedding(text):
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN manquant dans Render Environment Variables")

    payload = {"inputs": str(text or "")}

    response = requests.post(
        API_URL,
        headers=HEADERS,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Erreur Hugging Face: {response.status_code} - {response.text}"
        )

    data = response.json()

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (int, float)):
        return data

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        if len(data[0]) > 0 and isinstance(data[0][0], list):
            return np.mean(np.array(data[0]), axis=0).tolist()
        return data[0]

    raise RuntimeError(f"Format embedding invalide: {data}")


def cosine_score(text1, text2):
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    return float(cosine_similarity([emb1], [emb2])[0][0])


def normalize_text(value):
    value = str(value or "").lower().strip()
    value = unicodedata.normalize("NFD", value)
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    value = re.sub(r"[^a-z0-9+#. ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_skill_dict(skills):
    result = {}
    for skill, level in (skills or {}).items():
        skill_norm = normalize_text(skill)
        try:
            level_int = int(level)
        except Exception:
            level_int = 0
        if skill_norm:
            result[skill_norm] = max(0, min(level_int, 5))
    return result


def text_contains_any(text, words):
    text_norm = normalize_text(text)
    return any(normalize_text(word) in text_norm for word in words)


def extract_formation_title(value):
    """
    Accepte formationsSuivies sous plusieurs formats:
    - "Leadership et management"
    - {"titre": "..."}
    - {"title": "..."}
    - {"formation": "..."}
    - {"formation": {"titre": "..."}}
    """
    if isinstance(value, dict):
        nested = value.get("formation")
        if isinstance(nested, dict):
            return (
                nested.get("titre")
                or nested.get("title")
                or nested.get("nom")
                or nested.get("name")
                or ""
            )

        return (
            value.get("titre")
            or value.get("title")
            or value.get("nom")
            or value.get("name")
            or value.get("formation")
            or value.get("formationTitre")
            or value.get("formationTitle")
            or ""
        )

    return value


VIDEO_LIBRARY = {
    "communication": [
        {"titre": "Communication professionnelle", "urlYoutube": "https://www.youtube.com/watch?v=HAnw168huqA", "ordre": 1},
        {"titre": "Communication efficace au travail", "urlYoutube": "https://www.youtube.com/watch?v=8sjA90hvnQ0", "ordre": 2},
        {"titre": "Améliorer sa communication professionnelle", "urlYoutube": "https://www.youtube.com/watch?v=5yK_E3Yq5jA", "ordre": 3}
    ],
    "leadership": [
        {"titre": "Leadership", "urlYoutube": "https://www.youtube.com/watch?v=ktlTxC4QG8g", "ordre": 1},
        {"titre": "Gestion équipe", "urlYoutube": "https://www.youtube.com/watch?v=4a0FbQdH3dY", "ordre": 2},
        {"titre": "Management et leadership", "urlYoutube": "https://www.youtube.com/watch?v=Q2vQkHjS4xQ", "ordre": 3}
    ],
    "management": [
        {"titre": "Management d'équipe", "urlYoutube": "https://www.youtube.com/watch?v=Q2vQkHjS4xQ", "ordre": 1},
        {"titre": "Leadership", "urlYoutube": "https://www.youtube.com/watch?v=ktlTxC4QG8g", "ordre": 2},
        {"titre": "Gestion équipe", "urlYoutube": "https://www.youtube.com/watch?v=4a0FbQdH3dY", "ordre": 3}
    ],
    "recrutement": [
        {"titre": "Recrutement RH", "urlYoutube": "https://www.youtube.com/watch?v=HG68Ymazo18", "ordre": 1},
        {"titre": "Entretien de recrutement", "urlYoutube": "https://www.youtube.com/watch?v=6G8_qA8M8pQ", "ordre": 2},
        {"titre": "Sourcing candidats", "urlYoutube": "https://www.youtube.com/watch?v=4FQY3u4UxS0", "ordre": 3}
    ],
    "paie": [
        {"titre": "Gestion de la paie", "urlYoutube": "https://www.youtube.com/watch?v=b7OXULhF1pc", "ordre": 1},
        {"titre": "Bulletin de paie expliqué", "urlYoutube": "https://www.youtube.com/watch?v=zE51pYOTp2s", "ordre": 2},
        {"titre": "Charges sociales et salaire net", "urlYoutube": "https://www.youtube.com/watch?v=5cI-AkKy66I", "ordre": 3}
    ],
    "droit du travail": [
        {"titre": "Droit du travail", "urlYoutube": "https://www.youtube.com/watch?v=4Ko4b38N7gE", "ordre": 1},
        {"titre": "Contrat de travail", "urlYoutube": "https://www.youtube.com/watch?v=O_4LwZ2pJzQ", "ordre": 2},
        {"titre": "Droit social RH", "urlYoutube": "https://www.youtube.com/watch?v=R6NoL7cnkQY", "ordre": 3}
    ],
    "default": [
        {"titre": "Formation professionnelle", "urlYoutube": "https://www.youtube.com/watch?v=HAnw168huqA", "ordre": 1},
        {"titre": "Communication au travail", "urlYoutube": "https://www.youtube.com/watch?v=8sjA90hvnQ0", "ordre": 2},
        {"titre": "Développement des compétences", "urlYoutube": "https://www.youtube.com/watch?v=Q2vQkHjS4xQ", "ordre": 3}
    ]
}

# ─────────────────────────────────────────────────────────────
#  FORMATIONS EXTERNES — même structure que les formations internes
#  Utilisées quand le catalogue interne est épuisé
# ─────────────────────────────────────────────────────────────
FORMATIONS_EXTERNES = {
    "RH": [
        {
            "formationId": None,
            "formation": "Certification RH professionnelle",
            "title": "Certification RH professionnelle",
            "provider": "Coursera",
            "url": "https://www.coursera.org/search?query=ressources+humaines",
            "description": "Formation complète en gestion des ressources humaines : recrutement, paie, droit du travail.",
            "level": "Intermédiaire",
            "score": 0.80,
            "semanticScore": 0.80,
            "skillScore": 0.75,
            "priorityScore": 0.70,
            "type": "EXTERNE",
            "matchedSkills": ["recrutement", "rh", "gestion du personnel"],
            "reason": "Formation externe recommandée car le catalogue interne est épuisé.",
            "videos": [
                {"titre": "Introduction RH", "urlYoutube": "https://www.youtube.com/watch?v=HAnw168huqA", "ordre": 1},
                {"titre": "Recrutement RH", "urlYoutube": "https://www.youtube.com/watch?v=HG68Ymazo18", "ordre": 2}
            ]
        },
        {
            "formationId": None,
            "formation": "Recrutement et sourcing avancé",
            "title": "Recrutement et sourcing avancé",
            "provider": "LinkedIn Learning",
            "url": "https://www.linkedin.com/learning/search?keywords=recrutement",
            "description": "Maîtrisez les techniques modernes de recrutement et de sourcing de candidats.",
            "level": "Avancé",
            "score": 0.75,
            "semanticScore": 0.75,
            "skillScore": 0.70,
            "priorityScore": 0.65,
            "type": "EXTERNE",
            "matchedSkills": ["recrutement", "sourcing", "talent acquisition"],
            "reason": "Formation externe recommandée car le catalogue interne est épuisé.",
            "videos": [
                {"titre": "Techniques de recrutement", "urlYoutube": "https://www.youtube.com/watch?v=HG68Ymazo18", "ordre": 1},
                {"titre": "Sourcing candidats", "urlYoutube": "https://www.youtube.com/watch?v=4FQY3u4UxS0", "ordre": 2}
            ]
        },
        {
            "formationId": None,
            "formation": "Droit du travail et gestion sociale",
            "title": "Droit du travail et gestion sociale",
            "provider": "Udemy",
            "url": "https://www.udemy.com/courses/search/?q=droit+du+travail",
            "description": "Comprendre le cadre légal du droit du travail et la gestion administrative du personnel.",
            "level": "Débutant",
            "score": 0.72,
            "semanticScore": 0.72,
            "skillScore": 0.68,
            "priorityScore": 0.60,
            "type": "EXTERNE",
            "matchedSkills": ["droit du travail", "administration du personnel"],
            "reason": "Formation externe recommandée car le catalogue interne est épuisé.",
            "videos": [
                {"titre": "Droit du travail", "urlYoutube": "https://www.youtube.com/watch?v=4Ko4b38N7gE", "ordre": 1},
                {"titre": "Contrat de travail", "urlYoutube": "https://www.youtube.com/watch?v=O_4LwZ2pJzQ", "ordre": 2}
            ]
        }
    ],
    "IT": [
        {
            "formationId": None,
            "formation": "Spring Boot & Microservices",
            "title": "Spring Boot & Microservices",
            "provider": "Udemy",
            "url": "https://www.udemy.com/courses/search/?q=spring+boot",
            "description": "Développez des applications backend robustes avec Spring Boot et l'architecture microservices.",
            "level": "Avancé",
            "score": 0.82,
            "semanticScore": 0.82,
            "skillScore": 0.78,
            "priorityScore": 0.75,
            "type": "EXTERNE",
            "matchedSkills": ["spring", "java", "backend", "microservices"],
            "reason": "Formation externe recommandée car le catalogue interne est épuisé.",
            "videos": [
                {"titre": "Spring Boot tutorial", "urlYoutube": "https://www.youtube.com/watch?v=9SGDpanrc8U", "ordre": 1}
            ]
        },
        {
            "formationId": None,
            "formation": "Angular - Développement Frontend",
            "title": "Angular - Développement Frontend",
            "provider": "Coursera",
            "url": "https://www.coursera.org/search?query=angular",
            "description": "Maîtrisez Angular pour créer des interfaces web modernes et réactives.",
            "level": "Intermédiaire",
            "score": 0.78,
            "semanticScore": 0.78,
            "skillScore": 0.74,
            "priorityScore": 0.70,
            "type": "EXTERNE",
            "matchedSkills": ["angular", "typescript", "frontend"],
            "reason": "Formation externe recommandée car le catalogue interne est épuisé.",
            "videos": [
                {"titre": "Angular crash course", "urlYoutube": "https://www.youtube.com/watch?v=3dHNOWTI7H8", "ordre": 1}
            ]
        }
    ],
    "MANAGEMENT": [
        {
            "formationId": None,
            "formation": "Leadership et management d'équipe",
            "title": "Leadership et management d'équipe",
            "provider": "LinkedIn Learning",
            "url": "https://www.linkedin.com/learning/search?keywords=management",
            "description": "Développez vos compétences en leadership, gestion d'équipe et communication managériale.",
            "level": "Intermédiaire",
            "score": 0.77,
            "semanticScore": 0.77,
            "skillScore": 0.72,
            "priorityScore": 0.68,
            "type": "EXTERNE",
            "matchedSkills": ["leadership", "management", "communication"],
            "reason": "Formation externe recommandée car le catalogue interne est épuisé.",
            "videos": [
                {"titre": "Leadership", "urlYoutube": "https://www.youtube.com/watch?v=ktlTxC4QG8g", "ordre": 1},
                {"titre": "Management d'équipe", "urlYoutube": "https://www.youtube.com/watch?v=Q2vQkHjS4xQ", "ordre": 2}
            ]
        }
    ],
    "COMMERCIAL": [
        {
            "formationId": None,
            "formation": "Techniques de vente et négociation",
            "title": "Techniques de vente et négociation",
            "provider": "Udemy",
            "url": "https://www.udemy.com/courses/search/?q=vente+negociation",
            "description": "Maîtrisez les techniques de vente, la prospection client et la négociation commerciale.",
            "level": "Intermédiaire",
            "score": 0.76,
            "semanticScore": 0.76,
            "skillScore": 0.71,
            "priorityScore": 0.67,
            "type": "EXTERNE",
            "matchedSkills": ["vente", "negociation", "prospection"],
            "reason": "Formation externe recommandée car le catalogue interne est épuisé.",
            "videos": [
                {"titre": "Techniques de vente", "urlYoutube": "https://www.youtube.com/watch?v=8sjA90hvnQ0", "ordre": 1}
            ]
        }
    ],
    "FINANCE": [
        {
            "formationId": None,
            "formation": "Finance d'entreprise et contrôle de gestion",
            "title": "Finance d'entreprise et contrôle de gestion",
            "provider": "Coursera",
            "url": "https://www.coursera.org/search?query=finance+entreprise",
            "description": "Apprenez les fondamentaux de la finance d'entreprise, budgétisation et contrôle de gestion.",
            "level": "Intermédiaire",
            "score": 0.74,
            "semanticScore": 0.74,
            "skillScore": 0.70,
            "priorityScore": 0.65,
            "type": "EXTERNE",
            "matchedSkills": ["finance", "comptabilite", "controle de gestion"],
            "reason": "Formation externe recommandée car le catalogue interne est épuisé.",
            "videos": [
                {"titre": "Finance d'entreprise", "urlYoutube": "https://www.youtube.com/watch?v=HAnw168huqA", "ordre": 1}
            ]
        }
    ],
    "DATA": [
        {
            "formationId": None,
            "formation": "Power BI pour Data Analyst",
            "title": "Power BI pour Data Analyst",
            "provider": "Coursera",
            "url": "https://www.coursera.org/search?query=power%20bi%20data%20analyst",
            "description": "Analyse de données, tableaux de bord, reporting et visualisation avec Power BI.",
            "level": "Intermédiaire",
            "score": 0.80,
            "semanticScore": 0.80,
            "skillScore": 0.75,
            "priorityScore": 0.70,
            "type": "EXTERNE",
            "matchedSkills": ["power bi", "reporting", "dashboard", "data analysis"],
            "reason": "Formation externe recommandée selon le poste et les compétences ciblées.",
            "videos": [
                {"titre": "Power BI", "urlYoutube": "https://www.youtube.com/watch?v=AGrl-H87pRU", "ordre": 1}
            ]
        },
        {
            "formationId": None,
            "formation": "SQL pour l'analyse de données",
            "title": "SQL pour l'analyse de données",
            "provider": "Udemy",
            "url": "https://www.udemy.com/courses/search/?q=sql%20data%20analysis",
            "description": "Requêtes SQL, extraction de données, agrégations, analyse et reporting.",
            "level": "Intermédiaire",
            "score": 0.78,
            "semanticScore": 0.78,
            "skillScore": 0.74,
            "priorityScore": 0.70,
            "type": "EXTERNE",
            "matchedSkills": ["sql", "data analysis", "reporting"],
            "reason": "Formation externe recommandée selon le poste et les compétences ciblées.",
            "videos": [
                {"titre": "SQL Data Analysis", "urlYoutube": "https://www.youtube.com/watch?v=7S_tz1z_5bA", "ordre": 1}
            ]
        },
        {
            "formationId": None,
            "formation": "Python pour Data Science",
            "title": "Python pour Data Science",
            "provider": "Coursera",
            "url": "https://www.coursera.org/search?query=python%20data%20science",
            "description": "Python, pandas, analyse de données, visualisation et bases du machine learning.",
            "level": "Intermédiaire",
            "score": 0.76,
            "semanticScore": 0.76,
            "skillScore": 0.72,
            "priorityScore": 0.68,
            "type": "EXTERNE",
            "matchedSkills": ["python", "pandas", "data science", "machine learning"],
            "reason": "Formation externe recommandée selon le poste et les compétences ciblées.",
            "videos": [
                {"titre": "Python Data Science", "urlYoutube": "https://www.youtube.com/watch?v=LHBE6Q9XlzI", "ordre": 1}
            ]
        }
    ],
    "GENERAL": [
        {
            "formationId": None,
            "formation": "Développement personnel et professionnel",
            "title": "Développement personnel et professionnel",
            "provider": "Coursera",
            "url": "https://www.coursera.org/search?query=developpement+professionnel",
            "description": "Renforcez vos compétences transversales : communication, organisation, gestion du temps.",
            "level": "Débutant",
            "score": 0.70,
            "semanticScore": 0.70,
            "skillScore": 0.65,
            "priorityScore": 0.60,
            "type": "EXTERNE",
            "matchedSkills": ["communication", "organisation", "gestion du temps"],
            "reason": "Formation externe recommandée car le catalogue interne est épuisé.",
            "videos": [
                {"titre": "Développement des compétences", "urlYoutube": "https://www.youtube.com/watch?v=Q2vQkHjS4xQ", "ordre": 1}
            ]
        }
    ]
}


def get_formations_externes(domain, mode="EXTERNE"):
    """Retourne les formations externes du domaine détecté, en gardant le mode d'origine."""
    base = FORMATIONS_EXTERNES.get(domain, FORMATIONS_EXTERNES["GENERAL"])

    result = []
    for formation in base:
        item = formation.copy()
        item["type"] = mode
        item["source"] = "EXTERNE"
        result.append(item)

    return result


def get_all_formations_externes(mode="EXTERNE"):
    """Retourne toutes les formations externes, tous domaines confondus, sans doublons."""
    seen = set()
    result = []

    for formations in FORMATIONS_EXTERNES.values():
        for formation in formations:
            title_norm = normalize_text(formation.get("title") or formation.get("formation") or "")
            provider_norm = normalize_text(formation.get("provider") or "")
            key = f"{title_norm}|{provider_norm}"

            if key in seen:
                continue

            seen.add(key)
            item = formation.copy()
            item["type"] = mode
            item["source"] = "EXTERNE"
            result.append(item)

    return result


def external_formation_to_text(item):
    return " ".join([
        str(item.get("formation") or ""),
        str(item.get("title") or ""),
        str(item.get("provider") or ""),
        str(item.get("description") or ""),
        str(item.get("level") or ""),
        " ".join(item.get("matchedSkills") or [])
    ])


def lexical_overlap_score(query_text, item_text):
    query_words = set(normalize_text(query_text).split())
    item_words = set(normalize_text(item_text).split())

    ignored = {
        "formation", "formations", "poste", "professionnel", "professionnelle",
        "competences", "competence", "requises", "adapter", "adaptee",
        "developpement", "renforcer", "ameliorer", "booster", "pratique"
    }

    query_words = {w for w in query_words if len(w) > 2 and w not in ignored}
    item_words = {w for w in item_words if len(w) > 2 and w not in ignored}

    if not query_words or not item_words:
        return 0.0

    common = query_words.intersection(item_words)
    return len(common) / max(1, len(query_words))


def is_clearly_bad_external(poste, target_skills, item):
    """Filtre de sécurité métier pour éviter des recommandations clairement hors sujet."""
    context = normalize_text(" ".join([poste or "", " ".join(target_skills or [])]))
    item_text = normalize_text(external_formation_to_text(item))

    data_context_words = [
        "data analyst", "analyste data", "analyste donnees", "analyste données",
        "business intelligence", "power bi", "tableau", "sql", "reporting",
        "dashboard", "data", "analyse", "analytics", "python", "pandas"
    ]

    dev_only_words = [
        "spring", "angular", "frontend", "front end", "backend", "back end",
        "microservices", "developpeur", "développeur", "java backend"
    ]

    if any(w in context for w in data_context_words):
        if any(w in item_text for w in dev_only_words):
            data_item_words = ["data", "sql", "power bi", "reporting", "analytics", "python", "pandas", "dashboard"]
            if not any(w in item_text for w in data_item_words):
                return True

    rh_context_words = ["rh", "ressources humaines", "recrutement", "paie", "personnel"]
    if any(w in context for w in rh_context_words):
        if any(w in item_text for w in dev_only_words):
            return True

    finance_context_words = ["finance", "comptable", "comptabilite", "comptabilité", "audit", "budget"]
    if any(w in context for w in finance_context_words):
        if any(w in item_text for w in dev_only_words):
            return True

    return False


def rank_external_formations(domain, mode, query_text, target_skills, poste, max_results=MAX_RESULTS):
    """
    Classe les formations externes globalement par similarité sémantique.
    On ne se limite pas au domaine détecté : cela marche pour n'importe quel poste.
    """
    candidates = get_all_formations_externes(mode=mode)
    ranked = []

    for item in candidates:
        if is_clearly_bad_external(poste, target_skills, item):
            continue

        item_text = external_formation_to_text(item)

        try:
            semantic = cosine_score(query_text, item_text)
        except Exception:
            semantic = 0.0

        lexical = lexical_overlap_score(
            " ".join([query_text or "", poste or "", " ".join(target_skills or [])]),
            item_text
        )

        final_score = (semantic * 0.75) + (lexical * 0.25)

        # Bonus si une compétence ciblée apparaît explicitement dans la formation externe
        item_text_norm = normalize_text(item_text)
        matched = []
        for skill in target_skills or []:
            skill_norm = normalize_text(skill)
            if skill_norm and skill_norm in item_text_norm:
                matched.append(skill)

        if matched:
            final_score += 0.08

        final_score = max(0.0, min(final_score, 0.99))

        item_copy = item.copy()
        item_copy["type"] = mode
        item_copy["source"] = "EXTERNE"
        item_copy["semanticScore"] = round(semantic, 3)
        item_copy["skillScore"] = round(lexical, 3)
        item_copy["score"] = round(final_score, 3)

        if matched:
            item_copy["matchedSkills"] = matched

        item_copy["reason"] = (
            "Formation externe recommandée par similarité avec le poste "
            f"{poste or 'ciblé'} et les compétences à développer."
        )

        ranked.append(item_copy)

    ranked = sorted(ranked, key=lambda x: x.get("score", 0), reverse=True)

    # Seuil souple : si rien ne passe, on garde quand même les 3 meilleurs pour éviter écran vide
    filtered = [item for item in ranked if item.get("score", 0) >= 0.35]

    if not filtered:
        filtered = ranked[:3]

    return filtered[:max_results]

def build_videos_for_recommendation(matched_skills, formation_title, poste, domain):
    search_text = " ".join([
        " ".join(matched_skills or []),
        formation_title or "",
        poste or "",
        domain or ""
    ])
    search_norm = normalize_text(search_text)
    for key, videos in VIDEO_LIBRARY.items():
        if key != "default" and normalize_text(key) in search_norm:
            return videos
    return VIDEO_LIBRARY["default"]


def detect_domain(poste, user_skills, required_skills):
    text = normalize_text(poste)
    text += " " + " ".join(user_skills.keys())
    text += " " + " ".join(required_skills.keys())

    domains = {
        "RH": ["rh", "ressources humaines", "responsable rh", "recrutement", "paie", "droit du travail", "administration du personnel", "gestion du personnel", "sirh", "talent acquisition"],
        "IT": ["developpeur", "développeur", "java", "spring", "angular", "backend", "frontend", "informatique", "software", "devops"],
        "DATA": ["data analyst", "data scientist", "business analyst", "analyste data", "analyste donnees", "analyste données", "power bi", "tableau", "sql", "excel", "etl", "big data", "python", "pandas", "machine learning", "data warehouse", "reporting", "dashboard", "analytics", "business intelligence"],
        "COMMERCIAL": ["commercial", "vente", "prospection", "relation client", "crm", "negociation", "négociation"],
        "FINANCE": ["finance", "comptable", "comptabilite", "comptabilité", "controle de gestion", "audit", "budget"],
        "MARKETING": ["marketing", "seo", "communication digitale", "reseaux sociaux", "réseaux sociaux", "branding"],
        "MANAGEMENT": ["manager", "management", "leadership", "chef de projet", "gestion de projet", "responsable equipe", "responsable équipe"]
    }

    scores = {}
    for domain, keywords in domains.items():
        scores[domain] = sum(1 for keyword in keywords if normalize_text(keyword) in text)

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "GENERAL"


FORBIDDEN_BY_DOMAIN = {
    "RH": ["java", "spring", "angular", "typescript", "javascript", "backend", "frontend", "api", "docker", "devops", "programmation", "developpeur", "développeur", "sql"],
    "DATA": ["spring", "angular", "frontend", "backend", "microservices", "developpeur", "développeur", "react", "vue", "devops"],
    "COMMERCIAL": ["java", "spring", "angular", "backend", "frontend", "docker", "devops"],
    "FINANCE": ["java", "spring", "angular", "backend", "frontend", "docker", "devops"],
    "MARKETING": ["java", "spring", "angular", "backend", "frontend", "docker", "devops"]
}


def is_forbidden_for_domain(domain, formation_text):
    return text_contains_any(formation_text, FORBIDDEN_BY_DOMAIN.get(domain, []))


def analyze_gap(user_skills, required_skills):
    gaps = {}
    for skill, required_level in required_skills.items():
        current_level = user_skills.get(skill, 0)
        gap = required_level - current_level
        if gap > 0:
            gaps[skill] = {
                "requiredLevel": required_level,
                "currentLevel": current_level,
                "gap": gap
            }
    return gaps


def formation_to_text(formation):
    return " ".join([
        str(formation.get("titre") or ""),
        str(formation.get("description") or ""),
        str(formation.get("domaine") or ""),
        str(formation.get("niveau") or ""),
        " ".join(formation.get("competences") or [])
    ])


def build_query(mode, poste, user_skills, required_skills, gaps):
    poste_norm = normalize_text(poste)
    inferred_skills = []

    if "rh" in poste_norm or "ressources humaines" in poste_norm:
        inferred_skills = [
            "recrutement", "ressources humaines", "gestion du personnel",
            "administration du personnel", "droit du travail", "paie",
            "communication interne", "gestion des conflits", "formation professionnelle", "sirh"
        ]

    if mode == "GAP_POSTE":
        if gaps:
            target_skills = list(gaps.keys())
        elif required_skills:
            target_skills = list(required_skills.keys())
        elif inferred_skills:
            target_skills = inferred_skills
        else:
            target_skills = []

        query = " ".join([
            f"poste professionnel {poste}",
            "compétences requises",
            " ".join(target_skills),
            "formation adaptée au poste",
            "développement compétences métier"
        ])
        return query, target_skills

    weak_skills = [skill for skill, level in user_skills.items() if level <= 2]

    if weak_skills:
        target_skills = weak_skills
    elif inferred_skills:
        target_skills = inferred_skills
    else:
        target_skills = list(user_skills.keys())

    query = " ".join([
        f"poste professionnel {poste}",
        "booster renforcer améliorer compétences",
        " ".join(target_skills),
        "formation pratique progression professionnelle"
    ])
    return query, target_skills


def matched_skills_for_formation(target_skills, formation):
    formation_text = normalize_text(formation_to_text(formation))
    matched = []
    for skill in target_skills:
        skill_norm = normalize_text(skill)
        if skill_norm and skill_norm in formation_text:
            matched.append(skill)
    return matched


def recommend_existing_formations(payload):
    mode = payload.get("mode", "GAP_POSTE")
    poste = payload.get("poste") or ""

    user_skills = normalize_skill_dict(payload.get("userSkills", {}))
    required_skills = normalize_skill_dict(payload.get("requiredSkills", {}))

    formations_suivies = [
        normalize_text(extract_formation_title(x))
        for x in payload.get("formationsSuivies", []) or []
    ]

    formations = payload.get("formations", []) or []
    domain = detect_domain(poste, user_skills, required_skills)
    gaps = analyze_gap(user_skills, required_skills)

    query_text, target_skills = build_query(
        mode=mode,
        poste=poste,
        user_skills=user_skills,
        required_skills=required_skills,
        gaps=gaps
    )

    if not formations:
        return {
            "mode": mode,
            "poste": poste,
            "detectedDomain": domain,
            "gapSkills": gaps,
            "targetSkills": target_skills,
            "recommendations": [],
            "catalogueEpuise": False,
            "message": "Aucune formation active reçue depuis Spring Boot."
        }

    usable_formations = []

    for formation in formations:
        title = formation.get("titre") or formation.get("title") or ""
        title_norm = normalize_text(title)

        if title_norm and title_norm in formations_suivies:
            continue
        formation_text = formation_to_text(formation)
        if is_forbidden_for_domain(domain, formation_text):
            continue
        usable_formations.append(formation)

    # ─────────────────────────────────────────────────────────────
    #  CATALOGUE ÉPUISÉ → on retourne les formations externes
    #  avec la même structure que les formations internes
    # ─────────────────────────────────────────────────────────────
    if not usable_formations:
        nb_suivies = len(formations_suivies)
        nb_total = len(formations)
        catalogue_epuise = nb_suivies >= nb_total

        externes = rank_external_formations(
            domain=domain,
            mode=mode,
            query_text=query_text,
            target_skills=target_skills,
            poste=poste
        )

        if catalogue_epuise:
            message = (
                f"Félicitations ! Vous avez complété toutes les formations disponibles "
                f"({nb_suivies} formations). Voici des formations externes recommandées."
            )
        else:
            message = "Aucune formation interne compatible. Voici des formations externes recommandées."

        return {
            "mode": mode,
            "poste": poste,
            "detectedDomain": domain,
            "gapSkills": gaps,
            "targetSkills": target_skills,
            "recommendations": externes,
            "catalogueEpuise": catalogue_epuise,
            "source": "EXTERNE",
            "message": message
        }

    results = []

    for formation in usable_formations:
        formation_id = formation.get("id")
        titre = formation.get("titre") or "Formation sans titre"
        description = formation.get("description") or ""
        domaine = formation.get("domaine") or ""
        formation_text = formation_to_text(formation)

        semantic_score = cosine_score(query_text, formation_text)
        matched = matched_skills_for_formation(target_skills, formation)

        skill_score = len(matched) / max(1, len(target_skills)) if target_skills else 0.0

        domain_bonus = 0.0
        formation_text_norm = normalize_text(formation_text)
        domaine_norm = normalize_text(domaine)

        if domain == "RH":
            rh_words = [
                "rh", "ressources humaines", "recrutement", "paie",
                "droit du travail", "personnel", "sirh", "talent",
                "communication", "conflits", "management", "formation professionnelle"
            ]
            if any(normalize_text(w) in formation_text_norm for w in rh_words):
                domain_bonus = 0.18
            if "rh" in domaine_norm or "soft" in domaine_norm or "management" in domaine_norm:
                domain_bonus = max(domain_bonus, 0.12)

        priority_score = 0.50

        if mode == "GAP_POSTE":
            if matched:
                priority_score = 0.85
            elif domain_bonus > 0:
                priority_score = 0.70
            else:
                priority_score = 0.45

        elif mode == "BOOST_COMPETENCES":
            if matched:
                priority_score = 0.80
            elif semantic_score >= 0.62:
                priority_score = 0.65
            else:
                priority_score = 0.40

        final_score = (
            semantic_score * 0.55
            + skill_score * 0.20
            + priority_score * 0.15
            + domain_bonus
        )

        final_score = max(0.0, min(final_score, 0.99))

        if mode == "GAP_POSTE":
            if semantic_score < 0.50 and not matched and domain_bonus == 0:
                continue
            if final_score < 0.50:
                continue

        if mode == "BOOST_COMPETENCES":
            if semantic_score < 0.50 and not matched:
                continue
            if final_score < 0.50:
                continue

        displayed_matched_skills = matched if matched else target_skills[:2]

        reason = (
            f"Formation recommandée pour le poste {poste}. "
            f"Score sémantique : {round(semantic_score, 2)}. "
        )

        if matched:
            reason += "Compétences réellement détectées : " + ", ".join(matched) + "."
        elif domain_bonus > 0:
            reason += "Formation compatible avec le domaine métier du poste."
        else:
            reason += "Correspondance détectée par similarité sémantique."

        videos = build_videos_for_recommendation(
            matched_skills=displayed_matched_skills,
            formation_title=titre,
            poste=poste,
            domain=domain
        )

        results.append({
            "formationId": formation_id,
            "formation": titre,
            "title": titre,
            "provider": "BASE_INTERNE",
            "url": None,
            "description": description,
            "level": formation.get("niveau") or "auto",
            "score": round(final_score, 3),
            "semanticScore": round(semantic_score, 3),
            "skillScore": round(skill_score, 3),
            "priorityScore": round(priority_score, 3),
            "type": mode,
            "matchedSkills": displayed_matched_skills,
            "reason": reason,
            "videos": videos,
            "catalogueEpuise": False,
            "source": "INTERNE"
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:MAX_RESULTS]

    # Si des formations internes existent mais qu'aucune ne passe les seuils de score,
    # on bascule aussi vers les formations externes.
    if not results:
        externes = rank_external_formations(
            domain=domain,
            mode=mode,
            query_text=query_text,
            target_skills=target_skills,
            poste=poste
        )
        return {
            "mode": mode,
            "poste": poste,
            "detectedDomain": domain,
            "gapSkills": gaps,
            "targetSkills": target_skills,
            "recommendations": externes,
            "catalogueEpuise": True,
            "source": "EXTERNE",
            "message": "Aucune formation interne disponible ou assez pertinente. Voici des formations externes recommandées."
        }

    return {
        "mode": mode,
        "poste": poste,
        "detectedDomain": domain,
        "gapSkills": gaps,
        "targetSkills": target_skills,
        "recommendations": results,
        "catalogueEpuise": False,
        "source": "INTERNE",
        "message": "Recommandations générées."
    }


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        payload = request.json or {}
        response = recommend_existing_formations(payload)
        return jsonify(response), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Erreur interne agent IA",
            "message": str(e),
            "recommendations": []
        }), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "UP",
        "service": "formation-semantic-recommendation-agent",
        "model": MODEL_NAME,
        "version": "9.3.0",
        "logic": "semantic matching using Hugging Face Inference API",
        "features": [
            "recommend existing formations",
            "external formations fallback when catalogue exhausted",
            "global semantic ranking for external formations",
            "same JSON structure for internal and external formations",
            "new poste semantic understanding",
            "gap poste recommendations",
            "boost competence recommendations",
            "anti out-of-domain filter",
            "embeddable youtube videos"
        ]
    })


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Agent IA formations internes fonctionne",
        "version": "9.3.0",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend"
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)