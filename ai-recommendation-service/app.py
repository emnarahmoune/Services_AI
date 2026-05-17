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

    response = requests.post(
        API_URL,
        headers=HEADERS,
        json={"inputs": text},
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(f"Erreur Hugging Face: {response.status_code} - {response.text}")

    data = response.json()

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
            "recrutement",
            "ressources humaines",
            "gestion du personnel",
            "administration du personnel",
            "droit du travail",
            "paie",
            "communication interne",
            "gestion des conflits",
            "formation professionnelle",
            "sirh"
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
        normalize_text(x)
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
            "message": "Aucune formation active reçue depuis Spring Boot."
        }

    usable_formations = []

    for formation in formations:
        title = formation.get("titre") or formation.get("title") or ""

        if normalize_text(title) in formations_suivies:
            continue

        formation_text = formation_to_text(formation)

        if is_forbidden_for_domain(domain, formation_text):
            continue

        usable_formations.append(formation)

    if not usable_formations:
        return {
            "mode": mode,
            "poste": poste,
            "detectedDomain": domain,
            "gapSkills": gaps,
            "targetSkills": target_skills,
            "recommendations": [],
            "message": "Aucune formation compatible après filtrage métier."
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
            "videos": videos
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:MAX_RESULTS]

    return {
        "mode": mode,
        "poste": poste,
        "detectedDomain": domain,
        "gapSkills": gaps,
        "targetSkills": target_skills,
        "recommendations": results,
        "message": "Recommandations générées." if results else "Aucune formation assez pertinente trouvée."
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
        "version": "8.2.0",
        "logic": "semantic matching using Hugging Face Inference API",
        "features": [
            "recommend existing formations",
            "new poste semantic understanding",
            "new competence semantic understanding",
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
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend"
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)