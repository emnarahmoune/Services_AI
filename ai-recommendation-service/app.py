from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import unicodedata
import traceback
import re

app = Flask(__name__)
CORS(app)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)

MAX_RESULTS = 8


# ============================================================
# NORMALISATION
# ============================================================

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


# ============================================================
# VIDÉOS EMBEDDABLES
# ============================================================

VIDEO_LIBRARY = {
    "communication": [
        {
            "titre": "Communication professionnelle",
            "urlYoutube": "https://www.youtube.com/watch?v=HAnw168huqA",
            "ordre": 1
        },
        {
            "titre": "Communication efficace au travail",
            "urlYoutube": "https://www.youtube.com/watch?v=8sjA90hvnQ0",
            "ordre": 2
        },
        {
            "titre": "Améliorer sa communication professionnelle",
            "urlYoutube": "https://www.youtube.com/watch?v=5yK_E3Yq5jA",
            "ordre": 3
        }
    ],

    "communication interne": [
        {
            "titre": "Communication interne en entreprise",
            "urlYoutube": "https://www.youtube.com/watch?v=HAnw168huqA",
            "ordre": 1
        },
        {
            "titre": "Communication professionnelle",
            "urlYoutube": "https://www.youtube.com/watch?v=8sjA90hvnQ0",
            "ordre": 2
        },
        {
            "titre": "Gérer les échanges au travail",
            "urlYoutube": "https://www.youtube.com/watch?v=5yK_E3Yq5jA",
            "ordre": 3
        }
    ],

    "leadership": [
        {
            "titre": "Leadership",
            "urlYoutube": "https://www.youtube.com/watch?v=ktlTxC4QG8g",
            "ordre": 1
        },
        {
            "titre": "Gestion équipe",
            "urlYoutube": "https://www.youtube.com/watch?v=4a0FbQdH3dY",
            "ordre": 2
        },
        {
            "titre": "Management et leadership",
            "urlYoutube": "https://www.youtube.com/watch?v=Q2vQkHjS4xQ",
            "ordre": 3
        }
    ],

    "management": [
        {
            "titre": "Management d'équipe",
            "urlYoutube": "https://www.youtube.com/watch?v=Q2vQkHjS4xQ",
            "ordre": 1
        },
        {
            "titre": "Leadership",
            "urlYoutube": "https://www.youtube.com/watch?v=ktlTxC4QG8g",
            "ordre": 2
        },
        {
            "titre": "Gestion équipe",
            "urlYoutube": "https://www.youtube.com/watch?v=4a0FbQdH3dY",
            "ordre": 3
        }
    ],

    "recrutement": [
        {
            "titre": "Recrutement RH",
            "urlYoutube": "https://www.youtube.com/watch?v=HG68Ymazo18",
            "ordre": 1
        },
        {
            "titre": "Entretien de recrutement",
            "urlYoutube": "https://www.youtube.com/watch?v=6G8_qA8M8pQ",
            "ordre": 2
        },
        {
            "titre": "Sourcing candidats",
            "urlYoutube": "https://www.youtube.com/watch?v=4FQY3u4UxS0",
            "ordre": 3
        }
    ],

    "talent acquisition": [
        {
            "titre": "Talent Acquisition",
            "urlYoutube": "https://www.youtube.com/watch?v=HG68Ymazo18",
            "ordre": 1
        },
        {
            "titre": "Sourcing candidats",
            "urlYoutube": "https://www.youtube.com/watch?v=4FQY3u4UxS0",
            "ordre": 2
        },
        {
            "titre": "Entretien de recrutement",
            "urlYoutube": "https://www.youtube.com/watch?v=6G8_qA8M8pQ",
            "ordre": 3
        }
    ],

    "paie": [
        {
            "titre": "Gestion de la paie",
            "urlYoutube": "https://www.youtube.com/watch?v=b7OXULhF1pc",
            "ordre": 1
        },
        {
            "titre": "Bulletin de paie expliqué",
            "urlYoutube": "https://www.youtube.com/watch?v=zE51pYOTp2s",
            "ordre": 2
        },
        {
            "titre": "Charges sociales et salaire net",
            "urlYoutube": "https://www.youtube.com/watch?v=5cI-AkKy66I",
            "ordre": 3
        }
    ],

    "gestion de la paie": [
        {
            "titre": "Gestion de la paie",
            "urlYoutube": "https://www.youtube.com/watch?v=b7OXULhF1pc",
            "ordre": 1
        },
        {
            "titre": "Bulletin de paie expliqué",
            "urlYoutube": "https://www.youtube.com/watch?v=zE51pYOTp2s",
            "ordre": 2
        },
        {
            "titre": "Charges sociales et salaire net",
            "urlYoutube": "https://www.youtube.com/watch?v=5cI-AkKy66I",
            "ordre": 3
        }
    ],

    "droit du travail": [
        {
            "titre": "Droit du travail",
            "urlYoutube": "https://www.youtube.com/watch?v=4Ko4b38N7gE",
            "ordre": 1
        },
        {
            "titre": "Contrat de travail",
            "urlYoutube": "https://www.youtube.com/watch?v=O_4LwZ2pJzQ",
            "ordre": 2
        },
        {
            "titre": "Droit social RH",
            "urlYoutube": "https://www.youtube.com/watch?v=R6NoL7cnkQY",
            "ordre": 3
        }
    ],

    "gestion du personnel": [
        {
            "titre": "Gestion du personnel RH",
            "urlYoutube": "https://www.youtube.com/watch?v=HG68Ymazo18",
            "ordre": 1
        },
        {
            "titre": "Administration du personnel",
            "urlYoutube": "https://www.youtube.com/watch?v=4Ko4b38N7gE",
            "ordre": 2
        },
        {
            "titre": "Communication RH",
            "urlYoutube": "https://www.youtube.com/watch?v=HAnw168huqA",
            "ordre": 3
        }
    ],

    "administration du personnel": [
        {
            "titre": "Administration du personnel",
            "urlYoutube": "https://www.youtube.com/watch?v=4Ko4b38N7gE",
            "ordre": 1
        },
        {
            "titre": "Gestion du personnel RH",
            "urlYoutube": "https://www.youtube.com/watch?v=HG68Ymazo18",
            "ordre": 2
        },
        {
            "titre": "Contrat de travail",
            "urlYoutube": "https://www.youtube.com/watch?v=O_4LwZ2pJzQ",
            "ordre": 3
        }
    ],

    "gestion des conflits": [
        {
            "titre": "Gestion des conflits",
            "urlYoutube": "https://www.youtube.com/watch?v=Q2vQkHjS4xQ",
            "ordre": 1
        },
        {
            "titre": "Communication professionnelle",
            "urlYoutube": "https://www.youtube.com/watch?v=HAnw168huqA",
            "ordre": 2
        },
        {
            "titre": "Leadership et influence",
            "urlYoutube": "https://www.youtube.com/watch?v=ktlTxC4QG8g",
            "ordre": 3
        }
    ],

    "sirh": [
        {
            "titre": "SIRH et digitalisation RH",
            "urlYoutube": "https://www.youtube.com/watch?v=HG68Ymazo18",
            "ordre": 1
        },
        {
            "titre": "Gestion RH digitale",
            "urlYoutube": "https://www.youtube.com/watch?v=HAnw168huqA",
            "ordre": 2
        },
        {
            "titre": "Organisation RH",
            "urlYoutube": "https://www.youtube.com/watch?v=Q2vQkHjS4xQ",
            "ordre": 3
        }
    ],

    "default": [
        {
            "titre": "Formation professionnelle",
            "urlYoutube": "https://www.youtube.com/watch?v=HAnw168huqA",
            "ordre": 1
        },
        {
            "titre": "Communication au travail",
            "urlYoutube": "https://www.youtube.com/watch?v=8sjA90hvnQ0",
            "ordre": 2
        },
        {
            "titre": "Développement des compétences",
            "urlYoutube": "https://www.youtube.com/watch?v=Q2vQkHjS4xQ",
            "ordre": 3
        }
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
        if key == "default":
            continue

        key_norm = normalize_text(key)

        if key_norm in search_norm:
            return [
                {
                    "titre": video["titre"],
                    "urlYoutube": video["urlYoutube"],
                    "ordre": video["ordre"]
                }
                for video in videos
            ]

    return [
        {
            "titre": video["titre"],
            "urlYoutube": video["urlYoutube"],
            "ordre": video["ordre"]
        }
        for video in VIDEO_LIBRARY["default"]
    ]


# ============================================================
# DOMAINES
# ============================================================

def detect_domain(poste, user_skills, required_skills):
    text = normalize_text(poste)
    text += " " + " ".join(user_skills.keys())
    text += " " + " ".join(required_skills.keys())

    domains = {
        "RH": [
            "rh", "ressources humaines", "responsable rh", "recrutement",
            "paie", "droit du travail", "administration du personnel",
            "gestion du personnel", "sirh", "talent acquisition"
        ],
        "IT": [
            "developpeur", "développeur", "java", "spring", "angular",
            "backend", "frontend", "informatique", "software", "devops"
        ],
        "COMMERCIAL": [
            "commercial", "vente", "prospection", "relation client",
            "crm", "negociation", "négociation"
        ],
        "FINANCE": [
            "finance", "comptable", "comptabilite", "comptabilité",
            "controle de gestion", "audit", "budget"
        ],
        "MARKETING": [
            "marketing", "seo", "communication digitale", "reseaux sociaux",
            "réseaux sociaux", "branding"
        ],
        "MANAGEMENT": [
            "manager", "management", "leadership", "chef de projet",
            "gestion de projet", "responsable equipe", "responsable équipe"
        ]
    }

    scores = {}

    for domain, keywords in domains.items():
        score = 0

        for keyword in keywords:
            if normalize_text(keyword) in text:
                score += 1

        scores[domain] = score

    best = max(scores, key=scores.get)

    if scores[best] == 0:
        return "GENERAL"

    return best


FORBIDDEN_BY_DOMAIN = {
    "RH": [
        "java", "spring", "angular", "typescript", "javascript",
        "backend", "frontend", "api", "docker", "devops",
        "programmation", "developpeur", "développeur", "sql"
    ],
    "COMMERCIAL": [
        "java", "spring", "angular", "backend", "frontend", "docker", "devops"
    ],
    "FINANCE": [
        "java", "spring", "angular", "backend", "frontend", "docker", "devops"
    ],
    "MARKETING": [
        "java", "spring", "angular", "backend", "frontend", "docker", "devops"
    ]
}


def is_forbidden_for_domain(domain, formation_text):
    forbidden = FORBIDDEN_BY_DOMAIN.get(domain, [])
    return text_contains_any(formation_text, forbidden)


# ============================================================
# LOGIQUE IA
# ============================================================

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

    weak_skills = [
        skill
        for skill, level in user_skills.items()
        if level <= 2
    ]

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

    print("========== IA REQUEST ==========")
    print("MODE =", mode)
    print("POSTE =", poste)
    print("DOMAIN =", domain)
    print("USER SKILLS =", user_skills)
    print("REQUIRED SKILLS =", required_skills)
    print("TARGET SKILLS =", target_skills)
    print("NB FORMATIONS =", len(formations))

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

    print("NB FORMATIONS UTILISABLES =", len(usable_formations))

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

    query_embedding = model.encode(query_text, convert_to_tensor=True)

    formation_texts = [
        formation_to_text(f)
        for f in usable_formations
    ]

    formation_embeddings = model.encode(formation_texts, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, formation_embeddings)[0].cpu().numpy()

    results = []

    for index, formation in enumerate(usable_formations):
        formation_id = formation.get("id")
        titre = formation.get("titre") or "Formation sans titre"
        description = formation.get("description") or ""
        domaine = formation.get("domaine") or ""

        formation_text = formation_to_text(formation)
        semantic_score = float(similarities[index])

        matched = matched_skills_for_formation(target_skills, formation)

        if target_skills:
            skill_score = len(matched) / max(1, len(target_skills))
        else:
            skill_score = 0.0

        domain_bonus = 0.0
        formation_text_norm = normalize_text(formation_text)
        domaine_norm = normalize_text(domaine)

        if domain == "RH":
            rh_words = [
                "rh",
                "ressources humaines",
                "recrutement",
                "paie",
                "droit du travail",
                "personnel",
                "sirh",
                "talent",
                "communication",
                "conflits",
                "management",
                "formation professionnelle"
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

    print("NB RECOMMANDATIONS =", len(results))
    for r in results:
        print(r["type"], r["formationId"], r["formation"], r["score"], r["matchedSkills"])

    return {
        "mode": mode,
        "poste": poste,
        "detectedDomain": domain,
        "gapSkills": gaps,
        "targetSkills": target_skills,
        "recommendations": results,
        "message": (
            "Recommandations générées."
            if results
            else "Aucune formation assez pertinente trouvée."
        )
    }


# ============================================================
# API
# ============================================================

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
        "version": "8.1.0",
        "logic": "semantic matching on existing internal formations",
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
    app.run(port=5000, debug=True)
