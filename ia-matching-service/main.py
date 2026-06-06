from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import re
import unicodedata
import os
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="IA Matching CV Offre", version="3.2.0")

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/pipeline/feature-extraction"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}


class IaMatchingRequestDTO(BaseModel):
    candidatureId: Optional[int] = None
    offreId: Optional[int] = None
    employeId: Optional[int] = None
    titrePoste: Optional[str] = ""
    description: Optional[str] = ""
    competencesRequises: List[str] = []
    technologiesRequises: List[str] = []
    experienceMin: Optional[int] = 0
    niveauEtude: Optional[str] = ""
    cvText: Optional[str] = ""


class IaMatchingResponseDTO(BaseModel):
    texteExtrait: str
    competencesDetectees: List[str]
    technologiesDetectees: List[str]
    experiencesDetectees: List[str]
    anneesExperienceEstimees: int
    resumeProfil: str
    pointsForts: List[str]
    pointsFaibles: List[str]
    scoreGlobal: int
    scoreCompetences: int
    scoreTechnologies: int
    scoreExperience: int
    scoreFormation: int
    niveauCompatibilite: str
    competencesCorrespondantes: List[str]
    competencesManquantes: List[str]
    technologiesCorrespondantes: List[str]
    technologiesManquantes: List[str]
    justificationIa: str
    recommandationIa: str


def get_embedding(text: str):
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN manquant dans les variables Render")

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": str(text or "")},
            timeout=30  # ✅ réduit de 60 à 30
        )
    except requests.Timeout:
        raise RuntimeError("HuggingFace timeout — service indisponible après 30s")

    if response.status_code != 200:
        raise RuntimeError(f"Erreur HuggingFace: {response.status_code} - {response.text}")

    data = response.json()

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (int, float)):
        return data

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        if len(data[0]) > 0 and isinstance(data[0][0], list):
            return np.mean(np.array(data[0]), axis=0).tolist()
        return data[0]

    raise RuntimeError(f"Format embedding invalide: {data}")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_cv_into_chunks(cv_text: str, max_words: int = 80) -> List[str]:
    words = cv_text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def semantic_score_between_requirement_and_cv(requirement: str, cv_chunks: List[str]) -> float:
    if not requirement or not cv_chunks:
        return 0.0

    req_emb = get_embedding(requirement)
    best_score = 0.0

    for chunk in cv_chunks:
        chunk_emb = get_embedding(chunk)
        score = cosine_similarity([req_emb], [chunk_emb])[0][0]
        best_score = max(best_score, float(score))

    return best_score


def exact_match_score(requirement: str, cv_text: str) -> float:
    req = normalize_text(requirement)
    cv = normalize_text(cv_text)
    return 1.0 if req and req in cv else 0.0


def calculate_items_matching(required_items, cv_text, cv_chunks, threshold=0.50):
    matched, missing, details = [], [], {}

    for item in required_items:
        exact = exact_match_score(item, cv_text)
        score = 1.0 if exact == 1.0 else semantic_score_between_requirement_and_cv(item, cv_chunks)

        percentage = max(0, min(100, round(score * 100)))
        details[item] = percentage

        if score >= threshold:
            matched.append(item)
        else:
            missing.append(item)

    return matched, missing, details


def calculate_average_score(details: dict) -> int:
    # ✅ Score neutre si aucune compétence requise
    if not details:
        return 70
    return round(sum(details.values()) / len(details))


MONTHS_FR_EN = {
    "janvier": 1, "fevrier": 2, "février": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "aout": 8, "août": 8,
    "septembre": 9, "octobre": 10, "novembre": 11, "decembre": 12, "décembre": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9,
    "october": 10, "november": 11, "december": 12
}


def month_to_number(month_name: str) -> int:
    return MONTHS_FR_EN.get(normalize_text(month_name), 1)


def months_between(start_year: int, start_month: int, end_year: int, end_month: int) -> int:
    diff = abs((end_year * 12 + end_month) - (start_year * 12 + start_month))
    return diff if diff != 0 else 1


def extract_experience_section(text: str) -> str:
    patterns = [
        r"experiences professionnelles(.+?)(formation|education|competences|skills|projets|certifications|langues|$)",
        r"experience professionnelle(.+?)(formation|education|competences|skills|projets|certifications|langues|$)",
        r"experiences(.+?)(formation|education|competences|skills|projets|certifications|langues|$)",
        r"experience(.+?)(formation|education|competences|skills|projets|certifications|langues|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)

    return text


def estimate_experience_months(cv_text: str) -> int:
    text = normalize_text(cv_text)
    if not text:
        return 0

    experience_months = []

    direct_year_patterns = [
        r"(\d+)\s*ans?\s*d[' ]?experience",
        r"(\d+)\s*annees?\s*d[' ]?experience",
        r"experience\s*[:\-]?\s*(\d+)\s*ans?",
        r"(\d+)\s*years?\s*of\s*experience",
        r"(\d+)\s*yrs?\s*experience",
        r"(\d+)\s*ans?\s*en\s*(developpement|informatique|it|java|angular|spring|python|data|rh|finance)"
    ]

    for pattern in direct_year_patterns:
        for match in re.findall(pattern, text):
            try:
                value = match[0] if isinstance(match, tuple) else match
                experience_months.append(int(value) * 12)
            except ValueError:
                pass

    direct_month_patterns = [
        r"(\d+)\s*mois\s*d[' ]?experience",
        r"(\d+)\s*months?\s*of\s*experience"
    ]

    for pattern in direct_month_patterns:
        for match in re.findall(pattern, text):
            try:
                experience_months.append(int(match))
            except ValueError:
                pass

    month_names = "|".join(MONTHS_FR_EN.keys())
    month_year_range_pattern = rf"({month_names})\s+(20\d{{2}}|19\d{{2}})\s*[-–—]\s*({month_names})\s+(20\d{{2}}|19\d{{2}})"

    for start_month, start_year, end_month, end_year in re.findall(month_year_range_pattern, text):
        try:
            diff = months_between(
                int(start_year), month_to_number(start_month),
                int(end_year), month_to_number(end_month)
            )
            if 0 < diff <= 480:
                experience_months.append(diff)
        except ValueError:
            pass

    year_range_patterns = [
        r"(20\d{2}|19\d{2})\s*[-–—]\s*(20\d{2}|19\d{2})",
        r"(20\d{2}|19\d{2})\s*/\s*(20\d{2}|19\d{2})",
        r"(20\d{2}|19\d{2})\s*a\s*(20\d{2}|19\d{2})",
        r"(20\d{2}|19\d{2})\s*à\s*(20\d{2}|19\d{2})",
    ]

    for pattern in year_range_patterns:
        for y1, y2 in re.findall(pattern, text):
            try:
                diff_years = abs(int(y2) - int(y1))
                months = 6 if diff_years == 0 else diff_years * 12
                if 0 < months <= 480:
                    experience_months.append(months)
            except ValueError:
                pass

    current_year = datetime.now().year
    current_month = datetime.now().month

    since_patterns = [
        r"depuis\s+(20\d{2}|19\d{2})",
        r"poste\s+actuel\s+depuis\s+(20\d{2}|19\d{2})",
        r"actuellement\s+depuis\s+(20\d{2}|19\d{2})",
        r"since\s+(20\d{2}|19\d{2})",
        r"from\s+(20\d{2}|19\d{2})"
    ]

    for pattern in since_patterns:
        for start in re.findall(pattern, text):
            try:
                months = months_between(int(start), 1, current_year, current_month)
                if 0 < months <= 480:
                    experience_months.append(months)
            except ValueError:
                pass

    experience_section = extract_experience_section(text)
    section_total_months = 0

    for y1, y2 in re.findall(r"(20\d{2}|19\d{2})\s*[-–—/]\s*(20\d{2}|19\d{2})", experience_section):
        try:
            diff_years = abs(int(y2) - int(y1))
            months = 6 if diff_years == 0 else diff_years * 12
            if 0 < months <= 480:
                section_total_months += months
        except ValueError:
            pass

    if section_total_months > 0:
        experience_months.append(section_total_months)

    return max(experience_months) if experience_months else 0


def calculate_experience_score(cv_text: str, experience_min: int):
    experience_months = estimate_experience_months(cv_text)
    experience_years = round(experience_months / 12)

    if not experience_min or experience_min <= 0:
        return (50, 0) if experience_months <= 0 else (70, experience_years)

    required_months = experience_min * 12

    if experience_months <= 0:
        return 0, 0

    ratio = experience_months / required_months

    if ratio >= 1:
        return 100, experience_years

    score = round(ratio * 100)
    return max(20, score), experience_years


FORMATION_LEVELS = {
    "bac": 1, "baccalaureat": 1, "baccalauréat": 1,
    "bac+2": 2, "bts": 2, "dut": 2, "technicien superieur": 2, "technicien supérieur": 2,
    "bac+3": 3, "licence": 3, "bachelor": 3,
    "bac+5": 5, "master": 5, "mastere": 5, "mastère": 5,
    "ingenieur": 5, "ingénieur": 5, "engineering": 5,
    "doctorat": 8, "phd": 8, "doctorate": 8
}


def detect_formation_level(cv_text: str) -> int:
    cv = normalize_text(cv_text)
    detected = [level for keyword, level in FORMATION_LEVELS.items() if normalize_text(keyword) in cv]
    return max(detected) if detected else 0


def required_formation_level(niveau_etude: str) -> int:
    return FORMATION_LEVELS.get(normalize_text(niveau_etude), 0)


def calculate_formation_score(cv_text: str, niveau_etude: str) -> int:
    required = required_formation_level(niveau_etude)

    if required <= 0:
        return 70

    detected = detect_formation_level(cv_text)

    if detected <= 0:
        return 0

    if detected >= required:
        return 100

    return max(20, min(round((detected / required) * 100), 100))


def calculate_offer_cv_similarity(request: IaMatchingRequestDTO, cv_text: str) -> int:
    offer_text = f"""
    Poste : {request.titrePoste}
    Description : {request.description}
    Compétences : {", ".join(request.competencesRequises)}
    Technologies : {", ".join(request.technologiesRequises)}
    Niveau d'étude : {request.niveauEtude}
    Expérience minimale : {request.experienceMin}
    """

    if not cv_text.strip():
        return 0

    offer_emb = get_embedding(offer_text)

    # ✅ Correction — couper proprement au dernier espace
    cv_short = cv_text[:5000].rsplit(' ', 1)[0] if len(cv_text) > 5000 else cv_text
    cv_emb = get_embedding(cv_short)

    similarity = float(cosine_similarity([offer_emb], [cv_emb])[0][0])
    return max(0, min(100, round(similarity * 100)))


def resolve_niveau(score: int) -> str:
    if score >= 85:
        return "EXCELLENT"
    if score >= 70:
        return "TRES_BON"
    if score >= 55:
        return "BON"
    if score >= 40:
        return "MOYEN"
    return "FAIBLE"


def resolve_recommandation(score: int) -> str:
    if score >= 85:
        return "Profil fortement recommandé pour ce poste."
    if score >= 70:
        return "Profil intéressant, à considérer pour un entretien."
    if score >= 55:
        return "Profil acceptable, mais certaines compétences doivent être validées."
    if score >= 40:
        return "Profil partiellement compatible avec l'offre."
    return "Profil peu compatible avec les critères actuels."


@app.post("/api/matching/analyze", response_model=IaMatchingResponseDTO)
def analyze_matching(request: IaMatchingRequestDTO):
    cv_text = request.cvText or ""
    cv_chunks = split_cv_into_chunks(cv_text)

    competences_correspondantes, competences_manquantes, competences_details = calculate_items_matching(
        request.competencesRequises, cv_text, cv_chunks
    )

    technologies_correspondantes, technologies_manquantes, technologies_details = calculate_items_matching(
        request.technologiesRequises, cv_text, cv_chunks
    )

    score_competences = calculate_average_score(competences_details)
    score_technologies = calculate_average_score(technologies_details)

    score_experience, annees_experience = calculate_experience_score(cv_text, request.experienceMin or 0)
    score_formation = calculate_formation_score(cv_text, request.niveauEtude or "")
    score_semantique_global = calculate_offer_cv_similarity(request, cv_text)

    score_global = round(
        score_competences * 0.30
        + score_technologies * 0.30
        + score_experience * 0.15
        + score_formation * 0.10
        + score_semantique_global * 0.15
    )

    score_global = max(0, min(100, score_global))
    niveau = resolve_niveau(score_global)

    points_forts = []
    points_faibles = []

    if competences_correspondantes:
        points_forts.append("Compétences compatibles détectées : " + ", ".join(competences_correspondantes))

    if technologies_correspondantes:
        points_forts.append("Technologies compatibles détectées : " + ", ".join(technologies_correspondantes))

    if score_experience >= 80:
        points_forts.append(f"Expérience estimée compatible : {annees_experience} an(s).")

    if score_formation >= 80:
        points_forts.append("Niveau d'étude compatible avec l'offre.")

    if competences_manquantes:
        points_faibles.append("Compétences faibles ou non détectées : " + ", ".join(competences_manquantes))

    if technologies_manquantes:
        points_faibles.append("Technologies faibles ou non détectées : " + ", ".join(technologies_manquantes))

    if score_experience < 80 and request.experienceMin:
        points_faibles.append(
            f"Expérience estimée insuffisante : {annees_experience} an(s), {request.experienceMin} demandé(s)."
        )

    if score_formation < 80 and request.niveauEtude:
        points_faibles.append(
            f"Niveau d'étude demandé : {request.niveauEtude}. Niveau équivalent non clairement détecté dans le CV."
        )

    experiences_detectees = []
    if annees_experience > 0:
        experiences_detectees.append(f"{annees_experience} an(s) d'expérience estimée")

    resume_profil = (
        f"Analyse IA du profil pour le poste '{request.titrePoste}'. "
        f"Le modèle a comparé sémantiquement le contenu du CV avec les critères de l'offre."
    )

    justification = (
        f"Score global : {score_global}%. "
        f"Score compétences : {score_competences}%. "
        f"Score technologies : {score_technologies}%. "
        f"Score expérience : {score_experience}%. "
        f"Expérience estimée : {annees_experience} an(s). "
        f"Score formation : {score_formation}%. "
        f"Score sémantique global CV/offre : {score_semantique_global}%."
    )

    return IaMatchingResponseDTO(
        texteExtrait=cv_text,
        competencesDetectees=competences_correspondantes,
        technologiesDetectees=technologies_correspondantes,
        experiencesDetectees=experiences_detectees,
        anneesExperienceEstimees=annees_experience,
        resumeProfil=resume_profil,
        pointsForts=points_forts,
        pointsFaibles=points_faibles,
        scoreGlobal=score_global,
        scoreCompetences=score_competences,
        scoreTechnologies=score_technologies,
        scoreExperience=score_experience,
        scoreFormation=score_formation,
        niveauCompatibilite=niveau,
        competencesCorrespondantes=competences_correspondantes,
        competencesManquantes=competences_manquantes,
        technologiesCorrespondantes=technologies_correspondantes,
        technologiesManquantes=technologies_manquantes,
        justificationIa=justification,
        recommandationIa=resolve_recommandation(score_global)
    )


@app.get("/health")
def health():
    return {
        "status": "UP",
        "service": "ia-matching-service",
        "model": "HuggingFace Inference API - paraphrase-multilingual-MiniLM-L12-v2",
        "version": "3.1.0"
    }