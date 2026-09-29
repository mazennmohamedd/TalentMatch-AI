from pathlib import Path
import ast
import sys
import joblib
import numpy as np
import pandas as pd
 
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
 
 
PROJECT_DIR = Path(__file__).resolve().parents[2]
 
DATA_PATH = PROJECT_DIR / "Data" / "final_matching_results.csv"
 
MODELS_DIR = PROJECT_DIR / "Models"
 
PYTHON_FILES_DIR = PROJECT_DIR / "Python Files"
APPLICATION_DIR = Path(__file__).resolve().parent
 
if str(PYTHON_FILES_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_FILES_DIR))
 
if str(APPLICATION_DIR) not in sys.path:
    sys.path.insert(0, str(APPLICATION_DIR))
 
 
from project_utils import (
    clean_text, tokenize_text, document_vector, build_skill_index,
    find_skills_in_text, skill_match_score, scale_value, combine_scores
)
 
from cv_parser import extract_cv_text
 
 
df = pd.read_csv(DATA_PATH)
 
 
application_columns = [
    "candidate_record_id",
    "job_position_name",
    "job_text",
    "tfidf_similarity",
    "word2vec_similarity",
    "skill_coverage_percentage",
    "skill_coverage_score",
    "matched_skills",
    "missing_skills",
    "final_match_percentage"
]
 
 
application_df = df[application_columns].copy()
 
 
def _to_list(value):
    """The CSV stores lists as strings, e.g. "['python', 'sql']".
    Convert them back to real lists (empty list if missing or invalid)."""
 
    if isinstance(value, list):
        return value
 
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            return []
 
    return []
 
 
application_df["matched_skills"] = application_df["matched_skills"].apply(_to_list)
application_df["missing_skills"] = application_df["missing_skills"].apply(_to_list)
 
 
tfidf_vectorizer = joblib.load(
    str(MODELS_DIR / "tfidf_vectorizer.pkl")
)
 
word2vec_model = Word2Vec.load(
    str(MODELS_DIR / "word2vec_model.model")
)
 
skill_resources = joblib.load(
    str(MODELS_DIR / "skill_resources.pkl")
)
 
skill_index = build_skill_index(skill_resources["skill_vocabulary"])
skill_synonyms = skill_resources["skill_synonyms"]
scaling = joblib.load(str(MODELS_DIR / "score_scaling.pkl"))
 
 
def rank_candidates(job_position, top_k=5):
 
    results = application_df[
        application_df["job_position_name"] == job_position
    ].copy()
 
    return results.sort_values(
        "final_match_percentage",
        ascending=False
    ).head(top_k)
 
 
def get_job_text(job_position):
 
    job_rows = application_df[
        application_df["job_position_name"]
        .astype(str)
        .str.lower()
        == job_position.lower()
    ].copy()
 
    if job_rows.empty:
        return None
 
    job_rows = job_rows[
        job_rows["job_text"].notna()
    ]
 
    if job_rows.empty:
        return None
 
    job_text = job_rows.iloc[0]["job_text"]
 
    return clean_text(job_text)
 
 
def calculate_cv_match(candidate_text, job_position):
 
    job_text = get_job_text(
        job_position
    )
 
    if job_text is None:
        raise ValueError(
            "Job position was not found or has no job text."
        )
 
    candidate_text = clean_text(
        candidate_text
    )
 
    if not candidate_text.strip():
        raise ValueError(
            "No readable text was found in the CV."
        )
 
    # TF-IDF similarity
    tfidf_score = cosine_similarity(
        tfidf_vectorizer.transform([candidate_text]),
        tfidf_vectorizer.transform([job_text])
    )[0][0]
 
    # Word2Vec similarity
    candidate_vector = document_vector(
        tokenize_text(candidate_text),
        word2vec_model
    )
 
    job_vector = document_vector(
        tokenize_text(job_text),
        word2vec_model
    )
 
    if (
        np.linalg.norm(candidate_vector) > 0
        and
        np.linalg.norm(job_vector) > 0
    ):
        word2vec_score = cosine_similarity(
            [candidate_vector],
            [job_vector]
        )[0][0]
    else:
        word2vec_score = 0.0
 
    # Skills (same helpers as Notebook 06)
    candidate_skills = find_skills_in_text(
        candidate_text,
        skill_index,
        skill_synonyms
    )
 
    required_skills = find_skills_in_text(
        job_text,
        skill_index,
        skill_synonyms
    )
 
    coverage, matched_skills, missing_skills = skill_match_score(
        candidate_skills,
        required_skills
    )
 
    # Final score (same logic as Notebook 07)
    tfidf_scaled = scale_value(
        tfidf_score,
        **scaling["tfidf_similarity"]
    )
 
    word2vec_scaled = scale_value(
        word2vec_score,
        **scaling["word2vec_similarity"]
    )
 
    final_match_score = combine_scores(
        tfidf_scaled,
        word2vec_scaled,
        coverage
    )
 
    # JSON cannot carry NaN, so use None when the job has no recognized skills
    no_coverage = pd.isna(coverage)
 
    return {
        "job_position_name": job_position,
        "tfidf_similarity": float(tfidf_score),
        "word2vec_similarity": float(word2vec_score),
        "skill_coverage_score": (
            None if no_coverage else float(coverage)
        ),
        "skill_coverage_percentage": (
            None if no_coverage else float(coverage * 100)
        ),
        "final_match_score": float(final_match_score),
        "final_match_percentage": float(final_match_score * 100),
        "matched_skills": list(matched_skills),
        "missing_skills": list(missing_skills)
    }
 
 
app = FastAPI(
    title="TalentMatch AI",
    description=(
        "CV and Job Matching and Candidate Ranking System"
    ),
    version="1.0"
)
 
 
class RankingRequest(BaseModel):
 
    job_position_name: str = Field(
        min_length=1
    )
 
    top_k: int = Field(
        default=5,
        ge=1,
        le=20
    )
 
 
@app.get("/")
def home():
 
    return {
        "message": "TalentMatch AI API is running"
    }
 
 
@app.post("/rank")
def rank_job(
    request: RankingRequest
):
 
    results = rank_candidates(
        request.job_position_name,
        request.top_k
    )
 
    if results.empty:
 
        return {
            "message": "Job position not found.",
            "results": []
        }
 
    # NaN is not valid JSON, so replace it with None
    records = (
        results.astype(object)
        .where(results.notna(), None)
        .to_dict(orient="records")
    )
 
    return {
        "job_position_name": (
            request.job_position_name
        ),
        "number_of_results": len(results),
        "results": records
    }
 
 
@app.get("/data/job-positions")
def get_job_positions():
 
    positions = sorted(
        application_df[
            "job_position_name"
        ]
        .dropna()
        .unique()
        .tolist()
    )
 
    return {
        "job_positions": positions
    }
 
 
@app.post("/match-cv")
async def match_cv(
    file: UploadFile = File(...),
    job_position_name: str = ""
):
 
    if not job_position_name.strip():
 
        raise HTTPException(
            status_code=400,
            detail="Job position is required."
        )
 
    if not file.filename:
 
        raise HTTPException(
            status_code=400,
            detail="CV file is required."
        )
 
    filename = file.filename.lower()
 
    if not (
        filename.endswith(".pdf")
        or
        filename.endswith(".docx")
    ):
 
        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF and DOCX files are supported."
            )
        )
 
    try:
 
        file_bytes = await file.read()
 
        candidate_text = extract_cv_text(
            file_bytes,
            file.filename
        )
 
        result = calculate_cv_match(
            candidate_text,
            job_position_name
        )
 
        result["filename"] = file.filename
 
        return result
 
    except HTTPException:
 
        raise
 
    except Exception as error:
 
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )