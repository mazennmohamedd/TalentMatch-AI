import re
import numpy as np
 
# ---------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------
 
def clean_text(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text
 
 
def tokenize_text(text):
    text = clean_text(text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()
 
 
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _STOP
except Exception:  # sklearn missing -> no stop-word removal
    _STOP = frozenset()
 
 
def document_vector(tokens, model, remove_stopwords=True):
    """Average Word2Vec vector of a document.
 
    Accepts a token list (preferred). If a plain string is passed by mistake,
    it is tokenized instead of being iterated character by character.
    """
    if isinstance(tokens, str):
        tokens = tokenize_text(tokens)
 
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOP]
 
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)
 
 
# ---------------------------------------------------------------
# Skill helpers
# ---------------------------------------------------------------
 
# Words that are too generic to count as a "skill" on their own.
GENERIC_SKILLS = {
    "management", "developer", "engineer", "executive", "manager",
    "experience", "development", "design", "support", "sales",
    "analysis", "skills", "communication", "work", "team", "business",
    "software", "data", "system", "systems", "services", "service",
    "training", "testing", "planning", "research", "operations",
}
 
SINGLE_LETTER_SKILLS = {"c", "r"}
 
 
def normalize_skill_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    words = [w.rstrip(".") for w in text.split()]
    return " ".join(w for w in words if w)
 
 
def apply_synonyms(text, synonyms):
    if not synonyms:
        return text
    for old, new in synonyms.items():
        pattern = r"(?<!\w)" + re.escape(old) + r"(?!\w)"
        text = re.sub(pattern, new, text)
    return text
 
 
def clean_skill_vocabulary(skill_counter, min_freq=3, max_words=4):
    """Keep only real, reasonably frequent, short skills."""
    vocab = set()
    for skill, freq in skill_counter.items():
        skill = normalize_skill_text(skill)
        if not skill or freq < min_freq:
            continue
        n_words = len(skill.split())
        if n_words > max_words:
            continue
        if skill in GENERIC_SKILLS:
            continue
        if re.fullmatch(r"skill \d+", skill):          # dataset placeholders
            continue
        if len(skill) < 2 and skill not in SINGLE_LETTER_SKILLS:
            continue
        vocab.add(skill)
    return sorted(vocab)
 
 
def build_skill_index(skill_vocabulary, max_words=4):
    """Pre-compute a lookup structure (call once, reuse many times)."""
    return frozenset(
        tuple(s.split())
        for s in skill_vocabulary
        if 0 < len(s.split()) <= max_words
    )
 
 
def find_skills_in_text(text, skill_vocabulary, synonyms=None, max_words=4):
    """Longest-match, NON-overlapping skill extraction.
 
    'human resource management' -> ['human resource management']
    (not also 'human resource' and 'management').
    """
    index = (
        skill_vocabulary
        if isinstance(skill_vocabulary, frozenset)
        else build_skill_index(skill_vocabulary, max_words)
    )
 
    text = apply_synonyms(normalize_skill_text(text), synonyms)
    words = text.split()
 
    found, i = set(), 0
    while i < len(words):
        for n in range(min(max_words, len(words) - i), 0, -1):
            phrase = tuple(words[i:i + n])
            if phrase in index:
                found.add(" ".join(phrase))
                i += n
                break
        else:
            i += 1
    return sorted(found)
 
 
def skill_match_score(candidate_skills, required_skills, partial_threshold=0.5):
    """Coverage with partial credit.
 
    Returns (coverage, matched, missing). coverage is NaN when the job has no
    recognized required skills.
    """
    if not required_skills:
        return np.nan, [], []
 
    candidate_set = set(candidate_skills)
    matched, missing, total = [], [], 0.0
 
    for req in required_skills:
        req_words = set(req.split())
 
        if req in candidate_set:
            matched.append(req)
            total += 1.0
            continue
 
        best = max(
            (len(req_words & set(c.split())) / len(req_words)
             for c in candidate_set),
            default=0.0,
        )
 
        if best >= 1.0:
            matched.append(req)
            total += 1.0
        elif best >= partial_threshold:
            matched.append(f"{req} (partial)")
            total += best
        else:
            missing.append(req)
 
    return total / len(required_skills), matched, missing
 
 
# ---------------------------------------------------------------
# Score combination
# ---------------------------------------------------------------
 
W_TFIDF, W_W2V, W_SKILL = 0.25, 0.25, 0.50
 
 
def scale_value(x, low, high):
    """Map a raw similarity onto 0-1 using saved dataset percentiles."""
    if high <= low:
        return 0.0
    return float(min(1.0, max(0.0, (x - low) / (high - low))))
 
 
def combine_scores(tfidf_scaled, w2v_scaled, skill_coverage):
    """If the job has no recognized skills, drop that term and renormalize."""
    if skill_coverage is None or np.isnan(skill_coverage):
        return 0.5 * tfidf_scaled + 0.5 * w2v_scaled
    return (
        W_TFIDF * tfidf_scaled
        + W_W2V * w2v_scaled
        + W_SKILL * skill_coverage
    )