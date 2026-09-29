# 🎯 TalentMatch AI

## AI-Powered CV & Job Matching and Candidate Ranking System

TalentMatch AI is a machine-learning-based recruitment support system that analyzes the compatibility between candidates and job positions.

The system combines multiple NLP and machine-learning techniques to calculate a final candidate-job matching score and rank candidates according to their compatibility with a selected job position.

The application also allows users to upload a new CV in **PDF or DOCX format** and compare it with a selected job position.

> **Important:** TalentMatch AI is a decision-support system. The matching score is a project-defined compatibility score and should not be treated as a hiring probability or as the sole basis for recruitment decisions.

---

# 1. Project Overview

TalentMatch AI provides two main functions:

### 1. Candidate Ranking

- Select a job position.
- Retrieve existing candidates from the project dataset.
- Calculate their matching results.
- Rank candidates according to their final matching percentage.
- View matched and missing skills.

### 2. New CV Matching

- Upload a new PDF or DOCX CV.
- Extract the CV text.
- Compare the CV with a selected job position.
- Calculate TF-IDF similarity.
- Calculate Word2Vec similarity.
- Calculate skill coverage.
- Produce a final matching percentage.
- Display matched and missing skills.

---

# 2. Project Objectives

The main objectives of TalentMatch AI are:

- Analyze CV and job-description text automatically.
- Measure textual similarity between candidates and jobs.
- Identify relevant skills in candidate CVs and job requirements.
- Combine multiple matching techniques into one final score.
- Rank candidates according to their calculated compatibility.
- Provide an interactive application for demonstrating the matching system.
- Allow new CVs to be analyzed without manually modifying the dataset.
- Provide a reusable inference pipeline for new CVs.

---

# 3. Machine Learning Pipeline

The project is organized into a sequence of stages:

```text
Raw Data
   ↓
01 - Data Analysis
   ↓
02 - Preprocessing
   ↓
03 - TF-IDF Baseline
   ↓
04 - Classification
   ↓
05 - Word2Vec
   ↓
06 - Skills Matching
   ↓
07 - Final Matching & Ranking
   ↓
08 - Evaluation & Error Analysis
   ↓
09 - Inference Pipeline
   ↓
10 - Application
   ↓
FastAPI + Streamlit
4. Matching Methodology

TalentMatch AI combines three main components to calculate the final matching score.

4.1 TF-IDF Similarity

TF-IDF is used to represent candidate and job-description text as numerical vectors.

Cosine similarity is then used to measure the similarity between the candidate and the job vectors.

The result is:

TF-IDF Similarity

TF-IDF provides a lexical similarity baseline based on the importance of words in the text.

4.2 Word2Vec Similarity

Word2Vec is used to represent words as dense vector representations.

Document vectors are generated from the Word2Vec model and cosine similarity is used to calculate semantic similarity between the candidate and the job.

The result is:

Word2Vec Similarity

Word2Vec provides a semantic similarity component that can capture relationships between related words.

4.3 Skill Coverage

The system extracts skills from both:

Candidate CV
Job requirements

The system then determines:

Matched Skills
Missing Skills
Skill Coverage Score

Skill coverage is calculated as:

Skill Coverage =
Number of Matched Skills
------------------------
Number of Required Skills

When required skills are available, the coverage is also displayed as a percentage.

The project also supports partial skill matching where applicable.

5. Final Matching Score

The final matching score combines the three matching components:

Final Score =
    0.25 × TF-IDF Similarity
  + 0.25 × Word2Vec Similarity
  + 0.50 × Skill Coverage

The final score is converted to a percentage:

Final Match Percentage =
Final Score × 100

The current project uses:

TF-IDF          → 25%
Word2Vec        → 25%
Skill Coverage  → 50%

Skill coverage receives the largest weight because explicit required-skill matching is a major component of the final ranking.

These weights are a project design choice and do not represent an objective hiring formula.

6. Classification

The project also includes a classification stage.

The classification task uses the project matching score to create two match-quality categories:

Good Match
Poor Match

A matching-score cutoff of:

0.70

is used to define the two categories.

The classification stage includes:

Dummy Classifier baseline
Logistic Regression
Random Forest
Accuracy
Precision
Recall
F1-score
Classification reports
Confusion matrices

The TF-IDF vectorization and classification models are implemented using leakage-safe training pipelines.

Saved classification resources are stored in the Models/ directory.

7. Evaluation and Error Analysis

The project includes a dedicated evaluation and error-analysis stage.

The evaluation includes:

Matching score distributions
Correlation analysis
Difficult-case analysis
Top and lowest matching cases
Precision@5
Recall@5
NDCG@5
Comparison between TF-IDF baseline, Word2Vec, and final matching
Potential false-positive and false-negative analysis

Evaluation results are stored in:

Data/evaluation_error_analysis.csv
Data/ranking_evaluation_results.csv

The ranking evaluation uses the project's available reference matching signal.

These evaluation results should be interpreted as evaluation of the project's matching methodology rather than as ground-truth human hiring decisions.

8. Data Leakage Prevention

The project includes train/test separation in the TF-IDF baseline.

The TF-IDF vectorizer is fitted using the training portion of the data and then used to transform the remaining data.

The train/test split is also used when learning the scaling ranges for the final matching components.

The final matching score scaling parameters are learned from the training portion rather than from the complete dataset.

This helps prevent evaluation information from influencing the learned preprocessing and scaling steps.

9. Application Architecture

The final application consists of two main layers:

                     TalentMatch AI
                           │
             ┌─────────────┴─────────────┐
             │                           │
        Streamlit UI                FastAPI
             │                           │
             └─────────────┬─────────────┘
                           │
                      ML Models
                           │
             ┌─────────────┼─────────────┐
             │             │             │
           TF-IDF       Word2Vec    Skill Matching
             │             │             │
             └─────────────┼─────────────┘
                           │
                    Final Results
Streamlit

Streamlit provides the user interface.

It allows the user to:

Select a job position.
Select the number of candidates.
Rank candidates.
View final matching results.
View skill coverage.
View matched skills.
View missing skills.
Upload a new CV.
Select a target job.
Analyze the uploaded CV.
FastAPI

FastAPI provides the backend API.

It handles:

Candidate ranking
Job-position retrieval
New CV matching
CV file processing
Communication between the application and the matching pipeline
10. API Endpoints

The application currently provides the following endpoints.

GET /

Checks whether the API is running.

Example:

http://127.0.0.1:8000/
POST /rank

Ranks existing candidates for a selected job position.

Example input:

{
    "job_position_name": "data engineer",
    "top_k": 5
}

The endpoint returns matching candidates and their matching results.

GET /data/job-positions

Returns the available job positions.

This endpoint is used by Streamlit to populate the job-position selection controls.

POST /match-cv

Matches a newly uploaded CV against a selected job position.

Supported file formats:

PDF
DOCX

The endpoint calculates:

TF-IDF similarity
Word2Vec similarity
Skill coverage
Final matching score
Final matching percentage
Matched skills
Missing skills
Uploaded filename

The Streamlit interface focuses on the final matching result and skill coverage rather than exposing the individual TF-IDF and Word2Vec percentages.

11. CV Processing

The CV parser supports PDF and DOCX files.

The processing workflow is:

Uploaded CV
    ↓
File Type Detection
    ↓
PDF / DOCX Text Extraction
    ↓
Text Cleaning
    ↓
TF-IDF
    ↓
Word2Vec
    ↓
Skill Matching
    ↓
Final Score

The parser rejects unsupported file types.

12. Project Structure
TalentMatch-AI/

│
├── .gitignore
├── README.md
├── requirements.txt
│
├── .vscode/
│   └── settings.json
│
├── app/
│   └── 10_application/
│       ├── api.py
│       ├── app.py
│       └── cv_parser.py
│
├── Data/
│   ├── classification_results.csv
│   ├── evaluation_error_analysis.csv
│   ├── final_matching_results.csv
│   ├── processed_data.csv
│   ├── ranking_evaluation_results.csv
│   ├── resume_data_for_ranking.csv
│   ├── skills_matching_results.csv
│   ├── tfidf_baseline_results.csv
│   └── word2vec_results.csv
│
├── Demo/
│   └── start_talentmatch.bat
│
├── Models/
│   ├── classification_tfidf_vectorizer.pkl
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── score_scaling.pkl
│   ├── skill_resources.pkl
│   ├── tfidf_vectorizer.pkl
│   └── word2vec_model.model
│
└── Python Files/
    ├── 01_data_analysis.ipynb
    ├── 02_preprocessing.ipynb
    ├── 03_tfidf_baseline.ipynb
    ├── 04_classification.ipynb
    ├── 05_word2vec.ipynb
    ├── 06_skills_matching.ipynb
    ├── 07_final_matching_and_ranking.ipynb
    ├── 08_evaluation_and_error_analysis.ipynb
    ├── 09_inference_pipeline.ipynb
    ├── 10_application.ipynb
    └── project_utils.py
13. Dataset and Final Results

The final matching results contain:

Rows: 9544
Columns: 43

The final matching score was validated to ensure that:

No final scores are missing.
No final scores are below 0.
No final scores are above 100.

The final results are stored in:

Data/final_matching_results.csv

The project also generates additional results:

Data/classification_results.csv
Data/evaluation_error_analysis.csv
Data/ranking_evaluation_results.csv
Data/skills_matching_results.csv
Data/tfidf_baseline_results.csv
Data/word2vec_results.csv

The matching score is a project-defined compatibility score and should not be interpreted as a ground-truth hiring probability.

14. Model Resources

The project uses the following saved resources:

Models/

├── classification_tfidf_vectorizer.pkl
├── logistic_regression_model.pkl
├── random_forest_model.pkl
├── score_scaling.pkl
├── skill_resources.pkl
├── tfidf_vectorizer.pkl
└── word2vec_model.model

The trained Word2Vec model contains approximately:

4,958 words
15. Installation

Make sure Python is installed.

From the project root:

pip install -r requirements.txt
16. Running the Application
Step 1 — Start FastAPI

Open a terminal in the project root:

uvicorn api:app --app-dir app/10_application --reload --port 8000

FastAPI will normally be available at:

http://127.0.0.1:8000
Step 2 — Open API Documentation

FastAPI provides interactive Swagger documentation at:

http://127.0.0.1:8000/docs
Step 3 — Start Streamlit

Open another terminal in the project root:

streamlit run app\10_application\app.py

Streamlit will normally be available at:

http://localhost:8501
17. Using the Application
Candidate Ranking
Open the Streamlit application.
Select a job position.
Select the number of candidates.
Click Rank Candidates.
Review the ranking results.

The interface displays:

Candidate ID
Final Match %
Skill Coverage %
Matched Skills
Missing Skills
New CV Matching
Open the Streamlit application.
Navigate to Match a New CV.
Upload a PDF or DOCX CV.
Select the target job position.
Click Analyze CV.
Review the calculated matching result.

The interface displays:

Final Match
Skill Coverage
Matched Skills
Missing Skills
18. Example API Workflow

For a new CV:

CV File
   ↓
FastAPI /match-cv
   ↓
CV Text Extraction
   ↓
Text Cleaning
   ↓
TF-IDF Similarity
   ↓
Word2Vec Similarity
   ↓
Skill Matching
   ↓
Final Matching Score
   ↓
JSON Response
   ↓
Streamlit
19. Technologies Used
Technology	Purpose
Python	Main programming language
Pandas	Data processing
NumPy	Numerical operations
Scikit-learn	TF-IDF, classification and similarity
Gensim	Word2Vec
Joblib	Model and resource loading
pypdf	PDF CV extraction
python-docx	DOCX CV extraction
FastAPI	Backend API
Uvicorn	API server
Streamlit	User interface
Requests	Communication between Streamlit and FastAPI
20. Project Workflow

The complete system workflow is:

Candidate CV
     +
Job Description
     │
     ▼
Text Preprocessing
     │
     ├───────────────┐
     ▼               ▼
  TF-IDF          Word2Vec
     │               │
     └───────┬───────┘
             │
             ▼
      Skill Matching
             │
             ▼
      Score Combination
             │
             ▼
       Final Match %
             │
             ▼
      Candidate Ranking

For a newly uploaded CV:

PDF / DOCX
    ↓
Text Extraction
    ↓
Preprocessing
    ↓
Matching Pipeline
    ↓
Final Match %
    ↓
Matched / Missing Skills
21. Responsible AI

TalentMatch AI is designed as a recruitment decision-support system.

The system should not be used as the sole basis for employment decisions.

Important considerations include:

Human review should remain part of the recruitment process.
Matching scores should not be treated as objective measures of candidate quality.
Protected characteristics should not be used as matching features.
The system may produce incorrect matches or miss relevant experience.
CV formatting and extraction quality can affect the results.
The final score is a project-defined compatibility measure, not a probability of employment success.
22. Limitations

Current limitations include:

CV text extraction quality may vary between different document formats.
Word2Vec similarity depends on the vocabulary learned by the model.
Exact and partial skill matching depends on the available skill resources.
The final matching weights are manually defined project parameters.
The available reference matching signal does not represent independent human hiring decisions.
The system may produce false positives or false negatives.
The system does not replace professional recruitment judgment.
23. Future Work

Possible future improvements include:

Supporting additional CV file formats.
Improving CV information extraction.
Using transformer-based NLP models.
Improving semantic similarity using contextual embeddings.
Adding explainable matching recommendations.
Adding candidate filtering and search.
Adding authentication and user management.
Deploying the system as a publicly accessible web application.
Expanding the system with additional recruitment analytics.
Improving evaluation using independently labeled human-reviewed data.
24. Project Outcome

TalentMatch AI provides an end-to-end CV and job matching system that connects an NLP and machine-learning pipeline with a functional application.

The system allows users to:

Analyze existing candidates.
Rank candidates for specific job positions.
Upload new CVs.
Compare new CVs with selected jobs.
Calculate textual similarity.
Analyze skill coverage.
Identify matched and missing skills.
Generate a final matching percentage.
Access the matching functionality through FastAPI.
Use the system through a Streamlit interface.

The project demonstrates an end-to-end workflow from data preprocessing and NLP modeling to evaluation, inference, API development, and interactive application deployment.

25. Disclaimer

TalentMatch AI is a decision-support system developed as an academic machine-learning project.

Its matching scores are intended to assist human evaluation and should not be treated as the sole basis for recruitment decisions.

The final matching score represents the compatibility methodology implemented in this project and does not represent a guaranteed measure of candidate suitability, job performance, or employment probability.