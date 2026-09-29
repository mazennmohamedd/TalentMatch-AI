
import ast
import requests
import pandas as pd
import streamlit as st


API_URL = "http://127.0.0.1:8000"


def render_skill_badges(skills, kind="matched"):
    if isinstance(skills, str):
        try:
            skills = ast.literal_eval(skills)
        except (ValueError, SyntaxError):
            skills = [skills]

    if not skills or skills == ["None"]:
        st.write("None")
        return

    if kind == "matched":
        background = "rgba(46, 125, 50, 0.12)"
        color = "#66bb6a"
        border = "rgba(46, 125, 50, 0.35)"
        icon = "✓"
    else:
        background = "rgba(198, 40, 40, 0.12)"
        color = "#ef5350"
        border = "rgba(198, 40, 40, 0.35)"
        icon = "×"

    badges = ""

    for skill in skills:
        skill = str(skill).strip()
        badges += f"""
        <span style="
            display:inline-block;
            padding:7px 12px;
            margin:4px 5px 4px 0;
            border-radius:20px;
            background:{background};
            color:{color};
            border:1px solid {border};
            font-family:Arial, sans-serif;
            font-size:14px;
            font-weight:500;
        ">
            {icon} {skill}
        </span>
        """

    st.markdown(
        f'<div style="line-height:2.2;">{badges}</div>',
        unsafe_allow_html=True
    )


st.set_page_config(
    page_title="TalentMatch AI",
    page_icon="🎯",
    layout="wide"
)


st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        margin-top: 0;
        opacity: 0.7;
    }

    .section-title {
        font-size: 25px;
        font-weight: 600;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .candidate-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-bottom: 20px;
    }

    .match-score {
        font-size: 32px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🎯 TalentMatch AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered CV & Job Matching and Candidate Ranking System'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

st.info(
    "TalentMatch AI analyzes candidate-job compatibility using "
    "semantic matching and skill coverage."
)


# --------------------------------------------------
# GET JOB POSITIONS FROM FASTAPI
# --------------------------------------------------

try:

    response = requests.get(
        f"{API_URL}/data/job-positions",
        timeout=10
    )

    if response.status_code == 200:

        job_positions = response.json().get(
            "job_positions",
            []
        )

    else:

        st.error(
            f"FastAPI returned error {response.status_code}."
        )

        job_positions = []


except requests.exceptions.ConnectionError:

    st.error(
        "FastAPI is not running. "
        "Please start FastAPI on port 8000."
    )

    job_positions = []


except requests.exceptions.RequestException:

    st.error(
        "Could not connect to the FastAPI server."
    )

    job_positions = []


# --------------------------------------------------
# SEARCH SECTION
# --------------------------------------------------

if job_positions:

    st.markdown(
        '<div class="section-title">'
        '🔎 Candidate Search'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([3, 1])

    with col1:

        selected_job = st.selectbox(
            "Select Job Position",
            job_positions
        )

    with col2:

        top_k = st.number_input(
            "Number of Candidates",
            min_value=1,
            max_value=20,
            value=5,
            step=1
        )

    st.write("")

    rank_button = st.button(
        "🚀 Rank Candidates",
        use_container_width=True
    )


    # --------------------------------------------------
    # RANK CANDIDATES
    # --------------------------------------------------

    if rank_button:

        with st.spinner(
            "Analyzing candidates..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/rank",
                    json={
                        "job_position_name": selected_job,
                        "top_k": int(top_k)
                    },
                    timeout=30
                )


                if response.status_code == 200:

                    data = response.json()

                    results = data.get(
                        "results",
                        []
                    )


                    # --------------------------------------------------
                    # NO RESULTS
                    # --------------------------------------------------

                    if not results:

                        st.warning(
                            data.get(
                                "message",
                                "No candidates found."
                            )
                        )


                    else:

                        results_df = pd.DataFrame(
                            results
                        )


                        # --------------------------------------------------
                        # SUCCESS MESSAGE
                        # --------------------------------------------------

                        st.success(
                            f"Found {len(results_df)} matching "
                            f"candidates for **{selected_job}**."
                        )


                        # --------------------------------------------------
                        # MATCHING OVERVIEW
                        # --------------------------------------------------

                        st.markdown(
                            '<div class="section-title">'
                            '📊 Matching Overview'
                            '</div>',
                            unsafe_allow_html=True
                        )


                        best_match = results_df[
                            "final_match_percentage"
                        ].max()


                        average_match = results_df[
                            "final_match_percentage"
                        ].mean()


                        average_skill_coverage = results_df[
                            "skill_coverage_percentage"
                        ].mean()


                        metric1, metric2, metric3, metric4 = (
                            st.columns(4)
                        )


                        with metric1:

                            st.metric(
                                "👥 Candidates",
                                len(results_df)
                            )


                        with metric2:

                            st.metric(
                                "🏆 Best Match",
                                f"{best_match:.2f}%"
                            )


                        with metric3:

                            st.metric(
                                "📊 Average Match",
                                f"{average_match:.2f}%"
                            )


                        with metric4:

                            if pd.notna(
                                average_skill_coverage
                            ):

                                st.metric(
                                    "🎯 Avg Skill Coverage",
                                    f"{average_skill_coverage:.2f}%"
                                )

                            else:

                                st.metric(
                                    "🎯 Avg Skill Coverage",
                                    "N/A"
                                )


                        # --------------------------------------------------
                        # CANDIDATE COMPARISON CHART
                        # --------------------------------------------------

                        st.markdown(
                            '<div class="section-title">'
                            '📈 Candidate Comparison'
                            '</div>',
                            unsafe_allow_html=True
                        )


                        chart_data = results_df[
                            [
                                "candidate_record_id",
                                "final_match_percentage"
                            ]
                        ].copy()


                        chart_data[
                            "candidate_record_id"
                        ] = chart_data[
                            "candidate_record_id"
                        ].astype(str)


                        chart_data = chart_data.set_index(
                            "candidate_record_id"
                        )


                        st.bar_chart(
                            chart_data,
                            use_container_width=True
                        )


                        # --------------------------------------------------
                        # RANKING TABLE
                        # --------------------------------------------------

                        st.markdown(
                            '<div class="section-title">'
                            '🏆 Candidate Ranking'
                            '</div>',
                            unsafe_allow_html=True
                        )


                        ranking_table = results_df[
                            [
                                "candidate_record_id",
                                "job_position_name",
                                "final_match_percentage",
                                "skill_coverage_percentage",
                                "tfidf_similarity",
                                "word2vec_similarity"
                            ]
                        ].copy()


                        ranking_table.insert(
                            0,
                            "Rank",
                            range(
                                1,
                                len(ranking_table) + 1
                            )
                        )


                        ranking_table = ranking_table.rename(
                            columns={
                                "candidate_record_id":
                                    "Candidate ID",

                                "job_position_name":
                                    "Job Position",

                                "final_match_percentage":
                                    "Final Match %",

                                "skill_coverage_percentage":
                                    "Skill Coverage %"
                            }
                        )


                        st.dataframe(
                            ranking_table,
                            use_container_width=True,
                            hide_index=True
                        )


                        # --------------------------------------------------
                        # DETAILED CANDIDATES
                        # --------------------------------------------------

                        st.markdown(
                            '<div class="section-title">'
                            '👤 Candidate Details'
                            '</div>',
                            unsafe_allow_html=True
                        )


                        for index, row in results_df.iterrows():

                            candidate_id = row[
                                "candidate_record_id"
                            ]


                            final_match = row[
                                "final_match_percentage"
                            ]


                            skill_coverage = row[
                                "skill_coverage_percentage"
                            ]


                            tfidf = row[
                                "tfidf_similarity"
                            ]


                            word2vec = row[
                                "word2vec_similarity"
                            ]


                            st.markdown(
                                '<div class="candidate-card">',
                                unsafe_allow_html=True
                            )


                            col1, col2 = st.columns(
                                [3, 1]
                            )


                            with col1:

                                st.markdown(
                                    f"### #{index + 1} "
                                    f"Candidate {candidate_id}"
                                )

                                st.write(
                                    f"**Job Position:** "
                                    f"{row['job_position_name']}"
                                )


                            with col2:

                                st.markdown(
                                    '<div class="match-score">'
                                    f"{final_match:.2f}%"
                                    '</div>',
                                    unsafe_allow_html=True
                                )

                                st.caption(
                                    "Final Match"
                                )


                            # Match progress

                            progress_value = max(
                                0,
                                min(
                                    float(final_match) / 100,
                                    1
                                )
                            )


                            st.progress(
                                progress_value
                            )


                            if pd.notna(
                                skill_coverage
                            ):

                                st.metric(
                                    "Skill Coverage",
                                    f"{skill_coverage:.2f}%"
                                )

                            else:

                                st.metric(
                                    "Skill Coverage",
                                    "N/A"
                                )


                            st.write("")


                            # Skills

                            skill1, skill2 = st.columns(2)


                            with skill1:

                                st.markdown("### ✅ Matched Skills")

                                matched_skills = row.get(
                                    "matched_skills",
                                    []
                                )

                                if isinstance(matched_skills, str):
                                    try:
                                        import ast
                                        matched_skills = ast.literal_eval(matched_skills)
                                    except:
                                        matched_skills = [matched_skills]

                                if matched_skills:

                                    matched_html = ""

                                    for skill in matched_skills:

                                        matched_html += f"""
                                        <span style="
                                            display:inline-block;
                                            padding:7px 12px;
                                            margin:4px 5px 4px 0;
                                            border-radius:20px;
                                            background:rgba(46, 125, 50, 0.12);
                                            color:#66bb6a;
                                            border:1px solid rgba(46, 125, 50, 0.35);
                                            font-family:Arial, sans-serif;
                                            font-size:14px;
                                            font-weight:500;
                                        ">
                                            ✓ {skill}
                                        </span>
                                        """

                                    st.markdown(
                                        matched_html,
                                        unsafe_allow_html=True
                                    )

                                else:

                                    st.write("No matched skills.")


                            with skill2:

                                st.markdown("### ❌ Missing Skills")

                                missing_skills = row.get(
                                    "missing_skills",
                                    []
                                )

                                if isinstance(missing_skills, str):
                                    try:
                                        import ast
                                        missing_skills = ast.literal_eval(missing_skills)
                                    except:
                                        missing_skills = [missing_skills]

                                if missing_skills:

                                    missing_html = ""

                                    for skill in missing_skills:

                                        missing_html += f"""
                                        <span style="
                                            display:inline-block;
                                            padding:7px 12px;
                                            margin:4px 5px 4px 0;
                                            border-radius:20px;
                                            background:rgba(198, 40, 40, 0.12);
                                            color:#ef5350;
                                            border:1px solid rgba(198, 40, 40, 0.35);
                                            font-family:Arial, sans-serif;
                                            font-size:14px;
                                            font-weight:500;
                                        ">
                                            × {skill}
                                        </span>
                                        """

                                    st.markdown(
                                        missing_html,
                                        unsafe_allow_html=True
                                    )

                                else:

                                    st.write("No missing skills.")


                            st.markdown(
                                '</div>',
                                unsafe_allow_html=True
                            )


                else:

                    st.error(
                        f"FastAPI returned error "
                        f"{response.status_code}."
                    )


            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to FastAPI. "
                    "Make sure the FastAPI server is running."
                )


            except requests.exceptions.Timeout:

                st.error(
                    "The FastAPI request took too long."
                )


            except requests.exceptions.RequestException as error:

                st.error(
                    f"Request error: {error}"
                )


else:

    st.warning(
        "No job positions are available. "
        "Make sure FastAPI is running correctly."
    )



# --------------------------------------------------
# NEW CV MATCHING SECTION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">'
    '📄 Match a New CV'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Upload a PDF or DOCX CV and compare it with a selected "
    "job using the existing TalentMatch AI models."
)

uploaded_cv = st.file_uploader(
    "Upload CV",
    type=["pdf", "docx"],
    key="new_cv_upload"
)

if job_positions:

    cv_job_position = st.selectbox(
        "Select Target Job",
        job_positions,
        key="new_cv_job_position"
    )

    analyze_cv_button = st.button(
        "🔍 Analyze CV",
        use_container_width=True
    )

    if analyze_cv_button:

        if uploaded_cv is None:

            st.warning(
                "Please upload a PDF or DOCX CV."
            )

        else:

            files = {
                "file": (
                    uploaded_cv.name,
                    uploaded_cv.getvalue(),
                    uploaded_cv.type
                )
            }

            with st.spinner(
                "Extracting CV text and calculating the match..."
            ):
                try:
                    response = requests.post(
                        f"{API_URL}/match-cv",
                        params={"job_position_name": cv_job_position},
                        files=files,
                        timeout=120
                    )

                    if response.status_code == 200:
                        cv_result = response.json()
                        st.success("CV analyzed successfully.")
                        st.markdown(
                            '<div class="section-title">'
                            '🎯 CV Match Result'
                            '</div>',
                            unsafe_allow_html=True
                        )
                        st.metric(
                            "Final Match",
                            f"{cv_result['final_match_percentage']:.2f}%"
                        )
                        st.metric(
                            "Skill Coverage",
                            f"{cv_result['skill_coverage_percentage']:.2f}%"
                        )

                        skill1, skill2 = st.columns(2)

                        with skill1:

                            st.markdown("### ✅ Matched Skills")

                            matched_skills = cv_result.get(
                                "matched_skills",
                                []
                            )

                            if isinstance(matched_skills, str):
                                try:
                                    import ast
                                    matched_skills = ast.literal_eval(matched_skills)
                                except:
                                    matched_skills = [matched_skills]

                            if matched_skills:

                                matched_html = ""

                                for skill in matched_skills:

                                    matched_html += f"""
                                    <span style="
                                        display:inline-block;
                                        padding:7px 12px;
                                        margin:4px 5px 4px 0;
                                        border-radius:20px;
                                        background:rgba(46, 125, 50, 0.12);
                                        color:#66bb6a;
                                        border:1px solid rgba(46, 125, 50, 0.35);
                                        font-family:Arial, sans-serif;
                                        font-size:14px;
                                        font-weight:500;
                                    ">
                                        ✓ {skill}
                                    </span>
                                    """

                                st.markdown(
                                    matched_html,
                                    unsafe_allow_html=True
                                )

                            else:

                                st.write("No matched skills.")


                        with skill2:

                            st.markdown("### ❌ Missing Skills")

                            missing_skills = cv_result.get(
                                "missing_skills",
                                []
                            )

                            if isinstance(missing_skills, str):
                                try:
                                    import ast
                                    missing_skills = ast.literal_eval(missing_skills)
                                except:
                                    missing_skills = [missing_skills]

                            if missing_skills:

                                missing_html = ""

                                for skill in missing_skills:

                                    missing_html += f"""
                                    <span style="
                                        display:inline-block;
                                        padding:7px 12px;
                                        margin:4px 5px 4px 0;
                                        border-radius:20px;
                                        background:rgba(198, 40, 40, 0.12);
                                        color:#ef5350;
                                        border:1px solid rgba(198, 40, 40, 0.35);
                                        font-family:Arial, sans-serif;
                                        font-size:14px;
                                        font-weight:500;
                                    ">
                                        × {skill}
                                    </span>
                                    """

                                st.markdown(
                                    missing_html,
                                    unsafe_allow_html=True
                                )

                            else:

                                st.write("No missing skills.")

                    else:
                        try:
                            error_message = response.json().get(
                                "detail",
                                "CV matching failed."
                            )
                        except ValueError:
                            error_message = "CV matching failed."
                        st.error(error_message)

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Could not connect to FastAPI. "
                        "Make sure the FastAPI server is running."
                    )
                except requests.exceptions.Timeout:
                    st.error("The CV matching request took too long.")
                except requests.exceptions.RequestException as error:
                    st.error(f"Request error: {error}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.write("")

st.divider()

st.caption(
    "TalentMatch AI • CV & Job Matching and Candidate Ranking System"
)

