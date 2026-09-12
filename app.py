import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MedRisk AI",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD MODEL
# =========================================================

try:
    with open("patient_risk_model.pkl", "rb") as file:
        model = pickle.load(file)

except FileNotFoundError:
    st.error("patient_risk_model.pkl was not found.")
    st.info("Please run train_model.py first.")
    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* ==============================
   GLOBAL
   ============================== */

.stApp {
    background: #f4f7fb;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}


/* ==============================
   SIDEBAR
   ============================== */

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #dfe7f0;
}

[data-testid="stSidebar"] h1 {
    color: #123f6d;
    font-size: 28px;
    font-weight: 800;
}

[data-testid="stSidebar"] p {
    color: #718096;
}


/* ==============================
   HERO
   ============================== */

.hero-box {
    background: linear-gradient(135deg, #123f6d, #1976b8);
    padding: 34px 38px;
    border-radius: 22px;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(18, 63, 109, 0.15);
}

.hero-title {
    color: white;
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 7px;
}

.hero-subtitle {
    color: #dcecff;
    font-size: 16px;
}


/* ==============================
   PAGE TITLES
   ============================== */

.page-title {
    color: #123f6d;
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 5px;
}

.page-subtitle {
    color: #718096;
    font-size: 15px;
    margin-bottom: 25px;
}


/* ==============================
   STAT CARDS
   ============================== */

.stat-card {
    background: white;
    border: 1px solid #e0e8f1;
    border-radius: 17px;
    padding: 22px;
    min-height: 125px;
    box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
    transition: 0.2s ease;
}

.stat-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 9px 25px rgba(15, 23, 42, 0.09);
}

.stat-label {
    color: #718096;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

.stat-value {
    color: #123f6d;
    font-size: 30px;
    font-weight: 800;
    margin-top: 9px;
}

.stat-description {
    color: #9aa8b8;
    font-size: 12px;
    margin-top: 5px;
}


/* ==============================
   CONTENT CARD
   ============================== */

.content-card {
    background: white;
    border: 1px solid #e0e8f1;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 5px 18px rgba(15, 23, 42, 0.04);
}

.card-title {
    color: #123f6d;
    font-size: 20px;
    font-weight: 750;
}

.card-subtitle {
    color: #718096;
    font-size: 13px;
    margin-top: 4px;
    margin-bottom: 18px;
}


/* ==============================
   RISK BOX
   ============================== */

.low-risk {
    background: #eaf8f0;
    border: 1px solid #b8e4ca;
    color: #197044;
    border-radius: 15px;
    padding: 18px;
    text-align: center;
    font-size: 25px;
    font-weight: 800;
}

.high-risk {
    background: #fff0f0;
    border: 1px solid #f0c0c0;
    color: #c0392b;
    border-radius: 15px;
    padding: 18px;
    text-align: center;
    font-size: 25px;
    font-weight: 800;
}

.not-assessed {
    background: #eef5fc;
    border: 1px solid #cddfed;
    color: #1769aa;
    border-radius: 15px;
    padding: 18px;
    text-align: center;
    font-size: 22px;
    font-weight: 750;
}


/* ==============================
   PARAMETER CARDS
   ============================== */

.parameter-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 13px;
    padding: 16px;
    margin-bottom: 12px;
}

.parameter-title {
    color: #526273;
    font-size: 13px;
    font-weight: 700;
}

.parameter-value {
    color: #123f6d;
    font-size: 22px;
    font-weight: 800;
    margin-top: 4px;
}

.normal-status {
    color: #18804b;
    font-size: 12px;
    font-weight: 700;
}

.warning-status {
    color: #c17b16;
    font-size: 12px;
    font-weight: 700;
}

.high-status {
    color: #c0392b;
    font-size: 12px;
    font-weight: 700;
}


/* ==============================
   SIMULATOR
   ============================== */

.simulator-banner {
    background: linear-gradient(135deg, #eef6ff, #f5f1ff);
    border: 1px solid #dce6f2;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 22px;
}

.simulator-title {
    color: #315b91;
    font-size: 21px;
    font-weight: 800;
}

.simulator-text {
    color: #718096;
    font-size: 14px;
    margin-top: 7px;
}


/* ==============================
   AI EXPLAINABILITY
   ============================== */

.ai-box {
    background: #f4f1ff;
    border: 1px solid #ded5fa;
    border-radius: 16px;
    padding: 23px;
    margin-bottom: 22px;
}

.ai-title {
    color: #5b3c98;
    font-size: 20px;
    font-weight: 800;
}


/* ==============================
   DISCLAIMER
   ============================== */

.disclaimer {
    background: #fff9e8;
    border: 1px solid #f0d58b;
    border-radius: 13px;
    padding: 15px;
    color: #765f1d;
    font-size: 12px;
    line-height: 1.6;
}


/* ==============================
   BUTTONS
   ============================== */

.stButton > button {
    border-radius: 11px;
    min-height: 46px;
    font-weight: 700;
}


/* ==============================
   FOOTER
   ============================== */

.footer {
    text-align: center;
    color: #9aa8b8;
    font-size: 12px;
    padding: 30px 10px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def predict_patient(age, blood_pressure, glucose, bmi):

    patient = pd.DataFrame({
        "age": [age],
        "blood_pressure": [blood_pressure],
        "glucose": [glucose],
        "bmi": [bmi]
    })

    prediction = model.predict(patient)[0]

    risk_score = 0.0

    try:
        probabilities = model.predict_proba(patient)[0]
        classes = model.classes_

        probability_dict = dict(
            zip(classes, probabilities)
        )

        if "High" in probability_dict:
            risk_score = float(
                probability_dict["High"] * 100
            )

    except Exception:
        risk_score = 0.0

    return prediction, risk_score


def get_status(parameter, value):

    if parameter == "Age":

        if value < 45:
            return "Normal"

        elif value < 60:
            return "Attention"

        return "High"

    if parameter == "Blood Pressure":

        if value < 130:
            return "Normal"

        elif value < 140:
            return "Attention"

        return "High"

    if parameter == "Glucose":

        if value < 110:
            return "Normal"

        elif value < 140:
            return "Attention"

        return "High"

    if parameter == "BMI":

        if value < 25:
            return "Normal"

        elif value < 30:
            return "Attention"

        return "High"

    return "Normal"


def parameter_card(name, value):

    status = get_status(name, value)

    if status == "Normal":
        status_class = "normal-status"

    elif status == "Attention":
        status_class = "warning-status"

    else:
        status_class = "high-status"

    return f"""
<div class="parameter-card">
    <div class="parameter-title">{name}</div>
    <div class="parameter-value">{value}</div>
    <div class="{status_class}">Status: {status}</div>
</div>
"""


def risk_gauge(score):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={
                "suffix": "%",
                "font": {
                    "size": 40
                }
            },
            title={
                "text": "Estimated Risk"
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "bar": {
                    "thickness": 0.25
                },
                "steps": [
                    {
                        "range": [0, 40]
                    },
                    {
                        "range": [40, 70]
                    },
                    {
                        "range": [70, 100]
                    }
                ]
            }
        )
    )

    fig.update_layout(
        height=320,
        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 20
        }
    )

    return fig


def create_pdf(age, bp, glucose, bmi, prediction, score):

    try:

        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

    except ImportError:

        return None

    buffer = BytesIO()

    pdf = canvas.Canvas(
        buffer,
        pagesize=A4
    )

    width, height = A4

    pdf.setFont(
        "Helvetica-Bold",
        24
    )

    pdf.drawString(
        50,
        height - 60,
        "MedRisk AI"
    )

    pdf.setFont(
        "Helvetica",
        13
    )

    pdf.drawString(
        50,
        height - 85,
        "Patient Risk Assessment Report"
    )

    pdf.line(
        50,
        height - 100,
        width - 50,
        height - 100
    )

    y = height - 145

    pdf.setFont(
        "Helvetica",
        11
    )

    pdf.drawString(
        50,
        y,
        "Assessment Date: " +
        datetime.now().strftime(
            "%d-%m-%Y %H:%M"
        )
    )

    y -= 50

    pdf.setFont(
        "Helvetica-Bold",
        15
    )

    pdf.drawString(
        50,
        y,
        "Patient Parameters"
    )

    y -= 30

    pdf.setFont(
        "Helvetica",
        11
    )

    parameters = [
        f"Age: {age}",
        f"Blood Pressure: {bp}",
        f"Glucose: {glucose}",
        f"BMI: {bmi}"
    ]

    for item in parameters:

        pdf.drawString(
            70,
            y,
            item
        )

        y -= 25

    y -= 25

    pdf.setFont(
        "Helvetica-Bold",
        15
    )

    pdf.drawString(
        50,
        y,
        "AI Assessment"
    )

    y -= 30

    pdf.setFont(
        "Helvetica",
        11
    )

    pdf.drawString(
        70,
        y,
        f"Predicted Risk: {prediction}"
    )

    y -= 25

    pdf.drawString(
        70,
        y,
        f"Estimated Risk Score: {score:.1f}%"
    )

    y -= 60

    pdf.setFont(
        "Helvetica-Bold",
        13
    )

    pdf.drawString(
        50,
        y,
        "Disclaimer"
    )

    y -= 25

    pdf.setFont(
        "Helvetica",
        9
    )

    disclaimer = (
        "This report is generated by an educational machine "
        "learning demonstration. The model is trained on a "
        "small sample dataset and is not medically validated. "
        "It must not be used for medical diagnosis or treatment."
    )

    words = disclaimer.split()

    line = ""

    for word in words:

        if len(line) + len(word) < 90:

            line += " " + word

        else:

            pdf.drawString(
                50,
                y,
                line.strip()
            )

            y -= 15

            line = word

    if line:
        pdf.drawString(
            50,
            y,
            line.strip()
        )

    pdf.save()

    buffer.seek(0)

    return buffer


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    """
<h1>MedRisk AI</h1>
<p>Patient Risk Intelligence Platform</p>
""",
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "Dashboard",
        "Patient Assessment",
        "What-If Simulator",
        "AI Explainability"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
<div class="disclaimer">
<b>Educational Demonstration</b>
<br><br>
MedRisk AI is an academic machine learning project.
It is not a medical diagnosis system.
</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown(
        """
<div class="hero-box">
    <div class="hero-title">MedRisk AI</div>
    <div class="hero-subtitle">
        Intelligent Patient Risk Assessment Dashboard
    </div>
</div>
""",
        unsafe_allow_html=True
    )

    if st.session_state.last_result:

        result = st.session_state.last_result

        current_risk = result["risk"]
        current_score = result["score"]
        current_age = result["age"]
        current_bp = result["bp"]
        current_glucose = result["glucose"]
        current_bmi = result["bmi"]

    else:

        current_risk = "Not Assessed"
        current_score = 0.0
        current_age = 35
        current_bp = 120
        current_glucose = 100
        current_bmi = 24.0


    # ==========================
    # STAT CARDS
    # ==========================

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
<div class="stat-card">
    <div class="stat-label">Current Risk</div>
    <div class="stat-value">{current_risk}</div>
    <div class="stat-description">Latest AI assessment</div>
</div>
""",
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
<div class="stat-card">
    <div class="stat-label">Risk Score</div>
    <div class="stat-value">{current_score:.1f}%</div>
    <div class="stat-description">Estimated model probability</div>
</div>
""",
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
<div class="stat-card">
    <div class="stat-label">Assessments</div>
    <div class="stat-value">{len(st.session_state.history)}</div>
    <div class="stat-description">Current session</div>
</div>
""",
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            f"""
<div class="stat-card">
    <div class="stat-label">Patient Age</div>
    <div class="stat-value">{current_age}</div>
    <div class="stat-description">Years</div>
</div>
""",
            unsafe_allow_html=True
        )


    st.markdown("")


    # ==========================
    # RISK OVERVIEW
    # ==========================

    left, right = st.columns([1, 1])


    with left:

        st.markdown(
            """
<div class="content-card">
<div class="card-title">Risk Overview</div>
<div class="card-subtitle">
Current estimated patient risk
</div>
""",
            unsafe_allow_html=True
        )

        st.plotly_chart(
            risk_gauge(current_score),
            use_container_width=True
        )

        if current_risk == "Low":

            st.markdown(
                """
<div class="low-risk">
LOW RISK
</div>
""",
                unsafe_allow_html=True
            )

        elif current_risk == "High":

            st.markdown(
                """
<div class="high-risk">
HIGH RISK
</div>
""",
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
<div class="not-assessed">
NOT YET ASSESSED
</div>
""",
                unsafe_allow_html=True
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # ==========================
    # HEALTH PARAMETERS
    # ==========================

    with right:

        st.markdown(
            """
<div class="content-card">
<div class="card-title">Health Parameters</div>
<div class="card-subtitle">
Current patient measurements
</div>
""",
            unsafe_allow_html=True
        )

        st.markdown(
            parameter_card(
                "Age",
                current_age
            ),
            unsafe_allow_html=True
        )

        st.markdown(
            parameter_card(
                "Blood Pressure",
                current_bp
            ),
            unsafe_allow_html=True
        )

        st.markdown(
            parameter_card(
                "Glucose",
                current_glucose
            ),
            unsafe_allow_html=True
        )

        st.markdown(
            parameter_card(
                "BMI",
                current_bmi
            ),
            unsafe_allow_html=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # ==========================
    # QUICK FEATURES
    # ==========================

    st.markdown(
        """
<div class="content-card">
<div class="card-title">MedRisk AI Features</div>
<div class="card-subtitle">
Intelligent tools available in the platform
</div>
</div>
""",
        unsafe_allow_html=True
    )

    q1, q2, q3 = st.columns(3)

    with q1:

        st.info(
            "Patient Assessment\n\n"
            "Run an AI-based patient risk assessment."
        )

    with q2:

        st.info(
            "What-If Simulator\n\n"
            "Change patient values and see how the prediction changes."
        )

    with q3:

        st.info(
            "AI Explainability\n\n"
            "Understand the importance of each health parameter."
        )


# =========================================================
# PATIENT ASSESSMENT
# =========================================================

elif page == "Patient Assessment":

    st.markdown(
        "<div class='page-title'>Patient Assessment</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="page-subtitle">
Enter patient health information and generate an AI risk assessment.
</div>
""",
        unsafe_allow_html=True
    )


    st.markdown(
        """
<div class="content-card">
<div class="card-title">Patient Health Information</div>
<div class="card-subtitle">
Adjust the health parameters below
</div>
</div>
""",
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    with col1:

        age = st.slider(
            "Age",
            min_value=1,
            max_value=100,
            value=35
        )

        blood_pressure = st.slider(
            "Blood Pressure",
            min_value=50,
            max_value=220,
            value=120
        )


    with col2:

        glucose = st.slider(
            "Glucose",
            min_value=40,
            max_value=300,
            value=100
        )

        bmi = st.slider(
            "BMI",
            min_value=10.0,
            max_value=50.0,
            value=24.0,
            step=0.1
        )


    st.markdown("### Current Health Profile")


    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Age", age)

    with m2:
        st.metric("Blood Pressure", blood_pressure)

    with m3:
        st.metric("Glucose", glucose)

    with m4:
        st.metric("BMI", bmi)


    st.markdown("")


    if st.button(
        "Run AI Risk Assessment",
        type="primary",
        use_container_width=True
    ):

        prediction, score = predict_patient(
            age,
            blood_pressure,
            glucose,
            bmi
        )


        result = {
            "date": datetime.now().strftime(
                "%d-%m-%Y %H:%M"
            ),
            "age": age,
            "bp": blood_pressure,
            "glucose": glucose,
            "bmi": bmi,
            "risk": prediction,
            "score": score
        }


        st.session_state.last_result = result

        st.session_state.history.append(
            result
        )


        st.markdown("---")

        st.markdown(
            "### AI Assessment Result"
        )


        if prediction == "Low":

            st.markdown(
                f"""
<div class="low-risk">
LOW RISK
<br>
<span style="font-size:14px;">
Estimated Risk Score: {score:.1f}%
</span>
</div>
""",
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
<div class="high-risk">
HIGH RISK
<br>
<span style="font-size:14px;">
Estimated Risk Score: {score:.1f}%
</span>
</div>
""",
                unsafe_allow_html=True
            )


        st.markdown("")


        result_left, result_right = st.columns(2)


        with result_left:

            st.markdown(
                """
<div class="content-card">
<div class="card-title">Risk Visualization</div>
</div>
""",
                unsafe_allow_html=True
            )

            st.plotly_chart(
                risk_gauge(score),
                use_container_width=True
            )


        with result_right:

            st.markdown(
                """
<div class="content-card">
<div class="card-title">Parameter Analysis</div>
<div class="card-subtitle">
Individual parameter status
</div>
""",
                unsafe_allow_html=True
            )

            st.markdown(
                parameter_card(
                    "Age",
                    age
                ),
                unsafe_allow_html=True
            )

            st.markdown(
                parameter_card(
                    "Blood Pressure",
                    blood_pressure
                ),
                unsafe_allow_html=True
            )

            st.markdown(
                parameter_card(
                    "Glucose",
                    glucose
                ),
                unsafe_allow_html=True
            )

            st.markdown(
                parameter_card(
                    "BMI",
                    bmi
                ),
                unsafe_allow_html=True
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


        # ==========================
        # PDF REPORT
        # ==========================

        st.markdown("---")

        st.markdown(
            "### Patient Report"
        )

        pdf_file = create_pdf(
            age,
            blood_pressure,
            glucose,
            bmi,
            prediction,
            score
        )

        if pdf_file is not None:

            st.download_button(
                "Download PDF Report",
                data=pdf_file,
                file_name="MedRisk_AI_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        else:

            st.warning(
                "PDF support is not installed yet."
            )

            st.code(
                "pip install reportlab"
            )


# =========================================================
# WHAT-IF SIMULATOR
# =========================================================

elif page == "What-If Simulator":

    st.markdown(
        "<div class='page-title'>What-If Health Simulator</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="page-subtitle">
Experiment with different health values and see how the AI prediction changes.
</div>
""",
        unsafe_allow_html=True
    )


    st.markdown(
        """
<div class="simulator-banner">
<div class="simulator-title">
Interactive Risk Simulation
</div>
<div class="simulator-text">
Move the sliders and observe the simulated risk instantly.
</div>
</div>
""",
        unsafe_allow_html=True
    )


    sim1, sim2 = st.columns(2)


    with sim1:

        sim_age = st.slider(
            "Simulated Age",
            1,
            100,
            35,
            key="sim_age"
        )

        sim_bp = st.slider(
            "Simulated Blood Pressure",
            50,
            220,
            120,
            key="sim_bp"
        )


    with sim2:

        sim_glucose = st.slider(
            "Simulated Glucose",
            40,
            300,
            100,
            key="sim_glucose"
        )

        sim_bmi = st.slider(
            "Simulated BMI",
            10.0,
            50.0,
            24.0,
            0.1,
            key="sim_bmi"
        )


    sim_prediction, sim_score = predict_patient(
        sim_age,
        sim_bp,
        sim_glucose,
        sim_bmi
    )


    st.markdown("---")


    sim_left, sim_right = st.columns(2)


    with sim_left:

        st.markdown(
            """
<div class="content-card">
<div class="card-title">Simulated Patient</div>
<div class="card-subtitle">
Current simulated health profile
</div>
""",
            unsafe_allow_html=True
        )

        st.markdown(
            parameter_card(
                "Age",
                sim_age
            ),
            unsafe_allow_html=True
        )

        st.markdown(
            parameter_card(
                "Blood Pressure",
                sim_bp
            ),
            unsafe_allow_html=True
        )

        st.markdown(
            parameter_card(
                "Glucose",
                sim_glucose
            ),
            unsafe_allow_html=True
        )

        st.markdown(
            parameter_card(
                "BMI",
                sim_bmi
            ),
            unsafe_allow_html=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    with sim_right:

        st.markdown(
            """
<div class="content-card">
<div class="card-title">Simulation Result</div>
<div class="card-subtitle">
Live AI prediction
</div>
""",
            unsafe_allow_html=True
        )


        if sim_prediction == "Low":

            st.markdown(
                f"""
<div class="low-risk">
LOW RISK
<br>
<span style="font-size:14px;">
Risk Score: {sim_score:.1f}%
</span>
</div>
""",
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
<div class="high-risk">
HIGH RISK
<br>
<span style="font-size:14px;">
Risk Score: {sim_score:.1f}%
</span>
</div>
""",
                unsafe_allow_html=True
            )


        st.plotly_chart(
            risk_gauge(sim_score),
            use_container_width=True
        )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # ==========================
    # CURRENT VS SIMULATED
    # ==========================

    if st.session_state.last_result:

        st.markdown("---")

        st.markdown(
            "### Current vs Simulated"
        )


        current = st.session_state.last_result


        comparison = pd.DataFrame(
            {
                "Parameter": [
                    "Age",
                    "Blood Pressure",
                    "Glucose",
                    "BMI"
                ],
                "Current": [
                    current["age"],
                    current["bp"],
                    current["glucose"],
                    current["bmi"]
                ],
                "Simulated": [
                    sim_age,
                    sim_bp,
                    sim_glucose,
                    sim_bmi
                ]
            }
        )


        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )


        compare1, compare2 = st.columns(2)


        with compare1:

            st.metric(
                "Current Risk",
                current["risk"],
                f"{current['score']:.1f}%"
            )


        with compare2:

            st.metric(
                "Simulated Risk",
                sim_prediction,
                f"{sim_score:.1f}%"
            )


# =========================================================
# AI EXPLAINABILITY
# =========================================================

elif page == "AI Explainability":

    st.markdown(
        "<div class='page-title'>AI Explainability</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="page-subtitle">
Understand how the Decision Tree model uses patient parameters.
</div>
""",
        unsafe_allow_html=True
    )


    st.markdown(
        """
<div class="ai-box">
<div class="ai-title">
How does MedRisk AI make a prediction?
</div>
<br>
The Decision Tree model analyzes four patient parameters:
Age, Blood Pressure, Glucose and BMI.
It uses patterns learned from the training dataset
to classify the patient into a risk category.
</div>
""",
        unsafe_allow_html=True
    )


    # ==========================
    # FEATURE IMPORTANCE
    # ==========================

    feature_names = [
        "Age",
        "Blood Pressure",
        "Glucose",
        "BMI"
    ]


    try:

        importance = model.feature_importances_

        importance_df = pd.DataFrame(
            {
                "Parameter": feature_names,
                "Importance": importance
            }
        )

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=True
        )


        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=importance_df["Importance"],
                y=importance_df["Parameter"],
                orientation="h",
                text=[
                    f"{x:.2f}"
                    for x in importance_df["Importance"]
                ],
                textposition="auto"
            )
        )


        fig.update_layout(
            title="Model Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Parameter",
            height=420,
            margin={
                "l": 20,
                "r": 20,
                "t": 70,
                "b": 30
            }
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        most_important = importance_df.iloc[-1]


        st.markdown(
            f"""
<div class="content-card">
<div class="card-title">
Most Influential Parameter
</div>
<div class="card-subtitle">
Based on the trained Decision Tree model
</div>
<h2 style="color:#1769aa;">
{most_important["Parameter"]}
</h2>
<p style="color:#718096;">
Feature importance:
<b>{most_important["Importance"]:.2f}</b>
</p>
</div>
""",
            unsafe_allow_html=True
        )

    except Exception:

        st.warning(
            "Feature importance could not be displayed."
        )


    # ==========================
    # PARAMETERS
    # ==========================

    st.markdown(
        "### Model Parameters"
    )


    e1, e2, e3, e4 = st.columns(4)


    with e1:

        st.info(
            "Age\n\n"
            "Patient age used by the model."
        )


    with e2:

        st.info(
            "Blood Pressure\n\n"
            "Blood pressure value used by the model."
        )


    with e3:

        st.info(
            "Glucose\n\n"
            "Glucose level used by the model."
        )


    with e4:

        st.info(
            "BMI\n\n"
            "Body Mass Index used by the model."
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<div class="footer">
MedRisk AI · Academic Machine Learning Project
<br>
Decision Tree Based Patient Risk Classification
<br><br>
For educational demonstration only
</div>
""",
    unsafe_allow_html=True
)