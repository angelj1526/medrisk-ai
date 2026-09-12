import io
import pickle
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="MedRisk AI",
    page_icon="M",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: #F4F7FB;
}

[data-testid="stSidebar"] {
    background: #123B5D;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

h1, h2, h3 {
    color: #123B5D !important;
}

p, label {
    color: #425B70 !important;
}

.hero {
    background: white;
    padding: 30px;
    border-radius: 18px;
    border: 1px solid #DCE6EF;
    margin-bottom: 25px;
}

.hero h1 {
    margin: 0;
    color: #123B5D;
    font-size: 36px;
}

.hero p {
    margin-top: 8px;
    color: #61778A !important;
    font-size: 16px;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #DCE6EF;
    box-shadow: 0 3px 12px rgba(0,0,0,0.05);
}

.low {
    background: #E9F8EF;
    border: 1px solid #B9E5C9;
    color: #167044 !important;
    padding: 18px;
    border-radius: 14px;
    text-align: center;
    font-size: 20px;
    font-weight: 700;
}

.high {
    background: #FFF0EE;
    border: 1px solid #F0C3BC;
    color: #B43D35 !important;
    padding: 18px;
    border-radius: 14px;
    text-align: center;
    font-size: 20px;
    font-weight: 700;
}

.attention {
    background: #FFF7E5;
    border: 1px solid #EDD59B;
    color: #94620D !important;
    padding: 18px;
    border-radius: 14px;
    text-align: center;
    font-size: 20px;
    font-weight: 700;
}

.info {
    background: #EAF5FA;
    border: 1px solid #C9E1EC;
    color: #315B70 !important;
    padding: 16px;
    border-radius: 12px;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #DCE6EF;
    border-radius: 14px;
    padding: 18px;
}

[data-testid="stMetricLabel"] {
    color: #60788A !important;
}

[data-testid="stMetricValue"] {
    color: #123B5D !important;
}

button {
    border-radius: 10px !important;
}

.footer {
    text-align: center;
    color: #7A8D9D;
    font-size: 13px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("patient_data.csv")


@st.cache_resource
def load_model():
    with open("patient_risk_model.pkl", "rb") as f:
        return pickle.load(f)


try:
    data = load_data()
    model = load_model()

except Exception as e:
    st.error("Project files could not be loaded.")
    st.code(str(e))
    st.stop()


FEATURES = [
    "age",
    "blood_pressure",
    "glucose",
    "bmi"
]


# =========================================================
# PREDICTION
# =========================================================

def predict(age, bp, glucose, bmi):

    patient = pd.DataFrame([{
        "age": age,
        "blood_pressure": bp,
        "glucose": glucose,
        "bmi": bmi
    }])

    prediction = model.predict(patient)[0]

    probability = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(patient)[0]
        classes = list(model.classes_)

        if prediction in classes:
            index = classes.index(prediction)
            probability = float(probabilities[index])

    return str(prediction), probability


# =========================================================
# RISK STYLE
# =========================================================

def risk_class(risk):

    risk = risk.lower()

    if "low" in risk:
        return "low"

    if "high" in risk:
        return "high"

    return "attention"


# =========================================================
# PARAMETER STATUS
# =========================================================

def parameter_status(name, value):

    if name == "Age":

        if value <= 40:
            return "Normal"

        elif value <= 60:
            return "Attention"

        return "Elevated"

    if name == "Blood Pressure":

        if value < 130:
            return "Normal"

        elif value < 140:
            return "Attention"

        return "High"

    if name == "Glucose":

        if value < 110:
            return "Normal"

        elif value <= 125:
            return "Attention"

        return "High"

    if name == "BMI":

        if 18.5 <= value < 25:
            return "Normal"

        elif value < 30:
            return "Attention"

        return "High"

    return "Review"


# =========================================================
# PDF
# =========================================================

def create_pdf(name, age, bp, glucose, bmi, prediction, probability):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        "title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=colors.HexColor("#123B5D"),
        alignment=TA_CENTER,
        spaceAfter=15
    )

    normal = ParagraphStyle(
        "normal",
        parent=styles["Normal"],
        fontSize=11,
        leading=16,
        textColor=colors.HexColor("#425B70")
    )

    story = []

    story.append(Paragraph("MedRisk AI", title))
    story.append(
        Paragraph(
            "Patient Risk Assessment Report",
            title
        )
    )

    story.append(
        Paragraph(
            f"<b>Patient:</b> {name}",
            normal
        )
    )

    story.append(
        Paragraph(
            f"<b>Date:</b> {datetime.now().strftime('%d %B %Y')}",
            normal
        )
    )

    story.append(Spacer(1, 20))

    table_data = [
        ["Parameter", "Value", "Status"],
        [
            "Age",
            str(age),
            parameter_status("Age", age)
        ],
        [
            "Blood Pressure",
            str(bp),
            parameter_status("Blood Pressure", bp)
        ],
        [
            "Glucose",
            str(glucose),
            parameter_status("Glucose", glucose)
        ],
        [
            "BMI",
            str(bmi),
            parameter_status("BMI", bmi)
        ]
    ]

    table = Table(
        table_data,
        colWidths=[170, 120, 170]
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#123B5D")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#D5E1EA")
            ),
            (
                "TEXTCOLOR",
                (0, 1),
                (-1, -1),
                colors.HexColor("#425B70")
            ),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [
                    colors.white,
                    colors.HexColor("#F5F8FB")
                ]
            ),
            (
                "ALIGN",
                (1, 1),
                (1, -1),
                "CENTER"
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                9
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                9
            )
        ])
    )

    story.append(table)

    story.append(Spacer(1, 20))

    probability_text = (
        f"{probability * 100:.1f}%"
        if probability is not None
        else "Not available"
    )

    result = Table([
        ["AI Prediction", prediction],
        ["Model Probability", probability_text]
    ], colWidths=[220, 240])

    result.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#EAF3F8")
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#D5E1EA")
            ),
            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, -1),
                colors.HexColor("#123B5D")
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                10
            )
        ])
    )

    story.append(result)

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "<b>Note:</b> This application is an educational "
            "machine-learning demonstration and is not intended "
            "for clinical diagnosis or treatment decisions.",
            normal
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
        font-size:30px;
        font-weight:800;
        margin-bottom:4px;">
        MedRisk AI
        </div>

        <div style="
        font-size:13px;
        opacity:.8;">
        Patient Risk Intelligence
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "MENU",
        [
            "Dashboard",
            "Patient Assessment",
            "What-If Simulator",
            "AI Explainability",
            "Dataset"
        ]
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="
        font-size:12px;
        line-height:1.6;">
        Educational project<br>
        Decision Tree based assessment
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
        <div class="hero">
            <h1>MedRisk AI Dashboard</h1>
            <p>
            Intelligent patient risk assessment
            using machine learning.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    low = int(
        (data["risk"].str.lower() == "low").sum()
    )

    high = int(
        (data["risk"].str.lower() == "high").sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Records",
        len(data)
    )

    c2.metric(
        "Low Risk",
        low
    )

    c3.metric(
        "High Risk",
        high
    )

    c4.metric(
        "ML Model",
        "Decision Tree"
    )

    st.markdown("## Risk Overview")

    col1, col2 = st.columns(2)

    with col1:

        counts = data["risk"].value_counts()

        fig = px.pie(
            values=counts.values,
            names=counts.index,
            hole=0.5
        )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        averages = data[FEATURES].mean()

        avg_df = pd.DataFrame({
            "Parameter": [
                "Age",
                "Blood Pressure",
                "Glucose",
                "BMI"
            ],
            "Average": [
                averages["age"],
                averages["blood_pressure"],
                averages["glucose"],
                averages["bmi"]
            ]
        })

        fig = px.bar(
            avg_df,
            x="Parameter",
            y="Average",
            text="Average"
        )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("## Dataset Preview")

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# PATIENT ASSESSMENT
# =========================================================

elif page == "Patient Assessment":

    st.markdown("## Patient Risk Assessment")

    st.markdown(
        """
        <div class="info">
        Enter patient information and run the
        Decision Tree risk assessment.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    left, right = st.columns(2)

    with left:

        st.markdown("### Patient Information")

        name = st.text_input(
            "Patient Name",
            "Demo Patient"
        )

        age = st.number_input(
            "Age",
            1,
            100,
            40
        )

        bp = st.number_input(
            "Blood Pressure",
            60,
            220,
            120
        )

        glucose = st.number_input(
            "Glucose",
            40,
            300,
            100
        )

        bmi = st.number_input(
            "BMI",
            10.0,
            60.0,
            24.0,
            step=0.1
        )

        run = st.button(
            "Run Risk Assessment",
            type="primary",
            use_container_width=True
        )

    if run:

        prediction, probability = predict(
            age,
            bp,
            glucose,
            bmi
        )

        st.session_state["result"] = {
            "name": name,
            "age": age,
            "bp": bp,
            "glucose": glucose,
            "bmi": bmi,
            "prediction": prediction,
            "probability": probability
        }

    with right:

        st.markdown("### Assessment Result")

        result = st.session_state.get("result")

        if result is None:

            st.markdown(
                """
                <div class="info">
                Enter patient details and click
                <b>Run Risk Assessment</b>.
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            cls = risk_class(
                result["prediction"]
            )

            st.markdown(
                f"""
                <div class="{cls}">
                Predicted Risk: {result["prediction"]}
                </div>
                """,
                unsafe_allow_html=True
            )

            if result["probability"] is not None:

                probability = (
                    result["probability"] * 100
                )

                st.metric(
                    "Model Probability",
                    f"{probability:.1f}%"
                )

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=probability,
                        number={
                            "suffix": "%"
                        },
                        title={
                            "text": "Prediction Confidence"
                        },
                        gauge={
                            "axis": {
                                "range": [0, 100]
                            }
                        }
                    )
                )

                fig.update_layout(
                    height=260,
                    paper_bgcolor="white"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

    result = st.session_state.get("result")

    if result:

        st.markdown("## Parameter Analysis")

        values = [
            ("Age", result["age"]),
            ("Blood Pressure", result["bp"]),
            ("Glucose", result["glucose"]),
            ("BMI", result["bmi"])
        ]

        cols = st.columns(4)

        for col, (label, value) in zip(cols, values):

            status = parameter_status(
                label,
                value
            )

            with col:

                st.markdown(
                    f"""
                    <div class="card">
                    <b>{label}</b>
                    <h2>{value}</h2>
                    <p>{status}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("## Patient Report")

        pdf = create_pdf(
            result["name"],
            result["age"],
            result["bp"],
            result["glucose"],
            result["bmi"],
            result["prediction"],
            result["probability"]
        )

        st.download_button(
            "Download Patient PDF Report",
            data=pdf,
            file_name="medrisk_patient_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# =========================================================
# WHAT IF
# =========================================================

elif page == "What-If Simulator":

    st.markdown("## What-If Health Simulator")

    st.markdown(
        """
        <div class="info">
        Change the simulated values and see how the
        trained model responds.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    left, right = st.columns(2)

    with left:

        st.markdown("### Current Values")

        current_age = st.number_input(
            "Current Age",
            1,
            100,
            40,
            key="ca"
        )

        current_bp = st.number_input(
            "Current Blood Pressure",
            60,
            220,
            120,
            key="cb"
        )

        current_glucose = st.number_input(
            "Current Glucose",
            40,
            300,
            100,
            key="cg"
        )

        current_bmi = st.number_input(
            "Current BMI",
            10.0,
            60.0,
            24.0,
            step=0.1,
            key="cm"
        )

    with right:

        st.markdown("### Simulated Values")

        sim_age = st.slider(
            "Simulated Age",
            1,
            100,
            40
        )

        sim_bp = st.slider(
            "Simulated Blood Pressure",
            60,
            220,
            120
        )

        sim_glucose = st.slider(
            "Simulated Glucose",
            40,
            300,
            100
        )

        sim_bmi = st.slider(
            "Simulated BMI",
            10.0,
            60.0,
            24.0,
            step=0.1
        )

    current_prediction, current_probability = predict(
        current_age,
        current_bp,
        current_glucose,
        current_bmi
    )

    sim_prediction, sim_probability = predict(
        sim_age,
        sim_bp,
        sim_glucose,
        sim_bmi
    )

    st.markdown("## Comparison")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            f"""
            <div class="{risk_class(current_prediction)}">
            Current: {current_prediction}
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="{risk_class(sim_prediction)}">
            Simulated: {sim_prediction}
            </div>
            """,
            unsafe_allow_html=True
        )

    comparison = pd.DataFrame({
        "Scenario": [
            "Current",
            "Simulated"
        ],
        "Probability": [
            (current_probability or 0) * 100,
            (sim_probability or 0) * 100
        ]
    })

    fig = px.bar(
        comparison,
        x="Scenario",
        y="Probability",
        text="Probability",
        range_y=[0, 100]
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# EXPLAINABILITY
# =========================================================

elif page == "AI Explainability":

    st.markdown("## AI Explainability")

    st.markdown(
        """
        <div class="info">
        This section shows how the Decision Tree
        uses the available input features.
        </div>
        """,
        unsafe_allow_html=True
    )

    importance = getattr(
        model,
        "feature_importances_",
        None
    )

    if importance is not None:

        df = pd.DataFrame({
            "Parameter": [
                "Age",
                "Blood Pressure",
                "Glucose",
                "BMI"
            ],
            "Importance": importance
        })

        fig = px.bar(
            df,
            x="Importance",
            y="Parameter",
            orientation="h",
            text="Importance"
        )

        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside"
        )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            df.round(4),
            use_container_width=True,
            hide_index=True
        )

        strongest = df.loc[
            df["Importance"].idxmax(),
            "Parameter"
        ]

        st.info(
            f"The strongest learned feature in this "
            f"training dataset is: {strongest}."
        )


# =========================================================
# DATASET
# =========================================================

elif page == "Dataset":

    st.markdown("## Training Dataset")

    st.markdown(
        """
        <div class="info">
        These records are used by the Decision Tree
        model for this educational project.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("## Dataset Statistics")

    st.dataframe(
        data[FEATURES].describe().round(2),
        use_container_width=True
    )

    counts = (
        data["risk"]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "Risk",
        "Records"
    ]

    fig = px.bar(
        counts,
        x="Risk",
        y="Records",
        text="Records"
    )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
    MedRisk AI | Educational Machine Learning Project |
    Not for clinical diagnosis
    </div>
    """,
    unsafe_allow_html=True
)
