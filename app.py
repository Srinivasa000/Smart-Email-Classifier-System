import os
import streamlit as st
from dotenv import load_dotenv
from scipy.sparse import csr_matrix
import pandas as pd
import altair as alt

from src.model import MultiTargetModel
from src.data_preprocessing import clean_text, tokenize_and_stem
from src.imap_reader import IMAPReader

# -------------------- LOAD ENV --------------------
load_dotenv()

# -------------------- PAGE SETUP --------------------
st.set_page_config(
    page_title="Smart Email Classifier",
    page_icon="📧",
    layout="wide"
)

# -----------------------------------------------------
#                   GLOBAL CSS (PRO UI)
# -----------------------------------------------------
st.markdown("""
<style>

body {
    background-color: #f5f7fa !important;
}

h1{
    text-align:center;
    font-weight:900;
    font-size:40px !important;
    background: linear-gradient(90deg,#3a7bd5,#00d4ff);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    margin-bottom:5px;
}

.subtitle{
    text-align:center;
    color:#666;
    margin-top:-10px;
    margin-bottom:30px;
    font-size:18px;
}

.result-card{
    background:#ffffff;
    padding:22px;
    border-radius:14px;
    border:1px solid #e0e0e0;
    box-shadow:0px 3px 8px rgba(0,0,0,0.06);
    color:#111 !important;
}

.section-header{
    font-size:24px;
    font-weight:700;
    color:#333;
    margin-top:25px;
}

</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("<h1>Smart Email Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Category Detection • Department Routing • Priority Classification</p>",
            unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
MODEL_DIR = "models"
try:
    model = MultiTargetModel.load(MODEL_DIR)
except Exception as e:
    st.error(
        "❌ Failed to load model.\n\nMake sure your models folder contains:\n"
        "- vectorizer.joblib\n"
        "- models.joblib\n"
        "- encoders.joblib\n\n"
        f"Error: {e}"
    )
    st.stop()

# -------------------- NAVIGATION --------------------
page = st.sidebar.radio(
    "📌 Navigation",
    ["Classifier", "Inbox", "About"]
)

# -------------------- PIE CHART FUNCTION --------------------
def pie_chart(labels, probs):
    df = pd.DataFrame({"Class": labels, "Probability": probs})
    return alt.Chart(df).mark_arc(innerRadius=50).encode(
        theta="Probability",
        color=alt.Color("Class", legend=None),
        tooltip=["Class", alt.Tooltip("Probability", format=".3f")]
    ).properties(width=260, height=260)

# ============================================================
#                        CLASSIFIER PAGE
# ============================================================
if page == "Classifier":

    st.markdown("<div class='section-header'>✉️ Manual Email Classification</div>", unsafe_allow_html=True)

    subject = st.text_input("Email Subject")
    body = st.text_area("Email Body", height=200)

    classify_btn = st.button("🔍 Classify Email", use_container_width=True)

    if classify_btn:
        raw = subject + " " + body
        cleaned = clean_text(raw)
        processed = tokenize_and_stem(cleaned)

        tf = model.vectorizer.transform([processed])
        meta = csr_matrix([
            [len(subject.split()), len(body.split()), subject.count("!") + body.count("!")]
        ])

        results = model.predict(tf, meta)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Classification Results")

        # Layout rows
        card_row = st.columns(3)
        chart_row = st.columns(3)

        # Rename SPAM → CATEGORY
        mapping = {
            "spam": "category",
            "department": "department",
            "priority": "priority"
        }

        order = ["spam", "department", "priority"]

        for i, k in enumerate(order):
            display_name = mapping[k].upper()

            info = results[k]
            pred = info["prediction"]
            conf = round(info["probability"], 3)
            labels = info["classes"]
            probs = info["probs"]

            # ---------------- CARD ----------------
            with card_row[i]:
                st.markdown(f"""
                <div class="result-card">
                    <h4>{display_name}</h4>
                    <p style='font-size:17px;'>
                        <b>Prediction:</b> {pred}<br>
                        <b>Confidence:</b> {conf}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # ---------------- CHART ----------------
            with chart_row[i]:
                st.altair_chart(pie_chart(labels, probs), use_container_width=True)


# ============================================================
#                        INBOX PAGE
# ============================================================
elif page == "Inbox":

    st.markdown("<div class='section-header'>📥 Inbox Auto-Classification</div>", unsafe_allow_html=True)

    username = st.text_input("Email Username", os.getenv("EMAIL_USERNAME"))
    password = st.text_input("Email Password (App Password)", os.getenv("EMAIL_PASSWORD"), type="password")
    server = st.text_input("IMAP Server", os.getenv("IMAP_SERVER"))
    port = int(st.text_input("IMAP Port", os.getenv("IMAP_PORT") or "993"))

    if st.button("🔌 Fetch & Classify Last 20 Emails", use_container_width=True):
        reader = IMAPReader(username, password, server, port)

        with st.spinner("Fetching emails…"):
            try:
                emails = reader.fetch_recent(20)
            except Exception as e:
                st.error(f"IMAP error: {e}")
                st.stop()

        if not emails:
            st.info("Inbox empty.")
            st.stop()

        st.success(f"Fetched {len(emails)} emails.")

        # Loop emails
        for idx, em in enumerate(emails):

            subj = em["subject"]
            body = em["body"]
            sender = em["from"]

            st.markdown("---")
            st.markdown(f"### 📧 Email #{idx+1}")
            st.write(f"**From:** {sender}")
            st.write(f"**Subject:** {subj}")
            st.write(f"**Preview:** {body[:350]}...")

            # ================= CLASSIFY =================
            raw = subj + " " + body
            cleaned = clean_text(raw)
            processed = tokenize_and_stem(cleaned)
            tf = model.vectorizer.transform([processed])

            meta = csr_matrix([
                [len(subj.split()), len(body.split()), subj.count("!") + body.count("!")]
            ])

            results = model.predict(tf, meta)

            card_row = st.columns(3)
            chart_row = st.columns(3)

            order = ["spam", "department", "priority"]
            label_map = {"spam": "category", "department": "department", "priority": "priority"}

            for i, k in enumerate(order):

                display_name = label_map[k].upper()

                info = results[k]
                pred = info["prediction"]
                conf = round(info["probability"], 3)
                labels = info["classes"]
                probs = info["probs"]

                # -------- CARD --------
                with card_row[i]:
                    st.markdown(f"""
                    <div class="result-card">
                        <h4>{display_name}</h4>
                        <p><b>Prediction:</b> {pred}<br>
                        <b>Confidence:</b> {conf}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # -------- PIE CHART --------
                with chart_row[i]:
                    st.altair_chart(pie_chart(labels, probs), use_container_width=True)


# ============================================================
#                        ABOUT PAGE
# ============================================================
elif page == "About":

    st.markdown("<div class='section-header'>ℹ️ About This Project</div>", unsafe_allow_html=True)

    st.write("""
### Smart Email Classifier — End-to-End AI System

This project demonstrates a production-style email classification pipeline capable of:

#### ✔ Category Detection
Identify whether an email is:
- Spam  
- Ham  
- Notification  
- Promotional  
- or other custom categories

#### ✔ Department Routing
Automatically decide where the email should be routed:
- IT  
- HR  
- Finance  
- Sales  
- Support  

#### ✔ Priority Classification
Determine the urgency level:
- High  
- Medium  
- Low  

#### ✔ IMAP Inbox Integration
The system connects to any mailbox supporting IMAP:
- Gmail  
- Outlook  
- Office365  
- Yahoo (with app password)  

Fetches the last 20 emails and **auto-classifies each** with:
- Polished prediction cards  
- Confidence scores  
- Pie charts  

#### ✔ Tech Stack Used
- Python  
- Scikit-learn  
- Streamlit  
- IMAP (email integration)  
- TF-IDF vectorization  
- Custom preprocessing  
- Multi-output ML model  

#### ✔ Skills Demonstrated
- Machine Learning  
- NLP preprocessing  
- Real-world email integration  
- UI engineering  
- Modular software engineering  
- End-to-end pipeline design

""")

