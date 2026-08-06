import streamlit as st

import joblib


# Load the trained model and CountVectorizer (BoW)

model = joblib.load("NLP_newsprediction_model.joblib")

vectorizer = joblib.load("NLP_newsprediction_vectorizer.joblib")


st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)


# ----------------------------- CLASSIC NEWSPAPER UI -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap');

    .stApp {
        background-color: #dedbd4;
        background-image:
            repeating-linear-gradient(0deg, rgba(0,0,0,0.02) 0px, rgba(0,0,0,0.02) 1px, transparent 1px, transparent 4px);
    }

    /* All text black */
    html, body, .stApp, .stApp *,
    h1, h2, h3, h4, h5, h6, p, span, div, label, li,
    .stMarkdown, .stTextInput label, .stTextArea label {
        color: #000000 !important;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2.5rem !important;
        padding-bottom: 3rem !important;
        background-color: #eae7e0;
        border: 1px solid #000000;
        box-shadow: 6px 6px 0 rgba(0,0,0,0.18);
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }

    /* Masthead */
    h1 {
        font-family: 'Playfair Display', 'Times New Roman', serif !important;
        font-weight: 900 !important;
        font-size: 3.1rem !important;
        text-align: center;
        letter-spacing: 1px;
        margin-bottom: 0.2rem !important;
        border-bottom: 3px double #000000;
        border-top: 3px double #000000;
        padding: 0.6rem 0 0.8rem 0;
        text-transform: uppercase;
    }

    h2, h3 {
        font-family: 'Playfair Display', serif !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    p, label, div, span, input, textarea, button {
        font-family: 'Libre Baskerville', Georgia, serif !important;
    }

    .stMarkdown p {
        text-align: center;
        font-style: italic;
        font-size: 0.98rem;
    }

    .paper-rule {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 14px;
        font-size: 0.72rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        border-bottom: 1px solid #000;
        padding-bottom: 8px;
        margin-bottom: 26px;
        font-family: 'Libre Baskerville', serif;
    }

    /* Inputs */
    .stTextInput label, .stTextArea label {
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
    }

    .stTextInput input, .stTextArea textarea {
        background-color: #f6f4ef !important;
        border: 1px solid #000000 !important;
        border-radius: 0 !important;
        font-size: 1rem !important;
        box-shadow: inset 2px 2px 0 rgba(0,0,0,0.06);
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border: 2px solid #000000 !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* Button */
    .stButton { text-align: center; }
    .stButton > button {
        background-color: #000000 !important;
        color: #eae7e0 !important;
        border: 1px solid #000000 !important;
        border-radius: 0 !important;
        text-transform: uppercase;
        letter-spacing: 4px;
        font-weight: 700 !important;
        padding: 0.65rem 2.6rem !important;
        margin-top: 0.6rem;
        transition: all 0.15s ease-in-out;
    }
    .stButton > button * { color: #eae7e0 !important; }
    .stButton > button:hover {
        background-color: #eae7e0 !important;
        box-shadow: 4px 4px 0 #000000;
    }
    .stButton > button:hover * { color: #000000 !important; }

    /* Alerts as newsprint boxes */
    .stAlert {
        border-radius: 0 !important;
        border: 2px solid #000000 !important;
        background-color: #f6f4ef !important;
        box-shadow: 4px 4px 0 rgba(0,0,0,0.15);
    }
    .stAlert p {
        text-align: center;
        font-family: 'Playfair Display', serif !important;
        font-weight: 900 !important;
        font-size: 1.5rem !important;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* Progress bar (confidence meter) */
    .stProgress > div > div > div > div { background-color: #000000 !important; }
    .stProgress > div > div > div { background-color: #cdc9c0 !important; }

    /* How it works cards */
    .how-card {
        background-color: #f6f4ef;
        border: 1px solid #000000;
        border-top: 4px solid #000000;
        padding: 14px 16px;
        min-height: 168px;
        box-shadow: 3px 3px 0 rgba(0,0,0,0.15);
    }
    .how-card .step {
        font-size: 0.68rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        border-bottom: 1px solid #000;
        padding-bottom: 5px;
        margin-bottom: 8px;
        font-family: 'Libre Baskerville', serif;
    }
    .how-card .head {
        font-family: 'Playfair Display', serif;
        font-weight: 900;
        font-size: 1.05rem;
        line-height: 1.2;
        margin-bottom: 6px;
    }
    .how-card .body {
        font-size: 0.85rem;
        line-height: 1.45;
        text-align: justify;
    }

    .section-rule {
        border-top: 3px double #000000;
        margin: 34px 0 18px 0;
    }

    .conf-box {
        background-color: #f6f4ef;
        border: 1px solid #000000;
        padding: 12px 16px;
        margin-top: 10px;
        box-shadow: 3px 3px 0 rgba(0,0,0,0.15);
        font-family: 'Libre Baskerville', serif;
        font-size: 0.85rem;
    }
    .conf-num {
        font-family: 'Playfair Display', serif;
        font-weight: 900;
        font-size: 2.4rem;
        text-align: center;
        letter-spacing: 1px;
    }
    .conf-cap {
        text-align: center;
        font-size: 0.68rem;
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    footer, #MainMenu, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("📰 Fake News Detection")

st.markdown(
    '<div class="paper-rule"><span>Vol. I</span><span>&#10022;</span>'
    '<span>The Daily Verifier</span><span>&#10022;</span><span>Price: Free</span></div>',
    unsafe_allow_html=True,
)

st.write("Enter the news headline and article text below.")


headline = st.text_input("News Headline")


news_text = st.text_area(
    "News Text",
    height=250
)


if st.button("Predict"):

    if not headline.strip() or not news_text.strip():

        st.warning("Please enter both the headline and news text.")

    else:
        # Combine headline and article

        input_news = headline + " " + news_text

        # Convert text using CountVectorizer (BoW)

        input_vector = vectorizer.transform([input_news])

        # Predict

        prediction = model.predict(input_vector)[0]

        st.subheader("Prediction")

        # Change this depending on your labels
        # Works for numeric labels (1 / 0) and string labels ("REAL" / "FAKE")

        is_real = (prediction == 1) or (str(prediction).strip().upper() == "REAL")

        if is_real:

            st.success("✅ REAL NEWS")

        else:

            st.error("❌ FAKE NEWS")

        # ---------------------- CONFIDENCE ----------------------
        st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
        st.subheader("Confidence")

        try:
            probabilities = model.predict_proba(input_vector)[0]
            classes = list(model.classes_)
            confidence = float(probabilities[classes.index(prediction)]) * 100

            st.markdown(
                f'<div class="conf-num">{confidence:.2f}%</div>'
                f'<div class="conf-cap">Certainty for this verdict</div>',
                unsafe_allow_html=True,
            )
            st.progress(min(max(confidence / 100, 0.0), 1.0))

            breakdown = " &nbsp;&#10022;&nbsp; ".join(
                f"<b>{str(cls).upper()}</b>: {prob * 100:.2f}%"
                for cls, prob in zip(classes, probabilities)
            )
            st.markdown(
                f'<div class="conf-box" style="text-align:center;">{breakdown}</div>',
                unsafe_allow_html=True,
            )
        except AttributeError:
            st.markdown(
                '<div class="conf-box" style="text-align:center;">'
                "This model does not expose probability scores, so a confidence "
                "percentage cannot be reported.</div>",
                unsafe_allow_html=True,
            )


# ---------------------- HOW IT WORKS ----------------------
st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
st.subheader("How It Works")

steps = [
    (
        "Step One",
        "Copy The Story",
        "Paste the headline and the full article body. Both fields are joined into a "
        "single document, exactly the way the model was trained on the news dataset.",
    ),
    (
        "Step Two",
        "Bag Of Words",
        "A CountVectorizer converts the text into word-count features, turning the "
        "article into a numeric vocabulary vector the classifier can read.",
    ),
    (
        "Step Three",
        "The Verdict",
        "A Logistic Regression model weighs those word counts and returns its "
        "classification: REAL reporting or FAKE fabrication.",
    ),
    (
        "Step Four",
        "Confidence Meter",
        "The model's class probabilities are shown as a percentage, so you can see how "
        "certain the verdict is rather than trusting a bare label.",
    ),
]

cols = st.columns(2)
for index, (step, head, body) in enumerate(steps):
    with cols[index % 2]:
        st.markdown(
            f'<div class="how-card"><div class="step">{step}</div>'
            f'<div class="head">{head}</div>'
            f'<div class="body">{body}</div></div>',
            unsafe_allow_html=True,
        )
        st.write("")

st.markdown(
    '<div class="paper-rule" style="margin-top:26px;border-top:1px solid #000;'
    'border-bottom:none;padding-top:10px;"><span>Editor&#39;s Note</span>'
    '<span>&#10022;</span><span>Predictions are statistical, not proof</span></div>',
    unsafe_allow_html=True,
)
