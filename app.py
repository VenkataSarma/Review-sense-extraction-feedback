import os
import re
import spacy
import streamlit as st
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import bcrypt
from dotenv import load_dotenv
import pandas as pd
from transformers import pipeline
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
import string

from bson.binary import Binary
from PIL import Image
from io import BytesIO
# ---------------------- ENV & CONFIG ----------------------
load_dotenv()
MONGO_URI = os.getenv("MONGODB_URI")
USER_DB = os.getenv("USER_DB", "review_user_db")
ADMIN_DB = os.getenv("ADMIN_DB", "review_admin_db")
USER_COLL = "users"
ADMIN_COLL = "admins"
RESULTS_COLL = "analysis_results"
LOGS_COLL = "activity_logs"
FEEDBACK_COLL = "user_feedback"

if not MONGO_URI:
    st.error("❌ MONGODB_URI not set in .env file")
    st.stop()

st.set_page_config(page_title="Review Sensing System", layout="wide")

# ---------------------- SESSION STATE ----------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "page" not in st.session_state:
    st.session_state.page = "login"
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "role" not in st.session_state:
    st.session_state.role = None
# ---------------------- FEEDBACK SESSION STATE INIT ----------------------
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

if "show_feedback_box" not in st.session_state:
    st.session_state.show_feedback_box = False
# ---------------------- BATCH FEEDBACK SESSION STATE ----------------------
if "batch_analysis_done" not in st.session_state:
    st.session_state.batch_analysis_done = False

if "batch_result" not in st.session_state:
    st.session_state.batch_result = None

if "batch_feedback_submitted" not in st.session_state:
    st.session_state.batch_feedback_submitted = False

if "show_batch_feedback_box" not in st.session_state:
    st.session_state.show_batch_feedback_box = False


# ---------------------- CACHED RESOURCES ----------------------
@st.cache_resource
def get_db():
    client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)
    return client
client = get_db()
user_db = client[USER_DB]
admin_db = client[ADMIN_DB]

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")


@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

model = load_sentiment_model()


nlp = load_spacy_model()

# ---------------------- THEME COLORS ----------------------
def get_theme_colors():
    if st.session_state.theme == "dark":
        return {
            "bg": "#0e1117",
            "sidebar_bg": "#161b22",
            "text": "#ffffff",
            "muted_text": "#b0b0b0",
            "input_bg": "#1e242c",
            "card_bg": "#1e242c",
            "accent": "#0078d4",
            "hover": "#006bb3",
        }
    else:
        return {
            "bg": "#f8f9fb",
            "sidebar_bg": "#ffffff",
            "text": "#000000",
            "muted_text": "#333333",
            "input_bg": "#ffffff",
            "card_bg": "#ffffff",
            "accent": "#00a86b",
            "hover": "#00c37a",
        }

colors = get_theme_colors()
# ---------------------- ALERT COLORS BASED ON THEME ----------------------
alert_text_color = "#ffffff" if st.session_state.theme == "dark" else "#000000"
alert_bg = "#2e0000" if st.session_state.theme == "dark" else "#fdecea"
alert_border = "#ff4b4b" if st.session_state.theme == "dark" else "red"

st.markdown(f"""
<style>
div[data-testid="stAlert"] p {{
    color: {alert_text_color} !important;
}}

div[data-testid="stAlert"] {{
    background-color: {alert_bg} !important;
    border-left: 6px solid {alert_border} !important;
}}
</style>
""", unsafe_allow_html=True)


# ---------------------- CUSTOM STYLES ----------------------
st.markdown(f"""
<style>
body {{
    background-color: {colors['bg']};
    color: {colors['text']};
}}
[data-testid="stAppViewContainer"] {{
    background-color: {colors['bg']};
    color: {colors['text']};
}}
[data-testid="stSidebar"] {{
    background-color: {colors['sidebar_bg']};
    color: {colors['text']};
}}
label, .stMarkdown p, .stText, .stDataFrame, .stMetric {{
    color: {colors['text']} !important;
}}
.sidebar-title {{
    font-size: 22px;
    font-weight: 700;
    color: {colors['accent']};
    text-align: center;
    margin-bottom: 2rem;
}}
.stTextInput > div > div > input, textarea, select {{
    background-color: {colors['input_bg']} !important;
    color: {colors['text']} !important;
}}
.stButton>button {{
    background-color: {colors['accent']} !important;
    color: {colors['bg']} !important;
}}
.stButton>button:hover {{
    background-color: {colors['hover']} !important;
}}
.input-box, .result-box {{
    background: {colors['card_bg']};
    padding: 1.2rem;
    border-radius: 10px;
}}


</style>
""", unsafe_allow_html=True)

# ---------------------- RADIO BUTTON FIX ----------------------
st.markdown(f"""
<style>
div[role='radiogroup'] label p {{
    color: {colors['text']} !important;
}}
.stRadio > label {{
    color: {colors['text']} !important;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------- UTILITIES ----------------------
def ensure_indexes():
    user_db[USER_COLL].create_index("email", unique=True)
    admin_db[ADMIN_COLL].create_index("email", unique=True)
    user_db[RESULTS_COLL].create_index("user_email")
    user_db[FEEDBACK_COLL].create_index("user_email")

ensure_indexes()

def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password: str, password_hash: bytes) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash)

def create_account(role, name, email, password):
    pw_hash = hash_password(password)

    db = user_db if role == "user" else admin_db
    coll = USER_COLL if role == "user" else ADMIN_COLL

    try:
        db[coll].insert_one({
            "full_name": name,
            "email": email.lower(),
            "password_hash": pw_hash
        })
        save_log(email.lower(), "User Registered", f"Role: {role}")

        return True, "✅ Account created!"
        
    except DuplicateKeyError:
        return False, "⚠️ Email already exists."


def authenticate_account(role, email, password):
    db = user_db if role == "user" else admin_db
    coll = USER_COLL if role == "user" else ADMIN_COLL

    user = db[coll].find_one({"email": email.lower()})
    if not user:
        return False, "Account not found."

    if not check_password(password, user["password_hash"]):
        return False, "Wrong password."

    return True, user


def save_analysis_to_db(user_email, review_text, sentiment, confidence, aspects, aspect_pairs):

    user_db[RESULTS_COLL].insert_one({
        "user_email": user_email,
        "review": review_text,
        "sentiment": sentiment,
        "confidence": round(confidence * 100, 2),
        "aspects": aspects,
        "aspect_pairs": aspect_pairs,
        "created_at": datetime.now()
    })

    # ✅ ALSO LOG RESULT DETAILS
    save_log(
        user_email,
        "Review Analyzed",
        f"Text: {review_text[:60]} | Sentiment: {sentiment} | Confidence: {round(confidence*100,2)}%"
    )


# ---------------------- ASPECT EXTRACTION (ML-enhanced ABSA) ----------------------

# Canonical aspect vocabulary (keep and extend as needed)
aspect_keywords = {

    # =========================
    # DEVICE & HARDWARE
    # =========================
    "battery": [
        "battery", "battery life", "charge", "charging", "power", "drain"
    ],

    "camera": [
        "camera", "photo", "picture", "selfie", "lens", "zoom",
        "camera quality", "camera performance"
    ],

    "display": [
        "screen", "display", "resolution", "brightness",
        "touch", "glass", "screen size"
    ],

    "performance": [
        "performance", "slow", "fast", "lag", "hang",
        "speed", "response", "responsiveness", "crash", "freeze"
    ],

    # =========================
    # PRODUCT & DELIVERY
    # =========================
    "product": [
        "product", "item", "goods"
    ],

    "quality": [
        "quality", "material", "build", "design",
        "breathtaking", "outstanding", "defective",
        "faulty", "damaged", "cheap", "fake"
    ],

    "package": [
        "package", "packaging", "parcel", "box", "sealed"
    ],

    "delivery": [
        "delivery", "shipping", "arrival", "delivered",
        "late", "delay", "lost", "tracking"
    ],

    "order": [
        "order", "ordered", "received", "receiving",
        "cancel", "refund", "replacement", "return"
    ],

    # =========================
    # SERVICE & SUPPORT
    # =========================
    "support": [
        "service", "support", "customer service",
        "helpdesk", "staff", "representative",
        "helpline", "call center", "complaint", "manager"
    ],

    "experience": [
        "experience", "feedback", "interaction", "satisfaction"
    ],

    # =========================
    # APP & WEBSITE
    # =========================
    "app": [
        "app", "application", "interface", "ui",
        "login", "signup", "crash", "hang", "freeze"
    ],

    "website": [
        "website", "site", "page", "navigation", "loading",
        "broken", "not working"
    ],

    # =========================
    # FOOD & DINING
    # =========================
    "food": [
        "food", "meal", "taste", "dish", "snack",
        "delicious", "spicy", "fresh", "stale", "cold"
    ],

    "restaurant": [
        "restaurant", "hotel", "cafe", "dining",
        "ambience", "place", "environment"
    ],

    # =========================
    # MEDIA & ENTERTAINMENT
    # =========================
    "movie": [
        "movie", "film", "cinema", "story",
        "plot", "ending", "thrilling", "masterpiece"
    ],

    "book": [
        "book", "novel", "storyline", "author", "chapter"
    ],

    "playlist": [
        "playlist", "songs", "music", "track"
    ],

    # =========================
    # FLIGHT & TRAVEL
    # =========================
    "flight": [
        "flight", "airline", "plane", "boarding",
        "delay", "cancelled", "ticket", "refund"
    ],

    "airport": [
        "airport", "terminal", "check-in", "security"
    ],

    "staff_flight": [
        "crew", "attendant", "pilot", "air hostess"
    ],

    "comfort": [
        "seat", "comfort", "legroom", "space", "cleanliness"
    ],

    # =========================
    # TECHNICAL ISSUES
    # =========================
    "issue": [
        "issue", "problem", "error", "bug",
        "frustrating", "worst", "poor", "not good"
    ],
    "music": [
    "song", "songs", "music", "track", "album", "playlist"
],
"travel": [
    "vacation", "trip", "journey", "tour", "holiday", "travel"
],

}
    


# flatten set for quick membership checks
flattened_keywords = set(sum((v for v in aspect_keywords.values()), []))

def map_term_to_aspect(term: str):
    term = term.lower().strip()

    for asp, keys in aspect_keywords.items():
        for k in keys:
            k = k.lower()

            # Exact word match
            if term == k:
                return asp

            # Multi-word phrase match
            if " " in k and k in term:
                return asp

            # Token-level match
            if term in k.split():
                return asp

    return None


def extract_aspects_rule_based(text):
    """Return list of canonical aspect keys found by keyword matching (rule-based)."""
    detected = []
    lower = text.lower()
    for asp, keys in aspect_keywords.items():
        for k in keys:
            if re.search(rf"\b{re.escape(k)}\b", lower):
                detected.append(asp)
                break
    return detected

def _nearest_adj_for_noun(token):
    """
    Heuristic: find adjective that describes noun token:
    - left 'amod'
    - ADJ head with nsubj = noun
    - nearest adjective in same noun chunk or adjacent tokens
    """
    # amod modifier already handled elsewhere
    # Look for amod children
    amod = [w for w in token.lefts if w.dep_ == "amod"]
    if amod:
        return amod[0].text

    # Right adjectives that are part of 'is/was' constructions will get handled by acomp logic.
    # Fallback: search within noun chunk for adjectives
    chunk = None
    for ch in token.doc.noun_chunks:
        if token in ch:
            chunk = ch
            break
    if chunk:
        for w in chunk:
            if w.pos_ == "ADJ":
                return w.text

    # As last resort, look left/right within window 3 for an ADJ
    i = token.i
    for offset in (-3, -2, -1, 1, 2, 3):
        j = i + offset
        if 0 <= j < len(token.doc):
            w = token.doc[j]
            if w.pos_ == "ADJ":
                return w.text
    return None
def handle_negation(token):
    for child in token.children:
        if child.dep_ == "neg":
            return "not " + token.text.lower()
    return token.text.lower()


def extract_aspects_spacy(text):

    doc = nlp(text)
    pairs = []

    for token in doc:
        # Verb-based sentiment
        if token.pos_ == "VERB" and token.text.lower() in {
    "love", "hate", "like", "enjoy", "recommend", "prefer", "dislike"
}:
            for child in token.children:
                if child.dep_ == "dobj":  # direct object
                    asp = map_term_to_aspect(child.text)
                    if asp:
                        pairs.append((asp, token.text.lower()))
            break



    # collect all possible aspect terms for quick filtering
    valid_aspect_terms = set(sum(aspect_keywords.values(), []))

    for token in doc:
        if token.pos_ == "VERB" and token.text.lower() in {
            "love", "hate", "like", "enjoy", "recommend", "prefer", "dislike"
        }:
            for child in token.children:
                if child.dep_ in ("dobj", "pobj"):
                    asp_key = map_term_to_aspect(child.text)
                    if asp_key:
                        pairs.append((asp_key, token.text.lower()))
        # Case 1: adjective modifies noun → "camera good"
        if token.dep_ == "amod" and token.head.pos_ == "NOUN":
            head = token.head
            adj = token
            compound_parts = [w.text for w in head.lefts if w.dep_ == "compound"]
            full_term = " ".join(compound_parts + [head.text])

            asp_key = map_term_to_aspect(full_term) or map_term_to_aspect(head.text)
            if not asp_key and full_term.lower() not in valid_aspect_terms:
                continue  # skip non-aspect nouns

            final_aspect = asp_key if asp_key else full_term.lower()
            pairs.append((final_aspect, handle_negation(adj)))

        # Case 2: "battery is amazing"
        if token.dep_ == "nsubj" and token.head.pos_ == "ADJ":
            noun = token
            adj = token.head
            compound_parts = [w.text for w in noun.lefts if w.dep_ == "compound"]
            full_term = " ".join(compound_parts + [noun.text])
            asp_key = map_term_to_aspect(full_term) or map_term_to_aspect(noun.text)
            if not asp_key and full_term.lower() not in valid_aspect_terms:
                continue

            final_aspect = asp_key if asp_key else full_term.lower()
            pairs.append((final_aspect, handle_negation(adj)))

        # Case 3: "delivery was slow"
        if token.dep_ == "acomp":
            subs = [w for w in token.head.lefts if w.dep_ == "nsubj"]
            if subs:
                subj = subs[0]
                compound_parts = [w.text for w in subj.lefts if w.dep_ == "compound"]
                full_term = " ".join(compound_parts + [subj.text])
                asp_key = map_term_to_aspect(full_term) or map_term_to_aspect(subj.text)
                if not asp_key and full_term.lower() not in valid_aspect_terms:
                    continue

                final_aspect = asp_key if asp_key else full_term.lower()
                pairs.append((final_aspect, token.text.lower()))

    # Remove duplicates
    seen = set()
    unique_pairs = []
    for a, o in pairs:
        if (a, o) not in seen:
            unique_pairs.append((a, o))
            seen.add((a, o))

    return unique_pairs


# ---------------------- ASPECT SENTIMENT CLASSIFIER (ML-based) ----------------------

# threshold for applying neutral override
NEUTRAL_CONF_THRESH = 0.70

def _normalize_opinion_phrase(phrase: str):
    """Lower, strip punctuation and collapse whitespace for stable matching."""
    if phrase is None:
        return ""
    p = phrase.lower().strip()
    # remove leading/trailing punctuation
    p = p.strip(string.punctuation + " ")
    # collapse multiple spaces
    p = re.sub(r"\s+", " ", p)
    return p

def classify_aspect_sentiment(pairs):
    """
    Classify aspect sentiment using ML + lexical rules.
    Ensures words like 'slow', 'blurry', 'disappointing' always map correctly.
    """
    if not pairs:
        return []

    # Strong opinion lexicons
    positive_words = {
        "good", "great", "excellent", "amazing", "incredible",
        "fast", "bright", "smooth", "responsive", "sharp", "clear",
        "best", "awesome", "fantastic", "perfect", "love",
    "loved", "wonderful", "brilliant", "favorite",
    "superb", "outstanding", "breathtaking"
    }
    negative_words = {
        "bad", "poor", "slow", "disappointing", "terrible",
        "worst", "blurry", "dim", "laggy", "noisy",
        "disappointed", "disappointing", "disappoint",
    "frustrated", "frustrating",
    "annoyed", "angry", "hate",
    "horrible", "pathetic", "broken", "useless"
    }
    neutral_words = {
        "ok", "fine", "average", "decent",
        "satisfactory", "alright", "moderate", "fair",
        "acceptable", "normal", "standard"
    }

    results = []
    for aspect, opinion in pairs:
        # ✅ NEGATION HANDLING
        if opinion.startswith("not "):
            core = opinion.replace("not ", "").strip()

            if core in negative_words:
                results.append((aspect, "positive"))
                continue

            if core in positive_words:
                results.append((aspect, "negative"))
                continue

        phrase = _normalize_opinion_phrase(opinion)

        # -----------------------
        # 1️⃣ Strong Lexical Override (always applied)
        # -----------------------
        if phrase in positive_words:
            results.append((aspect, "positive"))
            continue

        if phrase in negative_words:
            results.append((aspect, "negative"))
            continue

        # -----------------------
        # 2️⃣ ML Model Prediction
        # -----------------------
        try:
            res = model(phrase)[0]
            raw_label = res.get("label", "")
            conf = float(res.get("score", 0))

            # Convert label
            if "LABEL_2" in raw_label or "POS" in raw_label:
                ml_sent = "positive"
            elif "LABEL_0" in raw_label or "NEG" in raw_label:
                ml_sent = "negative"
            else:
                ml_sent = "neutral"

            # -----------------------
            # 3️⃣ Neutral override ONLY if model weak & word is neutral
            # -----------------------
            if phrase in neutral_words and conf < 0.70:
                results.append((aspect, "neutral"))
            else:
                results.append((aspect, ml_sent))

        except:
            # Fallback simple rule
            if phrase in positive_words:
                results.append((aspect, "positive"))
            elif phrase in negative_words:
                results.append((aspect, "negative"))
            elif phrase in neutral_words:
                results.append((aspect, "neutral"))
            else:
                results.append((aspect, "neutral"))

    return results


# ---------------------- SENTIMENT HELPERS ----------------------
def normalize_label(label):
    label = label.upper()
    return {"LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE"}.get(label, label)

def adjust_label(label, score):
    return "NEUTRAL" if label == "NEGATIVE" and score < 0.80 else label

# ---------------------- SIDEBAR ----------------------
with st.sidebar:

    st.markdown("## 🧠 Review System")

    if st.session_state.user:

        if st.session_state.role == "admin":
            if st.button("📊 Dashboard"): st.session_state.page = "admin_dashboard"
            if st.button("👥 Users List"):
                st.session_state.page = "users_list"
            if st.button("📜 Activity / Logs"):
                st.session_state.page = "activity_logs"
            if st.button("👤 Profile"):
                st.session_state.page = "admin_profile"
            if st.button("🚪 Logout"):
                st.session_state.user = None
                st.session_state.role = None
                st.session_state.page = "login"
                st.rerun()

        else:
            if st.button("🏠 Dashboard"): st.session_state.page = "dashboard"
            if st.button("📊 Batch Analysis"): st.session_state.page = "batch_analysis"
            if st.button("👤 Profile"): st.session_state.page = "profile"
            if st.button("🧠 Active Learning"): st.session_state.page = "active_learning"
            if st.button("🚪 Logout"):
                st.session_state.user = None
                st.session_state.role = None
                st.session_state.page = "login"
                st.rerun()
    else:
        if st.button("🔑 Login"): st.session_state.page = "login"
        if st.button("📝 Register"): st.session_state.page = "register"


# ---------------------- PAGES ----------------------
def login_page():
    st.header("🔑 Login")

    role = st.radio("Login as:", ["User", "Admin"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        ok, user = authenticate_account(role.lower(), email, password)

        if ok:
            st.session_state.user = user
            st.session_state.role = role.lower()
            st.session_state.page = "dashboard"
            st.rerun()
            save_log(email.lower(), "User Login")

        else:
            st.error(user)


def register_page():
    st.header("📝 Register")

    role = st.radio("Register as:", ["User", "Admin"])
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    pwd = st.text_input("Password", type="password")
    cpwd = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if pwd != cpwd:
            st.error("Passwords do not match.")
        else:
            ok, msg = create_account(role.lower(), name, email, pwd)
            if ok:
               st.success(msg)
            else:
               st.error(msg)

#-------------ADMIN DASHBOARD-----------------#
def admin_dashboard():
    st.markdown("## 👑 Admin Dashboard")
    st.caption("Quick system overview")

    # ---------------- FETCH COUNTS ----------------
    total_feedbacks = user_db[RESULTS_COLL].count_documents({})
    corrected_feedbacks = user_db[RESULTS_COLL].count_documents({"active_learning": True})
    total_users = user_db[USER_COLL].count_documents({})

    # ---------------- STYLE ----------------
    box_color = "#ffffff" if st.session_state.theme == "light" else "#1e242c"
    text_color = colors["text"]

    # ---------------- SMALL STAT BOXES ----------------
    col1, col2, col3 = st.columns(3, gap="small")

    def stat_box(title, value, color):
        st.markdown(f"""
        <div style="
            background-color:{box_color};
            border-left:4px solid {color};
            padding:12px;
            border-radius:8px;
            height:90px;
            box-shadow:0 2px 5px rgba(0,0,0,0.08);
            display:flex;
            flex-direction:column;
            justify-content:center;">
            <div style="font-size:13px; color:{text_color};">{title}</div>
            <div style="font-size:24px; font-weight:700; color:{color};">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    with col1:
        stat_box("📩 Total Feedbacks", total_feedbacks, "#7B61FF")

    with col2:
        stat_box("✅ Corrected", corrected_feedbacks, "#2ECC71")

    with col3:
        stat_box("👥 Users", total_users, "#3498DB")

    st.markdown("---")

    # ---------------- SENTIMENT DISTRIBUTION ----------------
    st.markdown("### 📊 Sentiment Distribution (All Feedbacks)")

    sentiment_docs = list(user_db[RESULTS_COLL].find({}, {"_id": 0, "sentiment": 1}))
    if not sentiment_docs:
        st.info("No sentiment data found.")
        return

    sentiment_list = [
        s["sentiment"].lower()
        for s in sentiment_docs
        if "sentiment" in s and isinstance(s["sentiment"], str)
    ]

    counts = Counter(sentiment_list)
    labels = list(counts.keys())
    sizes = list(counts.values())

    color_map = {
        "positive": "#4CAF50",
        "negative": "#F44336",
        "neutral": "#2196F3"
    }
    colors_list = [color_map.get(lbl, "#999999") for lbl in labels]

    # ---------------- SMALL CENTER PIE ----------------
    st.markdown("<div style='display:flex; justify-content:center;'>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(2, 2))  # 50% size
    ax.pie(
        sizes,
        labels=labels,
        colors=colors_list,
        autopct="%1.1f%%",
        startangle=140,
        textprops={'fontsize': 7}
    )
    ax.axis("equal")
    st.pyplot(fig, use_container_width=False)

    st.markdown("</div>", unsafe_allow_html=True)


#-----users_list--------------------------#
def users_list_page():
    st.markdown("## 👥 Registered Users")

    users = list(user_db[USER_COLL].find())
    
    if len(users) == 0:
        st.info("No users found.")
        return

    for user in users:
        with st.container():
            col1, col2, col3 = st.columns([3, 3, 1])

            with col1:
                st.write(f"**Name:** {user.get('full_name','')}")

            with col2:
                st.write(f"**Email:** {user.get('email','')}")

            with col3:
                del_key = f"delete_{user['_id']}"

                if st.button("🗑 Delete", key=del_key):
                    # DELETE USER
                    user_db[USER_COLL].delete_one({"_id": user["_id"]})
                    save_log(
    st.session_state.user["email"],
    "User Deleted",
    user.get("email")
)


                    # DELETE USER'S ANALYSIS DATA
                    user_db[RESULTS_COLL].delete_many({"user_email": user.get("email")})

                    st.success(f"User {user.get('email')} deleted successfully.")
                    st.rerun()

            st.divider()

#--------ACTIVITY LOGS PAGE--------------#

def activity_logs_page():
    st.markdown("## 📜 System Activity Logs")

    logs = list(user_db[LOGS_COLL].find().sort("timestamp", -1))

    if not logs:
        st.info("No activity logs found.")
        return

    for log in logs:

        # ✅ Extract result first
        result = log.get("details", "No result logged")

        # ✅ Detect sentiment color
        if "POSITIVE" in result:
            sentiment_color = "#4CAF50"   # Green
        elif "NEGATIVE" in result:
            sentiment_color = "#F44336"   # Red
        else:
            sentiment_color = "#2196F3"   # Blue

        # ✅ User & Action
        st.markdown(f"""
        **👤 User:** {log.get("user_email")}  
        **⚡ Action:** {log.get("action")}  
        **🕒 Time:** {log.get("timestamp").strftime("%d %b %Y | %I:%M:%S %p")}
        """, unsafe_allow_html=True)

        # ✅ Result Box (Styled)
        st.markdown(f"""
        <div style="
            background:#f3f4f6;
            padding:10px;
            border-radius:6px;
            margin-top:6px;
            font-size:14px;
            border-left:4px solid {sentiment_color};
        ">
        <b>📝 Result</b><br>
        {result.replace("|", "<br>")}
        </div>
        """, unsafe_allow_html=True)

        st.divider()


#-------------------logs and activity------------------#
def save_log(user_email, action, details=""):
    user_db[LOGS_COLL].insert_one({
        "user_email": user_email,
        "action": action,
        "details": details,
        "timestamp": datetime.now()
    })

#-------------------------admin profile page------------------------------#
def admin_profile_page():
    st.header("👑 Admin Profile")

    if not st.session_state.user:
        st.warning("Please log in to view profile.")
        return

    admin_email = st.session_state.user["email"]
    profile = admin_db[ADMIN_COLL].find_one({"email": admin_email}) or {}

    # ---------- BASIC DETAILS ----------
    st.markdown("### Personal Details")

    name = st.text_input("Full Name", value=profile.get("full_name", ""))
    email = st.text_input("Email", value=profile.get("email", admin_email))
    organization = st.text_input("Organization", value=profile.get("organization", ""))

    # ---------- PASSWORD CHANGE ----------
    st.markdown("---")
    st.markdown("### 🔐 Change Password (Optional)")

    old_pwd = st.text_input("Current Password", type="password")
    new_pwd = st.text_input("New Password", type="password")
    confirm_pwd = st.text_input("Confirm New Password", type="password")

    # ---------- SAVE BUTTON ----------
    if st.button("💾 Save Changes"):

        # ✅ Update profile fields
        update_data = {
            "full_name": name.strip(),
            "email": email.strip().lower(),
            "organization": organization.strip(),
            "updated_at": datetime.now()
        }

        admin_db[ADMIN_COLL].update_one(
            {"email": admin_email},
            {"$set": update_data},
            upsert=True
        )

        # ✅ PASSWORD CHANGE LOGIC
        if old_pwd and new_pwd and confirm_pwd:

            if new_pwd != confirm_pwd:
                st.error("❌ New passwords do not match!")
                return

            admin = admin_db[ADMIN_COLL].find_one({"email": admin_email})

            if not check_password(old_pwd, admin["password_hash"]):
                st.error("❌ Current password is incorrect!")
                return

            hashed = hash_password(new_pwd)

            admin_db[ADMIN_COLL].update_one(
                {"email": admin_email},
                {"$set": {"password_hash": hashed}}
            )

            st.success("✅ Password changed successfully!")

            save_log(
                admin_email,
                "Admin Password Changed",
                "Password updated successfully"
            )

        # ✅ Update session if email changes
        if email.strip().lower() != admin_email:
            st.session_state.user["email"] = email.strip().lower()

        st.success("✅ Profile updated successfully!")
        st.rerun()
#----------------user feedback----------------#
def save_user_feedback(user_email, satisfied, thoughts, source, review_text=None, sentiment=None, aspects=None, aspect_pairs=None):
    user_db[FEEDBACK_COLL].insert_one({
        "user_email": user_email,
        "satisfied": satisfied,            # 👍/👎
        "thoughts": thoughts,              # optional text
        "source": source,                  # 'single' or 'batch'
        "review_text": review_text,        # full review
        "predicted_sentiment": sentiment,  # system sentiment label
        "aspects_detected": aspects,       # rule-based aspect list
        "aspect_pairs": aspect_pairs,      # aspect-opinion pairs
        "timestamp": datetime.now()
    })


#------------------------------#

def user_dashboard():
    st.markdown(f'<div class="main-title">💬 Review Analysis Dashboard</div>', unsafe_allow_html=True)
    st.write("Perform sentiment + aspect extraction on customer feedback.")

    # ---- FULL WIDTH LAYOUT SPLIT (50% - 50%) ----
    col1, col2 = st.columns([1, 1], gap="large")

    # ---------------- LEFT SIDE (INPUT BOX) ----------------
    with col1:
        st.markdown('<div class="input-box">', unsafe_allow_html=True)

        text = st.text_area(
            "Enter your feedback:",
            placeholder="Type review here...",
            height=300
        )

        analyze = st.button("🧠 Analyze Review", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- RIGHT SIDE (OUTPUT BOX) ----------------
    with col2:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        

        if analyze and text.strip():
            # 🔄 Reset feedback UI for new analysis
            st.session_state.analysis_done = False
            st.session_state.feedback_submitted = False
            st.session_state.show_feedback_box = False
            with st.spinner("Analyzing..."):
                res = model(text)
                raw_label = normalize_label(res[0]["label"])
                score = res[0]["score"]

                neutral_words = {"ok", "fine", "average", "decent", "satisfactory",
                                 "alright", "moderate", "fair", "acceptable",
                                 "normal", "standard"}
                lower_text = text.lower()

                if any(nw in lower_text for nw in neutral_words):
                    label = "NEUTRAL"
                else:
                    label = adjust_label(raw_label, score)

                
                # rule-based aspects
                aspects = extract_aspects_rule_based(text)

                # dependency-based pairs
                aspect_pairs = extract_aspects_spacy(text)

                # classify aspect sentiments
                aspect_sent = classify_aspect_sentiment(aspect_pairs)

                # Save to DB
                save_analysis_to_db(
                    st.session_state.user["email"] if st.session_state.user else "anonymous",
                    text,
                    label,
                    score,
                    aspects,
                    aspect_pairs
                )

                 # 4️⃣ ✅ NOW store everything in session_state (THIS WAS THE BUG)
                st.session_state.analysis_done = True
                st.session_state.analysis_result = {
                    "text": text,
                    "label": label,
                    "score": score,
                    "aspects": aspects,
                    "aspect_pairs": aspect_pairs,
                    "aspect_sent": aspect_sent 
                }
                st.session_state["latest_feedback"] = text
                st.session_state["latest_sentiment"] = label
                st.session_state["latest_confidence"] = score

                # ---- RESULT TEXT ----
                st.markdown(f"<p class='success-text'>Sentiment: {label}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='success-text'>Confidence: {round(score * 100, 2)}%</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='success-text'>Aspects Detected: {', '.join(aspects) if aspects else 'None'}</p>", unsafe_allow_html=True)

                if aspect_pairs:
                    st.write("### Aspect–Opinion Pairs:")
                    for a, o in aspect_pairs:
                        st.write(f"• {a} → {o}")

        st.markdown('</div>', unsafe_allow_html=True)

    # ----------- VISUALIZATIONS BELOW BOTH BOXES -----------
    # ----------- VISUALIZATIONS (SESSION SAFE) -----------
    if st.session_state.analysis_done:

        result = st.session_state.analysis_result
        aspect_sent = result["aspect_sent"]
        label = result["label"]
        score = result["score"]


        st.markdown("### 📊 Aspect Sentiment Visualization")

        if not aspect_sent:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pie(
            [round(score * 100, 2)],
            labels=[label],
            autopct="%1.1f%%",
            startangle=140
            )
            ax.axis("equal")
            st.pyplot(fig)

        else:
            colA, colB = st.columns(2)

            with colA:
                counts = Counter([s for _, s in aspect_sent])
                fig1, ax1 = plt.subplots()
                ax1.pie(
                    counts.values(),
                    labels=counts.keys(),
                    autopct="%1.1f%%",
                    startangle=140
                )
                ax1.axis("equal")
                st.pyplot(fig1)

            with colB:
                score_map = {"positive": 1, "neutral": 0, "negative": -1}
                aspects_order = [a for a, _ in aspect_sent]
                scores = [score_map[s] for _, s in aspect_sent]

                fig2, ax2 = plt.subplots(figsize=(6, 3))
                ax2.bar(aspects_order, scores)
                ax2.set_ylim(-1.1, 1.1)
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig2)

    # --------------------------------------------------------
    #          STEP 3: USER SATISFACTION FEEDBACK
    # --------------------------------------------------------
    # ---------------- USER SATISFACTION FEEDBACK ----------------
    if st.session_state.analysis_done:

        result = st.session_state.analysis_result

        st.markdown("---")
        feedback_container = st.container()
        with feedback_container:
            # ✅ CASE 1: Feedback already submitted → STOP UI here
            if st.session_state.feedback_submitted:
                st.success("🙏 Thank you for your feedback!")
            else:
                # ✅ CASE 2: Feedback not yet submitted
                st.markdown("### 📝 Are you satisfied with the result?")
                col1, col2 = st.columns(2)

        # 👍 YES
            with col1:
                if st.button("👍 Yes", key="single_yes"):
                    save_user_feedback(
                        user_email=st.session_state.user["email"],
                        satisfied=True,
                        thoughts="",
                        source="single",
                        review_text=result["text"],
                        sentiment=result["label"],
                        aspects=result["aspects"],
                        aspect_pairs=[(str(a), str(o)) for a, o in result["aspect_pairs"]]
                    ) 
                    st.session_state.feedback_submitted = True
                    

        # 👎 NO
            with col2:
                if st.button("👎 No", key="single_no"):
                    st.session_state.show_feedback_box = True

        # Textbox on NO
            if st.session_state.show_feedback_box:
                fb_text = st.text_area("Tell us what went wrong:", key="fb_text")

                if st.button("Submit Feedback", key="submit_fb"):
                    save_user_feedback(
                        user_email=st.session_state.user["email"],
                        satisfied=False,
                        thoughts=fb_text,
                        source="single",
                        review_text=result["text"],
                        sentiment=result["label"],
                        aspects=result["aspects"],
                        aspect_pairs=[(str(a), str(o)) for a, o in result["aspect_pairs"]]
                    )
                    st.session_state.feedback_submitted = True
                    st.session_state.show_feedback_box = False
                    



def batch_analysis_page():
    st.header("📊 Batch Analysis")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if not file:
        return

    # robust CSV read
    try:
        df = pd.read_csv(file, encoding="latin1", engine="python")
    except Exception:
        df = pd.read_csv(file, engine="python")

    st.dataframe(df.head())

    # select text column
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not text_cols:
        st.warning("No text columns detected in uploaded file.")
        return
    selected_col = st.selectbox("Choose text column", text_cols)

    # detect existing sentiment/label column if any
    label_candidates = [c for c in df.columns if c.lower() in ["airline_sentiment", "sentiment", "label"]]
    has_label = len(label_candidates) > 0
    label_col = label_candidates[0] if has_label else None

    if st.button("Run Analysis"):
        with st.spinner("Processing..."):
            all_aspect_sentiments = []  
            all_aspect_triplets = []    

            aspect_list_col = []
            aspect_pairs_col = []
            sentiment_col = []
            confidence_col = []

            for idx, row in df.iterrows():
                review = str(row[selected_col]) if pd.notna(row[selected_col]) else ""
                review = review.strip()

                # Overall sentiment
                if has_label:
                    full_label = str(row[label_col])
                    score = None
                else:
                    try:
                        out = model(review)[0]
                        full_label = normalize_label(out.get("label", "LABEL_1"))
                        score = round(out.get("score", 0) * 100, 2)
                    except Exception:
                        full_label = "NEUTRAL"
                        score = None

                sentiment_col.append(full_label)
                confidence_col.append(score)

                keyword_aspects = extract_aspects_rule_based(review)
                keyword_pairs = []

                # spaCy dependency extraction
                doc = nlp(review)
                spacy_pairs = []
                valid_aspect_terms = set(sum(aspect_keywords.values(), []))

                for token in doc:
                    if token.dep_ == "amod" and token.head.pos_ == "NOUN":
                        head = token.head
                        adj = token
                        compound_parts = [w.text for w in head.lefts if w.dep_ == "compound"]
                        full_term = " ".join(compound_parts + [head.text])
                        asp_key = map_term_to_aspect(full_term) or map_term_to_aspect(head.text)
                        final_aspect = asp_key if asp_key else full_term.lower()

                        if asp_key or full_term.lower() in valid_aspect_terms or head.text.lower() in valid_aspect_terms:
                            spacy_pairs.append((final_aspect, handle_negation(adj)))

                    if token.dep_ == "nsubj" and token.head.pos_ == "ADJ":
                        noun = token
                        adj = token.head
                        compound_parts = [w.text for w in noun.lefts if w.dep_ == "compound"]
                        full_term = " ".join(compound_parts + [noun.text])
                        asp_key = map_term_to_aspect(full_term) or map_term_to_aspect(noun.text)
                        final_aspect = asp_key if asp_key else full_term.lower()

                        if asp_key or full_term.lower() in valid_aspect_terms or noun.text.lower() in valid_aspect_terms:
                            spacy_pairs.append((final_aspect, handle_negation(adj)))

                    if token.dep_ == "acomp":
                        subs = [w for w in token.head.lefts if w.dep_ == "nsubj"]
                        if subs:
                            subj = subs[0]
                            compound_parts = [w.text for w in subj.lefts if w.dep_ == "compound"]
                            full_term = " ".join(compound_parts + [subj.text])
                            asp_key = map_term_to_aspect(full_term) or map_term_to_aspect(subj.text)
                            final_aspect = asp_key if asp_key else full_term.lower()

                            if asp_key or full_term.lower() in valid_aspect_terms or subj.text.lower() in valid_aspect_terms:
                                spacy_pairs.append((final_aspect, token.text.lower()))

                combined_pairs = keyword_pairs + spacy_pairs

                seen = set()
                final_pairs = []
                for a, o in combined_pairs:
                    if (a, o) not in seen:
                        final_pairs.append((a, o))
                        seen.add((a, o))

                if not final_pairs:
                    if keyword_aspects:
                        overall_sent = full_label.lower()
                        final_pairs = [(asp, overall_sent) for asp in keyword_aspects]
                        classified = [(asp, overall_sent) for asp in keyword_aspects]
                    else:
                        classified = []
                else:
                    classified = classify_aspect_sentiment(final_pairs)

                # Build triplets
                triplets = []
                for (a, o), (_, s) in zip(final_pairs, classified):
                    triplets.append((a, o, s))
                    all_aspect_sentiments.append((a, s))
                    all_aspect_triplets.append((a, o, s))

                if len(classified) < len(final_pairs):
                    for (a, o) in final_pairs[len(classified):]:
                        fallback_sent = full_label.lower()
                        triplets.append((a, o, fallback_sent))
                        all_aspect_sentiments.append((a, fallback_sent))
                        all_aspect_triplets.append((a, o, fallback_sent))

                aspect_pairs_col.append(triplets)
                aspect_list_col.append([a for a, _, _ in triplets])

                existing = user_db[RESULTS_COLL].find_one({
                    "review": review,
                    "user_email": st.session_state.user["email"]
                })

                if not existing:
                    save_analysis_to_db(
                        st.session_state.user["email"],
                        review,
                        full_label,
                        score / 100 if score else 0,
                        keyword_aspects,
                        triplets
                    )

            # Add outputs to dataframe
            df["Sentiment"] = sentiment_col
            df["Confidence"] = confidence_col
            df["Aspects"] = aspect_list_col
            df["Aspect_Pairs"] = aspect_pairs_col

            st.success("Batch Analysis Completed!")
            st.dataframe(df)

            # Visualization
            if all_aspect_sentiments:
                st.markdown("### 📊 Aspect Sentiment Distribution (Dataset)")
                sent_counts = Counter([s for _, s in all_aspect_sentiments])
            else:
                st.markdown("### 📊 Overall Sentiment Distribution (Dataset)")
                sent_counts = Counter([s.lower() for s in sentiment_col])

            labels = list(sent_counts.keys())
            sizes = list(sent_counts.values())

            color_map = {
                "positive": "#4CAF50",
                "negative": "#F44336",
                "neutral": "#2196F3"
            }
            colors_list = [color_map.get(lbl.lower(), "#999999") for lbl in labels]

            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pie( 
                sizes,
                labels=labels,
                colors=colors_list,
                autopct="%1.1f%%",
                startangle=140,
            )
            ax.axis("equal")
            st.pyplot(fig, use_container_width=False)
            
            # ✅ Persist batch analysis state (REQUIRED)
            st.session_state.batch_analysis_done = True
            st.session_state.batch_feedback_submitted = False
            st.session_state.show_batch_feedback_box = False
            st.session_state.batch_result = {
                "review_text": "BATCH_PROCESS",
    "sentiment": "MULTIPLE",
    "aspects": list(set(a for a, _ in all_aspect_sentiments)),
    "aspect_pairs": all_aspect_triplets
            }
            

        # ---------------------------------------------------------
        #                 STEP 4 — USER SATISFACTION FEEDBACK
        # ---------------------------------------------------------
    if st.session_state.batch_analysis_done:
        result = st.session_state.batch_result
        st.markdown("---")
        st.markdown("### 📝 Are you satisfied with the batch analysis results?")
        # If already submitted → stop here
        if st.session_state.batch_feedback_submitted:
            st.success("🙏 Thank you for your feedback!")
            return
        col1, col2 = st.columns(2)

        # 👍 YES
        with col1:
            if st.button("👍 (Batch)", key="batch_yes"):
                save_user_feedback(
                        user_email=st.session_state.user["email"],
                        satisfied=True,
                        thoughts="",
                        source="batch",
                        review_text=result["review_text"],
                        sentiment=result["sentiment"],
                        aspects=result["aspects"],
                        aspect_pairs=result["aspect_pairs"]
                    )
                st.session_state.batch_feedback_submitted = True
                

    # 👎 NO
        with col2:
            if st.button("👎 No (Batch)", key="batch_no"):
                st.session_state.show_batch_feedback_box = True

    # Text feedback on NO
        if st.session_state.show_batch_feedback_box:
            thoughts = st.text_area("Please share what went wrong:", key="batch_fb_text")

            if st.button("Submit Batch Feedback", key="submit_batch_fb"):
                save_user_feedback(
                    user_email=st.session_state.user["email"],
                    satisfied=False,
                    thoughts=thoughts,
                    source="batch",
                    review_text=result["review_text"],
                    sentiment=result["sentiment"],
                    aspects=result["aspects"],
                    aspect_pairs=result["aspect_pairs"]
                )
                st.session_state.batch_feedback_submitted = True
                st.session_state.show_batch_feedback_box = False
                







def profile_page():
    st.header("👤 Profile")

    if not st.session_state.user:
        st.warning("Please log in to view or edit profile.")
        return

    user_email = st.session_state.user["email"]
    profile = user_db[USER_COLL].find_one({"email": user_email}) or {}

    # Load saved photo
    def load_image(data):
        if not data:
            return None
        try:
            return Image.open(BytesIO(data))
        except:
            return None

    saved_img = load_image(profile.get("profile_image"))

    st.markdown("### Profile Photo")

    # ---------- PROFILE PHOTO (CLICKABLE) ----------
    if saved_img:
        st.image(saved_img, width=220)
    else:
        placeholder = Image.new("RGB", (220, 220), (240, 240, 240))
        st.image(placeholder, width=220)

    # Fake button that acts like clicking the image
    if st.button("📸 Change / View Photo"):
        st.session_state["open_photo_options"] = True

    # ---------- OPTIONS PANEL ----------
    if st.session_state.get("open_photo_options", False):
        st.markdown("---")
        st.markdown("### 🖼 Profile Photo Options")

        # View full image
        if saved_img:
            if st.button("👁 View Current Photo"):
                st.image(saved_img, caption="Full Size View")

        # Upload new image
        uploaded_img = st.file_uploader("Upload New Photo", type=["jpg", "jpeg", "png"])

        if uploaded_img:
            user_db[USER_COLL].update_one(
                {"email": user_email},
                {"$set": {"profile_image": Binary(uploaded_img.read())}}
            )
            st.success("Photo updated successfully!")
            st.session_state["open_photo_options"] = False
            st.rerun()

        if st.button("Close"):
            st.session_state["open_photo_options"] = False
            st.rerun()

    st.markdown("---")
    st.markdown("### Personal Details")

    # --------------------------------------
    #         EDITABLE USER DETAILS
    # --------------------------------------

    name = st.text_input("Full Name", value=profile.get("full_name", ""))
    email = st.text_input("Email", value=profile.get("email", user_email))

    dob_val = profile.get("dob")
    try:
        dob_date = pd.to_datetime(dob_val).date() if dob_val else None
    except:
        dob_date = None

    dob = st.date_input("Date of Birth", value=dob_date) if dob_date else st.date_input("Date of Birth")

    degree = st.text_input("Degree", value=profile.get("degree", ""))
    college = st.text_input("College", value=profile.get("college", ""))
    hobbies = st.text_area("Hobbies", value=profile.get("hobbies", ""))

    if st.button("Save Profile"):
        update_data = {
            "full_name": name.strip(),
            "email": email.strip().lower(),
            "dob": str(dob),
            "degree": degree.strip(),
            "college": college.strip(),
            "hobbies": hobbies.strip(),
            "updated_at": datetime.now()
        }

        old_email = user_email
        new_email = update_data["email"]

        user_db[USER_COLL].update_one(
            {"email": old_email},
            {"$set": update_data},
            upsert=True
        )

        # Sync email in session + analysis results
        if new_email != old_email:
            st.session_state.user["email"] = new_email
            user_db[RESULTS_COLL].update_many(
                {"user_email": old_email},
                {"$set": {"user_email": new_email}}
            )

        st.success("Profile updated successfully!")
        st.rerun()

def active_learning_page():
    box_color = "#E8F5E9" if st.session_state.theme == "light" else "#1E2A1E"
    text_color = "#000000" if st.session_state.theme == "light" else "#FFFFFF"

    st.header("🧠 Active Learning System")

    # ---------------- 🎓 DESCRIPTION BOX ----------------
    html = f"""
<div style="background-color:{box_color};
            color:{text_color};
            padding:25px;
            border-radius:12px;
            box-shadow:0 4px 8px rgba(0,0,0,0.2);
            font-size:17px;
            line-height:1.6;">

<h3>📘 What is Active Learning?</h3>

<p>
Active Learning is a machine learning approach where the model
selects the most confusing data and asks humans to correct only
those cases instead of labeling everything from scratch.
</p>

<h3>🎯 Purpose of Active Learning</h3>

<ul>
    <li>✅ Saves time</li>
    <li>✅ Improves model quality faster</li>
    <li>✅ Reduces manual workload</li>
    <li>✅ Focuses only on problematic data</li>
</ul>

<h3>🚀 How this page works?</h3>

<ol>
    <li>Your last analyzed feedback is shown here</li>
    <li>You can correct the sentiment if it's wrong</li>
    <li>The corrected value is saved into database</li>
</ol>

</div>
"""
    st.markdown(html, unsafe_allow_html=True)
    st.markdown("---")


    # ---------------- 🤖 SINGLE FEEDBACK CORRECTION ----------------
    if "latest_feedback" not in st.session_state:
        st.warning("Please analyze a review in Dashboard first.")
        return

    feedback = st.session_state["latest_feedback"]
    predicted = st.session_state["latest_sentiment"]
    confidence = st.session_state.get("latest_confidence", 0)

    st.subheader("✍ Correct Prediction")

    st.markdown("### 📝 Feedback")
    st.markdown(f"""
<div style="
    background-color:{box_color};
    color:{text_color};
    padding:15px;
    border-radius:8px;
    font-size:16px;
    border-left:5px solid #00a86b;">
{feedback}
</div>
""", unsafe_allow_html=True)


    st.markdown("### 🤖 Model Predicted Sentiment")
    st.write(f"**{predicted}**  (Confidence: {round(confidence*100,2)}%)")

    sentiment_options = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
    selected = st.selectbox(
        "Choose Correct Sentiment:",
        sentiment_options,
        index=sentiment_options.index(predicted)
    )

    # ---------------- 💾 SAVE BUTTON ----------------
    if st.button("✅ Save Correction To Database"):

        final_sentiment = selected

        user_db[RESULTS_COLL].update_one(
            {
                "review": feedback,
                "user_email": st.session_state.user["email"]
            },
            {
                "$set": {
                    "sentiment": final_sentiment,
                    "confidence": round(confidence * 100, 2),
                    "updated_at": datetime.now(),
                    "active_learning": True
                }
            },
            upsert=True
        )

        st.success(f"✅ Sentiment saved as: {final_sentiment}")
        save_log(
    st.session_state.user["email"],
    "Sentiment Corrected",
    f"Changed to {final_sentiment} | Review: {feedback[:80]}"
)



# ---------------------- ROUTING ----------------------
if st.session_state.user:

    if st.session_state.role == "admin":
        if st.session_state.page == "admin_dashboard":
            admin_dashboard()
        elif st.session_state.page == "users_list":
            users_list_page()
        elif st.session_state.page == "activity_logs":
            activity_logs_page()
        elif st.session_state.page == "admin_profile":
            admin_profile_page()
        else:
            admin_dashboard()

    else:
        if st.session_state.page == "dashboard": user_dashboard()
        if st.session_state.page == "batch_analysis": batch_analysis_page()
        if st.session_state.page == "profile": profile_page()
        if st.session_state.page == "active_learning": active_learning_page()
else:
    if st.session_state.page == "login": login_page()
    if st.session_state.page == "register": register_page()


# ---------------------- FOOTER ----------------------
st.markdown(
    f"<hr><center><small style='color:{colors['muted_text']};'>Built with ❤️ using Streamlit</small></center>",
    unsafe_allow_html=True,
)


