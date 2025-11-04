from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import cleantext
import os
import nltk
import time
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import re
from collections import Counter
from multiprocessing import Pool, cpu_count
import html
import secrets
import json
import hashlib
from datetime import datetime, timedelta

# Try to reuse existing email helper from csv_analyzer if available
try:
    from csv_analyzer import send_email_alert
except Exception:
    def send_email_alert(to_email, subject, message, attachments=None):
        # Fallback: no-op sender for environments without SMTP helper available
        return False, "Email helper (send_email_alert) not available"



# -------------------------
# Helper Functions (Fixed)
# -------------------------
def load_users():
    if os.path.exists("users.csv"):
        df = pd.read_csv("users.csv")
        # Trim spaces from emails and passwords
        df["email"] = df["email"].astype(str).str.strip().str.lower()
        df["password"] = df["password"].astype(str).str.strip()
        return df
    else:
        return pd.DataFrame(columns=["name", "email", "password"])

def save_user(name, email, password):
    email = email.strip().lower()          # normalize email
    password = str(password).strip()       # normalize password
    users = load_users()
    
    if email in users["email"].values:
        return False
    
    new_user = pd.DataFrame([[name, email, password]], columns=["name", "email", "password"])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv("users.csv", index=False)
    return True

def validate_user(email, password):
    email = email.strip().lower()          # normalize email
    password = str(password).strip()       # normalize password
    users = load_users()
    
    if email in users["email"].values:
        stored_password = users.loc[users["email"] == email, "password"].values[0]
        return stored_password == password
    
    return False


# -------------------------
# Forgot-password (OTP) helpers
# -------------------------
RESET_FILE = "reset_codes.json"
OTP_TTL_MIN = 15
MAX_SENDS_PER_HOUR = 3

def _load_reset_codes():
    try:
        if os.path.exists(RESET_FILE):
            return json.load(open(RESET_FILE, "r"))
    except Exception:
        pass
    return {}

def _save_reset_codes(d):
    try:
        json.dump(d, open(RESET_FILE, "w"))
    except Exception:
        pass

def _prune_reset_codes():
    codes = _load_reset_codes()
    now = datetime.utcnow().isoformat()
    changed = False
    for token, info in list(codes.items()):
        try:
            if info.get("expires_at") < now:
                codes.pop(token, None)
                changed = True
        except Exception:
            codes.pop(token, None)
            changed = True
    if changed:
        _save_reset_codes(codes)

def send_otp_to_email(email):
    codes = _load_reset_codes()
    now = datetime.utcnow()
    sends_last_hour = sum(1 for t,i in codes.items() if i.get("email")==email and datetime.fromisoformat(i.get("sent_at")) > (now - timedelta(hours=1)))
    if sends_last_hour >= MAX_SENDS_PER_HOUR:
        return False, "Rate limit reached. Try again later."
    code = f"{secrets.randbelow(10**6):06d}"
    token = secrets.token_urlsafe(16)
    expires_at = (now + timedelta(minutes=OTP_TTL_MIN)).isoformat()
    codes[token] = {
        "email": email,
        "code": code,
        "sent_at": now.isoformat(),
        "expires_at": expires_at,
        "attempts": 0
    }
    _save_reset_codes(codes)
    subject = "Your password reset code"
    body = f"Your password reset code is: {code}\nIt will expire in {OTP_TTL_MIN} minutes.\nIf you did not request this, ignore this email."
    ok, info = send_email_alert(email, subject, body)
    return ok, info if ok else (False, info)

def verify_otp(email, entered_code):
    codes = _load_reset_codes()
    now = datetime.utcnow()
    for token, info in list(codes.items()):
        if info.get('email') == email:
            try:
                if datetime.fromisoformat(info.get('expires_at')) < now:
                    codes.pop(token, None); _save_reset_codes(codes)
                    return False, "Code expired"
            except Exception:
                codes.pop(token, None); _save_reset_codes(codes)
                return False, "Code expired"
            if info.get('attempts', 0) >= 5:
                codes.pop(token, None); _save_reset_codes(codes)
                return False, "Too many attempts"
            if info.get('code') == entered_code:
                codes.pop(token, None); _save_reset_codes(codes)
                return True, "OK"
            else:
                info['attempts'] = info.get('attempts', 0) + 1
                codes[token] = info
                _save_reset_codes(codes)
                return False, "Invalid code"
    return False, "No code found for this email"

def hash_password(password: str) -> str:
    # Simple SHA256 hashing for now (replace with bcrypt in production)
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


# -------------------------
# Streamlit App Configuration
# -------------------------
st.set_page_config(page_title="Parallel Text Handling Processor", page_icon="💬", layout="centered")

# -------------------------
# Custom Styling
# -------------------------
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #D6EAF8, #EBF5FB);
            font-family: 'Segoe UI', sans-serif;
        }
        .main-title {
            text-align: center;
            font-size: 40px;
            color: #2E86C1;
            font-weight: 700;
            margin-bottom: 0px;
        }
        .sub-title {
            text-align: center;
            font-size: 20px;
            color: #555;
            margin-bottom: 40px;
        }
        .center-buttons {
            display: flex;
            justify-content: center;
            gap: 25px;
        }
        .stButton>button {
            background-color: #3498DB;
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            border: none;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #2874A6;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Landing Page Logic
# -------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "home"

if st.session_state["page"] == "home":
    st.markdown("<h1 class='main-title'>Parallel Text Handling Processor</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-title'>for Sentimental Analysis</h3>", unsafe_allow_html=True)

    # Center both buttons next to each other
    # Center both buttons properly side-by-side
    col_space1, col1, col2, col_space2 = st.columns([2, 1.2, 1.2, 2])

    with col1:
        login_btn = st.button("🔐 Login", use_container_width=True)
    with col2:
         signup_btn = st.button("📝 Sign Up", use_container_width=True)  # Non-breaking space between words

    if login_btn:
        st.session_state["page"] = "login"
    elif signup_btn:
        st.session_state["page"] = "signup"



# -------------------------
# SIGN UP PAGE
# -------------------------
elif st.session_state["page"] == "signup":
    st.sidebar.title("Navigation")
    if st.sidebar.button("🏠 Home"):
        st.session_state["page"] = "home"
    if st.sidebar.button("🔐 Login"):
        st.session_state["page"] = "login"

    st.title("Create an Account ✨")
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Register"):
        if name and email and password:
            if save_user(name, email, password):
                st.success("Account created successfully! Please login now.")
            else:
                st.warning("Email already registered. Try logging in.")
        else:
            st.error("Please fill all fields.")

# -------------------------
# LOGIN PAGE
# -------------------------
elif st.session_state["page"] == "login":
    st.sidebar.title("Navigation")
    if st.sidebar.button("🏠 Home"):
        st.session_state["page"] = "home"
    if st.sidebar.button("📝 Sign Up"):
        st.session_state["page"] = "signup"

    st.title("Login 🔑")
    email = st.text_input("Email").strip().lower()
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if validate_user(email, password):
            st.success(f"Welcome, {email}!")
            st.session_state["logged_in"] = True
            st.session_state["user_email"] = email
            st.session_state["page"] = "dashboard"
            
        else:
            st.error("Invalid email or password.")

    # Forgot password flow
    if st.button("Forgot password?"):
        st.session_state["show_forgot"] = True

    if st.session_state.get("show_forgot"):
        st.markdown("### Forgot password")
        fp_email = st.text_input("Enter your registered email", value="", key="fp_email")
        if st.button("Send code", key="send_code"):
            if not fp_email:
                st.error("Please enter an email.")
            else:
                ok, info = send_otp_to_email(fp_email.strip().lower())
                if ok:
                    st.success("Code sent. Check your email.")
                    st.session_state["fp_email_sent"] = fp_email.strip().lower()
                else:
                    st.error(f"Could not send code: {info}")

        if st.session_state.get("fp_email_sent"):
            entered = st.text_input("Enter the 6-digit code", key="fp_code")
            if st.button("Verify code", key="verify_code"):
                ok, info = verify_otp(st.session_state.get("fp_email_sent"), entered.strip())
                if ok:
                    st.success("Code verified — set a new password.")
                    new_pw = st.text_input("New password", type="password", key="fp_new1")
                    new_pw2 = st.text_input("Confirm password", type="password", key="fp_new2")
                    if st.button("Set password", key="fp_setpw"):
                        if new_pw and new_pw == new_pw2:
                            users = load_users()
                            email_val = st.session_state.get("fp_email_sent")
                            if email_val in users["email"].values:
                                # store password as plain text to match current validate_user behavior
                                users.loc[users["email"]==email_val, "password"] = str(new_pw).strip()
                                users.to_csv("users.csv", index=False)
                                st.success("Password updated. You can now login.")
                                st.session_state.pop("show_forgot", None)
                                st.session_state.pop("fp_email_sent", None)
                            else:
                                st.error("User not found.")
                        else:
                            st.error("Passwords do not match.")

# -------------------------
# DASHBOARD PAGE
# -------------------------
elif st.session_state["page"] == "dashboard":
    st.sidebar.title("Navigation")
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["page"] = "home"
       

    st.markdown("<h2 style='text-align:center;'>📊 Sentiment Analyzer Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;color:gray;'>Welcome to your workspace!</h3>", unsafe_allow_html=True)

    # ---------- Analyze Text UI (paste inside your dashboard section) ----------
    from textblob import TextBlob
    import re
    import pandas as pd
    import html
    
    # Import CSV analyzer module
    try:
        from csv_analyzer import analyze_csv_sentiment
        csv_module_available = True
    except ImportError as e:
        st.error(f"CSV analyzer module not found: {e}")
        csv_module_available = False

    # small stopword list (keeps things self-contained)
    STOPWORDS = {
    "the","and","for","with","that","this","have","had","was","were","will","would",
    "a","an","in","on","at","to","of","is","it","as","by","be","are","from","or","but",
    "not","so","if","we","they","you","i","can","my","our","their","me","him","her"
    }

    st.markdown("## ✨ Analyze Text")
    
    # First expander - Single text analysis
    with st.expander("📝 Analyze Single Text", expanded=True):
        text_input = st.text_area("Enter text here:", height=160, placeholder="Type/paste your review, tweet, comment, etc.")
        run = st.button("🚀 Run Analysis")
    
    # Second expander - CSV file analysis
    with st.expander("📊 Analyze CSV File", expanded=False):
        st.markdown("**Upload and analyze CSV files**")
        st.write("• Upload CSV files")
        st.write("• Select text columns")
        st.write("• Compare processing speeds")
        st.write("• Generate sentiment reports")
        csv_analyze = st.button("📁 Open CSV Analyzer")

    # Handle CSV analyzer button
    if csv_analyze:
        st.session_state['show_csv_analyzer'] = True
    
    # Show CSV analyzer if requested
    if st.session_state.get('show_csv_analyzer', False):
        st.markdown("---")
        if csv_module_available:
            # pass STOPWORDS into session_state so csv_analyzer can access it
            st.session_state['STOPWORDS'] = STOPWORDS
            analyze_csv_sentiment()
        else:
            st.error("CSV analyzer module is not available.")
        
        if st.button("🔙 Back to Dashboard"):
            st.session_state['show_csv_analyzer'] = False
            st.rerun()
    
    if run:
        raw = text_input.strip()
        if not raw:
            st.warning("Please enter some text to analyze.")
        else:
            # ---------- TextBlob overall ----------
            blob = TextBlob(raw)
            polarity = round(blob.sentiment.polarity, 3)        # -1 .. +1
            subjectivity = round(blob.sentiment.subjectivity, 3) # 0..1

            # Labels for polarity and subjectivity as in your screenshot
            if polarity > 0.05:
                overall_label = "positive"
                overall_color = "#1b8a2b"
                overall_emoji = "😊"
            elif polarity < -0.05:
                overall_label = "negative"
                overall_color = "#d62b2b"
                overall_emoji = "😟"
            else:
                overall_label = "neutral"
                overall_color = "#808080"
                overall_emoji = "😐"

            subjectivity_label = "subjective" if subjectivity >= 0.5 else "objective"

            # ---------- Word-level polarity (keywords) ----------
            # Tokenize words (keep original words for display)
            tokens = re.findall(r"\b\w[\w']*\b", raw)
            word_scores = {}
            polarity_map = {}  # Store polarity for each word for highlighting
            
            # Process each word and store polarity for highlighting
            for w in tokens:
                w_clean = w.lower().strip(".,!?;:\"'()[]{}")
                if len(w_clean) <= 2 or w_clean in STOPWORDS or w_clean.isnumeric():
                    continue
                # get polarity of single word using TextBlob
                wp = TextBlob(w_clean).sentiment.polarity
                polarity_map[w_clean] = wp  # Store polarity for highlighting
                
                # accumulate frequency-weighted polarity: store (count, sum_polarity)
                if w_clean in word_scores:
                    word_scores[w_clean][0] += 1
                    word_scores[w_clean][1] += wp
                else:
                    word_scores[w_clean] = [1, wp]

            # Build keyword list with magnitude and average polarity
            keywords = []
            for w, (count, sum_p) in word_scores.items():
                avg_p = sum_p / count  # Don't round here, keep precision
                # magnitude: use frequency * abs(avg_p) (simple heuristic)
                magnitude = count * abs(avg_p)
                keywords.append((w, magnitude, avg_p))

            # Sort by magnitude desc
            keywords_sorted = sorted(keywords, key=lambda x: x[1], reverse=True)

            # Create highlighted text: mark words positive (green) or negative (red)
            def highlight_token(tok):
                key = tok.lower().strip(".,!?;:\"'()[]{}")
                if key in polarity_map:
                    p = polarity_map[key]
                    if p > 0.05:
                        return f"<span style='background:#dff0df;padding:2px 4px;border-radius:4px;color:#064b2b'>{html.escape(tok)}</span>"
                    elif p < -0.05:
                        return f"<span style='background:#fbeaea;padding:2px 4px;border-radius:4px;color:#7a1313'>{html.escape(tok)}</span>"
                return html.escape(tok)

            # reconstruct highlighted HTML (preserve spacing)
            tokens_with_sep = re.split(r"(\s+)", raw)  # keep whitespace separators
            highlighted_parts = [highlight_token(t) if not re.match(r"\s+", t) else t for t in tokens_with_sep]
            highlighted_html = "<div style='line-height:1.8; font-size:15px; color:#333;'>" + "".join(highlighted_parts) + "</div>"

            # ---------- Display decorative result panels (grid: 2x2) ----------
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f"""
                    <div style="background:#fff9e6;border-left:6px solid {overall_color};padding:16px;border-radius:8px;">
                        <div style="font-size:14px;color:#555;margin-bottom:6px;">Overall Sentiment</div>
                        <div style="font-size:20px;font-weight:700;color:{overall_color};">{overall_emoji} {overall_label} ({polarity})</div>
                    </div>
                    """, unsafe_allow_html=True)

            with c2:
                st.markdown(
                    f"""
                    <div style="background:#f0f7ff;border-left:6px solid #3498DB;padding:16px;border-radius:8px;">
                        <div style="font-size:14px;color:#555;margin-bottom:6px;">Polarity</div>
                        <div style="font-size:20px;font-weight:700;color:#333;">{polarity}</div>
                        <div style="color:#777;margin-top:6px;">{ 'positive' if polarity>0.05 else ('negative' if polarity<-0.05 else 'neutral') }</div>
                    </div>
                    """, unsafe_allow_html=True)

            c3, c4 = st.columns(2)
            with c3:
                st.markdown(
                    f"""
                    <div style="background:#fff8f9;border-left:6px solid #E74C3C;padding:16px;border-radius:8px;">
                        <div style="font-size:14px;color:#555;margin-bottom:6px;">Subjectivity</div>
                        <div style="font-size:20px;font-weight:700;color:#333;">{subjectivity}</div>
                        <div style="color:#777;margin-top:6px;">{subjectivity_label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            with c4:
                st.markdown(
                    f"""
                    <div style="background:#f6fff6;border-left:6px solid #2ECC71;padding:16px;border-radius:8px;">
                        <div style="font-size:14px;color:#555;margin-bottom:6px;">Top Keywords</div>
                        <div style="font-size:16px;color:#333;font-weight:600;">{', '.join([k for k,_,_ in keywords_sorted[:6]]) if keywords_sorted else '—'}</div>
                        <div style="color:#777;margin-top:6px;">(highest magnitude)</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ---------- Debug information (temporary) ----------
            with st.expander("🔍 Debug Info (click to see)", expanded=False):
                st.write("**TextBlob Analysis:**")
                st.write(f"- Raw TextBlob Polarity: {blob.sentiment.polarity}")
                st.write(f"- Overall Polarity (rounded): {polarity}")
                st.write(f"- Overall Subjectivity: {subjectivity}")
                st.write(f"- Scale Position: {((polarity + 1) / 2) * 100:.1f}%")
                st.write(f"- Overall Label: {overall_label}")
                st.write(f"- Overall Emoji: {overall_emoji}")
                
                st.write("**Logic Check:**")
                st.write(f"- polarity > 0.05: {polarity > 0.05}")
                st.write(f"- polarity < -0.05: {polarity < -0.05}")
                st.write(f"- polarity between -0.05 and 0.05: {-0.05 <= polarity <= 0.05}")
                
                st.write("**Polarity Map (first 10 words):**")
                debug_items = list(polarity_map.items())[:10]
                for word, polarity in debug_items:
                    st.write(f"- '{word}': {polarity}")
                
                st.write("**Keywords Sorted (first 5):**")
                for i, (word, mag, score) in enumerate(keywords_sorted[:5]):
                    st.write(f"{i+1}. '{word}': magnitude={mag}, score={score}")

            # ---------- Show highlighted input text ----------
            st.markdown("**Highlighted text**")
            st.markdown(highlighted_html, unsafe_allow_html=True)
            st.markdown("---")

            # ---------- Enhanced Score range visualization ----------
            st.markdown("### 📊 Sentiment Score Visualization")
            
            # Create a more detailed visualization
            col_viz1, col_viz2, col_viz3 = st.columns([2, 1, 1])
            
            with col_viz1:
                # Calculate position on the scale (0 to 100%)
                # polarity ranges from -1 to +1, we need to map it to 0-100%
                scale_position = ((polarity + 1) / 2) * 100  # Convert -1 to +1 range to 0-100%
                
                # Ensure the position is within bounds (0-100%)
                scale_position = max(0, min(100, scale_position))
                
                st.markdown(f"""
                <div style="position:relative;margin:10px 0;">
                    <div style="width:100%;height:30px;background:linear-gradient(90deg,#d62b2b 0%, #f39c9c 20%, #cccccc 50%, #bfe6b8 80%, #2ecc71 100%); border-radius:15px;position:relative;border:2px solid #ddd;">
                        <div style="position:absolute;top:-8px;left:{scale_position}%;transform:translateX(-50%);width:0;height:0;border-left:10px solid transparent;border-right:10px solid transparent;border-bottom:20px solid #333;"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;margin-top:8px;font-size:12px;color:#666;">
                        <span>Very Negative (-1.0)</span>
                        <span>Neutral (0.0)</span>
                        <span>Very Positive (+1.0)</span>
                    </div>
                    <div style="text-align:center;margin-top:10px;font-size:14px;color:#333;background:#f0f0f0;padding:8px;border-radius:6px;">
                        <strong>Score: {polarity:.3f}</strong> | Position: {scale_position:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_viz2:
                st.metric("Polarity Score", f"{polarity:.3f}", delta=f"{polarity:.3f}")
            
            with col_viz3:
                st.metric("Subjectivity", f"{subjectivity:.3f}", delta=f"{subjectivity:.3f}")
            
            # Additional context
            st.markdown(f"""
            <div style="background:#f8f9fa;padding:15px;border-radius:8px;margin-top:10px;">
                <div style="font-size:14px;color:#555;">
                    <strong>Interpretation:</strong><br>
                    • <span style="color:#d62b2b;">Negative</span> (≤ -0.05): Sad/Unhappy sentiment<br>
                    • <span style="color:#666;">Neutral</span> (-0.05 to +0.05):  Balanced sentiment<br>
                    • <span style="color:#2ecc71;">Positive</span> (≥ +0.05): Happy/Upbeat sentiment<br>
                    • <strong>Current Result:</strong> {overall_emoji} {overall_label} ({polarity:.3f})<br>
                    • <strong>Subjectivity:</strong> {subjectivity_label} ({subjectivity:.3f})
                </div>
            </div>
            """, unsafe_allow_html=True)
            

            

            # ---------- Provide download of results as CSV ----------
            st.markdown("")
            if keywords_sorted:
                # Include all keywords (positive, negative, and neutral)
                all_keywords = []
                for word, magnitude, score in keywords_sorted:
                    all_keywords.append((word, magnitude, score))
                
                if all_keywords:
                    # Create DataFrame with all keywords
                    out_df = pd.DataFrame(all_keywords, columns=["keyword","magnitude","sent_score"])
                    # Round to 3 decimal places for display
                    out_df['magnitude'] = out_df['magnitude'].round(3)
                    out_df['sent_score'] = out_df['sent_score'].round(3)
                    # Add sentiment label column
                    out_df['sentiment'] = out_df['sent_score'].apply(lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))
                    csv = out_df.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download keywords CSV", csv, "keywords.csv", "text/csv")
                else:
                    st.info("No keywords found for download.")
