import streamlit as st
import pandas as pd
import time
from textblob import TextBlob
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import concurrent.futures
import numpy as np
from io import StringIO
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
import warnings
warnings.filterwarnings("ignore")

# Try to use the installed clean-text library if available
try:
    from cleantext import clean as _cleantext_clean
    CLEAN_TEXT_AVAILABLE = True
except Exception:
    _cleantext_clean = None
    CLEAN_TEXT_AVAILABLE = False

# Additional imports for NLP, multiprocessing cleaning, email and counting
import os
import tempfile
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from collections import Counter
import multiprocessing as mp
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag, word_tokenize
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

# Try to import sklearn CountVectorizer for frequency matrix; optional
try:
    from sklearn.feature_extraction.text import CountVectorizer
    SKLEARN_AVAILABLE = True
except Exception:
    CountVectorizer = None
    SKLEARN_AVAILABLE = False

# NLTK setup (download minimal packages if available)
if NLTK_AVAILABLE:
    try:
        nltk.data.find('tokenizers/punkt')
    except Exception:
        try:
            nltk.download('punkt', quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find('corpora/wordnet')
    except Exception:
        try:
            nltk.download('wordnet', quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except Exception:
        try:
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find('corpora/stopwords')
    except Exception:
        try:
            nltk.download('stopwords', quiet=True)
        except Exception:
            pass

_LEMMATIZER = WordNetLemmatizer() if NLTK_AVAILABLE else None
try:
    _STOPWORDS = set(stopwords.words('english')) if NLTK_AVAILABLE else set()
except Exception:
    _STOPWORDS = set()

# Email sender credentials (as provided)
SENDER_EMAIL = "davulurikaveri@gmail.com"
SENDER_PASSWORD = "klqm nzia qtsj gaes"

def send_email_alert(to_email, subject, message, attachments=None):
    """Send email via Gmail SMTP with optional attachments."""
    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))

        if attachments:
            for path in attachments:
                try:
                    with open(path, "rb") as f:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(path)}")
                    msg.attach(part)
                except Exception:
                    continue

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, to_email, msg.as_string())
        server.quit()
        return True, f"Email sent to {to_email}"
    except Exception as e:
        return False, str(e)


def _clean_and_tag_worker(item):
    """Worker for multiprocessing cleaning.
    Expects (idx, text) and returns (idx, cleaned_str, pos_tags, tokens_lemmatized, tokens_nostop).
    - tokens_nostop: tokens after stopword removal but BEFORE lemmatization (used for frequency counts)
    - tokens_lemmatized: tokens after lemmatization (used for cleaned_text / downstream analyses)
    """
    idx, text = item
    try:
        txt = str(text)
    except Exception:
        txt = ''
    # basic cleaning: remove punctuation/digits, lowercase
    txt = re.sub(r'[^\w\s]', ' ', txt)
    txt = re.sub(r'\d+', ' ', txt)
    txt = txt.lower()
    # tokenize
    tokens = []
    if NLTK_AVAILABLE:
        try:
            tokens = word_tokenize(txt)
        except Exception:
            tokens = re.findall(r"\b\w+\b", txt)
    else:
        tokens = re.findall(r"\b\w+\b", txt)
    # remove stopwords and single-char -> keep a copy before lemmatization
    tokens_nostop = [t for t in tokens if t and t not in _STOPWORDS and len(t) > 1]

    # lemmatize (produce tokens_lemmatized)
    tokens_lemmatized = tokens_nostop
    if NLTK_AVAILABLE:
        try:
            tokens_lemmatized = [_LEMMATIZER.lemmatize(t) for t in tokens_nostop]
        except Exception:
            pass

    # pos tagging (on lemmatized tokens)
    pos = []
    if NLTK_AVAILABLE and tokens_lemmatized:
        try:
            pos = pos_tag(tokens_lemmatized)
        except Exception:
            pos = []

    cleaned_str = " ".join(tokens_lemmatized)
    return (idx, cleaned_str, pos, tokens_lemmatized, tokens_nostop)


def parallel_clean_series(series, n_cores=None):
    """Parallel clean a pandas Series or list; returns dict idx -> {'cleaned_text','pos_tags','tokens'}"""
    items = []
    if hasattr(series, 'items'):
        for idx, val in series.items():
            items.append((idx, val))
    else:
        for i, val in enumerate(series):
            items.append((i, val))
    if not items:
        return {}
    workers = min(len(items), n_cores or mp.cpu_count())
    results = {}
    try:
        ctx = mp.get_context('spawn') if os.name == 'nt' else mp.get_context()
        with ctx.Pool(processes=workers) as pool:
            for out in pool.map(_clean_and_tag_worker, items):
                # worker now returns (idx, cleaned_str, pos, tokens_lemmatized, tokens_nostop)
                if not out:
                    continue
                if len(out) == 5:
                    idx, cleaned_str, pos, tokens_lemmatized, tokens_nostop = out
                else:
                    # backward compatible fallback
                    idx, cleaned_str, pos, tokens_lemmatized = out
                    tokens_nostop = tokens_lemmatized
                results[idx] = {
                    'cleaned_text': cleaned_str,
                    'pos_tags': pos,
                    'tokens': tokens_lemmatized,
                    'tokens_nostop': tokens_nostop
                }
    except Exception:
        # fallback sequential
        for idx, text in items:
            out = _clean_and_tag_worker((idx, text))
            if not out:
                continue
            if len(out) == 5:
                i, cleaned_str, pos, tokens_lemmatized, tokens_nostop = out
            else:
                i, cleaned_str, pos, tokens_lemmatized = out
                tokens_nostop = tokens_lemmatized
            results[i] = {
                'cleaned_text': cleaned_str,
                'pos_tags': pos,
                'tokens': tokens_lemmatized,
                'tokens_nostop': tokens_nostop
            }
    return results


def _get_original_texts(data):
    """Return an array-like of original texts whether `data` is a pandas Series or a list."""
    try:
        if hasattr(data, 'values'):
            return data.values
    except Exception:
        pass
    # Fallback: assume it's an iterable/list
    return list(data)

# Try to import plotly, fallback to matplotlib if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.info("💡 Install plotly for better interactive charts: `pip install plotly`")


def _safe_bar_chart(data, title=None):
    """Render a bar chart using Plotly (if available) or Matplotlib as a safe fallback.
    This avoids calling Streamlit's `st.bar_chart` which can import Altair in some envs.
    Accepts a pandas Series, DataFrame, or other array-like.
    """
    try:
        # Normalize pandas objects
        if isinstance(data, pd.Series):
            s = data
        elif isinstance(data, pd.DataFrame) and data.shape[1] == 1:
            # single-column DataFrame -> Series
            s = data.iloc[:, 0]
        else:
            s = None

        if PLOTLY_AVAILABLE:
            if s is not None:
                fig = px.bar(x=s.index.astype(str), y=s.values, title=title or "", labels={'x': '', 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
                return
            # DataFrame or other: let Plotly try to render
            fig = px.bar(data_frame=data, title=title or "")
            st.plotly_chart(fig, use_container_width=True)
            return

        # Fallback to matplotlib
        if s is not None:
            fig, ax = plt.subplots(figsize=(6, 4))
            # Attempt color mapping for common sentiment indices
            try:
                colors = [('#2ecc71' if x == 'positive' else '#e74c3c' if x == 'negative' else '#95a5a6') for x in s.index]
            except Exception:
                colors = None
            ax.bar(s.index.astype(str), s.values, color=colors)
            if title:
                ax.set_title(title)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            return

        # Generic fallback for DataFrame or other array-like
        try:
            df = pd.DataFrame(data)
            fig, ax = plt.subplots(figsize=(6, 4))
            df.plot(kind='bar', ax=ax)
            if title:
                ax.set_title(title)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            return
        except Exception:
            # Last resort: print raw data
            st.write(data)
            return
    except Exception as e:
        # If plotting helper itself fails, avoid crashing the app
        st.write(data)
        return

def analyze_csv_sentiment():
    """
    Main function for CSV sentiment analysis
    """
    st.markdown("## 📊 CSV File Sentiment Analysis")

    # --- Left-side navigation (small, non-intrusive) ---
    # Provide quick links to jump to different analysis views.
    nav_choice = st.sidebar.radio(
        "Navigate to:",
        ("Open CSV Analyzer", "Text Summaries", "TextBlob Analysis", "LLM Analysis", "Comparison"),
        index=0,
        key='csv_nav_choice'
    )
    # Persist last nav choice
    st.session_state['csv_nav'] = nav_choice

    # Sidebar mail button (always-visible button in the sidebar)
    # Keep this here so the user can send results at any time regardless of navigation
    if st.sidebar.button("📧 Mail Me"):
        mail_results()
        return

    # If the user selected a view other than the CSV uploader, try to open that view
    if nav_choice != "Open CSV Analyzer":
        saved_data = st.session_state.get('analyze_data')
        saved_column = st.session_state.get('analyze_column')
        if saved_data is None or saved_column is None:
            st.info("Please upload a CSV and run an analysis first via 'Open CSV Analyzer'.")
            return
        # Route to the requested view (call the underlying view functions directly)
        if nav_choice == "Text Summaries":
            text_summary(saved_data, saved_column)
            return
        if nav_choice == "TextBlob Analysis":
            textblob_analysis(saved_data, saved_column)
            return
        elif nav_choice == "LLM Analysis":
            llm_analysis(saved_data, saved_column)
            return
        elif nav_choice == "Comparison":
            comparison_analysis()
            return
    # File upload section
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload a CSV file containing text data to analyze"
    )

    df = None
    # If user uploaded file in this run, read it and persist in session_state
    if uploaded_file is not None:
        try:
            # Create a clean copy of the file content
            file_content = uploaded_file.getvalue()
            
            # Try different encodings to handle various CSV files
            try:
                # Try UTF-8 first
                decoded_content = file_content.decode('utf-8')
                df = pd.read_csv(StringIO(decoded_content))
            except UnicodeDecodeError:
                try:
                    # Try Latin-1
                    decoded_content = file_content.decode('latin-1')
                    df = pd.read_csv(StringIO(decoded_content))
                except UnicodeDecodeError:
                    # Try CP1252
                    decoded_content = file_content.decode('cp1252')
                    df = pd.read_csv(StringIO(decoded_content))
            
            # Persist the uploaded DataFrame so navigation/reruns keep it
            st.session_state['analyze_full_df'] = df
            st.success(f"✅ File uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            st.info("Please make sure your file is a valid CSV format.")

    # If no file was uploaded in this run, but a DataFrame exists in session_state from earlier, reuse it
    if df is None and 'analyze_full_df' in st.session_state:
        try:
            df = st.session_state['analyze_full_df']
            st.info(f"Using previously uploaded file (rows: {len(df)}). To upload a new file, use the uploader above.")
        except Exception:
            df = None

    if df is not None:
        try:
            # Display file information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                # uploaded_file may be None for persisted df; guard size display
                size_kb = (uploaded_file.size / 1024) if uploaded_file is not None else 0
                st.metric("File size", f"{size_kb:.1f} KB")
            
            
            
            # Display column information
            st.markdown("### 📋 Column Information")
            st.write("**Available columns:**")
            for i, col in enumerate(df.columns):
                st.write(f"{i+1}. **{col}** - {df[col].dtype}")
            
            # Column selection
            st.markdown("### 🎯 Select Column for Analysis")
            text_column = st.selectbox(
                "Choose the column containing text to analyze:",
                options=df.columns,
                help="Select the column that contains the text data you want to analyze for sentiment"
            )
            
            if text_column:
                # Display sample data
                st.markdown("### 📝 Sample Data Preview")
                st.dataframe(df[[text_column]].head(10))
                
                # Row selection slider
                st.markdown("### ⚙️ Analysis Settings")
                max_rows = len(df)  # Number of rows in the uploaded CSV file
                num_rows = st.slider(
                    "Number of rows to analyze:",
                    min_value=1,
                    max_value=max_rows,  # Use actual number of rows in CSV
                    value=min(100, max_rows),
                    help=f"Select how many rows you want to analyze (1 to {max_rows}). More rows will take longer to process."
                )
                
                # Get the data to analyze (preserve original indices)
                data_to_analyze = df[text_column].dropna().head(num_rows)

                # Offer an explicit Start Cleaning button so cleaning runs only after the user selects column & rows
                st.markdown("### 🧹 Cleaning Step")
                st.write("Click **Start Cleaning** to run multiprocessing NLTK cleaning (tokenize, remove stopwords, lemmatize, POS tag). This prepares cleaned_text and tokens used for summaries and for TextBlob/HF analyses.")

                if st.button("🔄 Start Cleaning"):
                    try:
                        with st.spinner("Cleaning text (multiprocessing)..."):
                            clean_map = parallel_clean_series(data_to_analyze)

                        # Build cleaned_series & maps preserving indices
                        cleaned_series = pd.Series({idx: info['cleaned_text'] for idx, info in clean_map.items()})
                        pos_map = {idx: info.get('pos_tags', []) for idx, info in clean_map.items()}
                        tokens_nostop_map = {idx: info.get('tokens_nostop', []) for idx, info in clean_map.items()}

                        # Filter out rows that are empty or only stopwords (use tokens_nostop)
                        def has_meaningful_tokens_idx(i):
                            toks = tokens_nostop_map.get(i, [])
                            return bool(toks and len(toks) > 0)

                        mask_idx = [idx for idx in list(cleaned_series.index) if has_meaningful_tokens_idx(idx)]
                        cleaned_filtered = cleaned_series.loc[mask_idx]
                        originals_filtered = data_to_analyze.loc[mask_idx]

                        # store full clean map and filtered results in session state for later use
                        st.session_state['analyze_clean_map'] = clean_map
                        st.session_state['analyze_cleaned'] = cleaned_filtered
                        st.session_state['analyze_originals'] = originals_filtered
                        st.session_state['analyze_pos_map'] = pos_map
                        st.session_state['analyze_tokens_nostop'] = tokens_nostop_map

                        # Persist selection so navigation (Text Summaries, TextBlob, LLM) can access uploaded data
                        st.session_state['analyze_data'] = data_to_analyze
                        st.session_state['analyze_column'] = text_column
                        st.session_state['analyze_full_df'] = df

                        st.success(f"Cleaning complete: {len(data_to_analyze)} / {len(data_to_analyze)} rows kept after removing stopword-only rows.")
                    except Exception:
                        # fallback to synchronous cleaning if multiprocessing fails
                        with st.spinner("Cleaning text (sequential fallback)..."):
                            cleaned_series = data_to_analyze.apply(clean_text)
                        pos_map = {idx: [] for idx in cleaned_series.index}
                        # basic token check
                        stopwords_set = set(st.session_state.get('STOPWORDS', set())) or _STOPWORDS
                        def has_meaningful_token(s):
                            try:
                                tokens = re.findall(r"\b\w[\w']*\b", str(s).lower())
                            except Exception:
                                tokens = []
                            if not tokens:
                                return False
                            return any(tok not in stopwords_set for tok in tokens)
                        mask = cleaned_series.apply(lambda x: has_meaningful_token(x))
                        cleaned_filtered = cleaned_series[mask]
                        originals_filtered = data_to_analyze[mask]
                        st.session_state['analyze_cleaned'] = cleaned_filtered
                        st.session_state['analyze_originals'] = originals_filtered
                        st.session_state['analyze_pos_map'] = pos_map
                        st.session_state['analyze_tokens_nostop'] = {idx: re.findall(r"\b\w[\w']*\b", str(cleaned_series.loc[idx]).lower()) for idx in cleaned_series.index}
                        st.success(f"Cleaning complete (fallback): {len(data_to_analyze)} / {len(data_to_analyze)} rows kept.")

                # Persist selection in session_state so UI (tabs) survives reruns
                if st.button("🚀 Start Analysis", type="primary"):
                    # Ensure cleaning has been performed; if not, advise user
                    if 'analyze_cleaned' not in st.session_state or st.session_state['analyze_cleaned'] is None or len(st.session_state['analyze_cleaned']) == 0:
                        st.warning("Please run 'Start Cleaning' first to prepare cleaned texts (removes stopword-only rows).")
                    else:
                        st.session_state['analysis_started'] = True
                        # store the pandas Series in session state (keeps .values available)
                        st.session_state['analyze_data'] = data_to_analyze
                        st.session_state['analyze_column'] = text_column
                        st.session_state['analyze_full_df'] = df  # keep full df for downloads/comparison
                        # cleaned/original pairs already stored in session_state by the cleaning step
                        analyze_data(data_to_analyze, text_column)

            # If analysis was started in a previous run, restore it (keeps tabs stable)
            if st.session_state.get('analysis_started'):
                saved_data = st.session_state.get('analyze_data', [])
                saved_column = st.session_state.get('analyze_column')
                if saved_data is not None and saved_column:
                    analyze_data(saved_data, saved_column)
                    
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            st.info("Please make sure your file is a valid CSV format.")

def analyze_data(data, column_name):
    """
    Perform sentiment analysis on the selected data
    """
    st.markdown("---")
    st.markdown("### 🔍 Analysis Results")
    
    # Create tabs for different analysis methods
    tab1, tab2, tab3 = st.tabs(["📊 TextBlob Analysis", "🤖 LLM Analysis", "📈 Comparison"])
    
    with tab1:
        textblob_analysis(data, column_name)
    
    with tab2:
        llm_analysis(data, column_name)
    
    with tab3:
        comparison_analysis()


def text_summary(data, column_name):
    """Show cleaned text, pos tags, top/least words and allow download."""
    st.markdown("#### Text Summaries")

    # Use cleaned data if available (must have run Start Cleaning). If not available, run cleaning now.
    cleaned = st.session_state.get('analyze_cleaned')
    pos_map = st.session_state.get('analyze_pos_map', {})
    clean_map = st.session_state.get('analyze_clean_map')
    tokens_nostop_map = st.session_state.get('analyze_tokens_nostop', {})
    if cleaned is None or len(cleaned) == 0:
        with st.spinner("Cleaning text (multiprocessing)..."):
            clean_map = parallel_clean_series(data)
        cleaned = pd.Series({idx: info['cleaned_text'] for idx, info in clean_map.items()})
        pos_map = {idx: info.get('pos_tags', []) for idx, info in clean_map.items()}
        tokens_nostop_map = {idx: info.get('tokens_nostop', []) for idx, info in clean_map.items()}

    # Build DataFrame for display
    try:
        orig_df = st.session_state.get('analyze_full_df')
        if orig_df is not None:
            # keep only rows analyzed if indices align
            display_idx = [i for i in cleaned.index if i in orig_df.index]
            df = orig_df.loc[display_idx].copy()
            df.index = display_idx
        else:
            df = pd.DataFrame({column_name: data})
    except Exception:
        df = pd.DataFrame({column_name: data})

    df['cleaned_text'] = df.index.map(lambda i: cleaned.get(i, ''))
    df['pos_tags'] = df.index.map(lambda i: pos_map.get(i, []))

    # Compute Top-20 words from cleaned_text using CountVectorizer (if available), otherwise fallback to Counter
    texts = df['cleaned_text'].fillna('').astype(str).tolist()
    top20 = []
    try:
        if SKLEARN_AVAILABLE and len(texts) > 0:
            # token_pattern captures words of length >=2
            cv = CountVectorizer(token_pattern=r"(?u)\b\w\w+\b")
            X = cv.fit_transform(texts)
            freqs = np.asarray(X.sum(axis=0)).ravel()
            terms = np.array(cv.get_feature_names_out())
            freq_series = pd.Series(freqs, index=terms).sort_values(ascending=False)
            top20 = list(freq_series.head(20).items())
        else:
            # fallback: simple whitespace split on cleaned_text
            total_counter = Counter()
            for t in texts:
                total_counter.update([w for w in str(t).split() if len(w) > 1])
            top20 = total_counter.most_common(20)
    except Exception:
        # graceful fallback to Counter in case CountVectorizer fails
        total_counter = Counter()
        for t in texts:
            total_counter.update([w for w in str(t).split() if len(w) > 1])
        top20 = total_counter.most_common(20)

    st.markdown("**Top 20 words**")
    if top20:
        top_series = pd.Series({w: c for w, c in top20})
        _safe_bar_chart(top_series, title="Top 20 words")
    else:
        st.info("No words to display")

    st.markdown("**Sample POS tags and cleaned text (first 50 rows)**")
    try:
        sample_df = df[[column_name, 'cleaned_text', 'pos_tags']].head(50)
        st.dataframe(sample_df, use_container_width=True)
    except Exception:
        st.dataframe(df.head(50), use_container_width=True)

    # Download full summary (original columns + cleaned_text + pos_tags)
    try:
        out_df = df.copy()
        # Ensure cleaned_text and pos_tags are present at the end
        if 'cleaned_text' in out_df.columns:
            out_df = out_df.drop(columns=['cleaned_text']) .join(out_df['cleaned_text'])
        if 'pos_tags' in out_df.columns:
            out_df = out_df.drop(columns=['pos_tags']) .join(out_df['pos_tags'])
        csv_data = out_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Text Summary CSV", csv_data, f"text_summary_{len(out_df)}_rows.csv", "text/csv")
    except Exception:
        st.info("Unable to prepare download file.")


def mail_results():
    """Send analysis results to registered user email from users.csv (column 'email')."""
    st.markdown("#### Mail Results")
    # determine recipient
    recipient = st.session_state.get('current_user_email')
    if not recipient:
        try:
            users_df = pd.read_csv(os.path.join(os.getcwd(), 'users.csv'))
            if 'email' in users_df.columns and len(users_df) > 0:
                recipient = users_df['email'].dropna().iloc[0]
        except Exception:
            recipient = None

    if not recipient:
        st.error("No recipient email found (users.csv missing or empty).")
        return

    # Prepare attachments from session_state full dfs
    attachments = []
    try:
        tb_full = st.session_state.get('textblob_results', {}).get('full_df')
        hf_full = st.session_state.get('hf_results', {}).get('full_df')
        if tb_full is not None:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            tb_full.to_csv(tf.name, index=True)
            attachments.append(tf.name)
        if hf_full is not None:
            tf2 = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            hf_full.to_csv(tf2.name, index=True)
            attachments.append(tf2.name)
    except Exception:
        attachments = []

    # Build summary text
    tb_counts = st.session_state.get('textblob_results', {}).get('sentiment_counts', pd.Series(dtype=int))
    hf_counts = st.session_state.get('hf_results', {}).get('sentiment_counts', pd.Series(dtype=int))
    tb_counts_str = tb_counts.to_string() if not tb_counts.empty else 'No data'
    hf_counts_str = hf_counts.to_string() if not hf_counts.empty else 'No data'

    # compute match/mismatch
    tb_map = {r.get('row_index'): r.get('sentiment') for r in st.session_state.get('textblob_results', {}).get('results', []) if isinstance(r, dict)}
    hf_map = {r.get('row_index'): r.get('sentiment') for r in st.session_state.get('hf_results', {}).get('results', []) if isinstance(r, dict)}
    common = set(tb_map.keys()).intersection(set(hf_map.keys()))
    matched = sum(1 for i in common if tb_map.get(i) == hf_map.get(i))
    mismatched = len(common) - matched
    approx_agreement = (matched / len(common) * 100) if len(common) > 0 else 0.0
    total_rows = len(st.session_state.get('analyze_data', [])) if st.session_state.get('analyze_data') is not None else 0

    body = f"""
Hello,

Please find attached the sentiment analysis results.

TextBlob Sentiment Distribution:
{tb_counts_str}

LLM Sentiment Distribution:
{hf_counts_str}

Total rows: {total_rows}
Matching sentiment labels: {matched}
Mismatched sentiment labels: {mismatched}
Approx. agreement: {approx_agreement:.2f}%

Best regards,
Sentiment Analyzer
"""

    ok, info = send_email_alert(recipient, "Sentiment Analysis Results", body, attachments=attachments)
    if ok:
        st.success(info)
    else:
        st.error(f"Error sending email: {info}")

    # cleanup temp files
    for p in attachments:
        try:
            os.remove(p)
        except Exception:
            pass

def textblob_analysis(data, column_name):
    """
    Perform TextBlob sentiment analysis with parallel processing comparison
    """
    st.markdown("#### TextBlob Sentiment Analysis")

    # If results already exist in session_state, display them instead of recomputing
    existing = st.session_state.get('textblob_results')
    if existing is not None and existing.get('full_df') is not None:
        # load stored values
        parallel_results = existing.get('results', [])
        parallel_time = existing.get('parallel_time') or existing.get('processing_time') or existing.get('parallel_time') or 0
        sequential_time = existing.get('sequential_time') or 0
        sentiment_counts = existing.get('sentiment_counts', pd.Series(dtype=int))
        full_df = existing.get('full_df')

        # Display timing comparison
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sequential Time", f"{sequential_time:.2f}s")
        with col2:
            st.metric("Parallel Time", f"{parallel_time:.2f}s")
        with col3:
            speedup = sequential_time / parallel_time if parallel_time > 0 else 1
            st.metric("Speedup", f"{speedup:.2f}x")

        # Sentiment distribution (display)
        col1, col2 = st.columns(2)
        with col1:
            try:
                if not sentiment_counts.empty and PLOTLY_AVAILABLE:
                    fig_pie = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="TextBlob Sentiment Distribution", color_discrete_map={'positive':'#2ecc71','negative':'#e74c3c','neutral':'#95a5a6'})
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    _safe_bar_chart(sentiment_counts, title="TextBlob Sentiment Distribution")
            except Exception:
                _safe_bar_chart(sentiment_counts, title="TextBlob Sentiment Distribution")

        with col2:
            try:
                _safe_bar_chart(sentiment_counts, title="TextBlob Sentiment Counts")
            except Exception:
                st.write(sentiment_counts)

        # Display sample and download
        st.markdown("**📋 Sample Results**")
        try:
            st.dataframe(full_df.head(10), use_container_width=True)
        except Exception:
            st.dataframe(pd.DataFrame(parallel_results).head(10), use_container_width=True)

        # Download
        st.markdown("### 📥 Download Results")
        try:
            csv_data = full_df.to_csv(index=True).encode('utf-8')
            st.download_button("📥 Download TextBlob Analysis Results", csv_data, f"textblob_sentiment_analysis_{len(full_df)}_rows.csv", "text/csv")
        except Exception:
            pass

        return

    # ---- otherwise compute (first run) ----
    # Sequential processing
    st.markdown("**🔄 Sequential Processing**")
    start_time = time.time()

    sequential_results = []
    # Support pandas Series with indices so we can compare by row index later
    if hasattr(data, 'items'):
        iterator = data.items()
    else:
        iterator = enumerate(data)

    for idx, text in iterator:
        if pd.notna(text) and str(text).strip():
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            if polarity > 0.05:
                sentiment = "positive"
            elif polarity < -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            sequential_results.append({
                'row_index': idx,
                'text': str(text)[:100] + "..." if len(str(text)) > 100 else str(text),
                'polarity': polarity,
                'sentiment': sentiment
            })

    sequential_time = time.time() - start_time

    # Parallel processing
    st.markdown("**⚡ Parallel Processing**")
    start_time = time.time()

    def analyze_single_text(item):
        # item may be (idx, text) for series iteration or just text
        if isinstance(item, tuple) and len(item) == 2:
            idx, text = item
        else:
            idx = None
            text = item
        if pd.notna(text) and str(text).strip():
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            if polarity > 0.05:
                sentiment = "positive"
            elif polarity < -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            return {
                'row_index': idx,
                'text': str(text)[:100] + "..." if len(str(text)) > 100 else str(text),
                'polarity': polarity,
                'sentiment': sentiment
            }
        return None

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        # If data is a Series, pass items() to executor.map so analyze_single_text receives (idx,text)
        if hasattr(data, 'items'):
            parallel_results = list(executor.map(analyze_single_text, data.items()))
        else:
            parallel_results = list(executor.map(analyze_single_text, data))

    # Filter out None results
    parallel_results = [r for r in parallel_results if r is not None]

    parallel_time = time.time() - start_time

    # Sentiment distribution
    sentiment_counts = pd.Series([r['sentiment'] for r in parallel_results]).value_counts()
    
    # Create visualization
    col1, col2 = st.columns(2)
    
    with col1:
        if PLOTLY_AVAILABLE:
            # Interactive pie chart using Plotly
            try:
                fig_pie = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="TextBlob Sentiment Distribution",
                    color_discrete_map={
                        'positive': '#2ecc71',
                        'negative': '#e74c3c',
                        'neutral': '#95a5a6'
                    }
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating pie chart: {e}")
                _safe_bar_chart(sentiment_counts, title="TextBlob Sentiment Distribution")
        else:
            # Fallback to matplotlib
            try:
                fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
                colors = ['#2ecc71' if x == 'positive' else '#e74c3c' if x == 'negative' else '#95a5a6' for x in sentiment_counts.index]
                ax_pie.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
                ax_pie.set_title("TextBlob Sentiment Distribution")
                st.pyplot(fig_pie)
            except Exception as e:
                st.error(f"Error creating pie chart: {e}")
                _safe_bar_chart(sentiment_counts, title="TextBlob Sentiment Distribution")
    
    with col2:
        if PLOTLY_AVAILABLE:
            # Interactive bar chart using Plotly
            try:
                fig_bar = px.bar(
                    x=sentiment_counts.index,
                    y=sentiment_counts.values,
                    title="TextBlob Sentiment Counts",
                    color=sentiment_counts.index,
                    color_discrete_map={
                        'positive': '#2ecc71',
                        'negative': '#e74c3c',
                        'neutral': '#95a5a6'
                    }
                )
                fig_bar.update_layout(showlegend=False, xaxis_title="Sentiment", yaxis_title="Count")
                st.plotly_chart(fig_bar, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating bar chart: {e}")
                _safe_bar_chart(sentiment_counts, title="TextBlob Sentiment Counts")
        else:
            # Fallback to matplotlib
            try:
                fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
                colors = ['#2ecc71' if x == 'positive' else '#e74c3c' if x == 'negative' else '#95a5a6' for x in sentiment_counts.index]
                ax_bar.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
                ax_bar.set_title("TextBlob Sentiment Counts")
                ax_bar.set_ylabel("Count")
                plt.xticks(rotation=45)
                st.pyplot(fig_bar)
            except Exception as e:
                st.error(f"Error creating bar chart: {e}")
                _safe_bar_chart(sentiment_counts, title="TextBlob Sentiment Counts")
    
    # Build full results DataFrame including original uploaded columns plus cleaned_text, pos_tags, score, sentiment
    indices = [r.get('row_index') for r in parallel_results if r.get('row_index') is not None]
    full_df = None
    orig_full_df = st.session_state.get('analyze_full_df')
    if orig_full_df is not None and len(indices) > 0:
        try:
            # select rows by original index and keep original columns
            full_df = orig_full_df.loc[indices].copy()
            # ensure index alignment
            full_df.index = indices
        except Exception:
            full_df = pd.DataFrame({column_name: st.session_state.get('analyze_data')})
    else:
        full_df = pd.DataFrame({column_name: st.session_state.get('analyze_data')})

    # Attach cleaned_text and pos_tags from session_state maps
    cleaned_map = st.session_state.get('analyze_cleaned', pd.Series(dtype=str))
    pos_map = st.session_state.get('analyze_pos_map', {})
    full_df['cleaned_text'] = [cleaned_map.get(idx, '') if hasattr(cleaned_map, 'get') else cleaned_map.loc[idx] if idx in getattr(cleaned_map, 'index', []) else '' for idx in full_df.index]
    full_df['pos_tags'] = [pos_map.get(idx, []) for idx in full_df.index]

    # Map polarity/score and sentiment from parallel_results
    score_map = {r['row_index']: r.get('polarity') for r in parallel_results if 'row_index' in r}
    sent_map = {r['row_index']: r.get('sentiment') for r in parallel_results if 'row_index' in r}
    full_df['score'] = [score_map.get(idx) for idx in full_df.index]
    full_df['sentiment'] = [sent_map.get(idx) for idx in full_df.index]

    # Store results in session state for comparison and download
    st.session_state['textblob_results'] = {
        'results': parallel_results,
        'sentiment_counts': sentiment_counts,
        'processing_time': parallel_time,
        'sequential_time': sequential_time,
        'parallel_time': parallel_time,
        'full_df': full_df
    }

    # Display sample of final full_df
    st.markdown("**📋 Sample Results**")
    try:
        st.dataframe(full_df.head(10), use_container_width=True)
    except Exception:
        st.dataframe(pd.DataFrame(parallel_results).head(10), use_container_width=True)

    # Download functionality
    st.markdown("### 📥 Download Results")
    try:
        csv_data = full_df.to_csv(index=True).encode('utf-8')
        st.download_button(
            "📥 Download TextBlob Analysis Results",
            csv_data,
            f"textblob_sentiment_analysis_{len(full_df)}_rows.csv",
            "text/csv",
            help=f"Download results for {len(full_df)} analyzed rows"
        )
        st.info(f"📊 **Analysis Summary:** {len(full_df)} rows analyzed | Processing time: {parallel_time:.2f}s")
    except Exception:
        st.info(f"📊 **Analysis Summary:** {len(parallel_results)} rows analyzed | Processing time: {parallel_time:.2f}s")

def llm_analysis(data, column_name):
    """
    Perform Hugging Face model sentiment analysis with parallel processing comparison
    """
    st.markdown("#### 🤖 Hugging Face Model Analysis")

    # If HF results already exist, display them instead of re-running
    existing = st.session_state.get('hf_results')
    if existing is not None and existing.get('full_df') is not None:
        sentiment_counts = existing.get('sentiment_counts', pd.Series(dtype=int))
        batch_time = existing.get('batch_time') or existing.get('processing_time') or 0
        sequential_time = existing.get('sequential_time') or 0
        model_name = existing.get('model_name', 'HuggingFace')
        display_df = existing.get('full_df')

        # Display timing comparison
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sequential Time", f"{sequential_time:.2f}s")
        with col2:
            st.metric("Batch Time", f"{batch_time:.2f}s")
        with col3:
            speedup = sequential_time / batch_time if batch_time > 0 else 1
            st.metric("Speedup", f"{speedup:.2f}x")

        # Sentiment distribution
        col1, col2 = st.columns(2)
        with col1:
            try:
                if not sentiment_counts.empty and PLOTLY_AVAILABLE:
                    fig_pie = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title=f"{model_name} Sentiment Distribution", color_discrete_map={'positive':'#2ecc71','negative':'#e74c3c','neutral':'#95a5a6'})
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    _safe_bar_chart(sentiment_counts, title=f"{model_name} Sentiment Distribution")
            except Exception:
                _safe_bar_chart(sentiment_counts, title=f"{model_name} Sentiment Distribution")

        with col2:
            try:
                _safe_bar_chart(sentiment_counts, title=f"{model_name} Sentiment Counts")
            except Exception:
                st.write(sentiment_counts)

        # Display sample and download
        st.markdown("**📋 Sample Results**")
        st.dataframe(display_df.head(10), use_container_width=True)
        st.markdown("### 📥 Download Hugging Face Results")
        try:
            csv_data = display_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download HF Analysis Results", csv_data, f"hf_sentiment_analysis_{len(display_df)}_rows.csv", "text/csv")
        except Exception:
            pass

        return

    # Model Configuration (choose one model and device)
    st.markdown("**⚙️ Model Configuration**")
    col1, col2 = st.columns(2)

    with col1:
        # give explicit key so selection persists across reruns
        model_choice = st.selectbox(
            "Choose Model:",
            [
                "nlptown/bert-base-multilingual-uncased-sentiment",
                "cardiffnlp/twitter-roberta-base-sentiment"
            ],
            help="Select the Hugging Face model for sentiment analysis",
            key='hf_model_choice'
        )

    with col2:
        device = st.selectbox(
            "Processing Device:",
            ["Auto", "CPU", "GPU (if available)"],
            help="Choose processing device (GPU recommended for large datasets)",
            key='hf_device_choice'
        )
    
    # Analysis settings
    st.markdown("**🔧 Analysis Settings**")
    batch_size = st.slider(
        "Batch Size for Processing:",
        min_value=1,
        max_value=min(32, len(data)),
        value=min(8, len(data)),
        help="Number of texts to process in each batch (higher = faster but more memory)"
    )
    
    # Run HF analysis and persist that a run was requested so output remains on reruns
    if st.button("🚀 Start Hugging Face Analysis", type="primary", key='hf_start_button'):
        st.session_state['hf_requested'] = True
        # Use cleaned data stored in session_state if available
        cleaned = st.session_state.get('analyze_cleaned')
        originals = st.session_state.get('analyze_originals')
        data_for_hf = cleaned if cleaned is not None and len(cleaned) > 0 else data
        originals_for_hf = originals if originals is not None and len(originals) > 0 else None
        perform_hf_analysis(data_for_hf, originals_for_hf, model_choice, device, batch_size)
    elif st.session_state.get('hf_requested'):
        # If a run was requested earlier, show results by re-running with stored params
        stored_model = st.session_state.get('hf_model_choice', model_choice)
        stored_device = st.session_state.get('hf_device_choice', device)
        cleaned = st.session_state.get('analyze_cleaned')
        originals = st.session_state.get('analyze_originals')
        data_for_hf = cleaned if cleaned is not None and len(cleaned) > 0 else data
        originals_for_hf = originals if originals is not None and len(originals) > 0 else None
        perform_hf_analysis(data_for_hf, originals_for_hf, stored_model, stored_device, batch_size)



def perform_hf_analysis(data, originals, model_name, device, batch_size):
    """
    Perform Hugging Face model sentiment analysis
    """
    st.markdown("---")
    st.markdown("### 🔄 Processing with Hugging Face Model...")
    
    # Show loading message
    with st.spinner(f"Loading {model_name} model..."):
        try:
            # Determine device
            if device == "Auto":
                device_id = 0 if torch.cuda.is_available() else -1
            elif device == "GPU (if available)":
                device_id = 0 if torch.cuda.is_available() else -1
            else:
                device_id = -1
            
            # Load the sentiment analysis pipeline
            if model_name == "distilbert-base-uncased":
                classifier = pipeline("sentiment-analysis", 
                                    model="distilbert-base-uncased-finetuned-sst-2-english",
                                    device=device_id)
            elif model_name == "roberta-base":
                classifier = pipeline("sentiment-analysis", 
                                    model="cardiffnlp/twitter-roberta-base-sentiment",
                                    device=device_id)
            elif model_name == "nlptown/bert-base-multilingual-uncased-sentiment":
                classifier = pipeline("sentiment-analysis", 
                                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                                    device=device_id)
            elif model_name == "cardiffnlp/twitter-roberta-base-sentiment":
                classifier = pipeline("sentiment-analysis", 
                                    model="cardiffnlp/twitter-roberta-base-sentiment",
                                    device=device_id)
            
            st.success(f"✅ Model {model_name} loaded successfully!")
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return
    
    # Sequential processing
    st.markdown("**🔄 Sequential Processing**")
    sequential_start = time.time()
    
    sequential_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # data may be a pandas Series with index; originals may be a Series with same index
    if hasattr(data, 'items'):
        seq_iter = list(data.items())
    else:
        seq_iter = list(enumerate(data))

    for i, (idx, text) in enumerate(seq_iter):
        if pd.notna(text) and str(text).strip():
            status_text.text(f"Processing text {i+1}/{len(seq_iter)} (Sequential)")
            orig_text = None
            if originals is not None:
                try:
                    orig_text = originals.loc[idx]
                except Exception:
                    # fallback: use text as original
                    orig_text = text
            else:
                orig_text = text
            result = analyze_single_text_hf(text, classifier, original_text=orig_text, row_index=idx)
            if result:
                sequential_results.append(result)
        progress_bar.progress((i + 1) / len(seq_iter))
    
    sequential_time = time.time() - sequential_start
    
    # Batch processing (more efficient for HF models)
    st.markdown("**⚡ Batch Processing**")
    batch_start = time.time()
    
    # Prepare texts for batch processing
    if hasattr(data, 'items'):
        texts_to_process = [str(t) for _, t in data.items() if pd.notna(t) and str(t).strip()]
        texts_indexes = [idx for idx, t in data.items() if pd.notna(t) and str(t).strip()]
    else:
        texts_to_process = [str(text) for text in data if pd.notna(text) and str(text).strip()]
        texts_indexes = list(range(len(texts_to_process)))
    
    progress_bar2 = st.progress(0)
    status_text2 = st.empty()
    
    batch_results = []
    
    # If cleaned map exists from earlier preprocessing, reuse it to avoid cleaning during timing
    cleaned_map_hf = st.session_state.get('analyze_cleaned')
    pos_map_hf = st.session_state.get('analyze_pos_map', {})

    # Process in batches
    for i in range(0, len(texts_to_process), batch_size):
        batch = texts_to_process[i:i + batch_size]
        batch_idxes = texts_indexes[i:i + batch_size]
        # Use precomputed cleaned text if available, otherwise clean per-batch
        if cleaned_map_hf is not None and len(getattr(cleaned_map_hf, 'index', [])) > 0:
            batch_cleaned = [str(cleaned_map_hf.loc[idx]) if idx in cleaned_map_hf.index else clean_text(str(batch[k])) for k, idx in enumerate(batch_idxes)]
        else:
            batch_cleaned = [clean_text(str(t)) for t in batch]
        status_text2.text(f"Processing batch {i//batch_size + 1}/{(len(texts_to_process)-1)//batch_size + 1}")
        
        try:
            # Process batch on cleaned texts
            batch_predictions = classifier(batch_cleaned)
            
            # Convert predictions to our format and include original + cleaned + row_index
            for j, prediction in enumerate(batch_predictions):
                raw_out = f"{prediction.get('label')} ({prediction.get('score'):.3f})"
                row_idx = texts_indexes[i + j] if (i + j) < len(texts_indexes) else None
                original_val = None
                try:
                    if originals is not None and row_idx is not None:
                        original_val = originals.loc[row_idx]
                except Exception:
                    original_val = batch[j]
                if original_val is None:
                    original_val = batch[j]
                result = {
                    'row_index': row_idx,
                    'original_text': original_val,
                    'cleaned_text': batch_cleaned[j],
                    'raw_output': raw_out,
                    'sentiment': map_sentiment_label(prediction['label']),
                    'confidence': round(prediction['score'], 3),
                    'raw_label': prediction['label'],
                    'raw_score': prediction['score']
                }
                batch_results.append(result)
                
        except Exception as e:
            st.error(f"Error processing batch: {str(e)}")
            # Process individually if batch fails
            for k, text in enumerate(batch):
                try:
                    result = analyze_single_text_hf(text, classifier)
                    if result:
                        batch_results.append(result)
                except:
                    continue
        
        progress_bar2.progress(min((i + batch_size) / len(texts_to_process), 1.0))
    
    batch_time = time.time() - batch_start
    
    # Clear progress indicators
    progress_bar.empty()
    progress_bar2.empty()
    status_text.empty()
    status_text2.empty()
    
    # Display timing comparison
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sequential Time", f"{sequential_time:.2f}s")
    with col2:
        st.metric("Batch Time", f"{batch_time:.2f}s")
    with col3:
        speedup = sequential_time / batch_time if batch_time > 0 else 1
        st.metric("Speedup", f"{speedup:.2f}x")
    
    # Sentiment distribution
    if batch_results:
        sentiment_counts = pd.Series([r['sentiment'] for r in batch_results]).value_counts()
        
        # Create visualization
        col1, col2 = st.columns(2)
        
        with col1:
            if PLOTLY_AVAILABLE:
                # Interactive pie chart using Plotly
                try:
                    fig_pie = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title=f"{model_name} Sentiment Distribution",
                        color_discrete_map={
                            'positive': '#2ecc71',
                            'negative': '#e74c3c',
                            'neutral': '#95a5a6'
                        }
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating pie chart: {e}")
                    _safe_bar_chart(sentiment_counts, title=f"{model_name} Sentiment Distribution")
            else:
                # Fallback to matplotlib
                try:
                    fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
                    colors = ['#2ecc71' if x == 'positive' else '#e74c3c' if x == 'negative' else '#95a5a6' for x in sentiment_counts.index]
                    ax_pie.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
                    ax_pie.set_title(f"{model_name} Sentiment Distribution")
                    st.pyplot(fig_pie)
                except Exception as e:
                    st.error(f"Error creating pie chart: {e}")
                    _safe_bar_chart(sentiment_counts, title=f"{model_name} Sentiment Distribution")
        
        with col2:
            if PLOTLY_AVAILABLE:
                # Interactive bar chart using Plotly
                try:
                    fig_bar = px.bar(
                        x=sentiment_counts.index,
                        y=sentiment_counts.values,
                        title=f"{model_name} Sentiment Counts",
                        color=sentiment_counts.index,
                        color_discrete_map={
                            'positive': '#2ecc71',
                            'negative': '#e74c3c',
                            'neutral': '#95a5a6'
                        }
                    )
                    fig_bar.update_layout(showlegend=False, xaxis_title="Sentiment", yaxis_title="Count")
                    st.plotly_chart(fig_bar, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating bar chart: {e}")
                    _safe_bar_chart(sentiment_counts, title=f"{model_name} Sentiment Counts")
            else:
                # Fallback to matplotlib
                try:
                    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
                    colors = ['#2ecc71' if x == 'positive' else '#e74c3c' if x == 'negative' else '#95a5a6' for x in sentiment_counts.index]
                    ax_bar.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
                    ax_bar.set_title(f"{model_name} Sentiment Counts")
                    ax_bar.set_ylabel("Count")
                    plt.xticks(rotation=45)
                    st.pyplot(fig_bar)
                except Exception as e:
                    st.error(f"Error creating bar chart: {e}")
                    _safe_bar_chart(sentiment_counts, title=f"{model_name} Sentiment Counts")
        
        # Prepare full results DataFrame and cleaned/display version
        full_results_df = pd.DataFrame(batch_results)

        # Add original text column for reference (handle Series or list)
        originals = _get_original_texts(data)
        if len(full_results_df) == len(originals):
            full_results_df['original_text'] = originals[:len(full_results_df)]

        # Ensure cleaned_text exists and is actually cleaned
        if 'cleaned_text' not in full_results_df.columns or full_results_df['cleaned_text'].isnull().all() or (full_results_df['cleaned_text'].fillna('') == full_results_df['original_text'].fillna('')).all():
            full_results_df['cleaned_text'] = full_results_df['original_text'].apply(clean_text)

        # Drop internal columns for display/download
        cols_to_drop = ['confidence', 'raw_label', 'raw_score']
        display_df = full_results_df.drop(columns=[c for c in cols_to_drop if c in full_results_df.columns], errors='ignore')

        # Remove cleaned_text from the display/download DataFrame (keep it in raw_results only)
        display_df_no_cleaned = display_df.drop(columns=['cleaned_text'], errors='ignore')

        # Store results in session state (display-friendly) and keep raw results if needed
        st.session_state['hf_results'] = {
            'results': display_df_no_cleaned.to_dict(orient='records'),
            'sentiment_counts': sentiment_counts,
            'processing_time': batch_time,
            'sequential_time': sequential_time,
            'batch_time': batch_time,
            'model_name': model_name,
            'raw_results': batch_results,
            'full_df': display_df_no_cleaned
        }

        # Display sample results (first 10 rows) WITHOUT cleaned_text
        st.markdown("**📋 Sample Results**")
        st.dataframe(display_df_no_cleaned.head(10), use_container_width=True)

        # Download functionality
        st.markdown("### 📥 Download Hugging Face Results")
        if not display_df_no_cleaned.empty:
            csv_data = display_df_no_cleaned.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download HF Analysis Results",
                csv_data,
                f"hf_sentiment_analysis_{len(display_df_no_cleaned)}_rows.csv",
                "text/csv",
                help=f"Download HF results for {len(display_df_no_cleaned)} analyzed rows"
            )

            # Show summary
            st.info(f"📊 **HF Analysis Summary:** {len(display_df_no_cleaned)} rows analyzed | Processing time: {batch_time:.2f}s | Model: {model_name}")
    else:
        st.warning("No results generated. Please try again.")

def analyze_single_text_hf(text, classifier, original_text=None, row_index=None):
    """
    Analyze sentiment of a single text using Hugging Face model
    """
    try:
        # 'text' here is expected to be cleaned text (perform_hf_analysis passes cleaned text)
        cleaned = text if isinstance(text, str) else str(text)
        # Truncate cleaned text if too long
        cleaned_short = cleaned[:512] if len(cleaned) > 512 else cleaned

        # Get prediction on cleaned text
        prediction = classifier(cleaned_short)[0]

        raw_out = f"{prediction.get('label')} ({prediction.get('score'):.3f})"

        return {
            'row_index': row_index,
            'original_text': original_text if original_text is not None else cleaned,
            'cleaned_text': cleaned,
            'raw_output': raw_out,
            'sentiment': map_sentiment_label(prediction['label']),
            'confidence': round(prediction['score'], 3),
            'raw_label': prediction['label'],
            'raw_score': prediction['score']
        }
        
    except Exception as e:
        st.error(f"Error analyzing text: {str(e)}")
        return None

def map_sentiment_label(label):
    """
    Map different model labels to standard sentiment labels
    """
    label_mapping = {
        'POSITIVE': 'positive',
        'NEGATIVE': 'negative', 
        'NEUTRAL': 'neutral',
        'LABEL_0': 'negative',  # Some models use LABEL_0/1/2
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive',
        '1 star': 'negative',   # Rating-based models
        '2 stars': 'negative',
        '3 stars': 'neutral',
        '4 stars': 'positive',
        '5 stars': 'positive'
    }
    
    return label_mapping.get(label, 'neutral')


def clean_text(text: str) -> str:
    """Simple cleaning: remove URLs, mentions, extra whitespace, and trim."""
    if not isinstance(text, str):
        text = str(text)
    # Prefer using clean-text library for robust cleaning
    if CLEAN_TEXT_AVAILABLE and _cleantext_clean is not None:
        try:
            cleaned = _cleantext_clean(text,
                                       fix_unicode=True,
                                       to_ascii=False,
                                       lower=False,
                                       no_line_breaks=True,
                                       no_urls=True,
                                       no_emails=True,
                                       no_phone_numbers=True)
        except Exception:
            cleaned = text
    else:
        cleaned = text

    # additional simple cleaning: remove mentions and stray hashtags, collapse whitespace
    cleaned = re.sub(r'@\w+', '', cleaned)
    cleaned = re.sub(r'#', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()
# ...existing code...
def comparison_analysis():
    """
    Compare TextBlob and Hugging Face model results (robust - catches exceptions so UI doesn't break)
    """
    import traceback
    try:
        st.markdown("TextBlob vs Hugging Face Model Comparison")

        # Check availability
        textblob_available = 'textblob_results' in st.session_state
        hf_available = 'hf_results' in st.session_state

        if not textblob_available and not hf_available:
            st.warning("Please run both TextBlob and Hugging Face analyses to see comparison results.")
            return

        # Safe getters
        textblob_data = st.session_state.get('textblob_results', {})
        hf_data = st.session_state.get('hf_results', {})

        tb_results = textblob_data.get('results', [])
        hf_results = hf_data.get('results', [])

        tb_counts = textblob_data.get('sentiment_counts', pd.Series(dtype=int))
        hf_counts = hf_data.get('sentiment_counts', pd.Series(dtype=int))

        tb_proc = textblob_data.get('processing_time')
        hf_proc = hf_data.get('processing_time')

        # TextBlob summary
        if textblob_available:
            st.markdown("**TextBlob Results:**")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Processing Time", f"{tb_proc:.2f}s" if tb_proc is not None else "N/A")
            with c2:
                st.metric("Total Analyzed", len(tb_results))
            with c3:
                most_common_tb = tb_counts.index[0] if (hasattr(tb_counts, 'index') and len(tb_counts) > 0) else "N/A"
                st.metric("Most Common", most_common_tb)

           
            try:
                if not tb_counts.empty:
                    _safe_bar_chart(tb_counts, title="TextBlob Sentiment Distribution")
                else:
                    st.info("No TextBlob sentiment counts to display.")
            except Exception:
                st.info("No TextBlob sentiment counts to display.")

        # HF summary
        if hf_available:
            hf_model_name = hf_data.get('model_name', 'HuggingFace')
            st.markdown(f"**🤖 Hugging Face Results ({hf_model_name}):**")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Processing Time", f"{hf_proc:.2f}s" if hf_proc is not None else "N/A")
            with c2:
                st.metric("Total Analyzed", len(hf_results))
            with c3:
                most_common_hf = hf_counts.index[0] if (hasattr(hf_counts, 'index') and len(hf_counts) > 0) else "N/A"
                st.metric("Most Common", most_common_hf)

           
            try:
                if not hf_counts.empty:
                    _safe_bar_chart(hf_counts, title=f"{hf_model_name} Sentiment Distribution")
                else:
                    st.info("No HF sentiment counts to display.")
            except Exception:
                st.info("No HF sentiment counts to display.")

        # Side-by-side & performance
        if textblob_available or hf_available:
            st.markdown("### 🔄 Side-by-Side Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**TextBlob Analysis**")
                try:
                    if not tb_counts.empty:
                        _safe_bar_chart(tb_counts, title="TextBlob Sentiment Distribution")
                    else:
                        st.info("No TextBlob counts")
                except Exception:
                    st.info("No TextBlob counts")
            with col2:
                st.markdown(f"**{hf_data.get('model_name','HuggingFace')} Analysis**")
                try:
                    if not hf_counts.empty:
                        _safe_bar_chart(hf_counts, title=f"{hf_data.get('model_name','HuggingFace')} Sentiment Distribution")
                    else:
                        st.info("No HF counts")
                except Exception:
                    st.info("No HF counts")

            # Performance comparison (safe reads)
            st.markdown("### ⚡ Performance Comparison")
            proc = st.session_state.get('processing_times', {})
            tb_seq = textblob_data.get('sequential_time') or proc.get('textblob_sequential') or proc.get('sequential_time')
            tb_par = textblob_data.get('processing_time') or proc.get('textblob_parallel') or proc.get('batch_time')
            hf_seq = hf_data.get('sequential_time') or proc.get('hf_sequential')
            hf_batch = hf_data.get('batch_time') or hf_data.get('processing_time') or proc.get('hf_batch')

            tdf = pd.DataFrame({
                'TextBlob Sequential': [tb_seq or 0],
                'TextBlob Parallel': [tb_par or 0],
                f"{hf_data.get('model_name','HF')} Sequential": [hf_seq or 0],
                f"{hf_data.get('model_name','HF')} Batch": [hf_batch or 0]
            })
            col1, col2 = st.columns(2)
            with col1:
                
                try:
                    _safe_bar_chart(tdf.T.rename(columns={0: 'seconds'}), title="Processing times (seconds)")
                except Exception:
                    st.write(tdf)

            # Match counts by sentiment using row_index maps
            with col2:
               
                tb_map = {r.get('row_index'): r.get('sentiment') for r in tb_results if isinstance(r, dict)}
                hf_map = {r.get('row_index'): r.get('sentiment') for r in hf_results if isinstance(r, dict)}
                sentiments = ['positive', 'negative', 'neutral']
                matched_counts = {s: 0 for s in sentiments}
                hf_counts_map = {s: 0 for s in sentiments}
                for rid, hfs in hf_map.items():
                    if hfs in sentiments:
                        hf_counts_map[hfs] += 1
                        if tb_map.get(rid) == hfs:
                            matched_counts[hfs] += 1
                match_df = pd.DataFrame({
                    'matched': [matched_counts[s] for s in sentiments],
                    'hf_total': [hf_counts_map[s] for s in sentiments]
                }, index=sentiments)
                _safe_bar_chart(match_df, title="Match counts by sentiment")

            # Agreement on overlapping rows
            tb_idx_set = {k for k in tb_map.keys() if k is not None}
            hf_idx_set = {k for k in hf_map.keys() if k is not None}
            common_idxs = sorted(list(tb_idx_set.intersection(hf_idx_set)))

            if common_idxs:
                st.markdown("### 🎯 Agreement on overlapping analyzed rows")
                agreements = 0
                total = len(common_idxs)
                disagreements = []
                for rid in common_idxs:
                    tb_s = tb_map.get(rid)
                    hf_s = hf_map.get(rid)
                    if tb_s == hf_s:
                        agreements += 1
                    else:
                        original_example = ''
                        for r in (hf_results + tb_results):
                            if isinstance(r, dict) and r.get('row_index') == rid:
                                original_example = r.get('original_text') or r.get('text') or ''
                                break
                        disagreements.append({
                            'row_index': rid,
                            'original_text': original_example,
                            'textblob': tb_s,
                            'hf': hf_s
                        })
                agreement_rate = (agreements / total) * 100 if total > 0 else 0
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Agreement Rate", f"{agreement_rate:.1f}%")
                with c2:
                    st.metric("Agreements", f"{agreements}/{total}")
                with c3:
                    st.metric("Disagreements", f"{total - agreements}/{total}")
                if disagreements:
                    st.markdown("### 🔍 Disagreement Examples (first 5)")
                    st.dataframe(pd.DataFrame(disagreements[:5]), use_container_width=True)
            else:
                st.info("No overlapping rows between TextBlob and HF results to compute agreement.")

    except Exception as exc:
        tb = traceback.format_exc()
        st.error("An error occurred inside comparison_analysis. See details below.")
        st.code(tb)
        print("Exception in comparison_analysis:", exc)
        print(tb)
        return
