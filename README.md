# Parallel-Text-Handling-Processor
This project was developed as part of the Infosys Springboard Internship.

**Domain:** AI / NLP / Data Processing | **Mode:** Web Application (Streamlit)

## 🔍 Project Overview
The *Parallel Text Handling Processor* is a web application designed to process, analyze, and interpret large text datasets efficiently.  
It integrates **Parallel Computing**, **Natural Language Processing (NLP)**, and **AI-based Sentiment Analysis** to extract meaningful insights from text.

The system allows users to:
- Analyze **single text input** interactively
- Upload **CSV files** for bulk analysis
- Perform **dual sentiment analysis** using TextBlob and Transformer models
- Visualize results using interactive charts
- **Download analyzed results**
- Receive **email reports** through automated SMTP integration

This project was developed as part of the **Infosys Springboard Internship**, following Agile methodology.

---

## 🎯 Objectives
- To handle large text datasets **efficiently** using Python multiprocessing.
- To provide **dual sentiment classification** (rule-based & AI-based).
- To develop a **simple and user-friendly web interface**.
- To automate **report generation and email delivery**.
- To create a **complete end-to-end text analytics workflow**.

---

## 🧠 Why Parallel Text Processing Matters
With the rapid growth of online data (reviews, social media, chats, forms), businesses need **fast and scalable** text understanding.  
Parallel text processing helps in analyzing large volumes of data **much faster**, supporting **real-time insights** for decision-making.

---

## 🌐 System Workflow
User Input / CSV Upload
↓
Text Cleaning & Normalization
↓
Sentiment Analysis (TextBlob + LLM)
↓
Result Comparison & Scoring
↓
Data Visualization & CSV Export
↓
Email Summary Report (SMTP)

## 🏗️ Technical Architecture
Frontend (UI Layer) → Streamlit
Backend (Processing) → Python, NLTK, TextBlob, Transformers
Parallel Execution → multiprocessing, ThreadPoolExecutor
Visualization → Plotly, Matplotlib
Storage & Data → Pandas, users.csv, reset_codes.json
Email Service → smtplib + email.message



---

## 🧩 Key Features
| Feature | Description |
|--------|-------------|
| 🔐 User Authentication | Login/Signup with OTP-based Password Reset |
| 📝 Single Text Analysis | Real-time TextBlob sentiment scoring |
| 📂 CSV Bulk Analysis | Parallel sentiment analysis for large datasets |
| 🤖 Dual Model Sentiment | TextBlob (rule-based) + LLM (AI-based) |
| 📊 Interactive Charts | Hover-based visual results (Plotly) |
| ⬇️ Export Results | Download output as CSV |
| 📧 Email Report | Automated results delivery to user email |

---

## ⚙️ Technologies Used

### Frontend
- Streamlit (UI)
- Plotly & Matplotlib (Interactive Graphs)

### Backend & NLP
- Python
- NLTK, TextBlob, Cleantext
- Hugging Face Transformers (LLM-based sentiment)
- PyTorch (Model backend)

### Performance
- multiprocessing
- concurrent.futures (ThreadPoolExecutor)

### Data & Storage
- Pandas, NumPy
- users.csv (User credentials)
- reset_codes.json (Temporary OTP mapping)

### Communication & Security
- smtplib & email.message (Email automation)
- hashlib + secrets (OTP Generation & validation)

---

## ▶️ How to Run the Project Locally
```bash
#1. Clone the Repository
git clone https://github.com/kaveri2025/Parallel-Text-Handling-Processor.git

# 2. Navigate into project
cd Parallel-Text-Handling-Processor

# 3. Install Required Libraries
pip install -r requirements.txt

# 4. Run the Application
streamlit run main.py

