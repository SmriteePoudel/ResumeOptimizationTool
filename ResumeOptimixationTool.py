import streamlit as st
from pdfminer.high_level import extract_text
from docx import Document
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy


nltk.download('punkt')
nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")


SKILL_KEYWORDS = ['python', 'java', 'sql', 'machine learning', 'excel',
                  'communication', 'teamwork', 'data analysis']

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

def get_similarity_score(resume_text, job_description):
    documents = [resume_text, job_description]
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(documents)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(float(similarity[0][0]) * 100, 2)

def extract_skills(text):
    doc = nlp(text.lower())
    found_skills = set()
    for token in doc:
        if token.text in SKILL_KEYWORDS:
            found_skills.add(token.text)
    return list(found_skills)


st.set_page_config(page_title="Resume Optimizer", layout="wide")
st.title("📄 Resume Optimization Tool")

uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=['pdf', 'docx'])
job_description = st.text_area("Paste the job description here")

if st.button("Optimize Resume"):
    if uploaded_file and job_description:
        if uploaded_file.name.endswith(".pdf"):
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = extract_text_from_docx(uploaded_file)

        cleaned_resume = preprocess_text(resume_text)
        cleaned_jd = preprocess_text(job_description)

        similarity = get_similarity_score(cleaned_resume, cleaned_jd)
        skills = extract_skills(resume_text)
        missing_skills = [skill for skill in SKILL_KEYWORDS if skill not in skills]

        st.subheader("🔍 Results")
        st.markdown(f"**Resume-JD Match Score:** `{similarity}%`")
        st.markdown(f"**Skills Found in Resume:** {', '.join(skills) if skills else 'None'}")
        st.markdown(f"**Missing Relevant Skills:** {', '.join(missing_skills) if missing_skills else 'None 🎉'}")
    else:
        st.warning("Please upload a resume and paste a job description.")
