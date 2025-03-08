# resume_screener.py
# Resume Screening functionality for HR AI Tools

import os
import re
import nltk
import PyPDF2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from transformers import pipeline

def download_nltk_resources():
    """Ensure required NLTK resources are available"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK resources...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

class ResumeScreener:
    def __init__(self):
        # Ensure NLTK resources are downloaded
        download_nltk_resources()
        
        # Initialize the zero-shot classification model
        print("Loading NLP model for resume screening...")
        self.nlp = pipeline("zero-shot-classification", 
                           model="facebook/bart-large-mnli")
        
        # Technical skills for software engineers
        self.technical_skills = [
            "Python", "Java", "JavaScript", "C++", "C#", "Go", "Ruby", "PHP",
            "SQL", "NoSQL", "MongoDB", "MySQL", "PostgreSQL", "Oracle", 
            "Docker", "Kubernetes", "AWS", "Azure", "GCP", "REST API",
            "GraphQL", "React", "Angular", "Vue", "Node.js", "Django", "Flask",
            "Spring Boot", "Microservices", "Git", "CI/CD", "Jenkins",
            "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
            "Agile", "Scrum", "Data Structures", "Algorithms", "OOP"
        ]
        
        # Initialize stop words
        self.stop_words = set(stopwords.words('english'))
        
    def extract_text_from_pdf(self, file):
        """Extract text from PDF resume"""
        text = ""
        try:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        except Exception as e:
            text = f"Error extracting text: {str(e)}"
        return text
    
    def extract_skills(self, text):
        """Extract skills from resume text"""
        skills = []
        for skill in self.technical_skills:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                skills.append(skill)
        return skills
    
    def extract_experience(self, text):
        """Extract years of experience from resume text"""
        patterns = [
            r'(\d+)(?:\+)?\s*(?:years?|yrs?)(?:\s+of)?\s+experience',
            r'experience\s+of\s+(\d+)(?:\+)?\s*(?:years?|yrs?)',
            r'(?:worked|working)\s+for\s+(\d+)(?:\+)?\s*(?:years?|yrs?)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return int(matches[0])
        
        return None
    
    def analyze_education(self, text):
        """Analyze education level from resume text"""
        education_levels = {
            "phd": ["phd", "ph.d", "doctor of philosophy"],
            "masters": ["masters", "ms", "m.s", "master of", "msc", "m.sc", "mba"],
            "bachelors": ["bachelors", "bachelor of", "bs", "b.s", "b.tech", "btech", "be", "b.e"],
            "associate": ["associate", "a.s", "as degree"]
        }
        
        education = []
        for level, keywords in education_levels.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                    education.append(level)
                    break
        
        return list(set(education))
    
    def match_job_description(self, resume_text, job_description):
        """Match resume with job description using cosine similarity"""
        # Vectorize the text
        vectorizer = CountVectorizer(stop_words='english')
        
        # Check if there's enough content
        if len(resume_text.split()) < 5 or len(job_description.split()) < 5:
            return 0.0
        
        vectors = vectorizer.fit_transform([resume_text, job_description])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return similarity
    
    def analyze_resume(self, file, job_description):
        """Main function to analyze the resume against a job description"""
        resume_text = self.extract_text_from_pdf(file) if file else ""
        
        if not resume_text:
            return {
                "error": "Could not extract text from the resume",
                "match_score": 0,
                "skills_matched": [],
                "skills_missing": [],
                "experience_years": None,
                "education": [],
                "recommendation": "Cannot analyze - resume text extraction failed"
            }
        
        # Extract skills from resume
        candidate_skills = self.extract_skills(resume_text)
        
        # Extract skills from job description
        required_skills = self.extract_skills(job_description)
        
        # Find matching and missing skills
        skills_matched = [skill for skill in candidate_skills if skill in required_skills]
        skills_missing = [skill for skill in required_skills if skill not in candidate_skills]
        
        # Extract experience
        experience_years = self.extract_experience(resume_text)
        
        # Analyze education
        education = self.analyze_education(resume_text)
        
        # Calculate match score
        jd_match_score = self.match_job_description(resume_text, job_description)
        skills_match_ratio = len(skills_matched) / len(required_skills) if required_skills else 0
        
        # Combined match score (weighted average)
        match_score = 0.4 * jd_match_score + 0.6 * skills_match_ratio
        match_percentage = round(match_score * 100, 2)
        
        # Recommendation
        if match_percentage >= 75:
            recommendation = "Strong Match - Recommend for interview"
        elif match_percentage >= 50:
            recommendation = "Moderate Match - Consider for interview"
        else:
            recommendation = "Low Match - Not recommended for this role"
        
        return {
            "match_score": match_percentage,
            "skills_matched": skills_matched,
            "skills_missing": skills_missing,
            "experience_years": experience_years,
            "education": education,
            "recommendation": recommendation
        }