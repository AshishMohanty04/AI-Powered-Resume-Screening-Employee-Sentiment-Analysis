# HR AI Tools: Resume Screening and Employee Sentiment Analysis
# This code uses Hugging Face's free models and Gradio for UI
# Adapted for VS Code

import os
import re
import nltk
import PyPDF2
import numpy as np
import pandas as pd
import gradio as gr
import torch
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Ensure NLTK resources are downloaded - only needs to run once
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK resources...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

# Call the function to download NLTK resources
download_nltk_resources()

# 1. Resume Screening Tool
class ResumeScreener:
    def __init__(self):
        # Initialize the sentiment analyzer for skill matching confidence
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

# 2. Employee Sentiment Analysis Tool
class EmployeeSentimentAnalyzer:
    def __init__(self):
        # Initialize sentiment analysis model
        self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                         model="distilbert-base-uncased-finetuned-sst-2-english")
        
        # Zero-shot classifier for topic classification
        self.topic_classifier = pipeline("zero-shot-classification", 
                                       model="facebook/bart-large-mnli")
        
        # Define topics for sentiment categorization
        self.topics = [
            "work-life balance", 
            "compensation", 
            "management", 
            "career growth", 
            "company culture", 
            "job satisfaction",
            "workload",
            "remote work",
            "benefits"
        ]
        
    def analyze_sentiment(self, text):
        """Analyze sentiment of employee feedback"""
        if not text.strip():
            return {"sentiment": "neutral", "score": 0.5}
        
        result = self.sentiment_analyzer(text)
        
        # Normalize sentiment to positive, negative, neutral
        label = result[0]["label"].lower()
        score = result[0]["score"]
        
        if label == "positive":
            sentiment = "positive"
        elif label == "negative":
            sentiment = "negative"
            score = 1 - score  # Invert score for negative sentiment
        else:
            sentiment = "neutral"
            score = 0.5
            
        return {"sentiment": sentiment, "score": score}
    
    def classify_topics(self, text):
        """Classify feedback into relevant topics"""
        if not text.strip():
            return {"topics": [], "scores": []}
        
        result = self.topic_classifier(text, self.topics)
        
        # Get top 3 topics if scores are high enough
        top_topics = []
        top_scores = []
        
        for topic, score in zip(result["labels"], result["scores"]):
            if score > 0.3:  # Only include topics with reasonable confidence
                top_topics.append(topic)
                top_scores.append(score)
            
            # Limit to top 3 topics
            if len(top_topics) >= 3:
                break
                
        return {"topics": top_topics, "scores": top_scores}
    
    def predict_attrition_risk(self, feedback_text, past_feedback=None):
        """Predict attrition risk based on sentiment and topics"""
        # Analyze current feedback
        sentiment = self.analyze_sentiment(feedback_text)
        topics = self.classify_topics(feedback_text)
        
        # Base risk on sentiment score (lower score = higher risk)
        base_risk = 1 - sentiment["score"]
        
        # Adjust risk based on topics mentioned
        topic_risk_weights = {
            "work-life balance": 0.8,
            "compensation": 0.9,
            "management": 0.7,
            "career growth": 0.9,
            "company culture": 0.6,
            "job satisfaction": 0.8,
            "workload": 0.7,
            "remote work": 0.5,
            "benefits": 0.6
        }
        
        # Calculate topic-based risk
        topic_risk = 0
        for topic, score in zip(topics["topics"], topics["scores"]):
            topic_weight = topic_risk_weights.get(topic, 0.5)
            topic_risk += topic_weight * score * base_risk
            
        # Normalize risk
        if topics["topics"]:
            topic_risk = topic_risk / len(topics["topics"])
        else:
            topic_risk = base_risk * 0.5
        
        # Combine base and topic risks
        final_risk = (base_risk * 0.6) + (topic_risk * 0.4)
        
        # Consider past feedback trends if available
        if past_feedback:
            past_sentiments = [self.analyze_sentiment(text)["score"] for text in past_feedback]
            avg_past_sentiment = sum(past_sentiments) / len(past_sentiments)
            
            # If current sentiment is significantly worse than past average, increase risk
            if sentiment["score"] < avg_past_sentiment - 0.2:
                final_risk += 0.2
                
            # If there's a declining trend, increase risk
            if len(past_sentiments) >= 2 and all(past_sentiments[i] > past_sentiments[i+1] for i in range(len(past_sentiments)-1)):
                final_risk += 0.15
        
        # Cap risk between 0 and 1
        final_risk = max(0, min(1, final_risk))
        
        # Classify risk level
        if final_risk >= 0.7:
            risk_level = "High"
        elif final_risk >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
            
        return {
            "risk_score": round(final_risk, 2),
            "risk_level": risk_level
        }
    
    def generate_recommendations(self, feedback_text):
        """Generate recommendations based on feedback analysis"""
        sentiment = self.analyze_sentiment(feedback_text)
        topics = self.classify_topics(feedback_text)
        attrition_risk = self.predict_attrition_risk(feedback_text)
        
        recommendations = []
        
        # Generate recommendations based on topics and sentiment
        for topic in topics["topics"]:
            if topic == "work-life balance" and sentiment["sentiment"] == "negative":
                recommendations.append("Review workload and consider flexible work arrangements")
                
            elif topic == "compensation" and sentiment["sentiment"] == "negative":
                recommendations.append("Review compensation package or provide other benefits")
                
            elif topic == "management" and sentiment["sentiment"] == "negative":
                recommendations.append("Provide management training or review team structure")
                
            elif topic == "career growth" and sentiment["sentiment"] == "negative":
                recommendations.append("Create clear career path and professional development opportunities")
                
            elif topic == "company culture" and sentiment["sentiment"] == "negative":
                recommendations.append("Work on team building activities and improve communication")
                
            elif topic == "job satisfaction" and sentiment["sentiment"] == "negative":
                recommendations.append("Review job responsibilities and align with employee strengths")
                
            elif topic == "workload" and sentiment["sentiment"] == "negative":
                recommendations.append("Assess workload distribution and consider additional resources")
                
            elif topic == "remote work" and sentiment["sentiment"] == "negative":
                recommendations.append("Evaluate remote work policy and provide necessary resources")
                
            elif topic == "benefits" and sentiment["sentiment"] == "negative":
                recommendations.append("Review benefits package and consider non-monetary incentives")
        
        # Add general recommendations based on risk level
        if attrition_risk["risk_level"] == "High":
            recommendations.append("Conduct one-on-one meeting to discuss concerns")
            recommendations.append("Consider retention bonus or other incentives")
            
        elif attrition_risk["risk_level"] == "Medium":
            recommendations.append("Schedule regular check-ins to monitor satisfaction")
            recommendations.append("Provide recognition for contributions and achievements")
        
        return list(set(recommendations))  # Remove duplicates
    
    def analyze_feedback(self, feedback_text, past_feedback=None):
        """Main function to analyze employee feedback"""
        if not feedback_text.strip():
            return {
                "error": "No feedback provided",
                "sentiment": {"sentiment": "neutral", "score": 0.5},
                "topics": {"topics": [], "scores": []},
                "attrition_risk": {"risk_score": 0.0, "risk_level": "Unknown"},
                "recommendations": ["Cannot analyze - no feedback provided"]
            }
        
        # Analyze sentiment
        sentiment = self.analyze_sentiment(feedback_text)
        
        # Classify topics
        topics = self.classify_topics(feedback_text)
        
        # Predict attrition risk
        attrition_risk = self.predict_attrition_risk(feedback_text, past_feedback)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(feedback_text)
        
        return {
            "sentiment": sentiment,
            "topics": topics,
            "attrition_risk": attrition_risk,
            "recommendations": recommendations
        }

# Create Gradio UI for Resume Screening
def create_resume_screener_ui():
    resume_screener = ResumeScreener()
    
    def process_resume(resume_file, job_description):
        if resume_file is None:
            return {
                "match_score": 0,
                "skills_matched": [],
                "skills_missing": [],
                "experience_years": None,
                "education": [],
                "recommendation": "Please upload a resume"
            }
        
        result = resume_screener.analyze_resume(resume_file, job_description)
        
        # Format output for display
        output = f"""
## Resume Analysis Results

### Match Score: {result['match_score']}%

### Skills Matched:
{', '.join(result['skills_matched']) if result['skills_matched'] else 'None'}

### Skills Missing:
{', '.join(result['skills_missing']) if result['skills_missing'] else 'None'}

### Experience: 
{f"{result['experience_years']} years" if result['experience_years'] else 'Not detected'}

### Education: 
{', '.join(result['education']).title() if result['education'] else 'Not detected'}

### Recommendation:
{result['recommendation']}
        """
        
        return output
    
    default_job_description = """
    Software Engineer
    
    Requirements:
    - 3+ years of experience in software development
    - Strong proficiency in Python and JavaScript
    - Experience with web frameworks like React, Angular, or Vue
    - Knowledge of databases (SQL, NoSQL)
    - Familiarity with cloud services (AWS, Azure, or GCP)
    - Experience with version control systems like Git
    - Understanding of data structures and algorithms
    - Bachelor's degree in Computer Science or related field
    - Experience with Docker and CI/CD pipelines is a plus
    """
    
    with gr.Blocks(title="Resume Screening Tool") as resume_app:
        gr.Markdown("# Resume Screening Tool for Software Engineer Positions")
        
        with gr.Row():
            with gr.Column():
                resume_file = gr.File(label="Upload Resume (PDF)")
                job_desc = gr.Textbox(
                    label="Job Description",
                    placeholder="Enter job description...",
                    value=default_job_description,
                    lines=10
                )
                submit_btn = gr.Button("Analyze Resume")
            
            with gr.Column():
                output = gr.Markdown(label="Analysis Result")
        
        submit_btn.click(
            fn=process_resume,
            inputs=[resume_file, job_desc],
            outputs=output
        )
    
    return resume_app

# Create Gradio UI for Employee Sentiment Analysis
def create_sentiment_analyzer_ui():
    sentiment_analyzer = EmployeeSentimentAnalyzer()
    
    def process_feedback(feedback, past_feedback=None):
        if not past_feedback:
            past_feedback = []
        
        result = sentiment_analyzer.analyze_feedback(feedback, past_feedback)
        
        # Create visualization for sentiment and risk
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Sentiment gauge
        sentiment_score = result["sentiment"]["score"]
        sentiment_colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]  # red, yellow, green
        sentiment_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("sentiment", sentiment_colors)
        sentiment_norm = plt.Normalize(0, 1)
        
        ax1.pie([sentiment_score, 1-sentiment_score], colors=[sentiment_cmap(sentiment_norm(sentiment_score)), 'lightgray'], 
               startangle=90, counterclock=False, wedgeprops=dict(width=0.3))
        ax1.text(0, 0, f"{int(sentiment_score*100)}%", ha='center', va='center', fontsize=20)
        ax1.set_title("Sentiment Score")
        
        # Risk gauge
        risk_score = result["attrition_risk"]["risk_score"]
        risk_colors = [(0, 1, 0), (1, 1, 0), (1, 0, 0)]  # green, yellow, red
        risk_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("risk", risk_colors)
        risk_norm = plt.Normalize(0, 1)
        
        ax2.pie([risk_score, 1-risk_score], colors=[risk_cmap(risk_norm(risk_score)), 'lightgray'], 
               startangle=90, counterclock=False, wedgeprops=dict(width=0.3))
        ax2.text(0, 0, f"{int(risk_score*100)}%", ha='center', va='center', fontsize=20)
        ax2.set_title("Attrition Risk")
        
        # Format output for display
        topics_str = ", ".join(result["topics"]["topics"]) if result["topics"]["topics"] else "None detected"
        
        recommendations_str = "\n".join([f"- {rec}" for rec in result["recommendations"]]) if result["recommendations"] else "None"
        
        output = f"""
## Employee Feedback Analysis Results

### Sentiment: {result["sentiment"]["sentiment"].title()}

### Key Topics Mentioned:
{topics_str}

### Attrition Risk: {result["attrition_risk"]["risk_level"]} ({result["attrition_risk"]["risk_score"] * 100:.0f}%)

### Recommended Actions:
{recommendations_str}
        """
        
        return output, fig
    
    default_feedback = """
    I'm feeling frustrated with my current role. The workload is too heavy, and I don't see a clear path for growth in this company. My manager doesn't provide enough feedback or support, and I feel that my skills are being underutilized. The compensation is also below market rate for someone with my experience. I'm thinking about looking for other opportunities.
    """
    
    with gr.Blocks(title="Employee Sentiment Analysis Tool") as sentiment_app:
        gr.Markdown("# Employee Sentiment Analysis Tool")
        
        with gr.Row():
            with gr.Column():
                feedback_text = gr.Textbox(
                    label="Employee Feedback",
                    placeholder="Enter employee feedback...",
                    value=default_feedback,
                    lines=10
                )
                submit_btn = gr.Button("Analyze Feedback")
            
            with gr.Column():
                output = gr.Markdown(label="Analysis Result")
                plot_output = gr.Plot()
        
        submit_btn.click(
            fn=process_feedback,
            inputs=[feedback_text],
            outputs=[output, plot_output]
        )
    
    return sentiment_app

# Combined app with tabs
def create_hr_ai_tools_app():
    with gr.Blocks(title="HR AI Tools") as app:
        gr.Markdown("# HR AI Tools")
        
        with gr.Tabs():
            with gr.TabItem("Resume Screening Tool"):
                create_resume_screener_ui()
            
            with gr.TabItem("Employee Sentiment Analysis Tool"):
                create_sentiment_analyzer_ui()
    
    return app

# Entry point for the application
def main():
    print("Starting HR AI Tools...")
    print("Loading models... (this may take a few moments)")
    
    # Create the app
    app = create_hr_ai_tools_app()
    
    # Launch the app
    print("Launching the web interface...")
    app.launch(share=False)  # Set share=True if you want to generate a public link
    
    print("Web interface closed.")

# Run the app if this file is executed directly
if __name__ == "__main__":
    main()