# sentiment_analyzer.py
# Employee Sentiment Analysis functionality for HR AI Tools

import nltk
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

class EmployeeSentimentAnalyzer:
    def __init__(self):
        # Ensure NLTK resources are downloaded
        download_nltk_resources()
        
        print("Loading sentiment analysis models...")
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