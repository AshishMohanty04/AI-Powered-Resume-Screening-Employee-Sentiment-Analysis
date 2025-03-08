 HR AI Tools

A comprehensive suite of AI-powered tools for HR professionals to streamline recruitment and employee management processes.

## Overview

HR AI Tools is a Python application that provides two main functionalities:
1. **Resume Screening Tool**: Automatically analyze resumes against job descriptions to identify the most suitable candidates
2. **Employee Sentiment Analysis Tool**: Analyze employee feedback to understand sentiment, identify concerns, and predict attrition risk

The application uses modern NLP techniques and machine learning models to provide insights that help HR professionals make more informed decisions.

## Features

### Resume Screening Tool
- Extract text from PDF resumes
- Identify technical skills and match them against job requirements
- Detect years of experience and education level
- Calculate overall match score based on resume and job description similarity
- Provide recommendations on candidate suitability

### Employee Sentiment Analysis Tool
- Analyze sentiment in employee feedback (positive, negative, neutral)
- Classify feedback into relevant HR topics (compensation, career growth, etc.)
- Predict attrition risk based on feedback content
- Generate actionable recommendations based on analysis
- Visualize sentiment and attrition risk scores

## Installation

1. Clone the repository:
```bash
git clone [https://github.com/yourusername/hr-ai-tools.git](https://github.com/AshishMohanty04/AI-Powered-Resume-Screening-Employee-Sentiment-Analysis.git)
cd hr-ai-tools
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyPDF2
- NLTK
- scikit-learn
- Transformers (Hugging Face)
- Matplotlib
- Gradio

A `requirements.txt` file might look like:

```
gradio>=3.50.2
matplotlib>=3.7.1
nltk>=3.8.1
PyPDF2>=3.0.0
scikit-learn>=1.3.0
transformers>=4.35.0
torch>=2.0.0
```

## Usage

Run the application with:

```bash
python main.py
```

This will start a local Gradio web server, typically on http://127.0.0.1:7860/

### Resume Screening

1. Upload a PDF resume
2. Enter or modify the job description
3. Click "Analyze Resume" to get results
4. Review the match score, matched skills, missing skills, and recommendations

### Employee Sentiment Analysis

1. Enter or paste employee feedback text
2. Click "Analyze Feedback" to get results
3. Review sentiment score, key topics, attrition risk, and recommended actions

## Architecture

The project consists of the following components:

- `resume_screener.py`: Contains the `ResumeScreener` class for resume analysis
- `sentiment_analyzer.py`: Contains the `EmployeeSentimentAnalyzer` class for feedback analysis
- `hr_ai_tools_app.py`: Creates the Gradio user interface for both tools
- `main.py`: Entry point for the application

## Technical Details

- Uses Hugging Face's transformers library for NLP tasks
- Implements zero-shot classification for topic identification
- Uses cosine similarity for matching resumes to job descriptions
- Integrates NLTK for natural language processing tasks
- Creates visualizations with Matplotlib
- Builds the UI with Gradio for easy interaction

## Future Enhancements

- Add support for more document formats (DOCX, TXT, etc.)
- Implement batch processing for multiple resumes
- Add historical tracking of employee sentiment over time
- Develop custom models fine-tuned specifically for HR tasks
- Add export functionality for reports and analytics
- Implement user authentication and multi-user support

## License

[Add your license information here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
