# hr_ai_tools_app.py
# Gradio UI for HR AI Tools

import gradio as gr
import matplotlib.pyplot as plt

# Import our modules
from resume_screener import ResumeScreener
from sentiment_analyzer import EmployeeSentimentAnalyzer

# Create Gradio UI for Resume Screening
def create_resume_screener_ui():
    resume_screener = ResumeScreener()
    
    def process_resume(resume_file, job_description):
        if resume_file is None:
            return "Please upload a resume"
        
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
    
    # Create and launch the app
    app = create_hr_ai_tools_app()
    print("Launching the web interface...")
    app.launch(share=False)  # Set share=True if you want to generate a public link
    
    print("Web interface closed.")

# Run the app if this file is executed directly
if __name__ == "__main__":
    main()