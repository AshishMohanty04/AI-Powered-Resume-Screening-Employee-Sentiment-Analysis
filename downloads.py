import subprocess
import sys

packages = [
    "transformers", "torch", "gradio", "PyPDF2",
    "pandas", "matplotlib", "scikit-learn", "nltk", "spacy"
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
