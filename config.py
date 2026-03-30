import os

# Model configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" # Switch to 'cpu' if no GPU is available

# Dataset configuration
MAX_PROMPTS = 100
DEBUG_PROMPTS = 25  # Start with a small number to verify the pipeline
GCG_SUFFIX = " !@#_xj99 !@#_xj99 !@#_xj99"
TARGET_CONTINUATION = "Sure" # Affirmative token evaluated for SHAP logit

# Defense configuration
DELETION_BUDGETS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]
EXPLAINERS = ["attention", "shap", "random"]

# Evaluation Judge API configuration
# Using Gemini by default here as the API Judge, alternative could be OpenAI
GEMINI_API_KEY = "YOUR_API_KEY_HERE" 
JUDGE_MODEL = "gemini-2.5-flash"

# Output configuration
RESULTS_DIR = "./results"
RAW_RESULTS_FILE = f"{RESULTS_DIR}/raw_results.csv"
METRICS_RESULTS_FILE = f"{RESULTS_DIR}/metrics_summary.csv"
HUMAN_RESULTS_FILE = f"{RESULTS_DIR}/metrics_summary.md"

# Batch processing
# Use this to slice exactly which prompts you want to run. 
# Changing this offset is safe because the CSV is set to "append" mode.
BATCH_START = 75
