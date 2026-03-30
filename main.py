import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

import config
from data_loader import load_dataset
from model_loader import load_model
from explainer_attention import get_attention_importance
from explainer_shap import get_shap_importance
from defense_pipeline import run_defense_regeneration, compute_adversarial_recovery
from evaluator import evaluate_harmfulness, rank_correlation

def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # 1. Loading
    tokenizer, model = load_model()
    dataset = load_dataset(tokenizer)
    
    results = []
    
    # 2. Loop through prompts
    print(f"Starting experiments on {len(dataset)} prompts (Batch starting at index {config.BATCH_START})...")
    for i, data in enumerate(tqdm(dataset)):
        prompt = data["prompt"]
        full_text = data["full_text"]
        input_ids = data["tokens"]
        suffix_mask = data["suffix_mask"]
        seq_len = len(input_ids)
        
        print(f"\n[Processing {i+1}/{len(dataset)} in Batch] Prompt: \"{prompt}\"")
        
        # Explainers Base Phase
        # --- Attention ---
        t0 = time.time()
        attn_scores = get_attention_importance(model, tokenizer, full_text, device=config.DEVICE)
        attn_time = time.time() - t0
        
        # --- SHAP ---
        t0 = time.time()
        # SHAP returns importance scores identically scaled 1:1 to input_ids
        shap_scores = get_shap_importance(model, tokenizer, input_ids, config.TARGET_CONTINUATION, device=config.DEVICE)
        shap_time = time.time() - t0
        
        # Explicit Sanity Check (Only on first debug prompt)
        if i == 0:
            print("\n" + "="*50)
            print("SANITY CHECK: TOP-5 EXPLAINER RANKINGS FOR PROMPT 0")
            print("="*50)
            
            def log_top_tokens(name, scores):
                top_idx = np.argsort(scores)[-5:][::-1] # descending
                print(f"--- Top 5 by {name.upper()} ---")
                for idx in top_idx:
                    tok_str = tokenizer.decode([input_ids[idx]], skip_special_tokens=True).strip()
                    is_suf = "YES" if suffix_mask[idx] == 1 else "NO"
                    print(f"Token: '{tok_str:<10}' | Score: {scores[idx]:.4f} | Is Suffix? {is_suf}")
                print()
                
            log_top_tokens("SHAP", shap_scores)
            log_top_tokens("Attention", attn_scores)
            
            with open(f"{config.RESULTS_DIR}/sanity_check_prompt_0.txt", "w") as scf:
                scf.write("SANITY CHECK RESULTS\n")
                scf.write(f"Prompt: {prompt}\n")
                scf.write(f"Total Suffix Tokens: {sum(suffix_mask)}\n\n")
                # Output can be extended if needed inside file too
            print("Sanity check verified and logged.\n" + "="*50 + "\n")
            
        # Agreement Metrics
        spearman = rank_correlation(attn_scores, shap_scores)
        
        # 3. Apply Defenses Across Budgets
        for budget in config.DELETION_BUDGETS:
            
            # Helper for executing an explainer
            def run_explainer(name, scores):
                mask, short_text, gen_resp = run_defense_regeneration(
                    model, tokenizer, full_text, input_ids, scores, budget, name, device=config.DEVICE
                )
                is_harmful = evaluate_harmfulness(prompt, gen_resp)
                recovery = compute_adversarial_recovery(scores, suffix_mask, budget)
                
                results.append({
                    "prompt_id": i,
                    "explainer": name,
                    "budget": budget,
                    "recovery_rate": recovery,
                    "harmful": is_harmful,
                    "shortened_text": short_text,
                    "response": gen_resp,
                    "exec_time": attn_time if name == "attention" else (shap_time if name == "shap" else 0),
                    "spearman_corr": spearman if name != "random" else 0
                })

            run_explainer("attention", attn_scores)
            run_explainer("shap", shap_scores)
            
            # Dummy scores for random (random generates indices internally, so passing None/Zeros is fine)
            run_explainer("random", [0]*seq_len)
            
    # 4. Save and summarize finding
    df = pd.DataFrame(results)
    file_exists = os.path.exists(config.RAW_RESULTS_FILE)
    df.to_csv(config.RAW_RESULTS_FILE, mode='a', header=not file_exists, index=False)
    print(f"\nBatch completed. Results appended to {config.RAW_RESULTS_FILE}")
    
    # Reload the entire accumulated dataset to generate the combined metrics file
    full_df = pd.read_csv(config.RAW_RESULTS_FILE)
    
    # Simple console summary
    metrics = full_df.groupby(["explainer", "budget"]).agg({
        "harmful": "mean",
        "recovery_rate": "mean",
        "exec_time": "mean"
    }).reset_index()
    
    metrics.to_csv(config.METRICS_RESULTS_FILE, index=False)
    metrics.to_markdown(config.HUMAN_RESULTS_FILE, index=False)
    print(f"Metrics Summary written to .csv and .md files.")
    print(metrics)
    
if __name__ == "__main__":
    main()
