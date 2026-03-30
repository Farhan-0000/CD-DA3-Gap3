import shap
import torch
import numpy as np

def get_shap_importance(model, tokenizer, input_ids, target_token_text, device="cuda"):
    """
    Computes SHAP values on the model's likelihood to output the `target_token_text`.
    Operates strictly via token indices (input_ids) to prevent tokenization drift and guarantee alignment.
    """
    target_token_id = tokenizer.encode(target_token_text, add_special_tokens=False)[0]
    seq_len = len(input_ids)
    
    def predict_target_prob(masks):
        """
        masks: 2D numpy array of shape (batch, seq_len) with 1s (keep) and 0s (drop)
        """
        probs = []
        for mask in masks:
            shortened_ids = [input_ids[i] for i in range(seq_len) if mask[i]]
            
            if len(shortened_ids) == 0:
                probs.append(0.0)
                continue
                
            inputs = torch.tensor([shortened_ids]).to(device)
            with torch.no_grad():
                outputs = model(inputs)
                
            # Logits of the very last token in the sequence
            next_token_logits = outputs.logits[0, -1, :]
            
            # Extract probability of the specific "Sure" token
            all_probs = torch.softmax(next_token_logits, dim=-1)
            target_prob = all_probs[target_token_id].item()
            probs.append(target_prob)
            
        return np.array(probs)

    # Use a tabular masker over feature presence (1) vs absence (0)
    # Background baseline is 'all zeros' (all tokens masked out)
    baseline = np.zeros((1, seq_len), dtype=int)
    masker = shap.maskers.Independent(baseline)
    
    # Restrict max_evals heavily for prototyping speed
    explainer = shap.Explainer(predict_target_prob, masker)
    
    # Compute SHAP for the "all ones" state (the full prompt)
    # Dynamically scale max_evals so it never crashes on prompts longer than 50 tokens
    # SHAP mathematically requires max_evals to be at least 2 * num_features + 1
    required_evals = 2 * seq_len + 50
    shap_values = explainer(np.ones((1, seq_len), dtype=int), max_evals=required_evals) 
    
    # Extract the importance values exactly matching the sequence length
    importance_scores = shap_values.values[0]
    
    return importance_scores
