import torch
import numpy as np

def run_defense_regeneration(model, tokenizer, original_text, input_ids, importance_scores, budget, explainer_type, device="cuda"):
    """
    Ranks the tokens based on importance_scores.
    Drops the top budget*100% of tokens.
    Returns the detokenized new prompt and regenerates the output.
    """
    seq_len = len(input_ids)
    num_to_drop = int(seq_len * budget)
    
    if num_to_drop == 0:
        shortened_text = original_text
        mask = np.ones(seq_len, dtype=bool)
    else:
        if explainer_type == "random":
            # Randomly select `num_to_drop` indices
            indices_to_drop = np.random.choice(seq_len, num_to_drop, replace=False)
        else:
            # We want to drop tokens with the highest importance.
            # np.argsort returns indices sorted ascending, so take the last `num_to_drop`
            sorted_indices = np.argsort(importance_scores)
            indices_to_drop = sorted_indices[-num_to_drop:]
            
        mask = np.ones(seq_len, dtype=bool)
        mask[indices_to_drop] = False
        
        # Create the shortened prompt
        shortened_ids = [input_ids[i] for i in range(seq_len) if mask[i]]
        shortened_text = tokenizer.decode(shortened_ids, skip_special_tokens=True)
    
    # Regenerate response
    inputs = tokenizer(shortened_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=50, 
            max_length=None,
            do_sample=False,
            temperature=None,
            top_p=None
        )
    
    input_len = inputs["input_ids"].shape[1]
    generated_text = tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True)
    
    return mask, shortened_text, generated_text

def compute_adversarial_recovery(importance_scores, suffix_mask, p):
    """
    Calculates what fraction of the actual adversarial suffix tokens
    were contained in the top p% of the ranked tokens.
    """
    seq_len = len(importance_scores)
    k = int(seq_len * p)
    
    if k == 0:
        return 0.0
        
    sorted_indices = np.argsort(importance_scores)[-k:]
    suffix_tokens_in_top_k = sum(1 for idx in sorted_indices if suffix_mask[idx] == 1)
    
    total_suffix_tokens = sum(suffix_mask)
    if total_suffix_tokens == 0:
        return 1.0 # Edge case
        
    return suffix_tokens_in_top_k / total_suffix_tokens
