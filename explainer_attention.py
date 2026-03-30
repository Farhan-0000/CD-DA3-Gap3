import torch
import numpy as np

def get_attention_importance(model, tokenizer, input_text, device="cuda"):
    """
    Runs the model once, extracts the last-layer attention weights,
    averages across heads, and returns the attention row 
    corresponding to the final token looking back over the sequence.
    """
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    # outputs.attentions is a tuple of (layer_1_attns, layer_2_attns, ...)
    # Each tensor is shape (batch_size, num_heads, sequence_length, sequence_length)
    attentions = outputs.attentions
    
    # Get last layer
    last_layer_attn = attentions[-1] # shape: (1, num_heads, seq_len, seq_len)
    
    # Average across all attention heads
    avg_heads_attn = last_layer_attn.mean(dim=1).squeeze(0) # shape: (seq_len, seq_len)
    
    # Get the last row: how much the final generated token attended to all previous tokens
    # Note: seq_len - 1 is the index of the final prompt token.
    final_token_attn_row = avg_heads_attn[-1, :] # shape: (seq_len,)
    
    # Convert to numpy and normalize if needed, though raw rank works fine
    importance_scores = final_token_attn_row.cpu().numpy()
    
    return importance_scores
