import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import config

def load_model():
    """
    Loads TinyLlama from Hugging Face with output_attentions=True.
    """
    print(f"Loading {config.MODEL_NAME} on {config.DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, 
        device_map=config.DEVICE, 
        torch_dtype=torch.float16,
        output_attentions=True  # CRITICAL for Gap 3: Enable attention extraction
    )
    
    # Ensure there is a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model.eval()
    return tokenizer, model
