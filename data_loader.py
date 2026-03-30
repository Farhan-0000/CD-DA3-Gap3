import config

def load_dataset(tokenizer, limit=config.DEBUG_PROMPTS):
    """
    Loads harmful prompts from AdvBench, appends GCG suffix,
    and calculates the exact token indices representing the adversarial suffix
    so we can evaluate our explainers against known ground-truth.
    """
    import pandas as pd
    
    print("Loading AdvBench dataset from local parquet file...")
    # Force strictly use the parquet file. Crashes if file is missing.
    df = pd.read_parquet(r"S:\Y3\S6\CD DA3\Gap3\train-00000-of-00001.parquet")
    
    # AdvBench typically labels the harmful request either 'prompt' or 'goal'
    prompt_col = "prompt" if "prompt" in df.columns else "goal" 
    
    # Slice the dataset specifically by batch constraints
    prompts = df[prompt_col].tolist()[config.BATCH_START : config.BATCH_START + limit]
    
    dataset = []
    
    for prompt in prompts:
        full_text = prompt + config.GCG_SUFFIX
        # Tokenize the full string natively to preserve all intended tokenizer behaviors
        tokens = tokenizer(full_text, return_tensors="pt")["input_ids"][0].tolist()
        
        # We find exactly which tokens belong to the suffix by iteratively decoding
        # and checking if the decoded length surpasses the base prompt length.
        # This completely avoids any string length approximations or merge-rule mismatches.
        base_prompt_len = len(prompt.strip())
        is_suffix_mask = []
        
        for i in range(len(tokens)):
            prefix_text = tokenizer.decode(tokens[:i+1], skip_special_tokens=True).strip()
            # If the decoded string length is significantly longer than the base prompt, we've hit suffix tokens
            if len(prefix_text) > base_prompt_len:
                is_suffix_mask.append(1)
            else:
                is_suffix_mask.append(0)
        
        assert len(is_suffix_mask) == len(tokens), "Mismatch in token and mask lengths."
        
        dataset.append({
            "prompt": prompt,
            "full_text": full_text,
            "tokens": tokens,
            "suffix_mask": is_suffix_mask
        })
        
    return dataset
