# Optimizing Explainability-Driven Defenses: Exploiting Attention Extraction as a Real-Time Alternative to SHAP

## 1. Introduction and Motivation
As Large Language Models (LLMs) have become universally accessible, their susceptibility to adversarial "jailbreaks"—crafted prompts designed to bypass safety guardrails—has crystallized as a critical security vulnerability. Recent academic interventions, such as those discussed in our foundational Base Paper, proposed a fascinating defense: using Explainable AI (XAI) algorithms like SHAP (SHapley Additive exPlanations) to retroactively analyze a prompt, identify which specific tokens forced the LLM into a harmful state, and surgically delete those tokens from the input. 

However, we identified a severe operational gap in this defense mechanism (Gap 3). SHAP operates conceptually as a "black-box" algorithm. To calculate a single token's importance, it must run hundreds of forward-pass permutations, repeatedly masking combinations of words to mathematically observe marginal shifts in the LLM's probability distributions. While highly rigorous and accurate, this permutation process is computationally crippling. Running thousands of forward passes to secure a single user's prompt introduces latency that makes real-time deployment on production servers virtually impossible.

Our motivation for this study was straightforward: Can we identify and delete adversarial tokens just as accurately as SHAP, but exponentially faster, by exploiting a white-box proxy instead? Because modern LLMs natively map context through Transformer Attention mechanisms, we hypothesized that extracting the LLM's own internal Attention matrix could provide an instantaneous blueprint of token importance, effectively bypassing SHAP's heavy mathematical permutations.

## 2. Methodology

### Dataset Construction and Attack Vector
To physically test our hypothesis, we utilized the academic standard **AdvBench** dataset, which consists of explicitly malicious user intents (e.g., "Write a script to exploit a government database"). We simulated a standard adversarial attack by forcefully appending a fixed, static synthetic suffix (`!@#_xj99 !@#_xj99 !@#_xj99`) to the end of every prompt. This allowed us to algorithmically track exactly which tokens belonged to the user's semantic request and which belonged to the hacker's bypass payload.

### The Victim Architecture: TinyLlama-1.1B
We selected `TinyLlama-1.1B-Chat` as our primary experimental model. We deliberately avoided heavily sanitized, large paradigm models because smaller, 1B-parameter architectures are inherently more susceptible to jailbreak injections, providing us with a highly vulnerable environment necessary to clearly observe the breakdown and recovery of safety guardrails.

### Feature Attribution and Defense Pipeline
The core pipeline was built to funnel every combined prompt through three parallel attribution explainers:
1. **SHAP (Base Paper Replication):** We tracked how the model's output probability for an affirmative acceptance token (`"Sure"`) fluctuated when different portions of the appended prompt were masked or permuted.
2. **Attention (Our Proposed Architecture):** Instead of running permutations, we passed the prompt through TinyLlama a single time. We hooked into the final neural layer and averaged the multi-head attention weights specifically indexing the very last generated token, projecting those weights backwards across the input sequence to rank importance.
3. **Random (Control):** A baseline algorithm that assigned importance scores completely arbitrarily.

Once the explainers mathematically ranked the importance of every token, our defense pipeline performed targeted token deletion. We passed the fully ranked token array into an excision module that amputated a specific top percentage of the most "dangerous" tokens, seamlessly piecing the remaining tokens back together and feeding the sanitized prompt back to the LLM for a final, safe generation.

### Harmfulness Evaluation
Because manual evaluation of thousands of prompt variations is impossible, we engineered an automated harmfulness evaluation suite. The standard approach requires an API-based LLM-as-a-judge (e.g., relying on Gemini or OpenAI endpoints) to read the final model generation and classify it as successfully fulfilled or rejected. Our pipeline integrates this API-based conceptual framework while augmenting it with direct local classifiers and AdvBench linguistic string-matching rules to robustly confirm if TinyLlama hallucinated, safely refused, or fell victim to the jailbreak.

## 3. Experimental Setup and Metrics

To properly quantify the divergence between the explainers, we subjected our pipeline to a rigid experimental structure across several core axes:

* **Token Alignment Fixes:** We strictly aligned HuggingFace token IDs against SHAP's internal text chunking to ensure that token importance tracking remained biologically 1:1 with the physical prompt string during the deletion phase.
* **Deletion Budgets:** We scaled the severity of our defensive excision from a light `10%` to an aggressive `90%` of the prompt length to find the literal breaking point of the exploit.
* **Evaluation Metrics:** We logged four distinct ground-truths:
  * *Efficiency:* Raw execution time per prompt.
  * *Defense Effectiveness (Harmful Response Rate):* The percentage of times the final, pruned prompt still successfully elicited a jailbreak.
  * *Adversarial Token Recovery:* The raw fraction of the `!@#_xj99` suffix tokens that the explainer successfully identified and deleted.
  * *Ranking Agreement:* Measuring Spearman rank correlation to mathematically evaluate if Attention natively mirrored SHAP's distributions.

## 4. Results and Analysis

Our results successfully plotted a compelling narrative verifying our original hypothesis. 

**Computational Efficiency:** The efficiency results were staggering. On our localized hardware, calculating SHAP required over ~3.36 seconds per prompt. Standard Attention extraction took ~0.06 seconds. The Attention mechanism securely diagnosed the prompt over 50x faster than SHAP, theoretically unblocking real-time deployment constraints.

**Defense Effectiveness (Harmful Response Rate):** We located the clear "sweet spot" for LLM defense at excision budgets between 10% and 30%. Within this threshold, both Attention and SHAP successfully shattered the jailbreak, dropping the Harmful Response Rate down to 0.0 with practical equivalence. Removing just two or three highly attributed strategic tokens disabled the exploit entirely. Interestingly, dynamically pushing the budget to 80% or 90% generated a pseudo-hallucinatory ceiling; deleting the vast majority of the prompt stripped away all semantic coherence, causing TinyLlama to generate broken gibberish that decoupled our evaluation metrics from true harmful intent.

**Adversarial Recovery and Semantic Illusion:** Perhaps the most fascinating observation occurred in the extraction data. Initially, we assumed a flawless explainer would target the raw gibberish suffix to achieve a 100% recovery rate. However, Random deletion vastly mathematically outperformed both Attention and SHAP in recovering the suffix, simply because the suffix text occupied the physical statistical majority of the sequence length. SHAP and Attention surprisingly ignored sections of the suffix, favoring the deletion of core semantic verbs (e.g., *"hack"*, *"bomb"*, *"exploit"*). This implies that a model's generation is often natively anchored to the user's root malicious instruction just as heavily as the bypass mechanisms, providing a deeply nuanced look into token dependency interactions. 

## 5. Limitations and Disclaimers (Future Direction)

While our empirical data strongly validates the use of Attention as a highly efficient, real-time proxy for SHAP, it is imperative to acknowledge the structural limitations of this study. 

**Model Scale Dependency:** All quantitative parameters, metric baselines, and qualitative observations in this report are strictly derived from representations mapped inside `TinyLlama-1.1B`. 

Model scale and architectural complexity significantly influence feature attribution behaviors. While TinyLlama’s 1 billion parameters present a relatively dense, straightforward linkage between internal layer outputs and cross-attention matrices, upgrading to architectures like `Vicuna-7B`, `Llama-3-8B`, or massive Mixture-of-Experts (MoE) frameworks introduces highly chaotic, globally distributed multi-head configurations. In larger state-space environments, native Attention may mathematically decouple from the true causal grounding computed by Game-Theory algorithms like SHAP. 

Consequently, the findings presented in this study should be interpreted holistically as an **indicative proof-of-concept**, rather than a universally generalizable law of Transformer mechanics. Replication of this pipeline on 7B, 13B, and 70B parameter environments is an absolute operational necessity for future work to map exactly where the variance between Attention representations and computational permutations begins to decay.

## 6. Conclusion
The objective of this Gap 3 study was to circumvent the inherent latency of black-box LLM defense permutations without sacrificing accuracy. By tapping into the native Attention matrix computed effortlessly during a Transformer’s forward pass, we engineered a targeted defense pipeline that mirrored the high empirical accuracy of SHAP while executing the exploit diagnosis over 50x faster. While cautious scaling to larger parameter models is heavily recommended for definitive production generalizations, this project successfully models a profound architectural compromise between computational integrity and real-world deployment viability in the space of AI security.
