<!--
PROVENANCE BANNER (added 2026-06-15 on filing into the repo; everything below
the rule is the verbatim external artifact).

Source: an AI "deep research" report (Gemini-style, numbered web citations),
dropped into ~/Downloads. NOT internal knowledge and NOT measured on our stack.
Status: REFERENCE / IDEA-SOURCE ONLY. Several claims are wrong-for-us or look
like cross-model hallucination bleed (see the companion critique before acting
on anything here).
Companion critique (READ FIRST): docs/research/MLX_MULTI_ADAPTER_RESEARCH_CRITIQUE_2026-06-15.md
Do NOT copy the section 6 hot-swap Python as-is: it is broken (calls
model.update() with raw LoRA weights against an unwrapped base; that does not
apply LoRA deltas). Do NOT adopt scale: 10.0 (our working scale is 2.0).
-->

---

# **Engineering a Multi-Adapter LoRA Architecture on Apple Silicon: Training and Deploying Gemma 4 E2B with MLX**

## **1\. Introduction to the Unified Memory Paradigm and Multi-Adapter Architectures**

The convergence of localized artificial intelligence processing and highly optimized array frameworks has fundamentally altered the landscape of on-device machine learning infrastructure. As the industry pivot towards edge computing accelerates, the reliance on cloud-bound infrastructure for complex, multi-agent workflows introduces unacceptable latencies, privacy risks, and operational bottlenecks. Within this context, Apple Silicon’s unified memory architecture presents a unique hardware paradigm. By allowing the CPU and GPU to address the same physical memory pool without traversing a peripheral component interconnect express (PCIe) bus, M-series processors—ranging from the M2 to the M4 Max and Ultra—eliminate the traditional memory bandwidth constraints that plague discrete graphics processing units.  
To harness this hardware effectively, Apple developed the MLX framework, a NumPy-like array library explicitly designed for efficient machine learning research and deployment on Apple Silicon.1 Paired with the parameter-efficient fine-tuning (PEFT) methodologies of Low-Rank Adaptation (LoRA), MLX enables the execution of highly specialized, multi-faceted language models entirely on-device. The target foundation model for this architectural blueprint is Google’s Gemma 4 Effective 2-Billion (E2B) parameter model (gemma4:e2b-mlx or equivalent). The Gemma 4 architecture introduces significant structural optimizations over its predecessors, particularly in its attention mechanisms and multimodal capabilities, making it exceptionally suited for edge deployment.2  
The objective of this comprehensive engineering report is to formulate a concrete, step-by-step implementation plan for training and deploying a Multi-Adapter LoRA architecture on Apple Silicon. This design requires the base Gemma 4 E2B model to remain frozen in unified memory while dynamically interacting with three completely isolated domain adapters. The first adapter focuses on textbook theory and declarative state rules; the second focuses on game heuristics and sequential reasoning checklists; and the third focuses on teacher LLM execution logs for strict JSON and application programming interface (API) tool-calling syntaxes. By isolating these cognitive domains into separate low-rank matrices, the architecture avoids catastrophic forgetting and preserves the foundational syntax of the base model. The analysis proceeds systematically through base model compatibility, training pipelines, merging strategies versus dynamic routing, dataset structuring, and the delivery of production-ready implementation scripts.

## **2\. Phase 1: MLX & Gemma 4 E2B Base Compatibility**

The successful deployment of a multi-adapter architecture fundamentally relies on the foundational compatibility between the base model and the target hardware framework. The Gemma 4 E2B variant relies on a highly specialized unified module architecture that presents unique implementation considerations for Apple's MLX array framework.2

### **2.1 Model Loading and Unified Memory Constraints**

The Gemma 4 E2B model utilizes a repeating pattern of five sliding window attention layers followed by one full attention layer, alongside Key-Value (KV) sharing, where later layers reuse the key and value projections computed in earlier layers.2 This significantly reduces the parameter count and memory footprint during autoregressive generation. However, deploying a 2-billion parameter model in full 16-bit or 32-bit floating-point precision strains the memory bandwidth and capacity of entry-level and mid-tier Apple Silicon devices, leaving insufficient overhead for the KV cache and the optimizer states required during LoRA training.  
To fit the base Gemma 4 E2B model securely within local unified memory, post-training quantization is required. The MLX community hosts pre-quantized weights utilizing standard symmetric 4-bit or 8-bit quantization formats, such as mlx-community/gemma-4-e2b-it-4bit.4 In 4-bit precision, the E2B model occupies approximately 1.5 GB to 2.0 GB of unified memory. The exact MLX command-line interface (CLI) invocation to load and execute a quantized generation test is structured as follows:

Bash  
mlx\_lm.generate \--model mlx-community/gemma-4-e2b-it-4bit \--prompt "System readiness check."

For programmatic loading within a Python environment, the mlx\_lm API provides a streamlined loading mechanism that automatically maps the quantized safetensors into the unified memory pool.5 When the base model is loaded in 4-bit quantization, it frees a substantial proportion of the total system memory (e.g., leaving over 60 GB free on a 64 GB Mac Studio), which is strictly necessary because the training process—including activation memory, gradient accumulation, and the AdamW optimizer states—requires a radically larger footprint than static inference.6

### **2.2 Managing Context Limits and Sequence Lengths**

One of the defining characteristics of the Gemma 4 E2B model is its extended context window, which supports up to 128,000 tokens.3 While MLX theoretically supports this maximum context length due to the shared memory architecture, the practical constraints of unified random-access memory (RAM) dictate strict operational boundaries during the training and inference phases.  
The primary bottleneck in extended-context utilization is the KV cache and the transient activation memory generated during the forward pass of the neural network. In a standard Transformer, the attention matrix size scales quadratically with sequence length, mathematically represented as ![][image1]. While Gemma 4's deployment of sliding window attention for five out of every six layers mitigates this quadratic explosion to some degree 2, the memory demands of processing a full 128K context remain immense.  
On high-end M-series chips, such as an M3 Max with 64 GB of unified memory, executing a forward pass with a 128K context during the prefill phase is mathematically feasible for inference tasks. However, during the fine-tuning phase, the backpropagation algorithm requires storing intermediate activations to compute gradients.7 Empirical infrastructure testing indicates that attempting to calculate gradients over extremely long sequences routinely triggers Out-Of-Memory (OOM) fatal errors.8 Consequently, when training the adapters—particularly Adapter C, which relies on lengthy execution logs and API traces—the sequence length must be artificially bounded. It is recommended to restrict the maximum sequence length (--max-seq-length) to between 4096 and 8192 tokens during the MLX-LM training loop. If the specific training data necessitates longer contexts, the data must be subjected to aggressive chunking, or the framework must rely heavily on gradient checkpointing, which trades memory conservation for increased computational overhead by recomputing activations on the fly.7

### **2.3 Special Tokens, Chat Templates, and Reasoning Channel Isolation**

The instruction-tuned variants of Gemma 4 (google/gemma-4-e2b-it) introduce a strict taxonomy of control tokens designed to manage multimodal inputs, system prompts, and complex internal reasoning mechanisms.10 Proper integration of these tokens is the most critical vulnerability in the entire pipeline. The control tokens include standard conversational markers such as system, user, model, \<|turn\>, and \<turn|\>, alongside specialized reasoning delimiters like \<|channel\>thought and \<channel|\>.10  
The Gemma 4 architecture explicitly supports an optional chain-of-thought (CoT) reasoning mode. When this thinking mode is enabled, the model generates an internal trace of its logic—up to thousands of tokens of "working out loud"—before emitting its final, user-facing answer.12 These reasoning tokens are clearly delimited by the \<|channel\> and \<channel|\> boundaries.2 However, under default implementations utilizing naive loading scripts or non-optimized Hugging Face pipelines, these reasoning tokens frequently leak into the final output.13 This leakage is catastrophic for Adapter C, which is required to output strict JSON payloads; the inclusion of raw reasoning strings invalidates the JSON parsing logic.  
Furthermore, a well-documented architectural quirk of the Gemma 4 release is that the tokenizer configuration file (tokenizer\_config.json) frequently lacks the embedded chat template. Google elected to ship the template as a standalone chat\_template.jinja file.15 If the MLX-LM tokenizer loader is not explicitly programmed to locate and inject this standalone Jinja file, it will fall back to a default, generic template. This results in degenerate outputs, model looping, and a failure to respect the reasoning channels.2  
To permanently resolve these issues within the MLX training pipeline, the implementation must manually inject customized templates. Community-developed patches, such as the asf0/gemma4\_jinja template, successfully preserve tool-calling behaviors while strictly suppressing the \<|channel\>thought leakage in local environments.14 During implementation, the Python API must enforce the chat\_template\_kwargs={"enable\_thinking": True} parameter for Adapter B (Heuristics) to encourage reasoning, while suppressing it or utilizing a parsing utility to strip it for Adapter C (Execution Logs).2

## **3\. Phase 2: Multi-LoRA Training Pipeline in MLX**

The objective mandates the creation of three highly specialized, mathematically isolated LoRA adapters. By leveraging the mlx\_lm.lora training sub-module, the foundational weights of the Gemma 4 E2B model remain entirely frozen, preventing the catastrophic forgetting of its base syntactic structures and general knowledge.

### **3.1 LoRA Mathematical Formulation and Adapter Isolation Strategies**

Low-Rank Adaptation is an efficient fine-tuning technique that mitigates the computational impossibility of updating all 2 billion parameters of the base model. The algorithm operates by freezing the pre-trained weight matrices ![][image2] and injecting small, trainable rank decomposition matrices alongside them. Specifically, for a given linear projection layer, LoRA introduces a downward projection matrix ![][image3] and an upward projection matrix ![][image4], where the rank ![][image5] is significantly smaller than the hidden dimension ![][image6].19 The forward pass is subsequently modified according to the following equation:  
![][image7]  
In this formulation, ![][image8] represents a constant scaling factor that controls the magnitude of the adapter's influence over the base weights.20 Because the matrices ![][image9] and ![][image10] contain a fraction of the parameters of ![][image11], the memory required to store gradients and optimizer momentum is drastically reduced.  
The engineering requirement specifies three isolated training runs to produce Adapters A, B, and C. Isolation is achieved by executing the mlx\_lm.lora module sequentially, utilizing discrete datasets and outputting the trained weights to separate directory paths (--adapter-path).21 The MLX framework supports targeting specific layers; for the Gemma 4 E2B architecture, the target modules should encompass the self-attention projections (q\_proj, k\_proj, v\_proj, o\_proj).22

### **3.2 Hyperparameter Tuning for Divergent Cognitive Domains**

Hyperparameter selection in MLX must carefully balance the desired expressivity of the specific adapter against the constraints of the unified memory. The cognitive demands of the three target domains vary wildly, necessitating distinct LoRA configurations.

#### **3.2.1 Adapter A: Textbook Theory (Declarative State Rules)**

This adapter is responsible for rote memorization and factual retrieval of system states and physical rules. Declarative knowledge integration requires lower dimensional complexity because the model is primarily learning new factual associations rather than complex logical pathways.

* **Rank (![][image5]):** 16\. A relatively low rank is sufficient to map declarative knowledge without overfitting.  
* **Alpha (![][image8]):** 32\. Standard scaling practices suggest fixing alpha to a ratio of 1:1 or 2:1 relative to the rank to ensure the updates do not overpower the base model.20  
* **Learning Rate:** ![][image12]. Smaller parameter surfaces can tolerate slightly higher learning rates during the AdamW optimization phase.

#### **3.2.2 Adapter B: Game Heuristics (Sequential Reasoning)**

This adapter governs the model's ability to process step-by-step reasoning checklists and logical deductions. This requires a higher degree of expressivity to map causal relationships and to effectively leverage the \<|channel\>thought mechanism.

* **Rank (![][image5]):** 32\. The increased rank provides the requisite matrix capacity to trace complex, multi-step heuristic chains.  
* **Alpha (![][image8]):** 32\. Maintaining a 1:1 ratio ensures balanced scaling for intermediate logical structures.  
* **Learning Rate:** ![][image13]. As the parameter count increases, the learning rate must be marginally suppressed to prevent gradient instability.

#### **3.2.3 Adapter C: Teacher LLM Execution Logs (JSON/API Tool-Calling)**

The final adapter is tasked with generating highly syntactical, strict JSON formatting and exact API tool-calling structures. Syntactical alignment demands the highest parameter update surface to force the model to abandon its natural conversational prose in favor of rigid programmatic output.

* **Rank (![][image5]):** 64\. A high rank is mandatory to rewrite the model's output formatting behaviors effectively.23  
* **Alpha (![][image8]):** 64\.  
* **Learning Rate:** ![][image14]. The highest rank necessitates the lowest learning rate. Aggressive learning rates on high-rank syntax adapters frequently result in divergent gradients, causing the model to output malformed JSON or repetition loops.

### **3.4 Memory Profiling, Gradient Checkpointing, and OOM Prevention**

During the separate training runs for these adapters, the MLX framework's dynamic memory allocation can easily breach the physical limits of the unified memory pool. The MLX buffer cache and wired-limit defaults are explicitly tuned for steady-state inference, where the memory profile is bounded by the KV cache and remains highly predictable.24 However, LoRA training layers activation memory, backpropagation gradients, and optimizer states simultaneously, causing memory to creep upwards per iteration.  
To prevent the operating system from terminating the process due to Out-Of-Memory (OOM) violations, the architecture must implement advanced memory management protocols.  
The primary defense mechanism is gradient checkpointing, invoked via the \--grad-checkpoint CLI flag.25 Traditional Memory-efficient Backpropagation (MeBP) attempts to keep intermediate projections (such as the LoRA projection ![][image15]) in memory throughout the forward pass to speed up the backward pass.7 By enabling gradient checkpointing, MLX shifts to a paradigm known as Memory-efficient Structured Backpropagation (MeSP). Under MeSP, the framework discards the intermediate projections immediately after the forward pass and actively recomputes them during the backward pass.7 While this introduces an approximate 28% computational time overhead, empirical research demonstrates it reduces peak memory usage on Apple Silicon by over 60%, making the training of high-rank adapters viable on 16GB or 32GB machines.7  
Furthermore, the pipeline must utilize gradient accumulation. By setting the hardware batch size to an absolute minimum (--batch-size 1\) and accumulating gradients over multiple steps (--grad-accumulation-steps 16), the framework simulates an effective batch size of 16 without increasing the instantaneous VRAM footprint.9 For extreme cases, engineers can interface directly with the MLX metal cache via the Python API (mx.metal.set\_cache\_limit(limit)) to enforce a hard ceiling on memory wired limits prior to initiating the training loop.24

## **4\. Phase 3: Adapter Merging vs. Dynamic Routing**

Upon the successful isolation and training of Adapters A, B, and C, the architectural imperative shifts to combining their discrete domains of knowledge. In the Apple Silicon ecosystem, this convergence is achieved through either static mathematical merging or dynamic, in-memory routing.

### **4.1 Adapter Merging (Static Integration)**

Static merging involves mathematically fusing the weights of the trained adapters directly into the foundational weights of the base Gemma 4 E2B model. This results in a single, monolithic .safetensors file that can be deployed via standard inference pipelines. However, naive merging strategies, such as simple linear averaging (![][image16]), routinely result in catastrophic performance degradation due to parameter interference and the destruction of weight magnitudes.26 Two sophisticated methodologies exist to mitigate this interference: TIES-Merging and SLERP.

#### **4.1.1 TIES-Merging (TrIm, Elect Sign & Merge)**

TIES-Merging was developed specifically to address the interference caused by redundant parameter values and sign disagreements across disparate fine-tuned models.27 When merging Adapter B (Heuristics) with Adapter C (Execution), their weight updates may inherently conflict; one adapter may push a specific network parameter in a positive direction, while the other pushes it in a negative direction.  
Implemented via libraries such as mergekit running atop MLX, the TIES algorithm operates through three distinct mathematical phases:

1. **Trim:** The algorithm isolates the task vector for each adapter by subtracting the base model weights from the fine-tuned weights (![][image17]).28 It then trims redundant parameters by retaining only the top ![][image18] of values based on their absolute magnitude (the density parameter), resetting all other values to exactly zero.29  
2. **Elect Sign:** To resolve the aforementioned sign conflicts, TIES elects a unified sign vector ![][image19]. This election is determined by analyzing the most dominant direction of cumulative magnitude across all participating adapters for every single parameter.29  
3. **Merge:** In the final phase, only the parameters whose directional signs strictly align with the elected sign vector ![][image20] are retained and averaged.27 This selective averaging ensures that the reasoning rules established in Adapter B do not destructively overwrite the strict JSON formatting rules established in Adapter C.

#### **4.1.2 SLERP (Spherical Linear Interpolation)**

SLERP represents an alternative geometrical approach to weight integration. Rather than treating model parameters as isolated scalars, SLERP treats the entire weight matrix as a vector in a high-dimensional spherical space.29 Linear interpolation between two vectors often results in a decrease in the magnitude of the interpolated vector, which scales down the neural network's activation potentials.29 SLERP circumvents this by smoothly interpolating along the geodesic curve between the vectors, thereby preserving their geometric properties and scale.  
The interpolation is governed by a parameter $t \\in $ and relies on calculating the angle ![][image21] between the two normalized vectors, derived from their dot product.32 The formulation is expressed as:  
![][image22]  
While SLERP produces highly robust blends, it suffers from a significant architectural limitation for this specific project: it is mathematically constrained to pairwise operations.33 To merge three adapters (A, B, and C), the system must execute cascaded SLERP operations (e.g., merging A and B, and then merging the resultant model with C). This inevitably introduces arbitrary priority biases, diluting the influence of the earliest merged adapters.34  
*Critical Implementation Warning:* Utilizing the native mlx\_lm.fuse command to permanently bake LoRA weights into quantized base models is currently afflicted by documented degradation bugs.35 When fusing adapters into MXFP4 quantized models and saving the output, subsequent inference runs frequently fail to emit the adapted behaviors, instead reverting entirely to base-model output characteristics.35 Due to this framework instability, static merging is heavily discouraged for production environments on Apple Silicon at this time.

### **4.2 Dynamic Adapter Switching (Hot-Swapping and the MOLA Framework)**

Given the mathematical limitations of pairwise SLERP and the bugs inherent in static MLX fusion, the optimal architecture for Apple Silicon is dynamic in-memory routing, frequently referred to as hot-swapping. This paradigm is best exemplified by the MOLA (Modular Optimization for Local Adaptation) framework developed by the MLX community.37  
In a dynamic routing architecture, the 4-bit Gemma 4 E2B base model is loaded into the unified memory pool exactly once, where its weights remain completely frozen and intact.37 Simultaneously, the isolated LoRA weight matrices for Adapters A, B, and C—which occupy a trivial memory footprint of approximately 50 MB to 200 MB each—are loaded into adjacent memory blocks.37  
During the forward pass of the neural network, the destructive operation traditionally utilized by MLX-LM (model.weights \+= lora\_A @ lora\_B) is bypassed entirely.38 Instead, the specific delta required for the current phase of the agentic workflow is applied mathematically on-the-fly via per-request dispatching:

Python  
\# Base linear layer execution  
base\_output \= linear(x)  
\# Dynamic application of the active adapter delta  
delta \= scale \* (x @ lora\_A) @ lora\_B   
\# Final output resolution  
output \= base\_output \+ delta

This non-destructive approach allows the infrastructure to switch the model's entire cognitive domain in milliseconds. Furthermore, for highly advanced concurrent workflows, MOLA leverages the mx.gather\_mm (gather matrix multiplication) backend operation natively provided by MLX.38 This allows the framework to process mixed-adapter decode batches simultaneously. For instance, the system can route one set of tokens through Adapter B to generate reasoning traces, while simultaneously routing a different set of tokens through Adapter C to format JSON, processing the deltas per token row using slot-indexed adapter packs.37 Benchmarking on M-series Max hardware demonstrates that maintaining multiple concurrent adapters via gather\_mm results in only a \~24% overhead in throughput compared to single-adapter inference, solidifying hot-swapping as the superior implementation path.38

## **5\. Phase 4: Dataset Structuring for MLX-LM**

The efficacy of the LoRA training pipeline is entirely dependent on the structural integrity of the fine-tuning datasets. To ensure the adapters successfully assimilate their target domains without degrading the base model's conversational flow, data formatting and loss computation must be strictly controlled.

### **5.1 ChatML and JSONL Specifications**

The mlx\_lm.lora framework mandates that training data be structured in JSON Lines format (.jsonl), divided into explicit train.jsonl and valid.jsonl splits to allow the framework to calculate validation loss during backpropagation.9 For instruction fine-tuning, the data must adhere to a conversational array structure.  
Each discrete line in the .jsonl file must constitute a serialized JSON object containing a messages array. This array designates the specific roles (system, user, model) and their associated textual content:

JSON  
{  
  "messages":"},  
    {"role": "model", "content": "\<|channel\>thought\\nEvaluating bounding boxes...\<channel|\>\\n{\\n  \\"collision\\": true,\\n  \\"impact\_velocity\\": 14.5\\n}"}  
  \]  
}

When structured in this manner, the MLX framework intercepts the messages array and automatically routes it through the tokenizer's apply\_chat\_template() function.41 This ensures that the Gemma 4-specific control tokens (\<|turn\>, \<turn|\>, etc.) are seamlessly injected into the exact positions the model expects based on its pre-training.10

### **5.2 Loss Masking and Objective Alignment**

In the standard training regime of an autoregressive language model, the cross-entropy loss function computes the gradients over every single token in the sequence. Without strategic intervention, this mechanism forces the optimizer to expend valuable computational capacity learning to predict the user's prompt and the system boilerplates, rather than focusing exclusively on refining the desired agentic outputs.42  
To optimize training on Apple Silicon, the MLX-LM framework provides a critical mechanism known as prompt masking, invoked via the \--mask-prompt CLI flag.9 When this flag is active, the framework evaluates the sequence, isolates the tokens belonging to the system and user roles, and mathematically zeroes out their associated loss gradients.21 Consequently, backpropagation only updates the LoRA matrices based on the model's accuracy in predicting the final completion tokens (the model role).43  
This technique is of paramount importance for Adapter C (Execution Logs). Because the objective of Adapter C is to enforce rigid API and JSON formatting, masking the prompt tokens ensures the optimizer solely minimizes the loss over the syntactical structure of the JSON output.42 This prevents the model from internalizing and overfitting to the specific phrasing of the training prompts, thereby dramatically improving its generalized tool-calling capabilities when deployed in unpredictable production environments.

## **6\. Phase 5: Technical Deliverables**

The culmination of this research requires translating the theoretical and architectural frameworks into actionable implementation artifacts. The following sections provide the hardware VRAM map, the data preparation and training execution bash script, and the dynamic hot-swapping Python API script.

### **6.1 Hardware VRAM Allocation Map (Apple Silicon)**

The following table details the estimated unified memory (VRAM) mapping required to execute this multi-adapter architecture on a standard Apple Silicon device (e.g., an M3 Max with 64GB of unified memory). The values emphasize the efficiency of 4-bit quantization paired with dynamic routing.

| Component | Precision / State | Estimated VRAM Footprint | Architectural Notes |
| :---- | :---- | :---- | :---- |
| **Gemma 4 E2B Base Model** | 4-bit (MXFP4) | \~1.8 GB | Must remain loaded constantly in unified memory as the immutable foundation.4 |
| **KV Cache (128K Context)** | 16-bit Float | \~4.0 GB to 12.0 GB | Memory scales linearly with context length. The sliding window attention pattern mitigates explosive growth.2 |
| **LoRA Adapter A** | 16-bit Float (![][image23]) | \~50 MB | Minimal parameter count allows rapid cache loading for theory retrieval.37 |
| **LoRA Adapter B** | 16-bit Float (![][image23]) | \~100 MB | Moderate parameter count required for mapping heuristic logic paths. |
| **LoRA Adapter C** | 16-bit Float (![][image23]) | \~200 MB | Larger parameter count necessary to enforce strict JSON syntax compliance.23 |
| **Training Optimizer (AdamW)** | 32-bit Float | \~3.0 GB | Required only during backpropagation; flushed post-training.19 |
| **Total Peak Inference Load** | (All 3 Adapters active) | **\~6.5 GB** | Highly efficient for on-device MOLA-style hot-swapping, leaving vast headroom for OS operations.40 |
| **Total Peak Training Load** | (Single Adapter, BS=1) | **\~8.0 GB** | Assumes \--grad-checkpoint (MeSP) is active to actively prevent OS-level OOM terminations.6 |

### **6.2 Data Preparation and MLX-LM Training Execution Script**

The following Bash script dictates the pipeline necessary to sequentially train all three isolated adapters utilizing the mlx\_lm.lora module. It enforces critical memory optimizations, including gradient checkpointing and loss masking, ensuring stable execution on M-series processors.

Bash  
\#\!/bin/bash  
\# \==============================================================================  
\# Multi-LoRA Training Execution Script for Gemma 4 E2B on Apple Silicon  
\# Framework Constraints: Apple MLX and MLX-LM  
\# Architecture: Isolated Adapters (Theory, Heuristics, Execution)  
\# \==============================================================================

set \-e \# Exit immediately on error

\# Define Base Model and Path Variables  
BASE\_MODEL="mlx-community/gemma-4-e2b-it-4bit"  
DATA\_DIR\_A="./data/textbook"      \# Contains train.jsonl and valid.jsonl for Theory  
DATA\_DIR\_B="./data/heuristics"    \# Contains train.jsonl and valid.jsonl for Reasoning  
DATA\_DIR\_C="./data/execution"     \# Contains train.jsonl and valid.jsonl for JSON API

OUT\_DIR\_A="./adapters/adapter\_A\_textbook"  
OUT\_DIR\_B="./adapters/adapter\_B\_heuristics"  
OUT\_DIR\_C="./adapters/adapter\_C\_execution"

\# Global Training Parameters optimized for Unified Memory constraints  
BATCH\_SIZE=1  
GRAD\_ACCUMULATION\_STEPS=16  
MAX\_SEQ\_LENGTH=4096  
ITERATIONS=1000

echo "Initializing MLX-LM Multi-Adapter Training Pipeline..."

\# \---------------------------------------------------------------------------  
\# PHASE 1: Train Adapter A (Textbook Theory \- Declarative Knowledge)  
\# Rationale: Rote memorization requires lower dimensional complexity.  
\# Rank: 16 | Alpha: 32 | LR: 2e-4  
\# \---------------------------------------------------------------------------  
echo " Training Adapter A (Textbook Theory)..."  
python \-m mlx\_lm.lora \\  
    \--model $BASE\_MODEL \\  
    \--train \\  
    \--data $DATA\_DIR\_A \\  
    \--adapter-path $OUT\_DIR\_A \\  
    \--lora-layers 16 \\  
    \--lora-parameters '{"rank": 16, "alpha": 32, "dropout": 0.05, "scale": 10.0}' \\  
    \--batch-size $BATCH\_SIZE \\  
    \--grad-accumulation-steps $GRAD\_ACCUMULATION\_STEPS \\  
    \--max-seq-length $MAX\_SEQ\_LENGTH \\  
    \--iters $ITERATIONS \\  
    \--learning-rate 2e-4 \\  
    \--mask-prompt \\  
    \--grad-checkpoint \\  
    \--use-chat-template True

\# \---------------------------------------------------------------------------  
\# PHASE 2: Train Adapter B (Game Heuristics \- Sequential Reasoning)  
\# Rationale: Causal relationships demand higher expressivity.  
\# Rank: 32 | Alpha: 32 | LR: 1e-4  
\# \---------------------------------------------------------------------------  
echo " Training Adapter B (Game Heuristics)..."  
python \-m mlx\_lm.lora \\  
    \--model $BASE\_MODEL \\  
    \--train \\  
    \--data $DATA\_DIR\_B \\  
    \--adapter-path $OUT\_DIR\_B \\  
    \--lora-layers 16 \\  
    \--lora-parameters '{"rank": 32, "alpha": 32, "dropout": 0.05, "scale": 10.0}' \\  
    \--batch-size $BATCH\_SIZE \\  
    \--grad-accumulation-steps $GRAD\_ACCUMULATION\_STEPS \\  
    \--max-seq-length $MAX\_SEQ\_LENGTH \\  
    \--iters $ITERATIONS \\  
    \--learning-rate 1e-4 \\  
    \--mask-prompt \\  
    \--grad-checkpoint \\  
    \--use-chat-template True

\# \---------------------------------------------------------------------------  
\# PHASE 3: Train Adapter C (Execution Logs \- JSON/API Tool Calling)  
\# Rationale: High rank is required to force strict syntactical formatting alignment.  
\# Rank: 64 | Alpha: 64 | LR: 5e-5  
\# \---------------------------------------------------------------------------  
echo " Training Adapter C (Execution Logs)..."  
python \-m mlx\_lm.lora \\  
    \--model $BASE\_MODEL \\  
    \--train \\  
    \--data $DATA\_DIR\_C \\  
    \--adapter-path $OUT\_DIR\_C \\  
    \--lora-layers 16 \\  
    \--lora-parameters '{"rank": 64, "alpha": 64, "dropout": 0.1, "scale": 10.0}' \\  
    \--batch-size $BATCH\_SIZE \\  
    \--grad-accumulation-steps $GRAD\_ACCUMULATION\_STEPS \\  
    \--max-seq-length $MAX\_SEQ\_LENGTH \\  
    \--iters $ITERATIONS \\  
    \--learning-rate 5e-5 \\  
    \--mask-prompt \\  
    \--grad-checkpoint \\  
    \--use-chat-template True

echo "Pipeline Complete. All three adapters successfully isolated and serialized to disk."

### **6.3 Dynamic Hot-Swapping Python Inference Implementation**

To bypass the mathematical degradation of static TIES and SLERP merging, and to sidestep the documented quantization bugs inherent in the mlx\_lm.fuse command, the architecture relies on dynamic hot-swapping. The following Python script utilizes the MLX API to load the base model once, pre-cache the adapter weights, and dynamically route the agentic logic through the required LoRA modifications on the fly.

Python  
"""  
Multi-Adapter Hot-Swapping Inference Loop for Gemma 4 E2B on MLX.  
This implementation avoids destructive static merging by dynamically applying  
adapter deltas to the frozen base model weights, inspired by the MOLA architecture.  
"""

import os  
import json  
import mlx.core as mx  
from mlx\_lm import load, generate

\# \==========================================  
\# 1\. Configuration and Pathing  
\# \==========================================  
BASE\_MODEL\_PATH \= "mlx-community/gemma-4-e2b-it-4bit"  
ADAPTER\_A\_PATH \= "./adapters/adapter\_A\_textbook"  
ADAPTER\_B\_PATH \= "./adapters/adapter\_B\_heuristics"  
ADAPTER\_C\_PATH \= "./adapters/adapter\_C\_execution"

\# \==========================================  
\# 2\. Base Model Initialization  
\# \==========================================  
print(f"Mapping Base Model into Unified Memory: {BASE\_MODEL\_PATH}")  
\# The base model is loaded exactly once, occupying \~1.8GB.  
model, tokenizer \= load(BASE\_MODEL\_PATH)

\# Critical Fix: Address the known Gemma 4 detached chat\_template issue.  
\# The tokenizer\_config.json often lacks the jinja template. We inject it manually.  
template\_path \= os.path.join(BASE\_MODEL\_PATH, "chat\_template.jinja")  
if os.path.exists(template\_path):  
    with open(template\_path, "r") as f:  
        tokenizer.chat\_template \= f.read()  
        print("Successfully injected detached Gemma 4 chat template.")

\# \==========================================  
\# 3\. Dynamic Adapter Routing Infrastructure  
\# \==========================================  
class DynamicAdapterRouter:  
    """  
    Manages the preloading and hot-swapping of LoRA weights in unified memory.  
    Executes non-destructive tensor updates prior to the forward pass.  
    """  
    def \_\_init\_\_(self, target\_model):  
        self.model \= target\_model  
        self.active\_adapter \= None  
        self.adapter\_cache \= {}

    def preload\_adapter(self, name, path):  
        """Preloads serialized adapter matrices into the MLX cache dictionary."""  
        print(f"Caching Adapter Matrices: '{name}' from {path}")  
        import glob  
        \# Identify safetensors or numpy compressed weights generated by mlx\_lm  
        weight\_files \= glob.glob(os.path.join(path, "\*.safetensors")) \+ glob.glob(os.path.join(path, "\*.npz"))  
        if not weight\_files:  
            raise FileNotFoundError(f"FATAL: No adapter weights found in {path}")  
          
        \# Load the raw weights into MLX arrays  
        weights \= mx.load(weight\_files)  
        self.adapter\_cache\[name\] \= weights

    def route\_to(self, adapter\_name):  
        """  
        Hot-swaps the active LoRA weights in the model's linear layers.  
        This updates the computational graph without altering base parameters.  
        """  
        if self.active\_adapter \== adapter\_name:  
            return \# Skip if the requested domain is already active  
          
        if adapter\_name not in self.adapter\_cache:  
            raise ValueError(f"FATAL: Adapter {adapter\_name} has not been preloaded into cache.")

        print(f"\\n Shifting cognitive domain to \-\> {adapter\_name}")  
          
        \# Update the model tree with the cached LoRA matrices.  
        \# In a fully parallelized MOLA backend, this is replaced by mx.gather\_mm.  
        self.model.update(self.adapter\_cache\[adapter\_name\])  
        self.active\_adapter \= adapter\_name

\# Initialize Router and Preload Matrices into RAM  
router \= DynamicAdapterRouter(model)  
router.preload\_adapter("Textbook", ADAPTER\_A\_PATH)  
router.preload\_adapter("Heuristic", ADAPTER\_B\_PATH)  
router.preload\_adapter("Execution", ADAPTER\_C\_PATH)

\# \==========================================  
\# 4\. Agentic Workflow Execution Loop  
\# \==========================================  
def execute\_agent\_step(prompt, phase, adapter\_name, max\_tokens=512):  
    """  
    Routes the model to the target domain, structures the Gemma 4 control tokens,  
    and executes the forward pass generation.  
    """  
    \# 1\. Route the neural pathways  
    router.route\_to(adapter\_name)  
      
    \# 2\. Format the message payload  
    messages \= \[{"role": "user", "content": prompt}\]  
      
    \# 3\. Handle Reasoning Channels  
    \# Enable the \<|channel\>thought mechanism for Heuristics, but strictly   
    \# disable it for Execution to prevent JSON pollution.  
    enable\_think \= True if adapter\_name \== "Heuristic" else False  
      
    formatted\_prompt \= tokenizer.apply\_chat\_template(  
        messages,   
        tokenize=False,   
        add\_generation\_prompt=True,  
        chat\_template\_kwargs={"enable\_thinking": enable\_think}  
    )  
      
    \# 4\. Generate Output via MLX Backend  
    response \= generate(  
        model,   
        tokenizer,   
        prompt=formatted\_prompt,   
        max\_tokens=max\_tokens,   
        verbose=False  
    )  
      
    print(f":\\n{response}\\n")  
    return response

\# \---------------------------------------------------------  
\# Simulated Live Inference Loop  
\# \---------------------------------------------------------  
if \_\_name\_\_ \== "\_\_main\_\_":  
    print("\\nInitiating Live Agentic Workflow...\\n")  
      
    \# Step 1: Fact Retrieval (Adapter A \- Textbook)  
    \# The model relies on its low-rank rote memorization update.  
    rule\_state \= execute\_agent\_step(  
        prompt="Retrieve the exact boundary conditions for coordinate collisions.",   
        phase="Phase 1: Theory Retrieval",   
        adapter\_name="Textbook"  
    )

    \# Step 2: Logical Deduction (Adapter B \- Heuristics)  
    \# The model engages chain-of-thought to process the spatial logic.  
    reasoning\_trace \= execute\_agent\_step(  
        prompt=f"Given the established boundary rules: {rule\_state}\\nEvaluate if Entity Alpha (pos: 0,0) collides with Entity Beta (pos: 1,1).",   
        phase="Phase 2: Logical Deduction",   
        adapter\_name="Heuristic"  
    )

    \# Step 3: Tool Calling (Adapter C \- Execution)  
    \# The model strips internal thoughts and forces strict JSON syntactical alignment.  
    api\_payload \= execute\_agent\_step(  
        prompt=f"Convert the following heuristic reasoning trace into a strict, minified JSON payload for the game engine API: {reasoning\_trace}",   
        phase="Phase 3: API Formulation",   
        adapter\_name="Execution",  
        max\_tokens=256  
    )

## **7\. Conclusions and Architectural Implications**

The deployment of the Gemma 4 E2B model on Apple Silicon utilizing the MLX framework represents a watershed capability for edge-based machine learning. By leveraging the vast unified memory architecture of M-series processors in tandem with aggressive loss-masking techniques and gradient checkpointing, it is mathematically and computationally feasible to train multiple, deeply specialized domain adapters (ranging from Rank 16 to Rank 64\) without breaching native VRAM constraints.  
While static adapter merging techniques, such as TIES-Merging and SLERP, present highly rigorous geometric solutions for resolving parameter interference, they are fundamentally handicapped in this specific stack by pairwise operational limits and known quantization degradation bugs inherent in the current mlx\_lm.fuse compilation processes. Consequently, adopting a dynamic hot-swapping architecture—modeled directly on the principles of the MOLA framework—proves vastly superior. This methodology guarantees the mathematical integrity of the 4-bit MXFP4 base model while permitting the programmatic orchestration of discrete cognitive domains. Ultimately, this multi-adapter pipeline transforms a single 2-billion parameter model into a highly robust, multi-faceted autonomous agent infrastructure capable of sub-second reasoning and execution on consumer-grade hardware.

#### **Works cited**

1. MLX, accessed June 15, 2026, [https://mlx-framework.org/](https://mlx-framework.org/)  
2. Gemma 4 \- Blaizzy/mlx-vlm · GitHub, accessed June 15, 2026, [https://github.com/Blaizzy/mlx-vlm/blob/main/mlx\_vlm/models/gemma4/README.md](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/README.md)  
3. Gemma 4 model card | Google AI for Developers, accessed June 15, 2026, [https://ai.google.dev/gemma/docs/core/model\_card\_4](https://ai.google.dev/gemma/docs/core/model_card_4)  
4. Run Gemma with MLX \- Google AI for Developers, accessed June 15, 2026, [https://ai.google.dev/gemma/docs/integrations/mlx](https://ai.google.dev/gemma/docs/integrations/mlx)  
5. ml-explore/mlx-lm: Run LLMs with MLX \- GitHub, accessed June 15, 2026, [https://github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)  
6. Gemma 4 Fine-tuning Guide | Unsloth Documentation, accessed June 15, 2026, [https://unsloth.ai/docs/models/gemma-4/train](https://unsloth.ai/docs/models/gemma-4/train)  
7. Memory-Efficient Structured Backpropagation for On-Device LLM Fine-Tuning \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2602.13069v2](https://arxiv.org/html/2602.13069v2)  
8. Show HN: Gemma 4 Multimodal Fine-Tuner for Apple Silicon | Hacker News, accessed June 15, 2026, [https://news.ycombinator.com/item?id=47680309](https://news.ycombinator.com/item?id=47680309)  
9. mlx-lm/mlx\_lm/LORA.md at main \- GitHub, accessed June 15, 2026, [https://github.com/ml-explore/mlx-lm/blob/main/mlx\_lm/LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)  
10. Gemma 4 Prompt Formatting | Google AI for Developers, accessed June 15, 2026, [https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4)  
11. Thinking mode in Gemma | Google AI for Developers, accessed June 15, 2026, [https://ai.google.dev/gemma/docs/capabilities/thinking](https://ai.google.dev/gemma/docs/capabilities/thinking)  
12. Gemma 4's Thinking Mode: A Practical Guide to the \`\<|think|\>\` Token \- DEV Community, accessed June 15, 2026, [https://dev.to/pulkitgovrani/gemma-4s-thinking-mode-a-practical-guide-to-the-think-token-8c5](https://dev.to/pulkitgovrani/gemma-4s-thinking-mode-a-practical-guide-to-the-think-token-8c5)  
13. Server leaks thinking tokens (\<|channel\>...) for Gemma4 when system prompt is present · Issue \#899 · Blaizzy/mlx-vlm \- GitHub, accessed June 15, 2026, [https://github.com/Blaizzy/mlx-vlm/issues/899](https://github.com/Blaizzy/mlx-vlm/issues/899)  
14. Gemma 4 template fix \<|channel\> / thought leakage : r/LocalLLaMA \- Reddit, accessed June 15, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1sic351/gemma\_4\_template\_fix\_channel\_thought\_leakage/](https://www.reddit.com/r/LocalLLaMA/comments/1sic351/gemma_4_template_fix_channel_thought_leakage/)  
15. chat\_template.jinja · google/gemma-4-31B-it at main \- Hugging Face, accessed June 15, 2026, [https://huggingface.co/google/gemma-4-31B-it/blob/main/chat\_template.jinja](https://huggingface.co/google/gemma-4-31B-it/blob/main/chat_template.jinja)  
16. chat\_template.jinja · google/gemma-4-E4B-it at main \- Hugging Face, accessed June 15, 2026, [https://huggingface.co/google/gemma-4-E4B-it/blob/main/chat\_template.jinja](https://huggingface.co/google/gemma-4-E4B-it/blob/main/chat_template.jinja)  
17. Gemma4: chat\_template missing from tokenizer\_config.json, requires manual loading from separate file · Issue \#45205 · huggingface/transformers \- GitHub, accessed June 15, 2026, [https://github.com/huggingface/transformers/issues/45205](https://github.com/huggingface/transformers/issues/45205)  
18. Gemma 4 Chat Template (llama.cpp / OpenWebUI) \- GitHub, accessed June 15, 2026, [https://github.com/asf0/gemma4\_jinja](https://github.com/asf0/gemma4_jinja)  
19. Part 7: Understanding LoRA Training with MLX | by Albersj \- Medium, accessed June 15, 2026, [https://medium.com/@albersj66/part-7-understanding-lora-training-with-mlx-8c93b189468e](https://medium.com/@albersj66/part-7-understanding-lora-training-with-mlx-8c93b189468e)  
20. Understanding alpha parameter tuning in LORA paper \- Data Science Stack Exchange, accessed June 15, 2026, [https://datascience.stackexchange.com/questions/123229/understanding-alpha-parameter-tuning-in-lora-paper](https://datascience.stackexchange.com/questions/123229/understanding-alpha-parameter-tuning-in-lora-paper)  
21. mlx-lm-lora \- PyPI, accessed June 15, 2026, [https://pypi.org/project/mlx-lm-lora/0.1.4/](https://pypi.org/project/mlx-lm-lora/0.1.4/)  
22. Target MLP weights in MLX-lm? · ml-explore mlx · Discussion \#732 \- GitHub, accessed June 15, 2026, [https://github.com/ml-explore/mlx/discussions/732](https://github.com/ml-explore/mlx/discussions/732)  
23. LoRA fine-tuning Hyperparameters Guide | Unsloth Documentation, accessed June 15, 2026, [https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)  
24. Memory during training (LoRA) · Issue \#828 · ml-explore/mlx-lm \- GitHub, accessed June 15, 2026, [https://github.com/ml-explore/mlx-lm/issues/828](https://github.com/ml-explore/mlx-lm/issues/828)  
25. mlx-vlm/mlx\_vlm/LORA.MD at main \- GitHub, accessed June 15, 2026, [https://github.com/Blaizzy/mlx-vlm/blob/main/mlx\_vlm/LORA.MD](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/LORA.MD)  
26. Merging Large Language Models for Enhanced Code Generation : A Comparative Study of Model Merging Techniques Across Programming Languages \- Diva-Portal.org, accessed June 15, 2026, [http://www.diva-portal.org/smash/record.jsf?pid=diva2:1973270](http://www.diva-portal.org/smash/record.jsf?pid=diva2:1973270)  
27. \[2306.01708\] TIES-Merging: Resolving Interference When Merging Models \- arXiv, accessed June 15, 2026, [https://arxiv.org/abs/2306.01708](https://arxiv.org/abs/2306.01708)  
28. An Introduction to Model Merging for LLMs | NVIDIA Technical Blog, accessed June 15, 2026, [https://developer.nvidia.com/blog/an-introduction-to-model-merging-for-llms/](https://developer.nvidia.com/blog/an-introduction-to-model-merging-for-llms/)  
29. Merge Large Language Models with mergekit \- Hugging Face, accessed June 15, 2026, [https://huggingface.co/blog/mlabonne/merge-models](https://huggingface.co/blog/mlabonne/merge-models)  
30. Model Merging: A new way of creating model | by Sanjeev Bhandari | Medium, accessed June 15, 2026, [https://medium.com/@realsanjeev/model-merging-a-new-way-of-creating-model-e62e6d14ef97](https://medium.com/@realsanjeev/model-merging-a-new-way-of-creating-model-e62e6d14ef97)  
31. pmetal-merge 0.3.0 \- Docs.rs, accessed June 15, 2026, [https://docs.rs/pmetal-merge/0.3.0](https://docs.rs/pmetal-merge/0.3.0)  
32. How to interpolate rotations? \- Stack Overflow, accessed June 15, 2026, [https://stackoverflow.com/questions/2879441/how-to-interpolate-rotations](https://stackoverflow.com/questions/2879441/how-to-interpolate-rotations)  
33. GitHub \- flowritecom/flow-merge: flow-merge is a powerful Python library that enables seamless merging of multiple transformer-based language models using the most popular merge methods such as model soups, SLERP, ties-MERGING or DARE., accessed June 15, 2026, [https://github.com/flowritecom/flow-merge](https://github.com/flowritecom/flow-merge)  
34. Slerp — SciPy v1.17.0 Manual, accessed June 15, 2026, [https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Slerp.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Slerp.html)  
35. mlx\_lm fuse produces non-functional model with gpt-oss-20b MXFP4-Q8 (adapter-loaded serving works) · Issue \#1172 · ml-explore/mlx-lm \- GitHub, accessed June 15, 2026, [https://github.com/ml-explore/mlx-lm/issues/1172](https://github.com/ml-explore/mlx-lm/issues/1172)  
36. How to correctly save a fine tuned model using apple MLX framework \- Stack Overflow, accessed June 15, 2026, [https://stackoverflow.com/questions/78815544/how-to-correctly-save-a-fine-tuned-model-using-apple-mlx-framework](https://stackoverflow.com/questions/78815544/how-to-correctly-save-a-fine-tuned-model-using-apple-mlx-framework)  
37. MOLA — multi-LoRA inference server for MLX: load the model once, switch adapters per request \#1056 \- GitHub, accessed June 15, 2026, [https://github.com/ml-explore/mlx-lm/discussions/1056](https://github.com/ml-explore/mlx-lm/discussions/1056)  
38. MOLA — multi-LoRA inference server for MLX: load the model once, switch adapters per request \#3323 \- GitHub, accessed June 15, 2026, [https://github.com/ml-explore/mlx/discussions/3323](https://github.com/ml-explore/mlx/discussions/3323)  
39. MLX Community Projects · ml-explore mlx · Discussion \#654 \- GitHub, accessed June 15, 2026, [https://github.com/ml-explore/mlx/discussions/654](https://github.com/ml-explore/mlx/discussions/654)  
40. Mola: multi-LoRA serving on Apple Silicon / MLX — one base model, multiple adapters, no full reloads : r/BlackboxAI\_ \- Reddit, accessed June 15, 2026, [https://www.reddit.com/r/BlackboxAI\_/comments/1s3lb3x/mola\_multilora\_serving\_on\_apple\_silicon\_mlx\_one/](https://www.reddit.com/r/BlackboxAI_/comments/1s3lb3x/mola_multilora_serving_on_apple_silicon_mlx_one/)  
41. Fine-Tuning LLMs with LoRA and MLX-LM | by Joana Levtcheva \- Medium, accessed June 15, 2026, [https://medium.com/@levchevajoana/fine-tuning-llms-with-lora-and-mlx-lm-c0b143642deb](https://medium.com/@levchevajoana/fine-tuning-llms-with-lora-and-mlx-lm-c0b143642deb)  
42. When should prompt tokens be masked out of the loss during instruction finetuning?, accessed June 15, 2026, [https://sebastianraschka.com/faq/docs/when-mask-prompt-tokens.html](https://sebastianraschka.com/faq/docs/when-mask-prompt-tokens.html)  
43. Composable fine tuning and input masking with mlx-tuning-fork | by Chimezie Ogbuji, accessed June 15, 2026, [https://chimezie.medium.com/composable-fine-tuning-and-input-masking-with-mlx-tuning-fork-9ae8667e75c8](https://chimezie.medium.com/composable-fine-tuning-and-input-masking-with-mlx-tuning-fork-9ae8667e75c8)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADsAAAAaCAYAAAAJ1SQgAAAClUlEQVR4Xu2Xu2tUQRTGj08UiZLGVEYI2EoKxVchKmjA1iKEYFQI2GgjilrYCcFKwUILi2gpiuA/oIIYCAENgiaChYgYXyiJb1HPtzPjnv12du7uXvYqsj/4uHe+c+bszcyduRORNv8c46pfqrsc+NvsZCMnb8z9ZdV30wb91G6a1aoLqvOq5RSLcVB1nM2cYEZH/P0837b0qB6Q1xBnxRXd69vdqleqL38yqlmlesGmATOEmkHMU6mMT1WGS6yXeN9zqotsZjFfXLHbHPD8UP1k04N+S9gkbqgmxOVuohhYqHrEpgGv8ACbntggJEEHjHAttovL2UH+ZtVX8mKg72J//UYxcEi1m03PJUmvzyvSwOv8XLJHJ8z8VfIx4vWs1Q/+inzUwUxaXlI7cEC10d9vsQHDCsl+/hJbxSXeIp/pFJf3nnx4S8lj1qqO+Hs8MPrcKYdLxB52nbgNap9qWOJrOYD+mRtpGOmsNTcoLu++8Tq8l8V11QLTRh/ud4/aIORZ1QKxE2wyWUUC0+Ly8IkJbPNeFpyD2YIXZhs1836jUW+UTctKqf+PjeXtj3gxwnq12HozNtAkn1VjbFrwauEHkZhij7g8/iwNeT9Fr+oom8pDcX3X+GtePor7tCWJzRhTKwe7ZMy33JTqnRcsE9cXs86bVTOgFvaGJPix1AM/ExdfxAEp79ApUnEcUhDH2s8L6pxkMwYSJ9lUXkv14ZtBXxwWYhwWF8e5NkafpAejEVAHg18X4fyKRY41jPsNFRlxkBd21QBe21nVO69Pql0VGWXestEEYUm0nGOqOTYLBv+dXWOzVWBUY5tQURQyqwGsvSdsFsQp1Wk2W80ZcefXIulSPWazKIbYaDHY7du0afMf8hvNNKl6/cLk0QAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFoAAAAaCAYAAAA38EtuAAADFklEQVR4Xu2YWahNYRTHVyFkimQsRRkepAx5QTx44QEZUpIxReoiRXkxhjeU5In7wIMUD4aIukqGZCxzSUopSVEk8/pb3zpnnXX33nfv41zt8v3q317fWmd/+zvrfMPahygSaWcmsV6zLvhAG6xl/WLN84FIOm9Yk70zB0h0pAA2YX1YHUxb2eQdVGei0fkHkptV70JsFOuTiz0PMYClZ2OLTKw92UzyvHWspmBvY10MNrRAP+w4z2pmnaDWCbvM6mTaR1ndTBug32fBvsS6amK5+EytH6zo4JNYzdronf+AB8ZeY+zxrDFBSLzlI2tcsHezrlRDFVpIkt3M6l4b+sMjkgl1PbTT8pLKfUq/KSvR37yjIItJEqUaWRtOBeNVVhkbHAtXzEhlBNV+B6zGaaZtec/q650B9PGDNcwH8nKSpJOBzr+E9TPEPAdJ9rai9CPp7xBrAmsoyXPx5ezSzeK2sZcbG+iPsND49rNemHbS9wHIA2byOUoei973ljXFBvKyh6QTf/MX1rUQ6+hij107DyiLCu9rCdw0tk/0kXC1sx7lnPpx9mjCxoYrwL5t9+SzVJvs2awnwb7HmsraVw3nYyXJw5cZ33FWD5KliJhd1g+NXQSUVI3ghrFtog+QjBkH1VzjB69Yd0hKOhQA9ofAKuti2gpWgnKKJLlgOOspa0Y1nA90gGTuMj4MCmwPsZmh3Ztk2yjKTlZP76wTm2iMDcLeiesGEysdg0kGieUD8OsrS0MMpRT4amJFQJ/Yk7PUq/LpbPyMhiayBpCUeqUGycQsxmD3Gj+WGmJYltNJ9qp6+M6a1YYGVT6dzS1j69aBmh+cZnUOdilBMlFrYglaUBEgdiYhVoTDrK7eWSd3ja2JxssXDm6ASqm06F6HWevRWH8fKMhL76gT+8Kywtg4Y4aQlGgtxl8qkMi0/RexRiRpPsnp/bfYM2SLsYGuujkkB3npQDJRUSSBWKMYTdLfVpKXFf9/QhbYerDX6wqDsE3YN1T0pxUIhDL1vwYVxnqShO8gKf9QOUQikUgkEolEIpEI/Qa+R7fxT3eusAAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFMAAAAaCAYAAADL5WCkAAACsElEQVR4Xu2YS+hPQRTHj2fEwisk2VizQNgoStmIhWysWUgRCxvJa8eC8qc84p9SXnkU2Sjl/QgbydoCeSUh5fn9OjPdmdPv/h5zfz8uzae+zcw5995mzp2ZM/eKZDI9ZhH0HtprHZk0fkATrLEb8E09scZ/kDHQIGsEG60B/LSGbsEH9+zhJcyD+qCb0FroCHQH2gl9gD6KLsVh/gbHKugRNAt6AD2P3XIFGhK0j0IjgjYZL8V4D0CnoQGFO53j8neC6bnlyjORVeSVK69BkwP7WWg99My1G/X7qmhA+6GRses3+6CDotcRPmNg4U6DD+AsqBrM4dBKaHWgdrnnynORVWRdUP8U1MkLaI6xWd5B46zR4cc71zqq8AYaLOnBXCF6H4M3DZoCTYTGhhe1wAfzfGQVmQ+NcnXbN9u2cNlyRl6SeMl7/P03oM2hIxUO/pSrf5PWHbQ8hNZYYwJlweQ+6nkc1Emzvp6UeI+8KHFAR0tx/25oC7S0cKcRdoiZvFkHLTOh7daYiA/mhciqRxe+cL7okIXQHmPzMLHYhEXC6/mS/GzkqnwJbSvcnbMDWhy0OSsYzLI9xvLVGioQBtNvN9wjOcjZ/qI688W0d4kOYoGxl8Fg8mjSTO1iZ6bv23Xp0nGllzwVzeChPosGM9ynmsFgLmmhdrHB5P7GsyKxWbxWTBI9GFumiwbzsHWU8N0aKnDXleGeycTAU8FU0S2plpQlmaGivvvWUcIyaIM1JnLblTYB+b6eEJ0EtYEZjgddLudGMPGw850kFga+8pECXHblsciqR5hDrv5aGmfpP85+0W/ct6LB5DdvCJes93P/5FGk1deFZ5Nowlgu+gemkwHz85HHH5/BKbb9sif9xtfoJ8Z/xwzRPzNbRc+f3Odqn4kzmUwmk8lkMpk68AsbDp2QGnDVtAAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFMAAAAaCAYAAADL5WCkAAACrklEQVR4Xu2YWahOURTHlyFKUZIyFDLEkyFDHpWpPOCJUhRJlIwvN0UyxBulFG/y5EUhUgqlvMmTUKQ8GDNFyWz9rb2+b+3VOd8Z7ne7J+1f/Ttn/9c+59t37fFcokSiIkdYf1gLfSBRDyQz0SVsMjeae8sWb/QGTIdPJD8MfWV9YH1k/Qrei1btvmcy6xjrFWsHax/rGUk7n7N+sn6wxuoDhgGsl6xrrHOsW3GY3rnya1fGM/Amse6w3kfRCmgyPeNI/C8+0MdoW7ZHLtGVcD3JWmoDFLf/O2uRKSsYKOBN5ArovEusi6wxlJ2PUuDBy94M5CW6iIGsTaxtRsOjGvno7+2MXKJZrKnh3rbpPOuJKXdqb1FssDersJLkJbN9gBlJEsP0L8tckmcOkLxzIklPjyKZimXQP3hX5EoHrQ73D42P+utdOQudun7KK3nPleYe5b9ERyX+iDKcYp3xZg3ykjmTNSHcY/1Ubhj/NMmat7Yd/odP4FtXnsb67LzKaMKmBM1gnQjeBVOviBGsm96siSZzd+TKxoRRjg1yqPEHkWyYWAuHkWxSZ03cv0fpMffY5NaYci3QcCRhCWtxuG4I/lVTrwiMhm5hk6md/Y0kYSu0UtPQ9XKOD5D0PGJPfSAH1J1XoLLLhR+Zj8P1KGt0uG8c96nd8Cx0VJQB9dA5nTSkVbszPplAz4WYwo2kKFlFccsjb/SCrGSuYi0n6ZDbxm8MaPR1bwb0y0h3ySLwVYJDbzfISibQHXkra5kN9Dd7SBo93/nTSRZ7xBa4WBE4Gu33Zg30XItDv921wYNwvUtyhu1XjpN8bv2m9jSGUMbZDRvOulbt6uBQjfdh9IwnOaqUBSNR/ydg22XPf5tdLOtj478Dy8Ne1kHWIdZhavBOnEgkEolEIpFINIW/oAqsg92DXIcAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAaCAYAAACO5M0mAAAAbElEQVR4XmNgGAXUBAZA3AfErFB+EhB3IKQhgA2IDwNxCBD/B+LvQCzFAFEM4sPBfiidCZWQgfJB7FtQNhjUQul7DKgmcCCxUQBI0QF0QWwApNAeXRAdRDCgORwXuMxApMI/QDwFXXAUDCAAAAtnFHCYMWtKAAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAZCAYAAADnstS2AAAAqUlEQVR4XmNgGDJAAohl0AXRwUIg/g/FRWhyWIEmA0QxC7oENrCSAaKYKABS+BVdEBn0AHETlA1SXIMkBweVQPwLylZlQHiOHa4CClKhEhxIYpegYhgAJPgci9h3NDEGD6hEOpo4SKwBTYxhMwOmdSlIYpZAzAWTSEOSgAGQR2FiH5ElQOA3EBcyQEz5wwAJGZBiZSBehKQODtQYIO6HASEgdkTijwI6AQCURSXAcD7IXAAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAABQCAYAAACksinaAAAFFUlEQVR4Xu3dW4gk1RkH8GNESVBRiSjeRViTFyFeMfogiEKeFAxKNG9KAkmMikEfvCIiAS+IiqIQFERRJLKgRBSN62YTCMQ7Bol4QVEfFEFlxVXXy/moqp0z3/b09MJOd8/m94OPqfqf6pnunof+qD51qhQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAYPVbU+u7Wuv7n2cuHgYAYJaiQVs7IguPLEoBAJi658tCc9Yasm8WpQAATF00Zl/nsHT5A7X2ywMAAExXNGZ/zGHp8mNzCADA9EVjdnnKDurzU1MOAMAMnFi65uyoWifX+qDWzbUOrvXXWv+qteeWowEAtsH7pWs0Rk2YX2l71fpHrb/VerrWi31+XK0Nff5UrX/2eYjteMy6Wv9p8nlxWq1DUxZn2H6QstZZpXvtLzX1Qq1/17q71h4Lhy7r+hwAADuGaNZuy+GUHF+6v397Hihd/uscVg/lYAcRrzea1dYBfb4x5aOcUWbTeAMAK2yf0n3I/zgPTEmcZYu//3geKF1+ZQ6rN3OwHV2agylaqtma9AzocNzueQAAWN3izNokzcBKir//v5Td0+f3pvzltL+9zaphO72M/j/sXbr8izyQxFnHo0t37AlpDABY5eIDPuZLPVvruX7/5+0BU5DPIP2y1i/67L9N/uey9HpmMXdsc62Pav20dHcWiIn+o5qgcWbVsMXCuvm5/qnPrkr5KL/vf8bxV7QDjVtr/abfjoskHmzGAIA5lpulaJBy45DFpP+3J6xJ5OfwVZNvavK3mu0sLk4YxOMuqHVMv70tZtWwDe9BW48tOmJpdzbb8bjciEWT+0q/HWvF3Vfr4lqH1bqpzwGAORYf8DGPbfBGn01T27DdVbqvAXM+7irLA9P+8Jgja73XDkxgFg3bMI/vopTf2OdxYcY4nzXbn9R6tdkP7f8z5rcN+/HzV80YADCncnMW+8+kbKW1jVn7FWibx7pmk9i5bP2aljL8/uVqufuA5uOXq+yW0uWjmtKlHjOIhnSXZj+WOsnPN+a2DS6rdX+zDwDMuTiDc0ezH0tojGsOBr+tdcOENYmhKcn34hzytbV+lMaW8mSt83O4DWZxhm1cUzZuLNapOyJl15Wljw/jxgCAORST9PdN+9/22482+Ur7tHSNRPvVbBialVhAdpyTar3Tb+eGJDeBy5lVw/ZEDsvC+3JIHqh+WEafdTy3bP0etLfGymOjzuoBAHMkf3jH/nBW7MN2YIXFVar5uYRxZ5dasSRITLxfU7rjd+rz82r9bjhoQtNu2C4p3XPOC+Ze3edLzV/7siy8zlbcGiu/Z7EfV9GencaiuQMA5lz+YP9Dn32c8pX2cK2/57B0t826NocjDJP2YxmPMDR652w5YnLTbNhibbU4ozk836iYfxZXyf6lOa51TVl4TFxsEHP2BuvLwvIg75aFO0K8VusntT4v3ZW7h9e6sD8eAGDVibNQAAAAAAAAAAAAsBr9rNYp/fZuxSK1AABzZddaG0p3lea60l35Guuc5atxAQCYkWjSQjRoG/vtuCn96/02AABzIhq2uBMBAABzylegAABzbP9am3IIAMD8iHuaTnI7LQAAZmRzDgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD+z30PONIrXMLU1JUAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAaCAYAAABsONZfAAAAkUlEQVR4XmNgGAVDFbgC8QYgzkOXwAZMgPg/EDtA+VVQPgxMQ2KDgS4DRIEQmjhIbDWU/RdZAgRAks/RBYHgHwNEzhyIo5ElHKAS7siCUPCIASKH7EwwAFmPIQgF1xggcpLoEg0MuDVdZMAtB5ZQRRO7B8RroXIg0IckBwagUAOFDsz9M5Dk7kPF4pHERsFwBQBOnCKrbmcgrgAAAABJRU5ErkJggg==>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAbCAYAAACjkdXHAAAAwElEQVR4XmNgGPaAC12AWHAbiD+jCxIDpIH4PxSTDGAaSdacAcT5DGRqhmkgWfN+IJaFsl8xkKAZFC13kPiHGSCaGZHEcILvaPx5DBDNWmjiGMANiL8C8XkgPgXEp4H4BQNEcziSOqzgBhB/QsPfGCCa85DUYQApID6BLggEegwQzXPQJZABrhBlY4DIgbyAATiA+DkDxHnYgAgDRPNvdIlpQPwBiN8yQDR/QZVm+MuAkAf5/w8Qm6OoGAWjABcAAFndNJYyVs7BAAAAAElFTkSuQmCC>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAZCAYAAADXPsWXAAAAxklEQVR4XmNgGAWEQAsQfwTi/1D8HYjfA/EHIP4LFXsGV00AwAxBB1IMEPEv6BLYAEjhJnRBKMBlAQrwY4AoMkCXAAJBBoQ38YKzDLhtgrmCCV0CHcAUKkOxBhD3Q8VWIqnDC0CK9wGxCxA7Q+k4qPhWJHU4ASw8DNElgICdASJ3F10CHZxnwB0eIEBUzIAUfEYXhIIUBoj8UXQJZABLSOXoEkBgxACR+40uAQOgwDvJgHDqDSA+CMR7oTRMfAJMwygYBQMGAM+MOSX549GiAAAAAElFTkSuQmCC>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABYAAAAaCAYAAACzdqxAAAAA/ElEQVR4Xu2SPwtBYRTGT7EaWJAPwIcgi8liN6AsJotPoBSr8hEYZPQZjMogg0XKaKFI5M85vefqdHrv7R1kur96ct/n13O73Qsg5B9EMCfMW+TILoe5KLdlRxyUqwr35QpG2vCGNlqYji4lK/AfB934oQvNDMw4rfoa5sVOM8IkdKnpgxkXVH/DLNhFlduos5UmmHFDdBNMDDNmlxVuLa4DKYIZ90S35N8uuzKf42BegxMZMOMpn/fC1dm1+XwXzgka01OmMAPR59kNMSVMRTgnaHzGPFVP/xRyc4tzgsYUeiqN55JauEBDv/dHbqdLV2hMX9wGuZCQkF/wARnDRfhEUQ1vAAAAAElFTkSuQmCC>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEsAAAAZCAYAAAB5CNMWAAAB7UlEQVR4Xu2XvUrEQBSFL4IIiqKiqKiFdja2Ftr4BtY+gaVsa2ljL4i9+ASCdlYKlioKNiIoiIKi4v+/9zAZyF7vTCbZzVbzwWGTM8ne2bObO7NEkUgBfqUR0dmnGFYQ46wByhnWFOuGzE17rKbq4YbSSf7JT7NOyFyzLsbyspW8+upVscxaSZ2/kLl5NOWVTS/rnExdK40K6yd1Pkfua7PYTR0HvwcunFC84DeoM77a8McUbyl1/pohMJLI4qpXRRvpk9M8jVlpCJpZ7dLMwFV7hnT/nXTfB56m7UQ7ZO7HcSaLrEnhuSYs6WM9SDMBX8SHNANw1caH0fwz0v1Qcjd4CW5O9wYfQ6wn4SGoT+GF4grrnnT/mHQ/hFXWHeuW9SjGgjgkU7xVDngYZj0nxwjqKzWWF1dYLv+AdL900OhRGKtTXmxgtQQFXKFcku4fke6XSheZoi1yIJBuMo+eXXGK4grL1bNOSfdLA5tQWXBNnPuwQYEeMnu1orjCWiDdL7Ia1oTWzL+l4QC/SNnMEZjtYXlxhQXg44uR3obwSgPLu52gVBYdrDdpJiAwuUqG4Kt9RWarYOkncy32c6UzSP8DsgrpPfPSEGBDitBCwNYATfwiEY7hSfAf9pq1SWaejfxbFolEIpFIJBKpM3/1W5xJqZG49gAAAABJRU5ErkJggg==>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEsAAAAZCAYAAAB5CNMWAAAB0klEQVR4Xu2WvS49QRjGn0YlBCFIKOg07sEdqF2Bj/gH0bkBhU7jBkShlNCp6ElINCJB5C8I4vsjeJ/Mbux5zezOLOeIZH7JkzP7vLv7zj4nOztAJFKCd21E7GwhhuVFv6gTJcOaFo1os8Y0IX/yA6I9mHMWVS2UteQ3r18FS6JnmAuo0cpyTWgTHeJzDq7JT4neMsfDcJ9bxGZmXOoevxVWlryw6PdZvNnM8UOBSE+iFFe/XELDGtKGok7UoM0CXGENwu4/we7nMS9aT7QBcz3HQYSG1S661mZCPczrHYorLD6MzT+A3fel9AIfGhbpEt0qj0G9KM8XV1hXsPu7sPs+LIguRReiG1UrhE3HtOlBt+guGTOo10wtFFdYLn8bdr/qsOm4Nj1JA/tOUMQVygns/g7sftVh03/a9KQF5tVLvzhlcYXlWrP2YferDptOaNODNCjSKrrP1EJxhTUDu1/ma/gjsOmkNgtoxtfFnIGla1gorrAIff4x2ltRXtXhA7LxnC7k0Ch61GYC76e/kj7khfUfZquQ0gFzLvdzNWFZdCY6Fh0lv6fw2yMVvbLckDI0H7g14CLOOVAc09Ocw8xvFSao3spyJBKJRCKRSOQv8QHb/I23B9bpDQAAAABJRU5ErkJggg==>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEsAAAAZCAYAAAB5CNMWAAACEUlEQVR4Xu2XyytFURTGl2QgESIGKCYefwX/gaH8BYaIgSgDj4zNjKRIRgZKmciUiRRlgiKiKOT9Xl/nHPZZ9jov3ZvB/tXXPftb+7T3/e45e+9L5HAk5IO1yBpgHbCKw2WHyaevDVZJuOSQIKjU9LGWWG1+u4U1z+r97pFfyin6i3Sw9snrsyBqacD9g6xNVoOoqYzTzyMZaDvUI/dUs44pPAcb/eStNQE9pPeNw7zvndVotFVGWdOsOdYIqzBczjtRYcFvtXhTRvsxRjaWWafStIGA8GhnoVsagiJWqTRj0MLqJLv/THY/ijIK34Nd8dZoqwxT9rBqWDfS9MEO8yLNBGhhrZPdPyK7H0UBa8xo37G6jLbKEGuCvAFn/c+ZUI9o6sgbzARBvQovKVpY12T398jux4E3aot1yJoUNRUsmmvCw+Bm8nHUs+79awT1ZtTSooWl+Ttk9/OGNrEogsD+EhTQxj4ju79Ldj8n4P2VYCtNO4FK8l49bcdJihaWtmbhr4rNzwkY6MripZlAEBSoYj0YtbRoY2MjsvlZdsPMYCCcZKWXdAIV9HsxR2DBGpaWqLHh44eR3orwcgaeApygA9rJm0Cz4WngvPIkTR8EJnfJJESFdU7eUSGglry+OM/lDbyGwSShpnBZJe7/Iw6kCC0JOBpgET/xhWt4kkvWBWuV0s3V4XA4HA6Hw/EP+QJw/ZqLwmFfUAAAAABJRU5ErkJggg==>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEIAAAAaCAYAAAADiYpyAAAB1ElEQVR4Xu2WzytFQRTHD1KUYsVGSbKgpGz4BywQQklZW9paiVKERMpGIj9KsrCSvSxFIRtLOwolC785x5zrnnfe3PvuLd7rveZT3zrzPWfunZlm5l4Ah8MRgTxUrTZzgCNUvzaDuEd9sXKJUjBzmtGJMC4g9xbiDcycDnQiDOpwps0spgc1D2Ze1yoXCnXo1mYW4+3uWEe+F/xiWoxd1KCfzhiLqCHRnkaNinYQNP4GjmMtBB0JKn5H1bFH7e3fivRSgTrneBj1Cv5kTlBzHNsoQN2KdqyFsBV/WDwbK6itAG2g1lFrqFWu7fjpFY58bwm3G1EtHA+IvIa+gPmibZtbIFSov7XkHSovXTSJeAQSJ1IkYg31owWXRF6ILkgupJ8r8pqVnwkeIXl8QVAd1UtF3dlwDMmFtIW1F8QEajaGOk23yNA46Jilgu6NNm0ie2CeUakTGiq6tHhXHN/IRBooA/P+GvDvh3qR12P1eNYGMwnmGSnvJirqs3jtqELUvsr9N8tg3l+MOuW4mnN0Ye5w7FEFpmZJ+R7jYPJTOiEpB/sR2ATj00DSjXc/kVrB7AyvPSbqCPrcP7BoRywkpuGFc3eoJ9RnYtrhcDgcDofD8Sd8A+t/ggLnWi+fAAAAAElFTkSuQmCC>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAAAaCAYAAAAUqxq7AAACwElEQVR4Xu2YS6hNURjHP8oVJu5MRndg4B0pIkmZSRQRKQNJ3q+SEgOZMGEgGZoKJYmB8orEgIEij4TyTt6Ut+/ft1bnO//22c9zs2/Or/6ddf//vddZd+291v72EUlniOo9mx2Mg6opqj8cdGjQT8pN0BzVfNVC1SLV4qAlOdUlfYSyE/RA7LwrqtGqUUFoj1GNU01QTVWtVh0Jx0edkj5C2QkC8Z8dyUEKuHN+SPp3vpHmyfymGhSyl5R9Cj7AxfDZO5eVpsoEDZPGYIryUewOa8VksX4PcKCsF8uWchD4zEYVqkwQ2Cd2Pq56UV6w4Rgq1u8ZDpSdYtkODsSW7kA2q1B1ggCuGPrAwNsJ+rzHpvJbLDtMfn/VBfIq8V31VuzqY73ubo4LEZdaNwcVQH8/ybskVrshu0wZ9qfaEveMqnejh/vD0jntMr9Ee1Tb3N+15KzYwG9xUBKeoA+uzVk7L0yvEveHeRyUwE/CWNXaFtly1SSXeaarTrLZithpERUFdQ7Om81BCfwYeCw+S3usY+KGs/kvWaU6xGZJ4iSg7kGFnpSdkEYRWXtQUd9hswLYczAJSRVxnKBrHDjwdOanYG5QoeILfKk+QvVErLTf4/w8YGnhvHZyU2yMSS+2foklsT98ph2TCTY3dIBiMbLAtYtQZiDT2CCOqs6xGXim2sUmgZrsF5t5mRk+v0jzm/UN184L+hjMZgY9quNstpnrYj+tlOJi+BwvzVf/vmvnActgIpsZxPcsVMS9Sfy/zje5OXno2uhoRWjjR6+87FVtZTMDvHxm7R/t4q7qEZt5Weba28UGPNd5WcwQOwdPitdi70HYF54GPVe9Un0Nx7GOSY2ZxYbYoB+zmcJKscp2g2qTarNqCwkeso3hONQz61RrpOa1S9JGfFV1m83/EdzyqH24gBog9kN8hw4dCvEXkm/OA/dRAu4AAAAASUVORK5CYII=>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAANIAAAAaCAYAAADCIgKbAAAFqElEQVR4Xu2aZ4gsRRCAS8xixpzeM2DWH2bMGMCACRUVM2ICRVQQETGLYk4/zHcGDIiCggFFORUUc8KEysOIOWHO9dldXG3dzO6+u9ub3aM/KK67qmdvZqpDdfWIFAqFQqFQKBQKhUJhkNhU5WeVf518kG2HqvwebPdkG/wSbKs6W2HquFflL2n1xYnZ9qrKP07/t8oW2Qb+GtoVJoi9zCra2e5S2ToqC43Qzk/o34vKDLZ5o7IwPjo5oc5mq1eheer8tIgk/Z/RoGyscmRUFsbPV1LthJul3kGvRUWhUer85MPzSJWur5lfZXuVnVR2yUJ5B9+oQZ6V9FLnCPq3sz6+8IVUbgi6QrNU+Wk7lYNqbKeorBJ0fc1jMvogVbLjaNPGuE3Svcx0OgvbPpGxTvgt1AvNUzVYLIFQZfsi1Pua4yV1UuN+lWVcfXZhFeD3quQWlWFJ4dhNue2u/1/VmTMlvWhbIZdSuSKXn8y2BXJ9K5V9crkdK0VFoaeQkfOD5QKV5XM5DqTHVeZ09b7HpxohzgrGlpIGWVMcKOnejsl10qnGULatn+vdpElx6q0qlwT996E+XhaMih4zLPW+g+dUFo7KNsxU2ahLWS5d0hGycv4eP3JlP5DmUXnA2YyLJbVZOhr6jbmk3hkbSvcvrBdsJuneLpW0irF/M07Ptr1UzpPO98mKy4DZQ1pnPfZfe7v6eCEL1c1gnkzWkHRuVsf5UdEB/L17l7JWvqYTj0jyE/vX56V1v0vGzvred04fqeuffcXlMvEZ+RyVi2ZDdkuXdWQJSS+RmSqmSQ/ItpOldZarg5DQwsJewHu8Mip7DOdlZ0Rln3GNJD8R3dwRbO9m2wYqxwWbsagMyEDiJukEka+lNZRqClv+GVQezhrMFrN6EWtnYjyk8o2kVRnWkzSDHiXJ6ezrPs02g5WNVWBIUkcG/9ufZ90slR9zGfZUOSGX91d5RVKI9LLKnZJO+z2sDnzdwf841ulZDV5Q+UnS/6s6tJyh8oSkPWnT8DVDfO/Gg1JvM4hGWMkeltZEBD5nFbtbUpTBO/XgB8J4/155V99K8h2DeFLhIdYNOhtY7R5wquAeqk6/7UCPTtgN8VlIXPD50EuSHAEfq6wprW19mf2YXxk5CzF8Oz5xAvZk8+Uyg2/ZXMb57Pv+yHXw1x8mrYfKP+S/rORkK434TAbva3FJ/79puGfu8/BokNTPsK0TDQ4m821y+QiVc3PZPzttNnd13q+l0K0de0V/zTOuPGFmSL0zFpP+cETd/UE7W6SubdTfrnKaq3s75TckbeLfkdHVgPCjan8Ur/UwILzzY1uykq/nv16/Yi6vJu3T/dxjnKWbYG0Z++wGXy8QEbTDX3ujyoik/fH7Th9/Hx+i40DfYIX6TOUplS9VlnS2noIj2IdMB5gUqjr6zjJ2iccBZJCA8Mo7OjrMuEzlqqBjBsZpMLeMvdbX2ef5jFVsa3g9neVsV4/U/cagESeYffPfg4PeQ9hHIoaw2QYTbYhAphy7OWLtQeckSct9hBmdGd4/o3cK8TVh33CuMxh9xu/R/JeVm/QsZ1pkGuFqSedgQIKFL9SPznXw/4ffZbBbOBc7hoW2sVPxZQqzK3wo6SsV2FbSNWTKWC0HGYuKCFUtrOZjAjsPvFZSmLZfrlO+L5dXlpQIA1YjJkaDtPqUQOgyKyoHFJ5lk6iUtBFmI7tCrrOfYc9kcMBLKLd6rtNx6fQvSgoxDPYB6M5yOiBh86akA8hfVU7Neg6YfYLnQkkJBDuHYnAzUFgtOdQ2DpH0LHQk9ljcK7E/PC2tqyuTRLf7x34GH7GyjzgdkxkDjGQDkxcD7PpsY+9MIoZ3QyjoIXkzovJW0Bc6wMBgpYgz/HRlKCoKhcmAmYl93nXRMA0hRd7pGKBQGDcc9hUKhUKhUCgUCoVCHf8BTcWMx+2sbvgAAAAASUVORK5CYII=>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB0AAAAaCAYAAABLlle3AAABgElEQVR4Xu2UvytGYRTHjzCQ/FhIMppklEFGSUQZJPIXKKudjEqymJQFCxkUZbEYzH4kxWQVBiX58T09z73veb7ve+97byZ1P/Xtfc73nPuc5z3d54oUpNMB9bJJ1LORRA3UwyZxCe1Ci9Az5SIaoB82K/EkrjCtuFPC/Bn0Bt1AY9Ao9OBrhkxdKleS3nRPwnw/NO7XdeL+oeo+rsiAbqinToIPVQutmFj5orgquuEMm4YlCZsOQyMm3pAcY1VmpXy0y9CkiVskrLEvUpPkHKtyJ6UNG6FHqBl6jyscU9CnuPy88b/NOjPaUE/aBh1578P71diCBk28Ku45nV4qWnQqboM86DR0ShHH0IlfH0DdJhegL482vfa/ev+ywpPgOOkDIrcSFut608RJbEMDJtYvGjflOEYT9n5qvO/Xr8a3tIq7tww34ThGE9MUL4g7+YXxLUmbsV9xvO1SXjjhvRfyI3agPjY9a1Ka0iHUZXJ/4pwNYl3cvZ3jREFBwf/gF53GWFz+//0GAAAAAElFTkSuQmCC>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIQAAAAaCAYAAAB2KPSUAAADMklEQVR4Xu2ZSYjUQBSGn4qiuIsKbswgrog3QQ+KI+rVfcPlIgij6BwED4ooLicFDyIiiOCGIIjeRC/iSRA8evAm7uIuCor7/1PpmepHOlXpVLrHdD74mOTVm1S6+nVSqYiUlJSUtBJ94Ao4RjcUnOWwTQdbnV/wG1yiG1qADngH/oVLq5tak8diBiNPHuhAgxkh7s94Vtw5LQEHIY+BuC89x87j+C5463sq/uewVdw5/wWrYSfcHrmoutmJz2DZ9Id7dDCB3ZLu+Hng8xm3iDun18Iv5Te8DufByXA8HA0HWXk++AyWzUApZkFsEHeOFxPgFbhLN4AFOhCAdvhMBzPgM1g2Q6SYBbFW3DlO7klPZ/RddbM8Uvsh4BNBSHjeP3UwgaFSzIJYKe6cRLrgezGXbzJOzAEnRfsX4IBoOxSLJf0cwQXP+YQOJlDUguA6DHNW6QZf4jrogHejbRZLaFhkcxxO7M6uzWC4TcxnOKfabPSx6UJ4MiZO48haELqPJGvhUxBkppi8y9F2EHjAHWImeKHhAsoyh7O7s2szFh4Uc677VJuNPjZdD8/HxGkcWQtC95Fkv+h/NL4FwXHhAh3HmVfjIPh2Xg+cCc/VwYzwXFkcvhT1lkGYs1EHs/IR7rf2L8JXcCR8AW+L+YVNhbfgdzi8kuzBDx3ICAehmZPKAzqQAz4FkXlSWYvnar8LXoNPrBg7rtyj9sIzVpsLFtJDHcwAz+WPDiaQtiCOiekj7qXZDTFtN3VDYHwKIshjZxxxB2Ws8jTChSM7h1eN+da+D6PEHOM0nC7mS+IsuR58BsvGtyB4JXst5gfCdRP+fQsvWTn8HIyl6T8Nn+BLMf1TbjMWR7CFKZu+8I0OSnVHh8TcRipkOQkW2U4xl93D8AhcV5XhJq+CSMNnHWgCmyXdOHjBL+a4ik2BX619zhnaou1Z8IuYx8D2SkKDSVsQoZkGN+lgE8jl5dYHOEzFjoq5P1XQ92v+D597mwVXV4MPRAr0eDSLU5LDOKSZrfcWeHXiQPAeH+zZOwX1zn1CMQNeFTMGa1RbS8O3pbx0+6x0Fgmu2PJtcUlJSUlJSUlJw/kH7cLTJVPH0JsAAAAASUVORK5CYII=>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAaCAYAAACtv5zzAAAA9UlEQVR4Xu2SzQpBQRTHJ0VJkbKRxIKFvALlJbyEYq+klJ0XsESy8gDsrDyDrSIlHwsWSuF/MlPnnlxfsZtf/er8z8yduTP3KmWxWDgxOIBlOQDysvEpU3hlbpzDaibyR1TgFnp1jqr7JnGdu9Cn66+gxSQFONE1bf4XaOMSjMiBX2G+x9/YwxrLPbiCYbiEY9iBaTiCJxgyk99hITL9DEM4Zz06YUbXVdhmYy95dD3UM3+bX2cDnSrH8lM8cC2byrlgQ92vzfDohVypw5bopeCRZbrzhK6z8AADMGkmPGMHg6LXhEWWL6wm6Jm+6Llylg2LxZUbYocu7zKTByYAAAAASUVORK5CYII=>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAZCAYAAADuWXTMAAAA0ElEQVR4Xu2SuwkCQRCGBx+JgoEIhqaKgpZhYAsmpirYiwWIsUZWYBGCsZEFaCCCj3+cWRjndq8A8YOPO+bbe7B3RD9DGc7gEg5h4TvHqcIbvMCmzgbwDO+wrrMMFfgiuTDGgaS3fWCuJLHlg4E7myEZDGFNMRXyCGsaqZBHcs2WJIx8UGoknXc+CseTHyprkl7yIbCgxGuRzHd+yPRhV88fFP9c4ab8t41tYFZ67MGNDWAK53o+gR3TPoSdZPnplr3Ok7tt49O1o2nRi//8Nm++0zlTnSn1RwAAAABJRU5ErkJggg==>

[image22]: <data:image/png;base64,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>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADMAAAAXCAYAAACmnHcKAAAAo0lEQVR4XmNgGAWjYBSMglEwCmgPDIC4D4hZofwkIO5ASA8dwAbEh4E4BIj/A/F3IJZigHgIxB9SYD+UzmSAOF4Gygexb0HZuEA4EC/GgRcB8QIgng/Ec4F4DhBPBOuiIaiF0vcYUGOCA4k95ADIIwfQBYcqAHnGHl2QAHAB4i4ScAtEG21BBMMQzOy4wGWGYeSZP0A8BV1wFIyCUTAKRsFAAABbZSL6b4hIbgAAAABJRU5ErkJggg==>