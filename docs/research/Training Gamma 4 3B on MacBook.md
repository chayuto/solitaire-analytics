# **Exhaustive Research Report: Fine-Tuning the Gemma 4 "\~3B" Architecture on Apple Silicon (MacBook M5 16GB)**

## **1\. Executive Summary**

The proliferation of open-weights foundational models has catalyzed a shift toward localized, on-device artificial intelligence, enabling developers to deploy frontier-level reasoning capabilities without reliance on cloud infrastructure. The release of the Gemma 4 family by Google DeepMind represents a structural evolution in this domain, introducing highly optimized parameter-efficiency mechanisms, native multimodality, and configurable reasoning states designed specifically for edge hardware. For researchers and engineers constrained by the memory limits of consumer hardware—specifically targeting the Apple MacBook M5 equipped with 16GB of unified memory—the process of fine-tuning models in the \~3-billion-parameter class demands a rigorous harmonization of algorithmic theory and hardware-specific memory management.  
This comprehensive technical report conducts a deep, investigative analysis into the fine-tuning pipelines for the Gemma 4 architecture on the M5 16GB platform. An immediate complication addressed herein is that the Gemma 4 family does not contain a standard, dense 3-billion-parameter (3B) model; rather, the "3B" classification is bracketed by specialized edge variants: the Gemma 4 E2B (2.3B effective parameters) and the Gemma 4 E4B (4.5B effective parameters), alongside a Mixture-of-Experts (MoE) variant, the 26B-A4B (3.8B active parameters).  
By synthesizing official Google training guidelines with bleeding-edge community frameworks—principally Apple's MLX array framework, the mlx-tune compatibility layer, and specialized toolkits like gemma-tuner-multimodal—this report delineates an exhaustive, end-to-end methodology. The analysis dissects the architectural nuances of the Gemma 4 edge models, evaluates the physical and computational constraints of the M5's unified memory architecture, resolves the severe framework discrepancies between PyTorch MPS and MLX, and provides a definitive walk-through for navigating Quantized Low-Rank Adaptation (QLoRA) without triggering catastrophic out-of-memory kernel panics.

## **2\. Architectural Deep Dive: Deconstructing the Gemma 4 "\~3B" Class**

To successfully optimize a training run on hardware with strict memory bounds, one must first deconstruct the target architecture. The user query specifies a "Gemma 4 3B" target. However, a review of the official Google DeepMind technical specifications reveals that the Gemma 4 lineage abandons standard dense scaling in this parameter bracket. Instead, the models that approximate the 3B capability class are structurally bifurcated into parameter-efficient edge models and sparse expert models.

### **2.1 The "Effective" Parameter Paradigm: E2B and E4B Architectures**

The nomenclature of the Gemma 4 edge models—E2B and E4B—utilizes the "E" prefix to denote "effective" parameters. This distinction is of paramount importance for memory budgeting during localized fine-tuning, as the effective parameter count dictates compute requirements, while the total parameter count dictates the physical memory footprint.  
The Gemma 4 E2B model operates with 2.3 billion effective parameters during any single forward pass, but its total parameter mass residing in memory is 5.1 billion. Similarly, the E4B variant utilizes 4.5 billion effective parameters while holding 8 billion total parameters in storage. This massive discrepancy between active computational weights and static memory weights is driven by a novel architectural advancement introduced in the Gemma 4 family: Per-Layer Embeddings (PLE).

### **2.2 The Mechanics and Implications of Per-Layer Embeddings (PLE)**

In standard transformer architectures, token IDs are mapped to dense vectors via a single embedding table located at the input layer. The resulting sequence of embeddings is then processed sequentially through the transformer's hidden layers. The Gemma 4 E2B and E4B models fundamentally alter this information flow.  
Per-Layer Embeddings act as a secondary, highly distributed embedding matrix that feeds a small, token-specific residual signal directly into every single decoder layer throughout the depth of the network. Mathematically, if h\_{l-1} represents the hidden state output of the preceding layer, the input to the subsequent layer l incorporates a layer-specific embedding vector E\_l(x\_{ids}) corresponding to the original input token IDs:  
The implications for fine-tuning are profound:

1. **Memory Saturation vs. Compute Efficiency:** The embedding tables required for PLE are enormous. In the case of the E2B model, the matrices holding these Per-Layer Embeddings account for more than half of all model parameters (roughly 2.8 billion of the 5.1 billion total). However, these parameters are only used for O(1) index lookups rather than dense, computationally expensive matrix multiplications. Therefore, while the model demands a high capacity of RAM to load, the floating-point operations per second (FLOPs) required during the forward and backward passes remain consistent with a much smaller 2.3B model.  
2. **Semantic Refinement:** From a qualitative perspective, PLE prevents information degradation in deep networks. Each layer receives its own dedicated channel to access token-specific information precisely when it becomes relevant to that layer's specific feature extraction focus, rather than forcing the initial embedding layer to pack all conceivable semantic contexts into a single upfront vector. During fine-tuning, updating these layer-based embedding vectors alongside the transformer blocks allows the model to rapidly acquire specialized, domain-specific knowledge.

### **2.3 Context Management: Dual RoPE and Shared KV Caches**

To support a massive context window of 128,000 tokens on highly constrained edge devices, the E2B and E4B models incorporate dual Rotary Position Embedding (RoPE) configurations. The architecture utilizes a standard RoPE application for local, sliding-window attention layers (which attend to the most recent 512 tokens), and a pruned RoPE configuration for global attention layers.  
Furthermore, the architecture employs a Shared Key-Value (KV) Cache mechanism. In this setup, the final N layers of the transformer reuse the exact key-value states generated by earlier layers, entirely eliminating the need for redundant KV projection matrices in the deeper stages of the model. For fine-tuning on a 16GB unified memory system, this architectural choice is highly beneficial, as it significantly reduces the memory overhead required to store activation states during the backward pass of backpropagation.

### **2.4 Native Multimodality: Vision and Audio Conformers**

Unlike prior generations that relied on external adapters, the Gemma 4 E2B and E4B models are natively multimodal, capable of processing text, images, and audio directly.

* **Vision Encoder:** The models feature a \~150M parameter vision encoder that utilizes learned 2D positional embeddings and multidimensional RoPE. Crucially, it preserves the original aspect ratios of input images and dynamically scales the encoding to fit various token budgets (70, 140, 280, 560, or 1120 tokens), allowing developers to balance visual resolution against memory consumption during multimodal fine-tuning.  
* **Audio Encoder:** The models integrate a \~300M parameter Universal Speech Model (USM) style conformer. This 12-layer audio tower processes 16kHz audio natively. This allows the model to perform Speech-to-Text (STT) and direct audio reasoning without requiring a separate transcription model like OpenAI's Whisper, massively simplifying the training pipeline for voice-enabled agents.

### **2.5 The Mixture-of-Experts (MoE) Illusion: 26B-A4B**

When searching for a "4B" parameter model in the Gemma 4 lineup, developers often encounter the Gemma 4 26B-A4B. The "A" in this nomenclature denotes "active parameters". The model utilizes a sparse Mixture-of-Experts architecture where a routing mechanism selects the top 8 experts out of a total of 128 available experts for each given token. While this results in only 3.8 billion active parameters during inference, the model houses a total of 25.2 billion parameters.  
The Gemma 4 MoE architecture is unique in that it maintains a dense Feed-Forward Network (FFN) that runs in parallel with the sparse experts, summing their outputs to ensure baseline capability regardless of routing choices. While highly efficient for inference on larger machines, **this model is fundamentally incompatible with fine-tuning on a 16GB MacBook**.  
During training, the optimizer must maintain states for all 25.2 billion parameters, regardless of how many are active per token. Official Unsloth documentation dictates that fine-tuning the 26B-A4B requires in excess of 40GB of VRAM. Attempting to load this model on a 16GB machine will instantly saturate the unified memory and the swap file, resulting in an immediate system failure. Therefore, the E2B and E4B models remain the exclusive, viable targets for the 16GB hardware profile.

## **3\. Hardware Profile Analysis: The MacBook M5 16GB Paradigm**

The M5 generation of Apple Silicon introduces specific hardware enhancements that directly impact the viability and speed of local Large Language Model (LLM) fine-tuning. Understanding the physical constraints of this hardware is essential for configuring the training environment correctly.

### **3.1 Memory Bandwidth and SSD Throughput**

The most significant bottleneck in LLM training—particularly during parameter-efficient methods like QLoRA, where the computational load is relatively light but memory access is constant—is memory bandwidth. The base MacBook M5 with 16GB of unified memory features an upgraded two-channel memory bandwidth of 153 \\text{ GB/s}. This represents a 27.5\\% increase over the 120 \\text{ GB/s} bandwidth found on the preceding M4 models. In practice, this enhancement translates directly to higher tokens-per-second (tok/s) throughput during the forward and backward passes of the training loop, as the GPU cores can fetch quantized weights and activation tensors from memory at a faster rate.  
Furthermore, the base M5 utilizes PCIe 4.0 architecture for its internal Solid State Drives (SSDs), doubling the read and write speeds compared to the PCIe 3.0 storage found in earlier base models. This storage speed is a critical safety net. When the 16GB unified memory pool is fully saturated by model weights, optimizer states, and sequence activations, the macOS kernel aggressively pages inactive memory blocks to the SSD via swap memory. While relying on swap memory inevitably degrades training speed, the PCIe 4.0 bandwidth ensures that these inevitable memory paging events—which often occur during sequence length spikes in the training data—do not result in total system paralysis.

### **3.2 The Unified Memory Constraint and Memory Math**

Apple Silicon does not utilize discrete Video RAM (VRAM). Instead, it employs a Unified Memory Architecture where the CPU, the GPU, and the specialized Neural Engine all share direct access to the same physical pool of memory.  
While the system possesses 16GB of physical RAM, the usable envelope for machine learning is significantly smaller. The macOS operating system, background daemons, and window server typically reserve 3 to 4 \\text{ GB} of memory at all times. Consequently, the actual working memory available for an ML training framework is strictly bound to approximately 12 \\text{ GB}.  
To fit a Gemma 4 edge model into this 12 \\text{ GB} envelope, 4-bit Quantized Low-Rank Adaptation (QLoRA) is an absolute mathematical necessity.

* **Gemma 4 E2B:** In 4-bit quantization, the 5.1 billion total parameters consume roughly 2.6 \\text{ GB} of memory just to hold the static weights.  
* **Gemma 4 E4B:** In 4-bit quantization, the 8 billion total parameters consume approximately 4.2 \\text{ GB} of memory.

However, static weights represent only a fraction of the total memory required during fine-tuning. The framework must also allocate memory for:

1. **LoRA Adapters:** The trainable rank decomposition matrices, usually kept in 16-bit (FP16) or 32-bit (FP32) precision.  
2. **Optimizer States:** The AdamW optimizer requires 32-bit precision to track momentum and variance for every trainable parameter, heavily penalizing memory.  
3. **Activations and Gradients:** The intermediate tensors generated during the forward pass that must be stored in memory to compute the gradients during the backward pass. Activation memory scales quadratically with sequence length.  
4. **KV Cache:** Required during evaluation steps.

When aggregating these requirements, the E4B model pushes the absolute physical limits of a 16GB machine, requiring severe restrictions on batch size and context length. The E2B model, consuming less base memory, offers a wider safety margin, allowing for the processing of longer multimodal sequences (such as audio or high-resolution images) without overflowing into swap memory.

## **4\. Resolving the Framework Divide: Official Documentation vs. Apple Silicon Reality**

The most treacherous obstacle in localized fine-tuning on Mac hardware is the vast discrepancy between official corporate guidelines and the realities of Apple Silicon. Following official documentation blindly on an M5 Mac will invariably lead to failed environments, system crashes, or agonizingly slow execution times.

### **4.1 The Failings of the Official PyTorch MPS Pathway**

Google's official guidelines for fine-tuning Gemma 4 models rely exclusively on the Hugging Face ecosystem (Transformers, TRL, PEFT) utilizing the PyTorch framework. For enterprise scaling, Google directs users to NVIDIA's NeMo framework, which officially recommends a cluster of 8 \\times \\text{H100 80GB} GPUs for optimal training of the larger Gemma variants.  
When executing PyTorch on Apple Silicon, the framework leverages the Metal Performance Shaders (MPS) backend to interface with the Mac's GPU. However, PyTorch's eager execution model and its nascent memory allocator for MPS exhibit severe flaws when operating at the edge of the unified memory limits. PyTorch frequently fails to release memory buffers aggressively enough, causing the unified memory pool to fill with orphaned tensors. This inevitably triggers the dreaded RuntimeError: MPS backend out of memory.  
To circumvent this, the PyTorch community relies on a highly dangerous environment variable: export PYTORCH\_MPS\_HIGH\_WATERMARK\_RATIO=0.0. This variable disables PyTorch's internal upper limit for memory allocations, forcing the framework to push all memory demands directly to the macOS kernel. On a constrained 16GB machine, this inevitably exhausts the physical RAM, maxes out the swap file on the SSD, and induces a hard system freeze or kernel panic, necessitating a physical reboot.  
Furthermore, PyTorch's support for INT4 weight-only quantization via torchao significantly lags in performance, failing to fully utilize the M5's memory bandwidth, resulting in sluggish training throughput.

### **4.2 The MLX Advantage and Community Frameworks**

To achieve stable, high-speed training on the M5, the deployment stack must pivot to MLX. MLX is an open-source array framework explicitly engineered by Apple's machine learning research team to maximize the efficiency of the unified memory architecture.  
Unlike PyTorch, MLX utilizes a lazy computation graph. This means that operations are merely recorded into a computational graph and are not executed—and crucially, memory is not allocated—until the values are explicitly required by an eval() statement. This allows the MLX compiler to optimize memory reuse, aggressively discarding intermediate activation tensors and keeping the memory footprint as low as mathematically possible. Benchmarks consistently demonstrate that MLX achieves superior throughput for quantized models compared to PyTorch MPS, often running up to three times faster for inference and significantly faster for gradient updates.  
For Gemma 4 fine-tuning on Mac, the ecosystem has standardized around three primary MLX-based solutions:

1. **mlx-lm / mlx-vlm:** The foundational libraries provided by the MLX team. They offer robust, low-level control over text and vision LoRA fine-tuning, natively supporting 4-bit quantization.  
2. **mlx-tune (formerly unsloth-mlx):** A highly sophisticated community wrapper that translates the industry-standard Unsloth API to the MLX backend. This library solves the "Context Switch" problem, allowing researchers to utilize standard Hugging Face SFTTrainer scripts with minimal modification while enjoying native Apple Silicon acceleration. mlx-tune automatically detects Gemma 4 models and applies the correct optimizations.  
3. **gemma-tuner-multimodal:** A specialized, open-source toolkit designed specifically for fine-tuning Gemma 4 across text, image, and audio modalities natively on Apple Silicon. It features built-in mitigations for Mac memory limits and allows streaming datasets from cloud storage (GCS/BigQuery) to avoid filling the Mac's internal SSD with terabytes of audio data.

For this comprehensive walk-through, the methodology will synthesize the APIs of mlx-tune and core MLX memory management strategies, as they represent the most robust intersection of compatibility and hardware safety.

## **5\. Exhaustive Training Walk-Through for Gemma 4 on M5 16GB**

### **5.1 Dataset Preparation and Formatting strictures**

Google's official guidelines mandate that supervised fine-tuning (SFT) data must adhere strictly to the conversational chat templates established during the model's initial instruction tuning. Gemma 4 introduces native support for system prompts—a feature absent in earlier Gemma iterations—enabling the creation of highly structured, controllable agentic workflows.  
Data must be meticulously curated into JSONL formats, traditionally split into train.jsonl, valid.jsonl, and test.jsonl. Each line in the dataset must represent a complete conversation structured around the ChatML format or Gemma's native control tokens:  
`{"messages":}`

If fine-tuning for multimodal audio applications utilizing the E2B or E4B Conformer tower, toolkits like gemma-tuner-multimodal and mlx-tune allow developers to pair raw .wav or .flac audio files directly with text representations. The audio is automatically resampled to 16kHz and processed natively through the UnslothVisionDataCollator, completely bypassing the need for traditional transcription models.

### **5.2 Model Initialization and Memory-Safe Quantization**

To fit within the 16GB unified memory envelope, the target model must be initialized in 4-bit precision immediately upon loading. Utilizing the mlx-tune API, the model is initialized via the FastLanguageModel class (or FastVisionModel if engaging the audio/vision encoders).  
In Python, the initialization protocol must dynamically inject 4-bit quantization layers into the model graph as the weights are streamed from the disk:  
`from mlx_tune import FastLanguageModel`

`# Initialize the model directly into 4-bit to prevent memory spikes`  
`model, tokenizer = FastLanguageModel.fr[span_52](start_span)[span_52](end_span)[span_57](start_span)[span_57](end_span)om_pretrained(`  
    `model_name="mlx-community/gemma-4-e2b-it-4bit",`  
    `max_seq_length=512, # Strictly bounded for 16GB safety`  
    `dtype=None,`  
    `load_in_4bit=True`  
`)`

The selection of the repository mlx-community/gemma-4-e2b-it-4bit (or the E4B equivalent) is critical. Attempting to load the full FP16 model from the official Google repository and quantifying it post-load will cause an immense memory spike during initialization, instantly triggering an out-of-memory error on the 16GB M5. The model must be pre-quantized and streamed from disk directly into quantized MLX arrays.

### **5.3 Defining Target Modules for Low-Rank Adaptation (LoRA)**

Low-Rank Adaptation (LoRA) is the mathematical engine of parameter-efficient fine-tuning. It freezes the pre-trained quantized weights and injects small, trainable rank decomposition matrices into the Transformer layers. For the Gemma 4 architecture, community consensus and official Unsloth parameters dictate targeting all linear layers to ensure maximum capability retention and reasoning fidelity.  
`model = FastLanguageModel.get_peft_model(`  
    `model,`  
    `r=16,`  
    `target_modules=["q_proj", "k_proj", "v_proj", "o_proj",`  
                    `"gate_proj", "up_proj", "down_proj"],`  
    `lora_alpha=32,`  
    `lora_dropout=0.05,`  
    `bias="none",`  
    `use_gradient_checkpointing="unsloth" # Offloads activations to save memory`  
`)`

*Architectural Insight on Target Modules:* In earlier eras of fine-tuning, researchers often targeted only the Query (q\_proj) and Value (v\_proj) projection matrices to save VRAM. However, empirical testing reveals that omitting the output (o\_proj) and MLP layers (gate\_proj, up\_proj, down\_proj) critically degrades the model's ability to learn complex reasoning pathways or adhere to new structural formats like strict JSON. Given the efficiency of MLX on the M5, targeting all linear modules is mathematically viable for the E2B model, provided that strict limitations are placed on sequence length.

### **5.4 Hyperparameter Optimization for the M5 Architecture**

The selection of hyperparameters dictates both the convergence stability of the model and the intense memory pressure applied to the MLX framework. Google's official text QLoRA guide for Gemma 4 proposes highly conservative baselines. Aggregating official documentation and robust community benchmarks yields the following optimized hyperparameter spectrum specifically tailored for the Apple Silicon M5 16GB environment :

| Hyperparameter | Recommended Value | Mathematical & Hardware Justification for M5 16GB |
| :---- | :---- | :---- |
| **Learning Rate (LR)** | 2 \\times 10^{-4} to 3 \\times 10^{-4} | QLoRA inherently requires higher learning rates than full precision fine-tuning. The optimizer must traverse a quantized, block-sparse loss landscape. 2 \\times 10^{-4} is the mathematically proven sweet spot for 3B class models. |
| **LoRA Rank (r)** | 16 | Rank defines the dimensionality of the trainable matrices. Lower ranks (r=8) lack the expressive capacity to alter the model's foundational reasoning; higher ranks (r=32) exponentially increase the memory footprint without delivering proportional quality gains. |
| **LoRA Alpha (\\alpha)** | 32 (2 \\times r) | Alpha controls the scaling of the LoRA updates against the frozen base weights. The fundamental equation \\Delta W \= \\frac{\\alpha}{r} AB implies that maintaining \\alpha \= 2r ensures that the magnitude of the gradient updates remains stable and predictable across layers. |
| **Batch Size** | 1 (Physical) | A physical batch size of 1 is an absolute necessity to prevent the activation memory from overflowing the unified memory pool during the backward pass. |
| **Gradient Accumulation** | 4 to 8 steps | Because a batch size of 1 leads to noisy, erratic gradients, accumulation steps are utilized to simulate an effective batch size of 4 to 8\. The optimizer only updates the weights after accumulating gradients over multiple micro-batches, saving vast amounts of memory at the cost of slight compute overhead. |
| **Epochs** | 1 to 3 | For high-quality, targeted datasets, 3 epochs are standard. Developers must actively monitor the validation loss for overfitting beyond epoch 2\. |
| **Max Sequence Length** | 512 to 1024 | Sequence length is the most dangerous hyperparameter. Activation memory scales quadratically with context length. On a 16GB Mac, exceeding 1024 tokens during training guarantees an out-of-memory crash. 512 is the safest baseline. |
| **Warmup Ratio** | 0.05 to 0.1 | Prevents early training instability by scaling the learning rate linearly from zero to the maximum over the first 5-10% of training steps, allowing the AdamW optimizer to calculate baseline momentums without diverging. |

### **5.5 Training Execution and Loss Monitoring**

The SFTTrainer class from the Hugging Face TRL library, elegantly wrapped by mlx-tune, manages the complex training loop natively on the M5's GPU and Neural Engine cores.  
A critical strategic implementation for maintaining memory hygiene is utilizing the train\_on\_responses\_only feature. In standard causal language modeling, the model attempts to predict every token in the sequence, including the user's prompt. By mathematically masking the prompt tokens during the cross-entropy loss calculation, the model avoids backpropagating gradients for the input context. This drastically reduces the computational burden and slashes the memory overhead required during the backward pass, preserving vital megabytes in the unified memory pool.  
Monitoring training metrics requires constant vigilance. The training loss measures the error on the active batch, while validation loss measures generalizability against unseen data.

* If the training loss oscillates violently without converging, the learning rate must be throttled.  
* If the loss drops to near-zero instantly in the first few steps and stalls, the learning rate is likely too high, or the dataset is insufficiently complex, causing the model to memorize the format rather than learn the reasoning.  
* If validation loss begins to rise while training loss falls, the model has entered catastrophic overfitting.

## **6\. Advanced Memory Management: Surviving the 16GB Limit**

The most significant technical hurdle in fine-tuning Gemma 4 on the M5 16GB is not compute speed—MLX can easily achieve inference generation speeds of 120 to 180 tokens per second depending on the configuration —but aggressive memory fragmentation resulting in catastrophic hardware failures.

### **6.1 Diagnosing the completeMemory() Kernel Panic**

When training highly parameterized models on heavily constrained Apple Silicon, researchers frequently encounter sudden, hard system reboots—the screen goes black without warning. Inspection of the macOS system logs via the terminal (log show \--predicate 'eventMessage contains "panic"' \--last 2h) often reveals a specific underflow error:  
panic(cpu X caller 0xfffffe...): "completeMemory() prepare count underflow" @IOGPUMemory.cpp:550.  
This is not a Python exception or an MLX warning; it is a hardware-level kernel panic originating deep within Apple's GPU memory driver (IOGPUMemory.cpp). This specific panic occurs because MLX's lazy evaluation paradigm allows the theoretical computational graph to expand faster than the system's garbage collector can physically reclaim unified memory. When the Metal GPU demands immediate access to memory that the macOS kernel has aggressively paged to the SSD swap file, the driver times out or underflows, forcing an unceremonious emergency shutdown to protect system integrity.

### **6.2 Bounding the MLX Allocator via Metal Configurations**

To prevent this fatal error on a 16GB M5, the MLX cache limits cannot be left to their default behaviors; they must be explicitly hard-coded into the training script before the model is loaded into memory. MLX provides lower-level APIs to throttle the Metal memory allocator.  
The standard system limit for the MLX working set size on a 16GB machine defaults to 1.5 times the recommended working size, which sits uncomfortably close to total system failure levels. To enforce absolute stability, community-derived optimizations dictate aggressively overriding these limits :  
`import mlx.core as mx`

`if mx.metal.is_available():`  
    `# Retrieve the OS's maximum recommended working set size`  
    `wired_limit = mx.metal.device_info()["max_recommended_working_set_size"]`  
      
    `# Restrict the wired limit to 90% to prevent total OS starvation`  
    `safe_wired_limit = int(wired_limit * 0.9)`  
    `print(f"Setting safe wired limit to {safe_wired_limit / 1e9:.2f} GB")`  
    `mx.metal.set_wired_limit(safe_wired_limit)`  
      
    `# Cap the absolute total memory limit strictly to the wired limit`  
    `mx.metal.set_memory_limit(safe_wired_limit)`  
      
    `# Aggressively reduce the cache limit to 50% of the wired limit`  
    `# This forces the MLX garbage collector to flush inactive tensors continuously`  
    `cache_limit = int(safe_wired_limit * 0.5)`  
    `mx.metal.set_cache_limit(cache_limit)`

By explicitly setting mx.metal.set\_cache\_limit() to a low threshold, the training script forces the MLX array framework to evaluate the computational graph and flush inactive tensors much sooner than it naturally would. While this incurs a minor performance penalty (approximately a 20\\% reduction in raw training speed due to the overhead of continuous allocation and deallocation cycles), it virtually eliminates the risk of kernel panics and system reboots, making unsupervised overnight training viable.

### **6.3 Explicit Cache Clearing and Garbage Collection in the Loop**

In addition to setting static limits at initialization, dynamic memory reclamation must be inserted directly into the core training loop. Between evaluation steps, or immediately following gradient accumulation phases, forcing the graph evaluation via mx.eval() and manually clearing the Metal cache ensures that orphaned activation tensors are destroyed before the next forward pass begins.  
`import gc`  
`import mlx.core as mx`

`#... Executed inside the training loop after an optimizer update step...`  
`# Force evaluation of all pending parameters in the computational graph`  
`mx.eval(model.parameters())`

`# Clear the Metal memory cache to destroy orphaned tensors`  
`mx.metal.clear_cache()`

`# Trigger Python's garbage collector`  
`gc.collect()`

It is vital to understand the side effects of mx.metal.clear\_cache(). Calling this function outright destroys the Key-Value (KV) cache states. During standard text generation, wiping the KV cache after every turn severely degrades latency, as the model must re-compute the entire prompt prefix from scratch. However, during the iterative, isolated micro-batches of supervised fine-tuning (SFT), retaining KV states between different training samples is unnecessary. Therefore, this complete flush guarantees that the next forward pass has the maximum possible memory overhead available, preserving the integrity of the 16GB limit.

## **7\. Model Evaluation, Export Logistics, and Deployment**

Upon the successful completion of the targeted training epochs, the resulting artifact is a set of LoRA adapters (typically saved as adapters.safetensors or adapters.npz). Because the base Gemma 4 weights remained entirely frozen in 4-bit precision during training, these adapter files are remarkably small (often under 100MB) and represent the isolated, fine-tuned behavioral delta.

### **7.1 Downstream Validation and Testing Protocols**

Evaluation of the fine-tuned Gemma 4 model must extend far beyond the raw perplexity metrics calculated during the training loop's validation phase. Official Google guidelines recommend subjecting the tuned artifact to three specific categories of boundary conditions:

1. **Success Tests:** Verifying the model can flawlessly execute the specific task it was trained for, such as adhering to complex JSON schemas, executing agentic tool-use commands, or parsing multimodal inputs accurately.  
2. **Failure Tests:** Ensuring the model correctly identifies and refuses explicitly prohibited requests, or handles out-of-domain inputs gracefully without hallucinating false confidence.  
3. **Boundary Tests:** Testing edge cases where inputs fall just inside or outside acceptable parameters to determine the fragility of the model's new behavior.

When properly fine-tuned, the Gemma 4 E2B model achieves astonishing performance benchmarks. Driven by the dense semantic representations allowed by the Per-Layer Embeddings, community benchmarking reveals that a tuned E2B can achieve 92.9% on classification tasks and 80.2% on information extraction F1 scores, effectively matching or beating larger 12B class models from competing lineages. It also exhibits flawless 100% prompt injection resistance when tuned for safety.

### **7.2 Exporting Constraints: Navigating the MLX GGUF Bug**

A critical, often-overlooked nuance of the Apple Silicon pipeline emerges during the final export phase. The prevailing community standard for local LLM deployment is the GGUF format, which is heavily optimized for inference engines like llama.cpp and Ollama.  
However, when fine-tuning a model that was inherently loaded in 4-bit quantization (e.g., utilizing the load\_in\_4bit=True flag to survive the 16GB limit), direct export to GGUF using the standard save\_pretrained\_gguf command **will fail** within both the mlx-tune and mlx-lm pipelines. This failure is not a bug in the training script, but a known, upstream architectural limitation in how MLX arrays attempt to directly translate natively quantized matrices into the GGUF format.  
To bypass this critical limitation, engineers must utilize one of two primary pathways :

1. **Native MLX Merging (Recommended for macOS):** If the deployed model will operate solely within the Apple hardware ecosystem (e.g., served locally via mlx\_lm.server), the developer should bypass GGUF entirely and call save\_pretrained\_merged(). This command dynamically fuses the LoRA adapters into the base weights, outputting a highly optimized, ready-to-serve Hugging Face format directory that runs natively on Metal.  
2. **Dequantization to FP16 (Required for GGUF):** To successfully export to the universal GGUF format, the model must be forced to dequantize its weights back to 16-bit precision during the save operation using the flag: model.save\_pretrained\_gguf("model", tokenizer, dequantize=True). This process will create a massive FP16 file on the SSD. Once exported, the user must utilize the external llama.cpp binary to manually re-quantize the file back down to a manageable size (e.g., ./llama-quantize model.gguf model-q4\_k\_m.gguf Q4\_K\_M).

Furthermore, for developers whose ultimate target is mobile or deeply embedded deployment (such as iOS, Android, or IoT devices), Google provides the LiteRT-LM framework. The specialized gemma-4-E2B-it-litert-lm format takes advantage of the E2B's unique architecture by mapping the 1.12GB of PLE embedding parameters directly to memory while keeping the 0.79GB of active text decoder weights isolated, representing a highly structured, memory-efficient alternative to GGUF for cross-platform edge applications.

## **8\. Conclusion**

The convergence of Google's Gemma 4 architectural advancements and Apple's M5 Silicon provides an unprecedented, localized capability for private, highly customized machine learning. For professional workflows constrained by a MacBook M5 featuring 16GB of unified memory, the objective of fine-tuning a "3B-class" model is most effectively and safely realized through the Gemma 4 E2B variant.  
Its innovative Per-Layer Embeddings (PLE) architecture allows it to punch significantly above its weight class, delivering frontier-level multimodal reasoning and structural fidelity while strictly adhering to the absolute constraints of a 16GB memory envelope.  
By deliberately bypassing the flawed memory management and severe limitations of the official PyTorch MPS pathway in favor of Apple's highly optimized MLX framework —and strategically leveraging translation utilities like mlx-tune —practitioners can access a stable, efficient, and exceptionally rapid training pipeline.  
Ultimately, successful fine-tuning on this hardware profile relies less on raw computational power and entirely on disciplined, algorithmic memory administration. The aggressive management of the MLX metal allocator (mx.set\_cache\_limit), the strategic imposition of 4-bit quantization, and the precise, mathematically sound tuning of hyperparameters collectively prevent catastrophic kernel panics, enabling developers to fully realize the transformative potential of edge AI on consumer hardware.

#### **Works cited**

1\. Gemma 4 model card | Google AI for Developers, https://ai.google.dev/gemma/docs/core/model\_card\_4 2\. Gemma 4 \- Google DeepMind, https://deepmind.google/models/gemma/gemma-4/ 3\. Welcome Gemma 4: Frontier multimodal intelligence on device \- Hugging Face, https://huggingface.co/blog/gemma4 4\. M5 MacBook Air vs. M4, M3, M2, M1: Should You Upgrade? \- CNET, https://www.cnet.com/tech/computing/macbook-air-upgrade-guide-m5-vs-m4-m3-m2-m1/ 5\. Fine-Tuning on a MacBook: MLX, 3 Minutes, 90 Examples, and a Model That Actually Works, https://florinelchis.medium.com/fine-tuning-on-a-macbook-mlx-3-minutes-90-examples-and-a-model-that-actually-works-7de0547da347 6\. google/gemma-4-E4B-it \- Hugging Face, https://huggingface.co/google/gemma-4-E4B-it 7\. GitHub \- ARahim3/mlx-tune: Fine-tune LLMs on your Mac with Apple Silicon. SFT, DPO, GRPO, Vision, TTS, STT, Embedding, and OCR fine-tuning — natively on MLX. Unsloth-compatible API., https://github.com/ARahim3/mlx-tune 8\. GitHub \- mattmireles/gemma-tuner-multimodal: Fine-tune Gemma 4 and 3n with audio, images and text on Apple Silicon, using PyTorch and Metal Performance Shaders., https://github.com/mattmireles/gemma-tuner-multimodal 9\. How to Fine-Tune Gemma 4: A Full Walkthrough with a Human Emotions Dataset, https://www.datacamp.com/tutorial/fine-tune-gemma-4 10\. Fine-Tune Gemma using Hugging Face Transformers and QloRA | Google AI for Developers, https://ai.google.dev/gemma/docs/core/huggingface\_text\_finetune\_qlora 11\. How My Local Coding Agent Crashed My Mac, and What I Learned About MLX Memory Management \- Medium, https://medium.com/@michael.hannecke/how-my-local-coding-agent-crashed-my-mac-and-what-i-learned-about-mlx-memory-management-e0cbad01553c 12\. Memory during training (LoRA) · Issue \#828 · ml-explore/mlx-lm \- GitHub, https://github.com/ml-explore/mlx-lm/issues/828 13\. Gemma 4, Phi-4, and Qwen3: Accuracy–Efficiency Tradeoffs in Dense and MoE Reasoning Language Models \- arXiv, https://arxiv.org/html/2604.07035v1 14\. Gemma 4 E2B vs E4B: The Edge Models That Run Audio and Vision on Your Phone, https://www.mindstudio.ai/blog/gemma-4-e2b-vs-e4b-edge-models-audio-vision-phone 15\. How does Per-Layer Embeddings improve Gemma 4? \- Milvus, https://milvus.io/ai-quick-reference/how-does-perlayer-embeddings-improve-gemma-4 16\. Per-Layer Embeddings: A simple explanation of the magic behind the small Gemma 4 models : r/LocalLLaMA \- Reddit, https://www.reddit.com/r/LocalLLaMA/comments/1sd5utm/perlayer\_embeddings\_a\_simple\_explanation\_of\_the/ 17\. Gemma 4 \- LM Studio, https://lmstudio.ai/models/gemma-4 18\. mlx-tune/README.md at main · ARahim3/mlx-tune · GitHub, https://github.com/ARahim3/unsloth-mlx/blob/main/README.md 19\. Gemma 4 Explained: How One Model Family Spans Phones and Frontier-Class Reasoning, https://louiswang524.github.io/blog/gemma-family/ 20\. Gemma 4 Fine-tuning Guide | Unsloth Documentation, https://unsloth.ai/docs/models/gemma-4/train 21\. Releases · ARahim3/mlx-tune \- GitHub, https://github.com/ARahim3/unsloth-mlx/releases 22\. State of PyTorch Hardware Acceleration 2025, https://tunguz.github.io/PyTorch\_Hardware\_2025/ 23\. MPS or MLX for Domestic AI? The Answer Will Surprise You | by Mike Koypish \- Medium, https://medium.com/@koypish/mps-or-mlx-for-domestic-ai-the-answer-will-surprise-you-df4b111de8a0 24\. Question about read/write speeds on the ORIGINAL M5 MacBook Pro \- Reddit, https://www.reddit.com/r/macbook/comments/1rsq6ym/question\_about\_readwrite\_speeds\_on\_the\_original/ 25\. MLX supports Qlora now : r/LocalLLaMA \- Reddit, https://www.reddit.com/r/LocalLLaMA/comments/18yz8kc/mlx\_supports\_qlora\_now/ 26\. Fine-Tuning LLMs Locally Using MLX LM: A Comprehensive Guide \- Level Up Coding, https://levelup.gitconnected.com/fine-tuning-llms-locally-using-mlx-lm-a-comprehensive-guide-6049fd3014bb 27\. Why is gemma4 using so much ram. : r/LocalLLaMA \- Reddit, https://www.reddit.com/r/LocalLLaMA/comments/1ssbzfc/why\_is\_gemma4\_using\_so\_much\_ram/ 28\. Unified Memory — MLX 0.31.2 documentation, https://ml-explore.github.io/mlx/build/html/usage/unified\_memory.html 29\. Profiling Apple Silicon Performance for ML Training \- arXiv, https://arxiv.org/pdf/2501.14925 30\. Unsloth-MLX \- Fine-tune LLMs on your Mac (same API as Unsloth) : r/LocalLLaMA \- Reddit, https://www.reddit.com/r/LocalLLaMA/comments/1q5mh84/unslothmlx\_finetune\_llms\_on\_your\_mac\_same\_api\_as/ 31\. litert-community/gemma-4-E2B-it-litert-lm \- Hugging Face, https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm 32\. mlx-tune/examples/48\_gemma4\_audio\_understanding.py at main \- GitHub, https://github.com/ARahim3/mlx-tune/blob/main/examples/48\_gemma4\_audio\_understanding.py 33\. Gemma 4 Fine-Tuning Guide: LoRA/QLoRA on Consumer GPUs | CloudInsight, https://cloudinsight.cc/en/blog/gemma-4-fine-tuning 34\. Performance of mtb-7b on mac M1 \- Beginners \- Hugging Face Forums, https://discuss.huggingface.co/t/performance-of-mtb-7b-on-mac-m1/67794 35\. Fine-Tuning Gemma 4 31B on CORD-v2 Receipts — End-to-End Guide — NeMo-AutoModel, https://docs.nvidia.com/nemo/automodel/latest/guides/vlm/gemma4.html 36\. MPS backend out of memory \- PyTorch Forums, https://discuss.pytorch.org/t/mps-backend-out-of-memory/183879 37\. Fine Tuning/GGML Quantiziation on Apple Silicon Guide : r/LocalLLaMA \- Reddit, https://www.reddit.com/r/LocalLLaMA/comments/15y9m64/fine\_tuningggml\_quantiziation\_on\_apple\_silicon/ 38\. Insufficient memory for Foundation… | Apple Developer Forums, https://origin-devforums.apple.com/forums/thread/789392 39\. PyTorch (MPS) is faster than MLX for training and inference for ResNets and Transformers (tested on 2 tasks) \#243 \- GitHub, https://github.com/ml-explore/mlx/issues/243 40\. Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU, https://machinelearning.apple.com/research/exploring-llms-mlx-m5 41\. MLX Integration \- Autohand Docs, https://autohand.ai/docs/integrations/mlx 42\. Memory reusing / garbage collection mechanism during a single eval · ml-explore mlx · Discussion \#912 \- GitHub, https://github.com/ml-explore/mlx/discussions/912 43\. Run Gemma with MLX \- Google AI for Developers, https://ai.google.dev/gemma/docs/integrations/mlx 44\. google/gemma-4-E2B-it \- Hugging Face, https://huggingface.co/google/gemma-4-E2B-it 45\. Fine-Tune Gemma 4 with LoRA & QLoRA: Complete Guide \- Lushbinary, https://lushbinary.com/blog/fine-tune-gemma-4-lora-qlora-complete-guide/ 46\. Fine-tuning Gemma-4-E2B on MacBook M3 \- Transformers \- Hugging Face Forums, https://discuss.huggingface.co/t/fine-tuning-gemma-4-e2b-on-macbook-m3/175228 47\. Full Model Fine-Tune using Hugging Face Transformers | Gemma, https://ai.google.dev/gemma/docs/core/huggingface\_text\_full\_finetune 48\. Choosing an On-Device LLM Runtime on Apple Silicon: A Decision Framework Beyond Benchmarks \- Medium, https://medium.com/@michael.hannecke/choosing-an-on-device-llm-runtime-on-apple-silicon-a-decision-framework-beyond-benchmarks-2449067b8b67 49\. Apple Silicon MLX LLM Inference Optimization Tutorial | Branch8, https://branch8.com/posts/apple-silicon-mlx-llm-inference-optimization-tutorial 50\. Metal — MLX 0.31.2 documentation, https://ml-explore.github.io/mlx/build/html/python/metal.html 51\. Seems like when generating, some memory usage cannot be correctly released. · Issue \#724 · ml-explore/mlx-examples \- GitHub, https://github.com/ml-explore/mlx-examples/issues/724 52\. Server clears Metal cache after every request, destroying KV prefix cache and forcing full re-prefill on every turn · Issue \#999 · Blaizzy/mlx-vlm \- GitHub, https://github.com/Blaizzy/mlx-vlm/issues/999 53\. Active memory continues to rise throughout training until the training run crashes · Issue \#1262 · ml-explore/mlx-examples \- GitHub, https://github.com/ml-explore/mlx-examples/issues/1262 54\. Fine-Tuning with LoRA or QLoRA \- ml-explore/mlx-examples \- GitHub, https://github.com/ml-explore/mlx-examples/blob/main/lora/README.md 55\. Gemma model fine-tuning | Google AI for Developers, https://ai.google.dev/gemma/docs/tune 56\. Benchmarked Gemma 4 E2B: The 2B model beat every larger sibling on multi-turn (70%), https://www.reddit.com/r/LocalLLaMA/comments/1sklc53/benchmarked\_gemma\_4\_e2b\_the\_2b\_model\_beat\_every/ 57\. It's Not the Size: Harness Design Determines Operational Stability in Small Language Models \- arXiv, https://arxiv.org/pdf/2605.12129