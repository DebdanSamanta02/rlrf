**Task: Implement RLRF for Vector Graphics Generation**

## **1\. Overview**

Implement a two-stage training pipeline to optimize an autoregressive Vision-Language Model (VLM) for generating Scalable Vector Graphics (SVG). The goal is to move beyond token-matching by incorporating evaluative feedback from rendered images using Group Relative Policy Optimization (GRPO).

## **2\. Fundamental Methodology**

### **Stage 1: Supervised Fine-Tuning (SVG-SFT)**

The model must first be adapted to the SVG domain by minimizing the negative log-likelihood of ground-truth SVG token sequences.

**Objective Function:**

$$\\mathcal{L}\_{SFT}(\\theta) \= \\mathbb{E}\_{x\_c \\sim \\mathcal{D}} \[-\\log p\_\\theta(x\_s | x\_c)\] \= \\mathbb{E}\_{x\_c \\sim \\mathcal{D}} \\left\[-\\sum\_{l=1}^{L} \\log p\_\\theta(x\_{s,l} | x\_{s,\<l}, x\_c)\\right\]$$

* **$x\_c$**: Input image or text prompt.

* **$x\_s$**: Ground-truth SVG token sequence.

### **Stage 2: Reinforcement Learning from Rendering Feedback (RLRF)**

Further refine the SFT model by maximizing a visual reward calculated from rendered SVG roll-outs. Use **GRPO** to eliminate the need for an external value network.

**GRPO Objective:**

$$\\mathcal{J}\_{GRPO}(\\theta) \= \\mathbb{E}\_{x\_c \\sim \\mathcal{D}} \\left\[\\frac{1}{G} \\sum\_{i=1}^{G} \\min(r\_i A\_i, \\text{clip}(r\_i, 1-\\epsilon, 1+\\epsilon)A\_i) \- \\beta D\_{KL}(p\_\\theta || p\_{\\theta\_{ref}})\\right\]$$

* **$r\_i$**: Probability ratio $\\frac{p\_\\theta(o\_i|x\_c)}{p\_{\\theta\_{old}}(o\_i|x\_c)}$.

* **$A\_i$**: Advantage, calculated as $R(x\_c, o\_i) \- \\frac{1}{G} \\sum\_{j=1}^{G} R(x\_c, o\_j)$.

* **$G$**: Number of roll-outs per input image (Target: 64).

* **$\\epsilon$**: Clipping threshold (Target: 0.4).

## ---

**3\. Reward Function Modeling**

The agent must implement a composite reward function $R\_{total} \= \\sum\_{i=1}^{K} w\_i R\_i$.

### **A. Image Reconstruction Reward ($R\_{img}$)**

Normalize the input image $I\_{in}$ and predicted rendering $I\_{pred}$ to zero mean and unit variance, then compute the L2 distance:

$$R\_{img} \= \\text{clip}\\left(1 \- \\frac{1}{N} ||I\_{in}^{norm} \- I\_{pred}^{norm}||\_2^2, \-1, 1\\right)$$

### **B. Code Efficiency Reward ($R\_{len}$)**

Penalize excessive token length relative to the ground-truth length $L\_{gt}$:

$$R\_{len} \= 1 \- \\left(\\frac{1}{L\_{gt}} \\max(0, L\_{pred} \- \\frac{L\_{gt}}{2})\\right)^2$$

### **C. Semantic Similarity Reward**

* **Im2SVG**: Use **DreamSim** cosine similarity.

* **Text2SVG**: Use **CLIP** similarity between the text prompt and the rendered image.

## ---

**4\. Execution Details**

### **Model Architecture**

* **Vision Encoder**: CLIP ViT or similar (e.g., Qwen2.5-VL ViT).

* **Projection Layer**: Linear layers to align visual tokens with LLM dimensions.

* **Language Model**: Decoder-only LLM (e.g., Qwen2.5-VL-7B or StarVector-1B).

### **Training Pipeline**

1. **Rendering Engine**: Integrate **CairoSVG** to rasterize predicted code into $224 \\times 224$ or adaptive resolution images.

2. **Sampling Strategy**: Implement autoregressive sampling with a temperature of **1.1** to ensure roll-out diversity.

3. **Data Curation**: For RLRF, filter for high-entropy images with at least 500 ground-truth SVG tokens.

4. **Hardware Optimization**: Use Fully Sharded Data Parallel (FSDP) and vLLM for high-throughput rollout generation.

## **5\. Constraints & Guardrails**

* **KL Penalty**: While the formula includes $\\beta D\_{KL}$, the paper suggests removing the KL term can improve reward learning for structured SVG syntax.

* **Reward Hacking Mitigation**:  
  * Enforce rendering at the reference image size regardless of predicted viewBox attributes.

  * In Text2SVG, strip \<text\> elements from the SVG before rendering to prevent the model from "cheating" by just writing the prompt.

* **Dynamic Max Length**: Set the maximum sampling length per batch to the longest ground-truth sequence plus a small threshold to avoid infinite loops.  
