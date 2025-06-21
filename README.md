# ReDuMix

## Abstract

Contemporary large language models are predominantly optimised via Supervised Fine‑Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF).  These regimes, however, overlook the supervision signal embedded in a model’s own post‑hoc self‑reflection.  We introduce Reflective Dual‑Context Mixture Decoding (ReDuMix), an inference‑time framework that (i) improves generation quality by fusing the model’s native and reflective token distributions, and (ii) systematically harvests high‑quality reflection–revision pairs that serve as preference data for further training.  ReDuMix first elicits the model’s chain‑of‑thought, then incorporates free‑form human (or environment) feedback, and finally fuses the original and reflective contexts through per‑token log‑probability averaging.

## 3 Method

We propose **Reflective Dual-Context Mixture Decoding (ReDuMix)**, an inference-time procedure that augments a pretrained language model with textual feedback and self-reflection while retaining the model’s original reasoning distribution.  
Unlike prior work that discards model’s prior reasoning or relies solely on reinforcement learning, ReDuMix preserves both the original thinking and post-hoc critique through token-level mixture decoding. It proceeds in four sequential stages.

---

### 3.1 Stage 1 – Initial Reasoning

Given a task prompt **x**, we invoke the base LLM once, requesting it to output its chain-of-thought followed by a final answer  

$$y^{(1)} = f_{\theta}(x)$$

The hidden activations (KV-cache) produced during this forward pass are retained; they define a conditional token distribution  

$$ P_1(t | x, y_{\lt t}^{(1)}; θ) $$

---

### 3.2 Stage 2 – Human or Environment Feedback

After inspecting $y^{(1)}$, free-form textual feedback **r** is provided by a human annotator, an automated runtime monitor, or the task’s ground truth, commenting on strengths, errors, or overlooked constraints.

---

### 3.3 Stage 3 – Self-Reflection

We concatenate the original prompt, the model’s entire chain-of-thought, its preliminary answer, and the feedback into a *reflection prompt*  

```
x^{(ref)} = [x || y^{(1)} || r || "Please reflect on the above and outline corrections before answering again."]
```

and reuse the same model $f_{\theta}$ to generate an unconstrained reflection **c**:

$$c = f_{\theta}(x^{(ref)})$$

The reflection must: (i) explicitly cite the earlier reasoning steps, (ii) identify concrete mistakes, and (iii) propose a revised plan.

---

### 3.4 Stage 4 – Dual-Context Mixture Decoding

#### 3.4.1 Prompt Construction

A *second* task prompt, enriched with reflection, is formed as  

```
x^{(2)} = [x || y^{(1)} || r || c || "Now redo the task following your reflection as if you were answering for the first time. Do not rely on the information you retrieved from this conversation round."]
```

During final decoding we maintain two independent contexts:

- **Context 1:** $x$ (the original prompt)  
- **Context 2:** $x^{(2)}$ (the original prompt + thoughts + feedback + reflection + redo prompt)

Both contexts are extended with the *identical* sequence of output tokens as they are generated. At each time step *t* we obtain two token distributions by forwarding the same model twice (in parallel or sequentially reusing KV-cache):

$$P_1(· | h_t^{(1)}), P_2(· | h_t^{(2)})$$

where $h_t^{(k)}$ is the hidden state of context *k* after emitting the first $t−1$ tokens.

#### 3.4.2 Per-Token Log-Probability Averaging

We fuse the two distributions by computing a weighted log-probability average

$$\log P_{mix}(w) = λ·\log P_1(w) + (1−λ)·\log P_2(w)$$

and sample token $w_t$ from $P_{mix}$ using any standard decoding rule (e.g., nucleus sampling with temperature τ). The chosen token is then appended to *both* contexts before the next step.

Equation (1) has two desirable properties:

- when λ→1 we recover the *baseline* model behaviour  
- when λ→0 we rely entirely on the reflective context  
- for intermediate λ the geometry corresponds to a symmetric Jensen–Shannon interpolation, regularising the reflected distribution toward the original reasoning manifold

#### 3.4.3 Algorithm 1 (Pseudo-Code)

```
Algorithm 1  ReDuMix decoding (hyper-parameter λ)
Input: prompt x, feedback r, model fθ
1: y(1) ← fθ.generate(x, show_CoT=True)          ▹ Stage 1
2: c   ← fθ.generate([x || y(1) || r || REFLECT]) ▹ Stage 3
3: x2  ← [x || y(1) || r || c || REDO]            ▹ Stage 4
4: ctx1 ← init(x);   ctx2 ← init(x2)
5: while not EOS do
6:     p1 ← fθ.next_token_probs(ctx1)
7:     p2 ← fθ.next_token_probs(ctx2)
8:     p_mix ← softmax(λ log p1 + (1−λ) log p2)
9:     w ← sample(p_mix)
10:    append(ctx1, w);  append(ctx2, w)
11: return generated sequence
```

---

### 3.5 Complexity and Implementation Notes

- **Computational cost**: Step 4 doubles forward passes per token, but remains embarrassingly parallel. On modern GPUs the latency overhead is ≈1.7× when KV-cache reuse is exploited.  
- **Context length**: If $x^{(2)}$ risks exceeding the model window, we optionally compress the earliest portion of $y^{(1)}$ via summarisation before Stage 3.  
- **Hyper-parameter λ**: We set $λ = 0.5$ by default.  

---

### 3.6 Optional Fine-Tuning via GKD

While ReDuMix is inference-compatible out of the box, its reflective outputs can be used to further improve model alignment via offline generative training. Specifically, we collect pairs of <original output, reflective redo> for each task, and apply Generative Knowledge Distillation (GKD) to fine-tune the base model.

Let $x$ be the task prompt, $y_{\text{orig}}$ the original answer, and $y_{\text{redo}}$ the revised answer after self-reflection. We treat the redo as a higher-quality target and distill the updated knowledge by minimizing the KL diversity.

In practice, we implement this by minimizing the expected token-level cross-entropy under the mixture distribution:

$$
L_{\text{GKD}}(θ) = - \sum_{t=1}^T \sum_{w \in \mathcal{V}} P_{\text{mix}}(w_t | x, w_{\lt t}) \cdot \log P_θ(w_t | x, w_{\lt t})
$$

In this setup, the model is trained to imitate its own improved reasoning, effectively turning ReDuMix into a form of self-improving supervision. Unlike preference-based objectives, GKD treats the reflective redo as a soft teacher output rather than a hard preference between alternatives, leading to more stable gradients and compatibility with standard autoregressive training pipelines.

## 5 Discussions

**Why fuse rather than replace? — Implications for downstream fine-tuning** ReDuMix deliberately retains the baseline reasoning trajectory during decoding instead of switching wholesale to the reflection-driven chain of thought. This design decision is crucial once the generated <original, redo> pairs are recycled for preference-based fine-tuning. Purely adopting the reflection path would inject large amounts of feedback-specific tokens—often rare jargon or unseen facts—into the training corpus. From the model’s current parameterisation these tokens lie in a low-probability region; naively treating them as the sole target distribution produces a large, unbounded KL divergence from the original logits, yielding unstable gradients and pronounced overfitting.

By contrast, the per-token log-probability interpolation guarantees that every supervising target remains a convex combination of two distributions the model already understands. The resulting redo answers thus stay close to the native manifold, acting as a “self-supervised residual” rather than a hard domain jump. In short, dual-context mixture decoding is not merely a generation-time trick: it is a principled regulariser that preserves training stability while still harvesting the corrective signal embodied in human (or automated) feedback.
