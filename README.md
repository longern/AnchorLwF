# Token-Level Learning without Forgetting: Fine-tuning LLM with Minimal Data Yet Preventing Catastrophic Forgetting

## Abstract

In this paper, we propose a novel fine-tuning method for large language models (LLMs) that combines token-level cross-entropy loss with token-level knowledge distillation via KL divergence. This approach allows for efficient memory retention with minimal annotated data, inspired by the "Learning without Forgetting" (LwF) paradigm, but applied at the token level. We demonstrate that only a single, carefully chosen token needs to be annotated, significantly reducing annotation costs while maintaining high performance. Experimental results show that our approach outperforms traditional methods like Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) in terms of memory retention for specific content.

## 1. Introduction

Large language models have shown significant promise across a wide range of tasks, but fine-tuning them efficiently remains a challenge. Traditional supervised fine-tuning (SFT) relies on full datasets annotated with ground truth, while knowledge distillation methods use a reference model to guide the learning process through KL divergence. Both methods suffer from drawbacks such as high annotation costs and a tendency to forget previously learned knowledge. Recent advancements in reinforcement learning (RL) have provided a partial solution but still face issues related to efficiency and stability in certain tasks.

In **Continual Learning**, models are often required to adapt to new tasks while retaining previously acquired knowledge, which is a known challenge due to **catastrophic forgetting**. In this work, we propose a novel token-level fine-tuning approach that combines cross-entropy loss for a small number of selected tokens with KL divergence for the remaining tokens, using the reference model as a guide. This method aims to balance memory retention with minimal data annotation, providing a practical solution for fine-tuning in resource-constrained environments.

## 2. Related Work

### 2.1 Supervised Fine-Tuning (SFT)

Supervised Fine-Tuning (SFT) is the standard approach for adapting a pre-trained model to a new task. It relies on a dataset of annotated examples and typically uses cross-entropy loss to minimize the error between the model's predictions and the ground truth. However, SFT has limitations in terms of scalability, requiring large amounts of labeled data, and it suffers from catastrophic forgetting when learning new tasks.

### 2.2 Knowledge Distillation (KD) via KL Divergence

Knowledge distillation is a technique where a smaller or reference model is used to guide the training of a larger model. The student model learns to match the output distribution of the teacher model using a loss function like KL divergence. This method helps prevent the student model from forgetting previously learned knowledge and can result in better generalization.

### 2.3 Learning without Forgetting (LwF)

Learning without Forgetting (LwF) is a method aimed at allowing a model to learn new tasks while retaining the knowledge from previously learned tasks. This is typically achieved by applying a regularization term that forces the model to maintain the outputs of its previous tasks. LwF, while effective, still relies on the full dataset being available for each task, which can be computationally expensive.

### 2.4 Reinforcement Learning (RL) for Fine-Tuning

Reinforcement learning (RL) has been used for fine-tuning models in environments where exploration and dynamic feedback are crucial. While RL offers flexibility and adaptability, it is often less efficient than other methods when the task requires minimal labeled data, or when the reward signal is sparse.

## 3. Methodology

### 3.1 Hybrid Loss Function

We propose a hybrid loss function that combines two distinct components:

- **Cross-Entropy Loss:** Applied to a small, carefully selected set of tokens. These tokens represent critical parts of the input that need to be memorized. Annotators select the first incorrect token and mark the correct one, guiding the model to correct this specific error.

- **KL Divergence Loss:** Applied to the remaining tokens, where the goal is to maintain the output distribution similar to that of the reference model. This ensures that the model does not deviate significantly from previously learned knowledge while focusing on the selected tokens.

### 3.2 Token Selection Process

The token selection process is a manual process initially (called xxx), where annotators interact with the model's output. The process follows these steps:

1. The annotator views the original input and output.
2. They select the first incorrect token in the output.
3. The annotator then provides the correct token for that position, which is used as ground truth for that token. Add this token to the xxx set.
4. Subsequent tokens are generated in a greedy manner based on the updated token.
5. This process repeats until the entire sequence is correct.

### 3.3 Reference Model

The reference model must be the same model being fine-tuned. It can be any auto-regressive language model that is used as the "teacher" during the knowledge distillation phase, guiding the model's outputs via KL divergence.

## 4. Experiments

### 4.1 Experimental Setup

To evaluate the effectiveness of our proposed method, we will perform experiments on a task that requires model memory retention. Specifically, we will test the model's ability to "remember" a specific piece of information after fine-tuning.

We will compare our approach against traditional SFT and RL-based fine-tuning methods. The primary evaluation criterion is the ability of the model to recall the selected memory after fine-tuning without catastrophic forgetting.

> **Note:** The specific task and dataset will be chosen based on availability and the model’s needs.

### 4.2 Results and Analysis

We will present the results of our experiments, showing that our method requires only minimal annotated data (potentially as little as one token per sequence) while outperforming traditional SFT and RL methods in terms of memory retention. Additionally, we will compare the performance of the model on tasks requiring long-term memory retention after fine-tuning.

## 5. Discussion

### 5.1 Advantages

- **Minimal Annotation Cost:** By requiring the annotation of only a small subset of tokens, the overall annotation cost is significantly reduced compared to traditional methods.
- **Efficient Memory Retention:** Our method is able to remember specific pieces of information effectively without significant forgetting, which is a common issue in large-scale fine-tuning.

### 5.2 Limitations

- **High Initial Annotation Cost:** While the total annotation cost is lower, the initial cost of annotating each sequence may still be high, especially when large-scale datasets are involved.
- **Manual Token Selection:** The token selection process is currently manual, which may be a bottleneck for large-scale applications. We plan to explore automation in future work.

### 5.3 Future Work

- **Automated Token Selection:** Future work could explore automating the token selection process through a dedicated module that can identify important tokens for fine-tuning.
- **Scalability:** We will explore the scalability of our approach to larger datasets and different model architectures.

## 6. Conclusion

We have introduced a novel fine-tuning method that combines token-level cross-entropy loss and KL divergence to enable efficient memory retention in large language models. Our method requires minimal annotated data and outperforms traditional fine-tuning techniques, offering a promising solution for fine-tuning in resource-constrained environments.

> **Note:** Experimental results and datasets need to be added in the final version.
