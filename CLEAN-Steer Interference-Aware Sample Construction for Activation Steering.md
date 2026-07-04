# CLEAN-Steer: Interference-Aware Sample Construction for Activation Steering

> **Data-level preprocessing for clean activation steering vectors**

------

## 📌 Quick Overview

**CLEAN-Steer** is a **data preprocessing framework** for activation steering in large language models. It addresses a critical but overlooked problem: **steering vectors extracted from naively constructed contrastive samples are often "contaminated" by interference dimensions** (e.g., topic, style, length) that systematically covary with the target behavior.

**Our insight**: Existing work (LayerNavigator, CDPO, StTP) addresses interference at the **layer selection**, **algorithm**, or **vector** level. CLEAN-Steer tackles it at the **data level** — before the steering vector is even extracted.

**Key claim**: Interference-aware sample construction is **orthogonal and complementary** to existing methods, and can be plugged into any existing activation steering pipeline.

------

## 🔬 Problem Illustration

### The Interference Problem in One Example

Suppose you want to steer a model to be **more optimistic** (target dimension). You construct contrastive pairs:

| Sample     | Optimism | Topic   | Activation (2D simplified) |
| :--------- | :------- | :------ | :------------------------- |
| Positive 1 | High     | Work    | [0.9, 0.1]                 |
| Positive 2 | High     | Weather | [0.8, 0.9]                 |
| Negative 1 | Low      | Work    | [0.1, 0.2]                 |
| Negative 2 | Low      | Weather | [0.0, 0.8]                 |

**If topic distribution is balanced**, the steering vector is:

```
v_pure = mean(pos) - mean(neg) = [0.80, 0.00]  ← Pure optimism!
```



**But if topic distribution is imbalanced** (e.g., 70% of positives are work-related, 70% of negatives are weather-related):

| Sample   | Optimism | Topic   | Activation    |
| :------- | :------- | :------ | :------------ |
| Pos 1-7  | High     | Work    | [0.9, 0.1] ×7 |
| Pos 8-10 | High     | Weather | [0.8, 0.9] ×3 |
| Neg 1-7  | Low      | Weather | [0.0, 0.8] ×7 |
| Neg 8-10 | Low      | Work    | [0.1, 0.2] ×3 |

```
pos_mean = [0.87, 0.34]
neg_mean = [0.03, 0.62]
v_contaminated = [0.84, -0.28]  ← Optimism + topic shift!
```



**Result**: The model becomes more optimistic, **but also shifts its topic distribution** — an unintended side effect.

------

## 🎯 Our Solution: Two Complementary Methods

### Method 1: Interference-Matched Pairing (IMP)

**Principle**: Balance interference dimensions across positive and negative samples **at the data construction stage**.

**When to use**: Interference is **discrete and labelable** (e.g., topic, person, tense).

**Mathematical guarantee**:

![image-20260704132501687](D:\AppData\Roaming\Typora\typora-user-images\image-20260704132501687.png)

### Method 2: Confound Projection Removal (CPR)

**Principle**: After extracting the steering vector, **project out** the interference direction in the activation space.

**When to use**: Interference is **continuous or hard to label** (e.g., style, length).

**Mathematical formulation**:

![image-20260704132533179](D:\AppData\Roaming\Typora\typora-user-images\image-20260704132533179.png)

### Combined Approach

**IMP + CPR** provides the best of both worlds: source-level prevention + vector-level purification.

------

**“Please Note:** The quantitative results reported in this README are **illustrative examples** based on preliminary experiments. Final numerical results may differ and will be updated upon completion of all experiments.”

## 📊 Key Experimental Findings

### Finding 1: Interference Types Have Different Impacts

| Interference Type | Alignment Drop vs. Clean Baseline |
| :---------------- | :-------------------------------- |
| Length            | -10%                              |
| Style             | -15%                              |
| Topic             | -20%                              |
| **Combined**      | **-30%**                          |

**Key insight**: Topic interference is the most harmful; combined interference is catastrophic.

### Finding 2: CLEAN-Steer Effectively Removes Interference

| Method           | Alignment Accuracy | Interference Shift | Purity Ratio |
| :--------------- | :----------------- | :----------------- | :----------- |
| Baseline         | 65%                | 0.55               | 1.18         |
| IMP (Matching)   | 82%                | 0.18               | 4.56         |
| CPR (Projection) | 84%                | 0.15               | 5.60         |
| **IMP + CPR**    | **88%**            | **0.08**           | **11.0**     |

### Finding 3: CLEAN-Steer is Orthogonal to Existing Methods

**2×2×2 factorial experiment (Sample × Extraction × Layer Selection)** :

| Sample          | Extraction | Layer Selection | Accuracy |
| :-------------- | :--------- | :-------------- | :------- |
| Standard        | MD         | LayerNav        | 85%      |
| Standard        | MD         | Top             | 82%      |
| Standard        | PCA        | LayerNav        | 80%      |
| Standard        | PCA        | Top             | 78%      |
| **CLEAN-Steer** | MD         | LayerNav        | **92%**  |
| **CLEAN-Steer** | MD         | Top             | 89%      |
| **CLEAN-Steer** | PCA        | LayerNav        | 88%      |
| **CLEAN-Steer** | PCA        | Top             | 86%      |

**Key insight**: CLEAN-Steer improves **all** extraction methods and **all** layer selection methods, with **maximum gain** when combined with LayerNavigator.

### Finding 4: CLEAN-Steer Shines in Low-Data Regimes

| Training Size | Baseline | CLEAN-Steer | Improvement |
| :------------ | :------- | :---------- | :---------- |
| 700 (Full)    | 65%      | 88%         | +23%        |
| 350 (50%)     | 58%      | 84%         | +26%        |
| 140 (20%)     | 48%      | 78%         | +30%        |
| 70 (10%)      | 40%      | 72%         | +32%        |

**Key insight**: The less data you have, the more valuable CLEAN-Steer becomes.

### Finding 5: CLEAN-Steer Generalizes Across Models and Tasks

| Model       | Baseline | CLEAN-Steer | Improvement |
| :---------- | :------- | :---------- | :---------- |
| Llama-3-8B  | 65%      | 88%         | +23%        |
| Qwen-2.5-7B | 63%      | 86%         | +23%        |
| Mistral-7B  | 62%      | 85%         | +23%        |

| Target Task    | Confounder | Baseline | CLEAN-Steer | Improvement |
| :------------- | :--------- | :------- | :---------- | :---------- |
| Optimism       | Topic      | 65%      | 88%         | +23%        |
| Optimism       | Style      | 62%      | 86%         | +24%        |
| Topic (Work)   | Sentiment  | 60%      | 85%         | +25%        |
| Style (Formal) | Topic      | 63%      | 87%         | +24%        |

------

## 🧩 Relationship to Existing Work

| Work               | Level           | What It Does                                 | How CLEAN-Steer Differs                            |
| :----------------- | :-------------- | :------------------------------------------- | :------------------------------------------------- |
| **LayerNavigator** | Layer Selection | Finds which layers to intervene on           | **Data preparation** — orthogonal                  |
| **CDPO**           | Algorithm       | Causal de-confounding at the algorithm level | **Data preprocessing** — can be used together      |
| **StTP / SKOP**    | Vector          | Projection during inference                  | **Pre-extraction purification** — different timing |
| **Tan et al.**     | Empirical       | Proves interference exists                   | **Provides a solution**                            |

**CLEAN-Steer's unique position**:

> "Existing work addresses interference **after** vector extraction (CDPO) or **during** inference (StTP). CLEAN-Steer prevents interference from entering the vector **before** extraction — a data-level approach that is **orthogonal and complementary** to all existing methods."