# LearnedFusion3D

# 📚 LearnedFeatureFusion3D

`LearnedFeatureFusion3D` is a multimodal neural network architecture that fuses 3D medical image features with structured metadata through learnable representations and flexible fusion strategies.

---

## 🧠 Architecture Overview

### 1. **Image Branch**
- Uses a 3D ResNet50 backbone to extract deep features from volumetric input (e.g., CT/MRI).
- Output features are projected to a 512-dimensional embedding via a fully connected layer with ReLU.
- Captures spatial and contextual information from the 3D image.

### 2. **Metadata Branch**
- Structured tabular metadata (e.g., age, gender, clinical scores) is passed through a two-layer MLP with ReLU activations.
- This transforms metadata into a 512-dimensional learned feature embedding.
- This branch is skipped when in `image_only` mode.

### 3. **Fusion Module**
- Combines image and metadata embeddings using one of the following strategies:
  - **`concat`**: Concatenates both 512-d embeddings → 1024-d input to classifier.
  - **`add`**: Elementwise addition of the two 512-d embeddings.
  - **`multiply`**: Elementwise multiplication of the two embeddings.
  - **`image_only`**: Only the image branch is used; metadata is ignored.

### 4. **Classifier**
- The fused feature vector is fed into a two-layer MLP classifier with ReLU and dropout.
- Produces a single logit output for binary classification.

---

## 📌 Fusion Modes Summary

| Mode        | Metadata Used | Fusion Method          |
|-------------|----------------|-------------------------|
| `image_only` | ❌              | Image features only     |
| `concat`     | ✅              | Concatenation           |
| `add`        | ✅              | Elementwise addition    |
| `multiply`   | ✅              | Elementwise multiplication |

---

# 🔥 Focal Loss

`FocalLoss` is used to handle class imbalance by reducing the contribution of easy examples and focusing training on hard ones.

## 📐 Formula

Given a binary classification with prediction probability $$p \in [0, 1] $$ and true label $$y \in [0, 1] $$, the Focal Loss is:

$$
\text{FL}(p, y) = -\alpha \cdot (1 - p_t)^\gamma \cdot \log(p_t)
$$

where:

$$
p_t = 
\begin{cases}
p & \text{if } y = 1 \\
1 - p & \text{if } y = 0
\end{cases}
$$

-  $$\alpha $$: Class balancing factor.
-  $$\gamma $$: Focusing parameter; larger values down-weight easy examples more.

## ✅ Features
- **Label smoothing**: Reduces overconfidence in predictions.
- **Dynamic weighting**: Computes $$\alpha$$ per batch based on label distribution.
- **Supports reduction modes**: `'none'`, `'mean'`, `'sum'`.

---

## 🧪 Use Case
This architecture and loss are particularly suited for **imbalanced classification tasks** in medical imaging, where:
- Visual features from 3D images are important;
- Clinical metadata improves prediction;
- The positive class (e.g. disease present) is rare.
