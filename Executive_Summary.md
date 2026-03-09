# Executive Summary: Implementing Asymmetric Cross-Modal Attention

**Student Project Proposal**

---

## Project Goal

This project implements and evaluates the framework proposed in *"Asymmetric Cross-Modal Attention for Multimodal Representation Learning"* (February 2026). The paper introduces a novel approach to combining different data modalities (e.g., images and text) by modeling **directional, asymmetric interactions** between them -- recognizing that the way an image informs text understanding is fundamentally different from how text informs image understanding.

The project will build a working implementation, train it on a Visual Question Answering (VQA) task, and rigorously compare it against four baselines -- including a deep-dive study of the legacy **Stacked Attention Network (SAN)** method -- to validate the paper's claims and contextualize its contributions within the field's evolution.

---

## Why This Matters

Current multimodal AI systems typically treat the interaction between data types as symmetric -- assuming images influence text the same way text influences images. This paper challenges that assumption. If validated, asymmetric attention could improve performance in critical applications like medical diagnosis (combining imaging with clinical notes), autonomous driving (combining cameras with depth sensors), and content moderation (combining images with text).

---

## System Architecture

```
 ┌───────────────────────────────────────────────────────────────────────┐
 │                                                                       │
 │   ┌──────────┐                                     ┌──────────┐      │
 │   │  Image    │                                     │  Text    │      │
 │   │  Input    │                                     │  Input   │      │
 │   │ (VQA img) │                                     │(Question)│      │
 │   └─────┬─────┘                                     └─────┬────┘      │
 │         │                                                 │           │
 │         ▼                                                 ▼           │
 │   ┌───────────┐                                   ┌───────────┐      │
 │   │  Image    │                                   │   Text    │      │
 │   │  Encoder  │                                   │  Encoder  │      │
 │   │(ViT/ResNet│                                   │(DistilBERT│      │
 │   │ pretrained)│                                  │ pretrained)│      │
 │   └─────┬─────┘                                   └─────┬─────┘      │
 │         │                                               │            │
 │         │  a ∈ R^(N×d)                   b ∈ R^(M×d)    │            │
 │         │                                               │            │
 │         │         ┌──────────────────────────┐          │            │
 │         ├────────►│  Cross-Attention Block 1 │◄─────────┤            │
 │         │         │  Q = a    K = b    V = b │          │            │
 │         │         │  "Image attends to Text" │          │            │
 │         │         └────────────┬─────────────┘          │            │
 │         │                     │ a←b                     │            │
 │         │                     │                         │            │
 │         │         ┌──────────────────────────┐          │            │
 │         ├────────►│  Cross-Attention Block 2 │◄─────────┤            │
 │         │         │  Q = b    K = a    V = a │          │            │
 │         │         │  "Text attends to Image" │          │            │
 │         │         └────────────┬─────────────┘          │            │
 │         │                     │ b←a                     │            │
 │         │                     │                         │            │
 │         │         ┌──────────────────────────┐          │            │
 │         │         │      Fusion Layer        │          │            │
 │         │         │    z = [a←b ; b←a]       │          │            │
 │         │         │     (concatenation)       │          │            │
 │         │         └────────────┬─────────────┘          │            │
 │         │                     │                         │            │
 │         │                     ▼                         │            │
 │         │         ┌──────────────────────────┐          │            │
 │         │         │   Answer Classifier      │          │            │
 │         │         │  MLP → Softmax → Top-K   │          │            │
 │         │         │     answer classes        │          │            │
 │         │         └──────────────────────────┘          │            │
 │                                                                      │
 └───────────────────────────────────────────────────────────────────────┘
```

The core innovation is the **two separate cross-attention blocks** operating in opposite directions. Block 1 lets the image representation selectively attend to the text (question), while Block 2 lets the text selectively attend to the image. These asymmetric, direction-specific interactions are then fused and passed to a classifier to predict the answer.

---

## Implementation Phases

| Phase | Duration | Description |
|-------|----------|-------------|
| **Phase 1: Foundation** | Weeks 1-2 | Environment setup, PyTorch fundamentals, data pipeline for VQA v2.0 dataset |
| **Phase 2: Baselines & Legacy Study** | Weeks 3-5 | Implement three simple baselines (early/late/symmetric fusion); study and implement the **Stacked Attention Network (SAN)**, a landmark 2016 VQA method, as a legacy comparison point |
| **Phase 3: Core Model** | Weeks 6-7 | Implement the asymmetric cross-modal attention framework with two directional cross-attention blocks |
| **Phase 4: Evaluation** | Weeks 8-9 | Full-scale training of all five models, quantitative comparison, ablation studies, and statistical analysis |
| **Phase 5: Visualization** | Weeks 10-11 | Attention heatmap generation, SAN vs. asymmetric side-by-side comparison, qualitative analysis |
| **Phase 6: Documentation** | Weeks 12-14 | Final report (including SAN study section), presentation slides, and optional interactive web demo |

---

## Expected Outcomes & Deliverables

1. **Working codebase** (Python/PyTorch) implementing the asymmetric cross-modal attention framework and four baselines (including SAN), hosted on GitHub
2. **Legacy method study** -- a 1-2 page analysis of Stacked Attention Networks: historical context, architecture, and limitations that motivate the paper's approach
3. **Quantitative comparison** across all five methods on VQA, measuring top-1 accuracy, top-5 accuracy, and per-category performance
4. **Attention visualizations** demonstrating asymmetric patterns, including side-by-side SAN vs. asymmetric attention maps on the same examples
5. **Ablation studies** testing model robustness under modality removal and noise injection
6. **Project report** (10-18 pages) and **presentation** (18-22 slides)
7. *(Optional)* Interactive Gradio web demo for live image + question input

---

## Resource Requirements

| Resource | Details | Cost |
|----------|---------|------|
| **Compute** | ~155 GPU hours total (Google Colab Free/Pro or Kaggle) | $0-25 |
| **Dataset** | VQA v2.0 (publicly available, Creative Commons license) | Free |
| **Software** | Python, PyTorch, HuggingFace Transformers, standard ML libraries | Free |
| **Time commitment** | 8-12 hours/week over 14 weeks | -- |

**Total estimated cost: $0-25** (primarily for optional Colab Pro GPU access during full training runs).

---

## Significance

This project bridges the gap between a theoretical research contribution and practical implementation. It provides:

- **Historical perspective** through the SAN legacy method study, showing how the field evolved from single-direction to asymmetric attention
- **Hands-on experience** with both classic and state-of-the-art multimodal deep learning
- **Rigorous experimental methodology** including five-way comparison, ablations, and statistical analysis
- **Reproducible research** with clean, documented code
- **Critical evaluation** of a recent paper's claims through independent implementation and comparison against an established method

The project is designed to be achievable for a student with AP Computer Science background while maintaining the rigor expected of research-level work.

---

*Estimated timeline: 14 weeks | Weekly commitment: 8-12 hours*
