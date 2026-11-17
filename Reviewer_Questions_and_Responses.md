# Response to Reviewers

These are proposed manuscript revisions, not statements that changes or new experiments have already been completed. Only this response document has been edited. Manuscript locations refer to the current `paper/sn-article.tex`; use the quoted anchors after line numbers shift. Preserve the original reviewer numbering. Comments exclusively about CLEVR or Flickr30k have been removed; mixed comments remain and their unsupported dataset-specific requests are identified explicitly.

Evidence shorthand used below:

- **F notebook:** `notebooks/train_evaluate_visualize_colab_frozen_5_gcp.ipynb`.
- **U notebook:** `notebooks/train_evaluate_visualize_colab_unfrozen_5_gcp_s7.ipynb`.
- **F metrics:** `outputs/results/frozen_encoder/metrics/`, with run prefix `clipB32_224_frozen_top3000_directional_parallel_v1_8d4983bd4bfe`.
- **U metrics:** `outputs/results/unfrozen_encoder/metrics/`, with run prefix `clipB32_224_fullft_top3000_directional_parallel_v1_ccb164279e25`.
- A suffix such as `*_s7_asymmetric_summary.json` identifies the exact file under the specified metrics directory and run prefix. Notebook cell indices are **zero-based JSON indices**, not execution counters. Both canonical notebooks use the same relevant cell indices unless stated otherwise.

The VQA predictions, configurations, and summaries describe `directional_parallel_v1`. Checkpoints under `artifacts/runs/clipB32_224_fullft_vqaScore_top3000_parammatched_sequential_s7/` belong to an older sequential experiment and cannot substitute for the current models. The proposed patches below form one coordinated revision: shared patches are referenced by reviewer item instead of being applied twice. Where new manuscript text is required in blue, use the existing `revision` environment around paragraph/block additions and `\rev{...}` for short prose changes; keep table environments and section commands outside text-color arguments.

## Reviewer #1

1. The asymmetric model applies one block per direction, whereas the symmetric control averages two independently parameterized blocks per direction. Thus, block applications and aggregation also differ. Add a conventional single-block shared baseline and an alternative parameter-matched control. Report exact parameter counts by component, FLOPs, training time, and peak memory.

## **Response:**

**Status: Partially addressable now.** The exact parameter counts, block applications, a scoped analytical FLOP count, observed run times, and allocated-memory peaks can be reported from the current code and saved histories. The requested new baseline results cannot be supplied from the saved parallel campaign: its active variants are only asymmetric and parameter-matched symmetric. A legacy single-shared-block class exists, but that is not a trained baseline result.

**Proposed revision — Section 3.4, after the paragraph ending “optimization dynamics” (lines 432–439):**

```latex
For VQA v2.0, both evaluated architectures contain exactly
6,306,816 fusion parameters, comprising two blocks of 3,153,408
parameters each. Each modality projection contains 393,728
parameters, and the answer classifier contains 2,063,800 parameters.
The projections, fusion module, and classifier therefore contain
9,158,072 parameters in total. These are the trained components in
the frozen setting; the pretrained encoders together contain
212,101,632 parameters. In the fully fine-tuned setting, the full
model contains 221,259,704 parameters, all marked trainable.
These counts apply to the VQA implementation.
The asymmetric model applies two cross-attention blocks per
example, one in each direction. The parameter-matched symmetric
model applies four blocks per example and averages two outputs
within each direction. Thus, this comparison matches learned
parameter counts but also changes block applications and output
aggregation; it does not isolate directional specialization alone.
```

**Proposed revision — Section 4.1, after Table 1 (after line 592):**

```latex
\paragraph{Observed VQA resource use.}
The following measurements come from the twelve recorded epochs
of each VQA run on an NVIDIA A100-SXM4-40GB. Time is the sum of
recorded epoch wall times, including training, validation-subset
evaluation, and checkpoint writing. It excludes preflight probes,
offline feature extraction, full-validation evaluation, and test
export. Memory is the maximum CUDA memory allocated by PyTorch
over the recorded epochs, in GiB; it is not total device memory
usage. Frozen runs train on cached encoder states with batch size
2,048; fine-tuned runs include both encoders with batch size 128.
\begin{center}
\small
\resizebox{\linewidth}{!}{%
\begin{tabular}{llrrrr}
\toprule
Setting & Seed & Sym. hours & Asym. hours & Sym. GiB & Asym. GiB \\
\midrule
Frozen & 7   & 1.092 & 1.105 & 7.858 & 4.492 \\
Frozen & 42  & 1.072 & 1.092 & 7.858 & 4.492 \\
Frozen & 123 & 0.302 & 0.164 & 7.858 & 4.492 \\
Fine-tuned & 7   & 1.879 & 1.800 & 7.107 & 6.894 \\
Fine-tuned & 42  & 1.850 & 1.815 & 7.107 & 6.894 \\
Fine-tuned & 123 & 2.342 & 2.433 & 7.107 & 6.894 \\
\bottomrule
\end{tabular}
}
\end{center}
These are observed run costs, rather than a controlled latency
benchmark. In particular, the large timing variation between
frozen runs prevents attributing their wall-time differences
solely to the fusion architecture.

\paragraph{Analytical fusion operation count.}
For one example with 50 image tokens, 20 text tokens, embedding
dimension 512, and feed-forward width 2,048, the asymmetric
fusion requires 222,248,960 multiply--accumulate operations for
its attention projections, attention matrix multiplications, and
feed-forward linear layers. Counting one multiply--accumulate as
two FLOPs gives 0.4445 GFLOPs. The parameter-matched symmetric
fusion repeats these operations for both shared blocks, giving
0.8890 GFLOPs under the same convention. These analytical counts
exclude encoders, modality projections, the classifier,
normalization, nonlinearities, softmax, residual additions, output
averaging, and backward computation; they are not end-to-end
training or inference FLOP measurements.
```

The analytical count is derived from the implemented block: for query/key lengths `q,k`, the counted MACs are `(10q + 2k)d^2 + 2qkd`; summing `(q,k)=(50,20),(20,50)` gives the asymmetric figure. Parameter counts were checked on CPU by instantiating the actual fusion and cached-feature VQA-head classes, without loading pretrained weights. The one-shared-block class has 3,153,408 fusion parameters and 6,004,664 projection/fusion/classifier parameters; do not add an accuracy row for it.

**Evidence:** F/U notebooks cells 1, 16, 18, and 26; F/U metrics `*_preflight.json`, all six `*_history.json` files per setting, and `*_summary.json`. The preflight fields `epoch_seconds` and `full_val_seconds` are extrapolated estimates from short probes; the table above deliberately uses recorded histories instead.

**Experiments needed when GPU access returns:** Train the conventional one-shared-block baseline and an alternative exactly parameter-matched control consisting of two shared blocks composed sequentially in each direction, rather than averaged in parallel. The latter has the same 6,306,816 fusion parameters and four block applications; it tests sensitivity to aggregation while introducing a depth difference that must be reported. Compare both with the existing two architectures on VQA using paired seeds 7, 42, and 123, the same encoders, head, splits, data order, training horizon, and validation selection rule. Start with frozen encoders; repeat under fine-tuning before making a claim about both settings. Measure steady-state forward and training-step latency after warmup, synchronized on the same GPU with identical batch sizes and precision; record allocated and reserved memory and separate encoder/feature-cache costs. Obtain full forward/backward FLOP profiles with explicit operator coverage. New controls and a clean runtime comparison remain outstanding even after the reporting patch above.

3. Dataset, loss, prediction head, training configuration, and sometimes backbone change across comparisons, preventing causal attribution to modality roles. Qualify this explanation consistently throughout the manuscript. Directional ablations, component-level sharing, gradient-conflict measurements, or controlled input perturbations would strengthen the evidence. Keep Figure 1 explicitly conceptual.

## **Response:**

**Status: Addressable now through claim calibration.** The experiments compare concrete architectures within each configuration. They do not manipulate modality roles independently of datasets, objectives, heads, and backbones, and they do not establish a learned mechanism. The following patches replace the strongest claims consistently; they do not change the reported accuracies.

**Proposed revision — replace the complete Abstract body (line 138, between `\abstract{` and its closing brace):**

```latex
Images and text can play distinct functional roles: a question
specifies what information is needed, while an image provides
evidence for answering it. Motivated by this distinction, we
empirically compare shared and direction-specific cross-modal
attention. All symmetric results use a parameter-matched control
with two independently parameterized blocks, each shared across
both directions, whose outputs are averaged within each direction.
The asymmetric architecture uses one independent block per
direction. The comparison matches fusion parameter counts but
differs in block applications and aggregation. We evaluate these
architectures on CLEVR and VQA v2.0 for question answering and
Flickr30k for image--text retrieval. The frozen question-answering
comparisons yield mean paired gains of 5.27 and 0.48 percentage
points on CLEVR and VQA v2.0, respectively. The fine-tuned VQA
gain is 0.23 percentage points. In the single frozen-encoder
Flickr30k run, the symmetric control exceeds asymmetric fusion
by 6.3 and 6.2 points in image-to-text and text-to-image Recall@1.
Retrieval uses a different training objective and configuration,
so this result is preliminary and does not isolate task effects.
Our results describe configuration-dependent differences between
the evaluated fusion designs. Distinct modality roles remain a
motivating hypothesis, rather than an experimentally established
explanation of these differences.
```

**Proposed revision — Section 1:**

1. Replace the paragraph beginning “The results reveal a task-dependent pattern” (line 192) with:

```latex
The evaluated configurations show different relative performances
for the two fusion designs. Asymmetric fusion improves the paired
question-answering results, while the symmetric control performs
better in the single Flickr30k retrieval run. These observations
motivate further investigation of directional parameter sharing,
but do not establish that modality roles cause the differences.
Across benchmarks, the dataset, objective, prediction head,
optimization configuration, and sometimes vision backbone also
change. Within each configuration, parameter matching controls
the number of learned fusion parameters, but block applications
and aggregation still differ between the architectures.
```

2. Replace the first active contribution, beginning “We develop a parameter-matched comparison” (line 204), with:

```latex
\item We empirically compare shared and direction-specific
cross-modal attention under an equal fusion parameter count.
We explicitly distinguish this parameter control from differences
in computation and output aggregation.
```

3. Replace the third active contribution, beginning “We provide evidence that directional parameter sharing should be chosen according to the task” (line 208), with:

```latex
\item We report configuration-dependent differences between the
two fusion designs, including paired question-answering results
and a preliminary single-run retrieval comparison. These results
motivate, but do not establish, an explanation based on distinct
modality roles.
```

4. Replace the final sentence of the study-objective paragraph, beginning “Matching the fusion parameter count allows us” (line 178), with:

```latex
Matching the fusion parameter count controls the number of
learned fusion parameters, but the comparison also differs in
block applications and averaging and therefore does not isolate
directional specialization as the cause of any improvement.
```

**Proposed revision — Section 7, replace the results paragraph beginning “The results show that the benefits depend on the task and dataset” (line 1226) and the final paragraph (line 1228):**

```latex
Under the evaluated frozen configurations, asymmetric fusion
improves CLEVR accuracy by 5.27 percentage points and VQA v2.0
accuracy by 0.48 points over the parameter-matched symmetric
control. The mean fine-tuned VQA gain is 0.23 points. In the
single Flickr30k retrieval run, the symmetric control achieves
higher Recall@1 in both directions. Because the benchmarks
differ in their data, objectives, heads, training configurations,
and sometimes backbones, these results do not establish that
task characteristics or distinct modality roles cause the
observed differences. Parameter matching also leaves differences
in block applications and aggregation.

Directional parameter sharing is therefore an architectural
choice to evaluate under a specified experimental configuration.
The functional-role account remains a motivating hypothesis.
Future work should test it through controlled directional
ablations and input interventions, together with alternative
shared controls and direct analysis of the learned interactions.
```

Keep Figure 1 explicitly conceptual using the caption under R1 additional comment e. Use the revised Section 5.5 wording under R1 additional comment b and the complete limitations replacement under R1 additional comment f; these close the corresponding claims outside the Abstract and Introduction.

**Evidence:** Current Sections 3.4, 4.1/Table 1, 5.1.2, and 5.4; F/U metrics `*_config.json` and `*_final_training_report.json`; F/U notebooks cell 16. Cross-benchmark facts above are qualifications of the manuscript's reported protocols, not independently reproduced CLEVR or Flickr30k results.

**Experiments needed only to make a stronger mechanistic claim:** On one fixed VQA backbone/data/objective configuration, train directional ablations and component-sharing variants using paired seeds and the same selection rule. Compare sharing the attention projections versus sharing the feed-forward sublayers, report the resulting parameter/compute differences, and measure the cosine similarity of the two directional loss gradients on shared parameters before summation. At evaluation, compare matched pairs with shuffled and controlled counterfactual pairs. Attention plots can describe associations but cannot alone establish a causal functional-role explanation. None of these experiments is required merely to apply the conservative text revisions above.

5. VQA: The reported gains of 0.48 ± 0.19 and 0.23 ± 0.16 percentage points require per-seed results, confidence intervals for paired differences, and computational-cost comparisons. Clearly distinguish standard deviations from confidence intervals.

## **Response:**

**Status: Addressable now for statistics and observed costs; controlled runtime profiling and new controls remain pending.** All twelve saved full-validation prediction arrays reproduce the corresponding summary accuracies within numerical precision. The six saved image-cluster bootstrap intervals also reproduce on CPU using 2,000 draws and RNG seed 2026.

**Proposed revision — Section 5.1.2, insert after the mean-gain sentence ending “fine-tuned encoders” (lines 793–795), before Table 3:**

```latex
The per-seed full-validation results are shown below. Accuracies
are percentages, gains and interval endpoints are percentage
points, and Sym. denotes the parameter-matched symmetric control.
Each 95\% interval resamples validation images with replacement,
retaining all questions belonging to each sampled image and
pairing the two models on the same resampled observations. We
use 2,000 bootstrap draws and random seed 2026. These intervals
measure evaluation uncertainty conditional on each trained
checkpoint pair, rather than uncertainty across training runs.
\begin{center}
\small
\resizebox{\linewidth}{!}{%
\begin{tabular}{llrrrl}
\toprule
Setting & Seed & Sym. & Asym. & Gain & Conditional 95\% CI \\
\midrule
Frozen & 7   & 59.4985 & 59.8632 & +0.3647 & [0.2232, 0.5097] \\
Frozen & 42  & 59.2118 & 59.9083 & +0.6965 & [0.5490, 0.8562] \\
Frozen & 123 & 59.1529 & 59.5384 & +0.3855 & [0.2244, 0.5439] \\
Fine-tuned & 7   & 65.3767 & 65.4981 & +0.1215 & [-0.0084, 0.2494] \\
Fine-tuned & 42  & 65.2046 & 65.6160 & +0.4114 & [0.2786, 0.5447] \\
Fine-tuned & 123 & 65.3117 & 65.4786 & +0.1669 & [0.0346, 0.3024] \\
\bottomrule
\end{tabular}
}
\end{center}
The paired gain is positive for every evaluated seed in both
encoder settings. Nevertheless, the fine-tuned seed-7 conditional
interval includes zero. The reported $0.48 \pm 0.19$ and
$0.23 \pm 0.16$ summarize the mean and sample standard deviation
of the three seedwise gains, not confidence intervals.
For a separate, exploratory estimate of training-seed uncertainty,
a Student-$t$ interval for the mean paired gain is [0.021, 0.944]
points for frozen encoders and [-0.154, 0.621] points for
fine-tuned encoders. These intervals use two degrees of freedom
and assume normally distributed seedwise differences. With only
three seeds, that assumption cannot be assessed reliably, so
the intervals should be interpreted cautiously and should not be
confused with the conditional image-bootstrap intervals.
```

The seed-level calculation is `mean(delta) ± t(0.975, df=2) × sample_SD(delta)/sqrt(3)`, with critical value 4.30265273. It uses the unrounded paired gains, not the displayed two-decimal headline values. Apply the parameter, observed-resource, and analytical-FLOP additions under R1–1 for the cost comparison.

**Evidence:** F/U metrics `*_final_training_report.json`, all twelve `*_s{seed}_{variant}_val_predictions.npz` arrays and associated summaries; F/U notebooks cell 38 (`paired_image_bootstrap`) and cell 26 (training/evaluation accounting). The arrays contain scores and identifiers, enabling this CPU analysis without model inference.

**Remaining GPU work:** More paired training seeds would improve estimation of training-run variability. Controlled runtime measurements and the additional baseline comparisons remain those specified under R1–1. No new training or GPU is needed to add the verified statistics above.

6. Table 4 presents selected checkpoints, with different seeds in the frozen comparison, rather than replicated paired results. Clarify validation and checkpoint-selection procedures for every benchmark, including any additional training before VQA test submission.

## **Response:**

**Status: Partially addressable now.** The VQA procedure is recoverable in full, including selection before full-validation scoring and test export. The current manuscript describes Flickr30k selection, but the underlying retrieval and CLEVR run records are not present alongside the canonical VQA records, so their provenance cannot be verified to the same standard.

**Proposed revision — Section 5.1.2, insert immediately after the paragraph defining the three paired seeds (after line 784):**

```latex
For each VQA architecture and seed, we train for twelve epochs
and select the epoch with the highest official VQA accuracy on
a fixed, unfiltered 10,000-question validation subset sampled
without replacement using seed 2026. The subset is identical
across architectures and training seeds. Ties within a run retain
the earlier epoch. We then evaluate that selected checkpoint on
all 214,354 validation questions; these full-validation scores do
not select the epoch. Table~\ref{tab:vqa_answer_prediction}
summarizes these selected-checkpoint scores, rather than
necessarily the final-epoch scores. The selected epochs are:
\begin{center}
\small
\begin{tabular}{llrr}
\toprule
Setting & Seed & Sym. epoch & Asym. epoch \\
\midrule
Frozen & 7   & 12 & 12 \\
Frozen & 42  & 12 & 12 \\
Frozen & 123 & 12 & 11 \\
Fine-tuned & 7   & 11 & 12 \\
Fine-tuned & 42  & 9  & 12 \\
Fine-tuned & 123 & 10 & 11 \\
\bottomrule
\end{tabular}
\end{center}
```

**Proposed revision — replace the Hidden-test Results paragraph, beginning “Table~\ref{tab:vqa_test} reports results obtained through” (lines 828–838):**

```latex
Table~\ref{tab:vqa_test} reports official EvalAI scores for
individual validation-selected checkpoints. For each architecture
and encoder setting, we select the seed with the highest
validation-subset accuracy at its best epoch; ties prefer the
lower seed and then the earlier epoch. The selected frozen
symmetric and asymmetric checkpoints use seeds 7 and 42,
respectively, both at epoch 12. The fine-tuned checkpoints both
use seed 42, at epochs 9 and 12 for the symmetric and asymmetric
models, respectively. We export predictions directly from these
checkpoints without additional training or train--validation
retraining. Each export contains one answer for each of the
447,793 test2015 questions and is scored using the official
answer-normalization and evaluation procedure. These are selected
individual runs, not replicated paired test-set estimates; the
frozen comparison additionally uses different seeds.
```

**Evidence:** F/U notebooks cell 23 (subset), cell 26 (`finish_run`, strict epoch improvement, and `select_campaign_checkpoints`), cells 32 and 73 (test dataset/export; in executed notebooks the export code is cell 74). F/U metrics per-seed summaries and `*_val_subset_indices.json`; all four metadata files under `outputs/evalai_submissions/{frozen_encoder,unfrozen_encoder}/`. Their epochs, seeds, fingerprints, and export hashes agree with this account. Test-standard answer-type results inherit the same selected-checkpoint scope.

**Not supportable yet:** For Table 2, recover the original CLEVR training scripts, checkpoint metadata, and epoch histories before stating “best” or “final.” Section 5.4 already says that retrieval checkpoints maximize the mean of the two Recall@1 values on the fixed 50-image subset; this is a reported protocol, not an independent verification from local run files. Obtain those source records before adding exact selected epochs or stronger procedural claims. Recovering existing records needs no GPU. If records/checkpoints cannot be recovered, a documented rerun is needed when GPU access returns; do not invent a common checkpoint rule for all three benchmarks.

7. Evaluation with missing inputs demonstrates input dependence but does not establish cross-modal reasoning or directional specialization. Specify how each modality is removed and add separately trained unimodal baselines, shuffled image–question pairs, or controlled counterfactuals.

## **Response:**

**Status: The implementation details and interpretation can be corrected now. Separately trained unimodal and shuffled/counterfactual results cannot be supplied from the current campaign.**

**Proposed revision — Section 5.2, replace the opening paragraph beginning “The second objective is to examine” (lines 889–898):**

```latex
The second objective is to measure sensitivity to the available
inputs using trained checkpoints, without retraining. The VQA
interventions differ between encoder settings. In frozen runs,
image blindness zeros the cached, unprojected image-token states
before the trainable image projection; text blindness zeros the
cached, unprojected text-token states before the text projection.
In fine-tuned runs, image blindness instead zeros the normalized
image tensor before the CLIP encoder, and text blindness replaces
every question token ID with the RoBERTa padding ID, 1, before
the text encoder. In both VQA text-blind settings, the text
attention mask is set to all ones, avoiding an all-masked
cross-attention softmax. These replacements remove the original
question content, but do not force every downstream activation
to zero: projection biases and, in the fine-tuned setting,
encoder transformations remain active. Full-input and ablated
scores use the same fixed 10,000-question validation subset.
These out-of-distribution interventions measure input dependence;
they do not establish cross-modal reasoning, learned directional
specialization, or the accuracy of separately trained unimodal
models. The implementation details in this paragraph apply to
VQA v2.0; the CLEVR intervention requires its own source-code
verification.
```

**Evidence:** F/U notebooks cell 57 (`evaluate_ablation`); cells 14 and 18 distinguish encoder input tensors, cached states, trainable projections, and pooling. In particular, the existing sentence claiming that all image-blind conditions zero “image features” is not an accurate description of fine-tuned VQA. The saved executed notebooks reproduce the displayed VQA ablation scores.

**Experiments needed when GPU access returns:** Train question-only and image-only VQA baselines with the same answer vocabulary, target construction, training data, horizon, selection subset, and seeds 7/42/123. Use each appropriate encoder and a comparable pooled classifier; report exact parameters instead of assuming equal capacity. For the current multimodal architectures, recover the matching parallel checkpoints and evaluate fixed derangements of validation image--question pairs, excluding pairings with the same image. Also evaluate a documented set of minimally changed question/image pairs with known expected answers. Report full-input versus perturbed accuracy with paired image-cluster intervals. Shuffling and counterfactual evaluation require inference, not retraining; they could run on CPU if the correct checkpoints and data were available. A GPU is primarily needed for practical throughput and new unimodal training. The older sequential checkpoints cannot establish these results for the paper's current comparison.

8. Frame the contribution as an empirical study of directional parameter sharing and clarify its distinction from prior co-attention and asymmetric fusion work. Specify pretrained checkpoints, pooling, masking, model dimensions, VQA targets and loss, out-of-vocabulary answer handling, retrieval loss, and batch sampling. Report question-truncation rates, assess longer CLEVR inputs, and provide code alongside a self-contained methodology.

## **Response:**

**Status: Partially addressable now.** VQA methods and empirical positioning can be made self-contained from the current implementation. Retrieval sampling details, ViT-B/16 pretraining provenance, and measured truncation rates are not established by the local VQA artifacts. Do not fill those gaps with assumptions.

**Proposed revision — Section 4, replace the first encoder paragraph (lines 465–474):**

```latex
For VQA v2.0, the vision checkpoint is
\texttt{openai/clip-vit-base-patch32}, loaded with
\texttt{CLIPVisionModel}, and the text checkpoint is
\texttt{roberta-base}, loaded with \texttt{RobertaModel} and its
RoBERTa tokenizer. We use the encoders' last hidden states,
rather than CLIP's global projected embedding. At resolution
$224\times224$, the vision encoder returns 49 patch tokens
and one class token. Both encoder hidden dimensions are 768,
and separate learned linear maps project them to 512 dimensions.
Questions are padded or truncated to 20 tokenizer positions,
including the tokenizer's special tokens. For CLEVR and Flickr30k,
the reported vision architecture is ViT-B/16, producing 196 patch
tokens and one class token at the same resolution. The reported
text architecture is RoBERTa-base.
```

The ViT-B/16 sentences retain the manuscript's reported architecture and token count without guessing a pretraining dataset. The requested exact identifier and pretraining source remain unresolved in this response; supply them only when the original records establish them.

**Proposed revision — Section 4, insert after the cross-attention-block paragraph (after line 491):**

```latex
In the VQA implementation, each direction is updated once from
the original projected encoder states; the two updates are
parallel, with no additional stacked fusion rounds. Each block
uses three LayerNorm modules, eight-head cross-attention at
dimension 512, and a feed-forward network
$512\rightarrow2048\rightarrow512$ with GELU, dropout, and
residual connections. The symmetric control averages two such
shared blocks within each direction; it does not stack them.
Text padding positions are excluded as keys/values when image
tokens query text. In the reverse direction all visual tokens
are available as keys/values. Text padding positions are excluded
from the final text pooling. The VQA classifier is
$1024\rightarrow512\rightarrow3000$, with ReLU and dropout 0.1
between the two linear layers. The output is a vector of answer
logits; the highest-scoring class supplies the predicted answer.
```

**Proposed revision — Section 3.5, insert after the pooling equation (after line 449):**

```latex
For VQA v2.0, image pooling is the arithmetic mean of the
49 updated spatial tokens, excluding the class token; the class
token remains present during cross-attention. Text pooling is a
masked mean over non-padding positions, including the tokenizer's
special tokens. If $m_j$ denotes the tokenizer attention mask,
then $\mathbf r^B=\sum_j m_j\widehat{\mathbf Z}^B_j/
\max(1,\sum_jm_j)$. The two 512-dimensional pooled vectors are
concatenated. These pooling details describe the verified VQA
implementation; they are not assumed for the other benchmarks.
```

**Proposed revision — Section 4, insert after the prediction-head paragraph (after line 496):**

```latex
\paragraph{VQA targets and sampling.}
The answer vocabulary consists of the 3,000 most frequent
normalized \texttt{multiple\_choice\_answer} labels in the training
annotations. For each training question and in-vocabulary answer
$a$, the target is $\min(c(a)/3,1)$, where $c(a)$ counts its
occurrences among the ten normalized annotator answers. We train
with binary cross-entropy with logits, averaged over batch and
answer dimensions. Training questions with no in-vocabulary
annotator answer are excluded. Validation retains every question,
including those whose answers are all out of vocabulary, and uses
the official leave-one-annotator-out VQA score rather than the
training target. Predictions are restricted to the training
vocabulary; an all-out-of-vocabulary validation question therefore
receives zero credit. Training examples are shuffled each epoch
using a dedicated generator seeded by the paired training seed
plus $1{,}000{,}003$ times the epoch number, matching data order
between the two architectures. We use AdamW with the learning
rates and batch sizes in Table~\ref{tab:hparams}, weight decay
0.01, gradient clipping at norm 1.0, and mixed precision. The
frozen setting uses no learning-rate scheduler. The fine-tuned
setting uses cosine decay with warmup for one epoch, capped at
10\% of total optimization steps. Both settings train for twelve
epochs and select checkpoints using the fixed validation subset.
```

Use the empirical framing under R1–3 and the verified related-work replacement under R2 minor c. The existing implementation-code URL already appears in the Introduction and Data availability; keep it and point readers to the two canonical notebooks above, rather than treating the archived notebooks as interchangeable reproductions.

**Evidence:** F/U notebooks cells 6, 8 (`build_answer_vocab`), 11 (`VQADataset`), 14, 16, 18, 23, and 26; F/U metrics `*_config.json`, the answer-vocabulary JSONs under each setting's `checkpoints/`, and the existing code-availability statements. These support VQA-specific descriptions; generic pooling/backbone claims for all benchmarks would overreach.

**Missing work and dependencies:**

- **No GPU needed:** Recover original CLEVR/Flickr30k configs and source code to identify the ViT-B/16 checkpoint, pooling/masking, fusion depth, retrieval loss reduction, how captions are sampled, and how multiple captions of the same image are treated in a batch. The manuscript currently states a bidirectional multi-positive contrastive loss with temperature 0.05; without the retrieval implementation, that statement is not enough to reconstruct its exact formula or sampler. Do not invent either.
- **No GPU needed once raw questions are restored:** Tokenize the full train/validation questions without truncation, with the exact tokenizer and special tokens, and report the number and fraction exceeding 20 positions, plus length distributions. The saved prediction arrays contain identifiers and scores, not question text, so they cannot establish a corpus truncation rate.
- **GPU inference or retraining:** Once the correct checkpoints/data are recovered, evaluate longer question limits (for example 40 positions) under both architectures, distinguishing inference-only sensitivity from retraining with the longer limit. Re-extract frozen text features for the new length. To claim a trained length effect, retrain paired seeds 7/42/123 with all other choices fixed. Longer CLEVR-input assessment specifically requires the missing CLEVR code/checkpoints. Until these steps are complete, do not claim that truncation is rare or harmless.

### Additional comments

b) Section 5.5 provides qualitative context rather than a direct baseline experiment. Rename the section to reflect its actual content.

## **Response:**

**Status: Fully addressable now.** This section supplies context and contains no direct large-model baseline experiment.

**Proposed revision — Section 5.5, replace the subsection heading (line 1198):**

```latex
\subsection{Objective 5: Relation to Large-Scale Multimodal Models}
```

**Proposed revision — replace the section's final sentence, beginning “By controlling the encoders” (line 1200):**

```latex
Our experiments compare shared and direction-specific fusion
under specified encoder, training, and parameter-budget
configurations. They neither benchmark against these large-scale
systems nor isolate modality roles as the cause of the observed
performance differences.
```

**Evidence:** The complete current Section 5.5 is a literature/context paragraph; its text and tables contain no new measurements against CLIP, Flamingo, BLIP-2, or LLaVA.

**Remaining work:** None for this revision; no GPU experiment is needed to rename and accurately describe the section. A future direct comparison would require a separate matched evaluation protocol, but is not required by this comment.

d) Define symmetric consistently as directional parameter sharing. Shared parameters do not imply identical attention maps or invariance to swapping modalities.

## **Response:**

**Status: Fully addressable now.** Define symmetry at the fusion-parameter level and avoid extending it to the complete heterogeneous vision--language model.

**Proposed revision — Section 1, insert after the definition paragraph beginning “An important architectural choice” (after line 159):**

```latex
Throughout this paper, symmetric refers only to sharing learned
cross-attention-block parameters between directions. It does not
require identical attention maps or outputs: the query and
key--value sequences differ between directions. Nor does this
terminology assert that the complete model is invariant to
swapping images and text, which use different encoders, masks,
and pooling operations.
```

**Proposed revision — Section 3.4, replace the two sentences beginning “Parameter sharing does not imply identical attention weights” (lines 427–431):**

```latex
The shared parameter sets constrain the transformations used
in both directions, not the attention maps or their outputs.
Those maps depend on the direction-specific query and key--value
inputs and can have different shapes. This design matches the
asymmetric fusion parameter count while retaining directional
parameter sharing; no modality-swap invariance of the full
vision--language model is assumed.
```

**Evidence:** F/U notebooks cell 16 uses the same `shared_experts` in both directions while reversing query/key-value inputs, and cell 18 applies different image and text pooling. A synthetic equal-shape swap check in cell 1 concerns the isolated fusion module under controlled inputs; it does not establish full-model invariance.

**Remaining work:** None; no GPU is required.

e) The medical example in Figure 1 is outside the evaluated domains. Consider a benchmark-aligned example and make clear that the illustration is conceptual.

## **Response:**

**Status: Addressable now through a caption revision.** A medical illustration can remain as a clearly labeled conceptual example without implying medical evaluation. Replacing it with an actual benchmark example is optional and requires a verified source image/question; it should not be fabricated from the illustration.

**Proposed revision — replace the complete Figure 1 caption, beginning “Using a medical visual question answering” (lines 167–168):**

```latex
\caption{\footnotesize Conceptual illustration of the hypothesis
that opposite cross-attention directions can support different
interactions. Throughout the paper, $A=I$ denotes image features
and $B=T$ denotes text features; $A\rightarrow B$ means that $A$
queries $B$. Thus, text querying image is $T\rightarrow I$
($B\rightarrow A$), and image querying text is $I\rightarrow T$
($A\rightarrow B$). A question can guide retrieval of visual
evidence, while visual evidence can condition text
representations. The medical scene and highlighted regions are
illustrative, not outputs or attention maps from the evaluated
models. No medical dataset is evaluated, and the depicted
functional roles are a motivating hypothesis rather than a
finding of this study.}
```

**Additional figure correction:** The embedded image itself currently reverses the A/B labels. Apply the exact label corrections under R2 major 5 when editing the figure asset. A caption alone must not be described as having corrected the raster labels.

**Evidence:** `paper/medical-example.png` was visually inspected; Section 4.2 lists the evaluated benchmarks, none medical. The present asset is a schematic, not evidence from the saved VQA runs.

**Remaining work:** No GPU is needed for the caption or conceptual artwork corrections. If a benchmark-aligned replacement is desired, use an identified VQA question and its corresponding image with a schematic highlighting explicitly labeled as illustrative. Actual model overlays would additionally require matching checkpoint/data provenance and inference; those are not necessary to satisfy the conceptual-example clarification.

f) Expand the limitations section to cover the two-block averaging control, cross-benchmark confounding, the small retrieval selection subset, and single-seed complementary analyses.

## **Response:**

**Status: Addressable now as a limitations revision.** The relevant control, evaluation, and cross-benchmark limitations can be stated without new experiments. This also supplies the shared limitations patch for R2 minor g.

**Proposed revision — replace Section 6's heading and all its prose, from `\section{Limitations and Mitigation Strategies}` through the paragraph ending “ablation analyses” (lines 1203–1221); remove the obsolete commented-out draft inside that span as well:**

```latex
\section{Limitations}

The parameter-matched symmetric control averages two shared
blocks in each direction, whereas the asymmetric model applies
one dedicated block per direction. The comparison matches learned
fusion parameter counts but also changes block applications and
aggregation. Equal parameter counts therefore do not establish
equal computation, effective capacity, or optimization behavior.
Additional shared controls and controlled resource measurements
are needed to separate these effects.

Across benchmarks, the data, learning objective, prediction head,
training configuration, and sometimes vision backbone differ.
Consequently, the relative results cannot be attributed solely
to task characteristics or modality roles. The functional-role
interpretation remains a motivating hypothesis. No systematic
analysis of the two directional attention maps or causal
interventions establishes distinct learned roles. Modality-blind
evaluation applies out-of-distribution input replacements to
trained models and measures input dependence, rather than the
performance of separately trained unimodal systems or proof of
cross-modal reasoning.

The primary VQA validation comparison uses three paired seeds,
which provides only a limited estimate of training-run variation.
Image-bootstrap intervals are conditional on the fitted
checkpoint pairs and do not replace seed-level uncertainty.
The hidden-test and modality-blind results use selected individual
checkpoints; the frozen VQA comparisons use different seeds for
the two architectures. Complementary results therefore do not
establish replicated architecture-level or category-level effects.

CLEVR uses synthetic images and controlled language, limiting
generalization to natural scenes. Its fine-tuned comparison uses
only two paired seeds, and the large gain mainly reflects lower
symmetric-model accuracy after fine-tuning; its magnitude should
not be treated as evidence that directional specialization has
been isolated. The cross-benchmark results also do not isolate
the effects of backbone and pretraining choices. The Flickr30k
result is based on one
frozen-encoder run per architecture under a different objective,
with checkpoint selection on only 50 validation images. This
limited selection subset and the absence of replicated retrieval
runs restrict conclusions about architecture preference. Corpus
truncation rates and sensitivity to longer inputs also remain
unmeasured in the available records.
```

**Evidence:** Section 3.4's two-block averaging definition; Table 1's different configurations; the seed counts and selection rules reported in Sections 5.1 and 5.4; F/U metrics `*_final_training_report.json`; export metadata; VQA ablation provenance. The CLEVR fine-tuning pattern follows the current Table 2, not a newly reproduced experiment. Source provenance is explicitly marked as incomplete rather than guessed.

**Remaining experiments:** R1–1 specifies alternative shared controls and profiling; R1–3 specifies controlled mechanism tests; R1–7 specifies unimodal/shuffled/counterfactual comparisons; R1–8 separates CPU truncation analysis and missing configurations from longer-input GPU work. For the mixed benchmark limitations retained here, additional paired retrieval runs and a larger fixed validation selection subset would test the reported retrieval preference, and recovering CLEVR training curves/configs should precede any fine-tuning rerun. No GPU access is required to add this honest limitations text now.

## Reviewer #2

### Major Issues

4. The functional-role interpretation is not examined directly. The Abstract states that asymmetric fusion is more effective “when modalities serve distinct functional roles.” Figure 1 and Section 1 build the motivation around these roles. Section 5.1.1 correctly notes that “the performance comparison alone does not establish that distinct functional roles have emerged.” Section 5.4 also offers its account of the retrieval result as one possible explanation. The architecture already produces attention maps for both directions (Section 3.3 and Figure 2). However, these maps are not analyzed. Please consider adding a short qualitative analysis of the two directional maps for a few CLEVR or VQA v2.0 examples, using existing checkpoints. Alternatively, please present the functional-role account in the Abstract and Conclusion as a motivating hypothesis rather than as a finding. Either option would be acceptable.

## **Response:**

**Status: Fully addressable now using the reviewer's stated alternative: present functional roles as a motivating hypothesis.** Current performance comparisons and the existence of attention-map outputs do not directly test that hypothesis. Saved qualitative figures do not, by themselves, establish a systematic analysis or causal interpretation.

**Proposed revision — Section 3.3, insert immediately after the paragraph ending “the resulting fused information” (after line 389):**

```latex
The availability of directional attention maps does not itself
establish functional specialization. We do not systematically
test whether the two maps encode distinct task roles, and the
reported accuracy differences do not identify that mechanism.
The functional-role account motivates the architectural
comparison rather than constituting a demonstrated result.
```

Apply the complete Abstract and Conclusion replacements under R1–3, which explicitly end the former with “Distinct modality roles remain a motivating hypothesis” and state the same scope in the Conclusion. Use the conceptual Figure 1 caption under R1 additional e and the complete Section 6 replacement under R1 additional f. These are coordinated patches, not additional alternate versions of those sections.

**Evidence:** F/U notebooks cell 16 can return directional maps, but active training/evaluation normally requests no attention weights. `return_attention=True` and the presence of visualization cells show a capability, not evidence that role specialization was established. The current manuscript offers the role interpretation without a systematic map analysis.

**Remaining work:** None is required to satisfy this comment using the hypothesis wording. An optional descriptive analysis would recover the exact selected parallel checkpoints, choose a fixed documented set of VQA examples, show both directional maps with token/patch aggregation and padding treatment stated, and include failures as well as successes. Inference would be practical on a GPU, but would not require retraining or by itself prove a causal mechanism. The mismatched older sequential checkpoints must not be used as evidence for the current architecture.

5. The notation for the two directions is inconsistent between figures. Section 3.3 defines A → B as modality A querying modality B. Figure 2 and Section 4 assign image features to ZA and text features to ZB. Under these definitions, A → B is image-to-text interaction. Figure 1, however, labels text-to-image interaction as A → B and image-to-text interaction as B → A. Since the paper is about the roles of the two directions, this mismatch may confuse readers. Please align the labels in Figure 1 with the notation used in Section 3 and Figure 2.

## **Response:**

**Status: Partially addressable by a manuscript line patch; the full correction also requires editing the figure asset.** The caption can be corrected now, but the reversed A/B labels are embedded in `paper/medical-example.png` and will remain wrong unless the artwork is edited. No GPU experiment is involved.

**Proposed revision — Section 3.2, insert immediately after the sentence introducing inputs from modalities A and B (after line 295):**

```latex
In the vision--language experiments and figures, $A=I$ denotes
the image modality and $B=T$ the text modality. As defined below,
the arrow denotes query-to-key/value direction: $A\rightarrow B$
is image querying text and $B\rightarrow A$ is text querying image.
```

Use the complete Figure 1 caption under R1 additional e, which already states the same convention. **Exact artwork replacements:**

- Left header: replace `Text-to-Image (A → B)` with `Text-to-Image (B → A)`.
- Right header: replace `Image-to-Text (B → A)` with `Image-to-Text (A → B)`.
- Keep the corresponding `T → I` and `I → T` block labels; those already match the query convention. Keep Section 3.3's existing definition that A queries B and the image/text assignment in the method figures.

**Evidence:** Visual inspection of `paper/medical-example.png`; Section 3.3 lines 316–331; Section 4's projected-image/text assignment and F/U notebooks cell 16. The issue is an inconsistent label, not an experiment requiring rerunning.

**Remaining work:** Apply the two artwork changes and visually verify the resulting Figure 1 when editing the manuscript/figure assets. This response-only task does not modify either asset or manuscript. A caption-only change should be described as a clarification, not completion of the embedded-label correction.

6. Some details needed to interpret the comparison are missing. Please state which epoch is reported in Tables 2 and 3, the final epoch or the best epoch on validation data. Section 5.1.2 describes epoch selection only for the hidden-test submission. Please report the number of fusion parameters in each architecture. Section 3.4 describes the budgets as comparable but gives no figures. The symmetric control runs four block passes per example instead of two, so training or inference time would also be useful. Please state the pretraining source of the ViT-B/16 encoder used for CLEVR and Flickr30k. Section 4 specifies CLIP only for the ViT-B/32 encoder used for VQA v2.0. Please define the pooling function in Section 3.5 and state the number of stacked fusion layers. Please explain how the text-blind condition is implemented. Section 5.2 states that question features are withheld from the fusion module, but it is not clear what replaces the text inputs in each block. Please give the seeds of the individual runs in Tables 5 to 7. Finally, please state whether the VQA v2.0 gain is positive for each of the three seeds, as is done for CLEVR in Section 5.1.1.

## **Response:**

**Status: Partially addressable now.** Every VQA-specific item can be grounded in the canonical code and saved results. The exact ViT-B/16 pretraining source, CLEVR epoch-selection procedure, and provenance of its individual diagnostic runs remain unverified without original records.

**Proposed revision — Table 3 caption (`tab:vqa_answer_prediction`, lines 800–804), replace its opening two sentences:**

```latex
VQA v2.0 accuracy (\%) on all 214,354 validation questions for
checkpoints selected by the highest accuracy on the fixed
10,000-question validation subset, rather than necessarily the
final epoch. Results are mean $\pm$ sample standard deviation
across paired seeds 7, 42, and 123.
```

Keep the existing caption's subsequent definition of Sym. and Gain. Apply the selected-epoch table and checkpoint-selection text under R1–6; the exact component counts and resource additions under R1–1; pooling and fusion-depth definitions under R1–8; and the complete modality-removal description under R1–7. Those are the definitive patches for these overlapping requests.

**Proposed revision — Table 6 caption (`tab:vqa_modality_ablation`, lines 963–965), replace the full caption:**

```latex
\caption{VQA v2.0 modality-blind accuracy (\%) on the fixed
10,000-question validation subset for selected individual
checkpoints. The frozen symmetric and asymmetric models use
seeds 7 and 42, respectively, both at epoch 12; the fine-tuned
models both use seed 42, at epochs 9 and 12, respectively.
Results are not averages across seeds. Sym. denotes the
parameter-matched symmetric control.}
```

**Proposed revision — Table 8 caption (`tab:vqa_answer_types`, lines 1061–1063), replace the full caption:**

```latex
\caption{VQA v2.0 test-standard accuracy (\%) by answer type for
the selected checkpoints in Table~\ref{tab:vqa_test}. The frozen
symmetric and asymmetric models use seeds 7 and 42, respectively;
both fine-tuned models use seed 42. Sym. denotes the
parameter-matched symmetric control. Gain is the
asymmetric-minus-symmetric difference in percentage points.
These selected individual runs are not paired-seed averages.}
```

The per-seed table and accompanying statement under R1–5 explicitly report that VQA gains are positive for every seed in both encoder settings. These additions also prevent treating the selected diagnostic scores as the full-validation headline scores.

**Evidence:** F/U metrics per-seed summaries and final reports, F/U notebooks cells 16/18/23/26/57, and all four test-export metadata files. Current Table 3's accuracies match the saved full-validation scores of selected epochs. Table 6's full-input values match the selected-subset scores (60.62/61.18 frozen and 65.92/66.35 fine-tuned).

**Not possible to resolve from current evidence:** The source contains comments suggesting seed 123 for frozen CLEVR diagnostics and seed 7 for fine-tuned diagnostics, but comments and figure filenames are not sufficient checkpoint provenance. Recover the original configs, selection logs, and predictions before putting those seeds into Tables 5 or 7 or declaring Table 2 to use best/final epochs. The ViT-B/16 architecture name alone does not establish its pretraining source. Obtaining this documentation requires no GPU; if the original experiment cannot be recovered, a documented rerun will require GPU access. The conservative limitations wording under R1 additional f records these gaps in the meantime.

### Minor Issues

c) The positioning against closely related designs could be more precise. Section 2 cites LXMERT [14] as an example of cross-modal attention. In its public implementation, the cross-modality layer applies a single cross-attention module in both directions. This is close to the shared design studied here. ViLBERT (Lu et al., NeurIPS 2019) instead uses separate co-attention streams for the two modalities. METER (Dou et al., CVPR 2022) compares co-attention and merged-attention fusion, using a CLIP-ViT and RoBERTa setup similar to the one used here. A brief discussion of these works would help readers see what the present comparison adds.

## **Response:**

**Status: Addressable now by a verified related-work insertion.** The comparison concerns a parameter-sharing choice within cross-modal fusion, not the invention of two-direction co-attention. LXMERT is closely related at the attention-sharing level, but its whole cross-modality layer is not the same shared block used here: it also has separate self-attention and feed-forward processing.

**Proposed revision — Section 2, insert immediately after the cross-modal-attention paragraph ending “matched fusion parameter budget” (after line 217):**

```latex
More directly related designs already make different sharing
choices within co-attention. LXMERT~\cite{tan2019lxmert} reuses
one cross-attention module in both directions within each
cross-modality layer, while retaining separate self-attention
and feed-forward processing for language and vision. In contrast,
ViLBERT~\cite{lu2019vilbert} uses two interacting modality streams
with separate visual and textual query, key, and value projections
in its co-attention implementation. METER~\cite{dou2022meter}
empirically compares merged-attention and co-attention fusion
and investigates CLIP vision encoders and RoBERTa text encoders.
Our study examines directional parameter sharing in a smaller
downstream fusion comparison, including an equal-parameter
two-block shared control. It does not introduce directional
co-attention itself, reproduce these full pretrained systems,
or isolate sharing from every aggregation and compute difference.
```

**Proposed bibliography addition — append these author-provided entries to `paper/references.bib` when applying the related-work patch.** The keys are absent from the current bibliography; the LXMERT key already exists. These entries were retrieved from the respective authors' repositories, rather than generated from recalled metadata:

```bibtex
@article{lu2019vilbert,
  title={ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks},
  author={Lu, Jiasen and Batra, Dhruv and Parikh, Devi and Lee, Stefan},
  journal={arXiv preprint arXiv:1908.02265},
  year={2019}
}

@inproceedings{dou2022meter,
  title={An Empirical Study of Training End-to-End Vision-and-Language Transformers},
  author={Dou, Zi-Yi and Xu, Yichong and Gan, Zhe and Wang, Jianfeng and Wang, Shuohang and Wang, Lijuan and Zhu, Chenguang and Zhang, Pengchuan and Yuan, Lu and Peng, Nanyun and Liu, Zicheng and Zeng, Michael},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022},
  url={https://arxiv.org/abs/2111.02387},
}
```

**Evidence:** LXMERT's [`LXRTXLayer.cross_att`](https://github.com/airsplay/lxmert/blob/master/src/lxrt/modeling.py#L405-L437) applies `visual_attention` twice while maintaining modality-specific self-attention/FFN modules. ViLBERT's [`BertBiAttention`](https://github.com/jiasenlu/vilbert_beta/blob/master/vilbert/vilbert.py#L538-L627) defines distinct `query1/key1/value1` and `query2/key2/value2`; the [NeurIPS 2019 proceedings page](https://papers.nips.cc/paper_files/paper/2019/hash/c74d97b01eae257e44aa9d5bade97baf-Abstract.html) confirms the paper and two-stream design. METER's [CVPR 2022 paper page](https://openaccess.thecvf.com/content/CVPR2022/html/Dou_An_Empirical_Study_of_Training_End-to-End_Vision-and-Language_Transformers_CVPR_2022_paper.html) describes the fusion study; its [official repository](https://github.com/zdou0830/METER) provides CLIP/RoBERTa configurations. BibTeX was retrieved from the [ViLBERT README citation block](https://github.com/jiasenlu/vilbert_beta#references) and [METER README citation block](https://github.com/zdou0830/METER#citation). The supplied ViLBERT BibTeX cites its preprint; its NeurIPS publication is independently verified above.

**Remaining work:** No GPU experiment is needed for this positioning revision. When these proposed changes are applied to the manuscript, add the bibliography entries and rebuild to check citation resolution. No bibliography file has been modified in this response-only pass.

d) The modality-blind ablations would benefit from a short discussion. In Table 5, text-blind accuracy is exactly 7.01% for all four CLEVR configurations. Identical values across four different models may indicate a constant prediction. Please state whether this is the case. In Table 6, frozen text-blind accuracy is 12.69% for the symmetric model and 19.83% for the asymmetric model. This gap of about 7 points is not discussed. Since zeroed or withheld inputs are not seen during training, these ablations mainly show that both models rely on both modalities. Please state this scope in Section 5.2.

## **Response:**

**Status: Partially addressable now.** The VQA ablation gap and its limited scope can be discussed directly. Equal CLEVR aggregate accuracies do not establish a constant prediction; a numerical coincidence must not be converted into a diagnosis without per-example outputs.

**Proposed revision — Section 5.2, append after the VQA results paragraph ending “all configurations” (after line 956):**

```latex
In the frozen text-blind VQA comparison, the asymmetric checkpoint
scores 19.83\% and the symmetric checkpoint 12.69\%, a difference
of 7.14 percentage points. This is a response to an artificial
input replacement, not evidence that the asymmetric architecture
is a better image-only model. The two frozen checkpoints were
selected from different seeds, and the ablated text states are
not inputs on which either model was trained. The fine-tuned
text-blind difference is instead -0.21 points. These selected-run
diagnostics therefore support input dependence but do not
establish a consistent architecture-level advantage or directional
specialization under missing modalities.
```

**Proposed revision — Section 5.2, replace the sentence beginning “Without question features, both architectures achieve” (lines 904–907):**

```latex
All four displayed CLEVR text-blind accuracies are 7.01\%.
These aggregate values alone do not establish whether the models
predict a constant answer. Per-question predictions and the
original ablation implementation are needed to determine the
prediction distribution and to check numerical validity.
```

Apply the exact VQA replacement procedure under R1–7 as well. Do not state that CLEVR uses the same padding-mask fix: only the VQA code has been inspected. The VQA implementation explicitly avoids an all-masked attention softmax, but that does not prove that a numerical problem occurred in the separate CLEVR experiment.

**Evidence:** Table 6 and F/U executed notebooks' ablation outputs give 19.83−12.69=7.14 and 25.09−25.30=−0.21 points. F/U final reports identify the selected VQA seeds. Table 5 supplies only aggregate CLEVR accuracies; no matching CLEVR per-question ablation predictions or source implementation is available locally.

**Remaining work:** First obtain the original CLEVR ablation predictions and calculate the number of unique predicted answers, their frequencies, and finite-logit/attention checks on CPU. If only its checkpoint is recoverable, rerun ablation inference with the original code and record those diagnostics; GPU inference is useful but retraining is unnecessary. For claims about architecture-wide missing-input performance, evaluate matching VQA checkpoints for all paired seeds and add separately trained unimodal baselines as specified under R1–7. Until then, the proposed wording explicitly leaves the constant-prediction question unresolved.

e) The Abstract defines symmetric fusion as sharing parameters across directions and separately mentions a parameter-matched symmetric control. All symmetric results in Section 5 come from the parameter-matched control. Please make this explicit in the Abstract. Results for a single shared block, if already available, would also show whether the additional fusion parameters change the symmetric model’s accuracy.

## **Response:**

**Status: Addressable now for the Abstract; single-shared-block accuracy remains unavailable.** All symmetric results in the manuscript refer to the two-block parameter-matched control. A class for the conventional single-block baseline is present, but it is excluded from the active parallel campaign and has no matched three-seed result in the saved evidence.

**Proposed revision — Abstract, replace the sentence “We include a parameter-matched symmetric control to distinguish the benefits of directional specialization from those of increased fusion capacity” (within line 138):**

```latex
All symmetric results use a parameter-matched control with two
independently parameterized blocks, each shared across both
directions, whose outputs are averaged within each direction.
```

This exact clarification is already included in the complete Abstract replacement under R1–3. If applying that replacement, do not also retain the original “We include” sentence or append this clarification a second time. Keep the following Abstract statement that computation and aggregation differ, so equal parameter count is not described as equal capacity.

**Evidence:** F/U notebooks cell 16 (`LegacySymmetricCrossModalFusion` versus `ParameterMatchedSymmetricCrossModalFusion`), cell 26 (`ACTIVE_MODEL_TYPES`), and all paired rows in F/U metrics `*_final_training_report.json`. The active variants are `symmetric_matched` and `asymmetric`.

**Experiments needed when GPU access returns:** Train the conventional shared-block baseline under the same paired seeds and selection rules as the current VQA models, as specified under R1–1. Its 3,153,408 fusion parameters and 6,004,664 trained projection/fusion/classifier parameters must be stated alongside accuracy, since this baseline has fewer parameters. No GPU is needed for the Abstract correction itself, and no baseline accuracy should be invented from an architecture definition.

f) Section 5.5 is titled “Comparison with Large-Scale Multimodal Baselines,” but it contains no comparison. It explains why such a comparison is outside the scope of the study, which is reasonable. Please rename the section, for example to “Relation to Large-Scale Multimodal Models.” The final sentence, which states that the experiments “show when direction-specific fusion is useful,” could also be softened in line with Major Issue 1.

## **Response:**

**Status: Fully addressable now.** Rename the section to describe its content and restrict the final statement to the evaluated architectural comparison.

**Proposed revision — Section 5.5, line 1198:**

```latex
\subsection{Objective 5: Relation to Large-Scale Multimodal Models}
```

**Proposed revision — replace the final sentence beginning “By controlling the encoders” (line 1200):**

```latex
Our experiments compare shared and direction-specific fusion
under specified encoder, training, and parameter-budget
configurations. They neither benchmark against these large-scale
systems nor isolate modality roles as the cause of the observed
performance differences.
```

These are the same heading and sentence replacements specified under R1 additional b; apply them once. The preliminary retrieval qualification belongs in the Abstract/Introduction/Conclusion replacements under R1–3, so this section should not reintroduce an unqualified “when” claim.

**Evidence:** Current Section 5.5 contains contextual discussion, not measurements against the named large-scale systems; Section 5.4 reports only a single seed and a different objective for retrieval.

**Remaining work:** None for this revision. No GPU experiment is required.

g) Section 6 is titled “Limitations and Mitigation Strategies,” but no mitigation strategies are described. Several limitations noted above are also missing. These include the single-run retrieval result, the decline of the symmetric model under fine-tuning on CLEVR, and the absence of a direct analysis of the attention maps. Please extend this section accordingly, or rename it.

## **Response:**

**Status: Fully addressable now through the complete Section 6 replacement under R1 additional f.** That replacement renames the section to `Limitations`, removes the claim of unspecified mitigation strategies, and explicitly includes the single-run retrieval result and small selection subset, the two-seed CLEVR fine-tuning decline, absence of a systematic directional-map analysis, aggregation/compute differences, cross-benchmark confounding, selected-run diagnostics, and uncertainty limits.

**Proposed revision — Section 6 heading (line 1203):**

```latex
\section{Limitations}
```

**Proposed key wording — include these sentences from the coordinated full replacement under R1 additional f:**

```latex
No systematic analysis of the two directional attention maps or
causal interventions establishes distinct learned roles.
```

```latex
Its fine-tuned comparison uses only two paired seeds, and the
large gain mainly reflects lower symmetric-model accuracy after
fine-tuning; its magnitude should not be treated as evidence that
directional specialization has been isolated.
```

Here “Its” refers to CLEVR in the immediately preceding sentence of the full replacement. The retrieval paragraph in that replacement explicitly states one frozen-encoder run per architecture, a different objective, and selection on 50 images. Apply the **entire** replacement rather than appending these excerpts to the old section: the old “These results support the consistency of the primary findings” sentence should not remain as a blanket conclusion about the unstable complementary results.

**Evidence:** Current Sections 5.1.1, 5.4, and 6; Table 2's symmetric decline from 82.58% to 64.93% versus the asymmetric change from 87.85% to 89.35%; F/U reports for the three-seed VQA comparison. These establish the reported scope, not explanations such as gradient conflict or overfitting, which remain untested.

**Remaining work:** No GPU is needed for the limitations revision. Stronger conclusions require the experiments and missing provenance already itemized under R1–1, R1–3, R1–7, R1–8, and R2 major 6. Recovering existing logs and configurations should precede rerunning training.
