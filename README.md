# CLIP-RL: What Contrastive Fine-Tuning Actually Does to a Multimodal Embedding Space

Retrieval accuracy is a single number, and it hides almost everything that matters about a vision-language representation. Two CLIP variants can differ by 4x in top-1 recall while their text encoders remain 99% representationally identical. A model can have the *highest* cross-modal similarity of any variant tested and still be among the worst at retrieval.

This repository is the instrument we built to study that gap: a measurement suite for the geometry of joint embedding spaces, plus a config-driven training framework so geometry can be observed while it changes, not only before and after.

## Findings

### Setup

Five publicly released CLIP variants, all evaluated on MS-COCO `val2017` (5,000 images, 5 captions each) with features extracted once and analysed offline: **TripletCLIP**, **NegCLIP**, **LaCLIP**, **NegCLIP++**, and OpenCLIP **ViT-B/32 trained on LAION-2B** as the un-fine-tuned reference point.

### 1. Retrieval and representation geometry move in opposite directions

The four fine-tuned variants lose most of the base model's retrieval ability, yet they score *higher* on cross-modal representational similarity and carry a *larger* modality gap.

| | TripletCLIP | NegCLIP | LaCLIP | NegCLIP++ | LAION-2B |
| --- | --- | --- | --- | --- | --- |
| I2T recall@1 | 0.142 | 0.122 | 0.098 | 0.110 | **0.565** |
| I2T recall@10 | 0.449 | 0.401 | 0.353 | 0.374 | **0.874** |
| T2I recall@1 | 0.113 | 0.082 | 0.071 | 0.084 | **0.389** |
| T2I recall@10 | 0.383 | 0.317 | 0.280 | 0.318 | **0.746** |
| Linear CKA, text vs image | **0.990** | 0.957 | 0.918 | 0.964 | 0.945 |
| Relative modality gap | **0.703** | 0.588 | 0.554 | 0.596 | 0.509 |

TripletCLIP is the clearest case: the highest cross-modal CKA of any model tested (0.990) and the largest modality gap (0.703), with roughly a quarter of the base model's recall@1. **High representational similarity between the two modalities does not mean the embeddings are usefully aligned for retrieval.** Any evaluation that reports only one of these will mis-rank these models.

### 2. Alignment and uniformity separate the variants that retrieval does not

| | TripletCLIP | NegCLIP | LaCLIP | NegCLIP++ | LAION-2B |
| --- | --- | --- | --- | --- | --- |
| Alignment | 1.332 | 1.332 | 1.382 | 1.341 | 1.385 |
| Uniformity, text | -1.406 | -1.214 | -1.541 | -1.219 | **-2.568** |
| Uniformity, image | -1.040 | -2.558 | -3.008 | -2.479 | -2.658 |

The base model spreads text embeddings far more evenly over the hypersphere than any fine-tuned variant, and TripletCLIP's image embeddings collapse the most (-1.040 against -2.658). Alignment barely moves across all five. In this family, fine-tuning is mostly a story about losing uniformity.

### 3. Fine-tuned variants converge on nearly the same representation

Pairwise linear CKA between models, on the same COCO features:

```
Text encoders                          Image encoders
        T      N      L     N++  L2B          T      N      L     N++  L2B
T    1.000  0.995  0.993  0.995 0.964    T  1.000  0.967  0.931  0.973 0.964
N    0.995  1.000  0.994  0.997 0.960    N  0.967  1.000  0.955  0.978 0.959
L    0.993  0.994  1.000  0.994 0.959    L  0.931  0.955  1.000    .      .
N++  0.995  0.997  0.994  1.000 0.960
L2B  0.964  0.960  0.959  0.960 1.000
```

Every fine-tuned pair sits at 0.99 or above on the text side, while each is only ~0.96 from the LAION-2B model. Four different fine-tuning objectives, four very different retrieval scores, and one shared representation. This is the Platonic Representation Hypothesis showing up within a single model family.

### 4. Encoders from different models are interchangeable after a learned map

If the representations really are that close, a text encoder from one model should be transplantable into another model's image space. We trained a small mapping from NegCLIP's text space into TripletCLIP's image space (100 iterations, final loss ~8e-4) and measured what survived:

| | CKA to TripletCLIP image space | Positive similarity | Negative similarity |
| --- | --- | --- | --- |
| TripletCLIP text, native | 0.945 | 0.787 | 0.682 |
| NegCLIP text, transported | **0.952** | 0.788 | 0.695 |

The transported encoder matches the native one on representational similarity and reproduces its positive and negative similarity structure. Cross-model encoder substitution is a linear-scale operation, not a retraining problem.

### 5. Information imbalance stops contrastive alignment from forming at all

A synthetic probe isolates the mechanism. We build two correlated Gaussian mixtures where the second view's component means are an affine map of the first, sample paired observations so that one view carries dimensions the other lacks, and train two residual-MLP encoders with InfoNCE.

With 10,000 mixture components and a one-dimensional shared signal, the objective never leaves the chance floor: loss plateaus at 5.53 against the ln(256) = 5.545 lower bound for a batch of 256, and retrieval stays at chance (I2T recall@1 = 0.002, recall@10 = 0.038 over 500 samples).

Contrastive learning cannot manufacture alignment that the data does not contain. When one view is informationally richer than the other, the objective is satisfied by doing nothing.

## The framework

Measurement is only half of it. The `exp/` pipeline exists so that geometry can be tracked through training rather than sampled at the endpoints.

**Experiments compose.** Training and evaluation are the same kind of object and run in the order the config lists them, so `evaluate` then `train` then `evaluate` is one file and every metric is directly comparable across the stages.

**Contrastive Knowledge Consolidation (CKC).** A fine-tuning scheme aimed at finding 3 and 4 above: if the pretrained representation is the thing worth keeping, train against it explicitly. A frozen deep copy of the model acts as a teacher, lightweight MLP projectors bridge the two spaces, and the student optimises a consolidation loss plus weighted InfoNCE. The teacher refreshes every `update_iter` steps.

**Components resolve by name.** Datasets, losses, metrics, and experiments are instantiated from strings in the config, so extending the framework means adding a module and naming it. `main.py` never changes.

## Metrics

| Metric | What it reports |
| --- | --- |
| `retrieval` | Image-to-text and text-to-image recall at 1, 5, 10 |
| `multimodal_retrieval` | Retrieval over a pooled image and text set, so cross-modal and within-modal neighbours compete directly |
| `uniformity` | Sampling-based uniformity estimate together with alignment, following the alignment and uniformity decomposition of contrastive learning |
| `platonic` | Cross-modal representational similarity: mutual k-NN, cycle k-NN, LCS k-NN, CKA and unbiased CKA, SVCCA, CKNNA, and edit-distance k-NN, built on unbiased HSIC |
| `eigenfunction` | Spectral and top-k statistics of the representation across training iterations, with plots written to disk |

Also implemented: relative modality gap (`compute_rmg_cos`), and a mean-centred rotation test that separates matched-pair similarity from mismatched-pair similarity.

## Repository layout

```
main.py                  Entry point. Reads a YAML config and runs the experiment list in order.
clip_train.yaml          Defaults, experiment sequence, model, preprocessor chain.

framework/               Abstract interfaces: ImageTextEncoder, Dataset, Experiment, Preprocessor.
model/clip.py            CLIP ViT-B/32 wrapper with L2-normalised encode_image and encode_text.
dataset/                 MS-COCO and CC12M loaders (CC3M via config) plus a naive loader for smoke tests.
preprocessor/            Image resize and normalise, text tokenisation at 77 tokens, batching.

exp/
  Evaluation.py          The metric suite above.
  ContrastiveLearning.py Standard InfoNCE fine-tuning.
  CKCLearning.py         Contrastive Knowledge Consolidation.
  ExpFactory.py          Resolves an experiment from its config name.

loss/
  infonce.py             Standard InfoNCE, and a pooled multimodal variant that masks self-similarity.
  uniformity.py          Sampling-based uniformity objective.
  ckc.py                 Teacher and student consolidation loss.

notebook/
  Feature Evaluation.ipynb    Findings 1 to 4.
  Information Imbalance.ipynb Finding 5.

utils/utils.py           Config loading, instantiation by name, GPU setup, logging.
```

## Getting started

```bash
git clone https://github.com/ziiihoooYe/CLIP_RL.git
cd CLIP_RL
pip install torch torchvision numpy pandas matplotlib pillow tqdm pyyaml datasets
pip install git+https://github.com/openai/CLIP.git

python main.py --config clip_train.yaml
```

The config is read top to bottom. Each `exp` entry names an experiment class, the dataset it runs on, and its arguments:

```yaml
default:
  gpu: [5, 6, 7]
  log_file: ckc_uni_log.log

exp:
  - class: evaluation           # measure the starting geometry
    dataset:
      class: coco_val_dataset
    args:
      sample_size: 5000
      metrics:
        - retrieval
        - uniformity
        - platonic:
            trials: 10
        - eigenfunction:
            save_path: ./eigenfunction_results

  - class: contrastive_learning  # then perturb it
    dataset:
      class: cc12m_train_dataset
    args:
      epochs: 50
      batch_size: 1024
      temperature_learnable: false

  - class: evaluation            # then measure again
    dataset:
      class: coco_val_dataset
    args:
      sample_size: 5000
      metrics: [retrieval, uniformity]

model:
  class: clip_vitb32

preprocessor:
  - class: image_preprocessor
    args: {image_size: 224}
  - class: text_preprocessor
    args: {cutoff_length: 77}
  - class: to_batch_preprocessor
```

Swap the middle stage for CKC:

```yaml
  - class: ckc_learning
    dataset:
      class: cc3m_train_dataset
    args:
      infonce_weight: 0.1
      batch_size: 5000
      max_iter: 50
      update_iter: 50      # refresh the frozen teacher every N iterations
      ckc_temperature: 0.07
```

## Notes

Numbers above come from the committed notebook outputs and are reproducible from them. Training data is Conceptual Captions 3M and 12M; evaluation is MS-COCO `val2017`.

Originated as a course project for CMU 11-777 Multimodal Machine Learning. The CLIP backbone is loaded through OpenAI's `clip` package; the framework, metrics, losses, and analyses in this repository are implemented here.
