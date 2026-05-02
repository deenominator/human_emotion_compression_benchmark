# How Small Is Too Small?
## Here we dive into benchmarking model compression thresholds for Hindi Emotion Classification on low resource Indian devices
> Studying quatization, prunin and knowlege distillation after applying it to Hindi NLP to identify the threshold of the model, i.e, the exact accuracy beyond which a model becomes completely useless

 ## PROBLEM STATEMENT 
 Over half the population of india has phones or internet under 2 GB of RAM and no GPU. Hindi NLP models like IndicBERT genreally require around 140-500 MB of memory and GPU to work, but this results in it being undeployable on most of the population's devices 
 
 Devlopers who build Hindi AI apps have no guideline on how to compress their model, whats would work before it becomes useless
 
 Here our results fill the gap and hope to help out developers 

## What We Did

We took ai4bharat/indic-bert model and fine tuned it for 7 class of hindi emotions, the dataset has been taken from kaggle.  
We then applied three comression techniques at different intensities and then noting the exact point whre the threshold is reached and the accuracy has a drop.

The technique we used were:
1. INT8 Quantization Post training
2. Pruning at 30%, 50% and 70% sparsity
3. Knowledge Distillation by 3 layer student from 12 layer teacher

For each techniques we studied the-
1. Macro F1 Scoew
2. Model size on the disk in MBs
3. Inference Latency
4. Peak RAM delta during inference

## RESULTS

| Method | F1 Score | Size (MB) | Latency (ms) | F1 Drop |
|---|---|---|---|---|
| **Baseline (FP32)** | 0.3064 | 141.9 | 9.23 | — |
| INT8 Quantization | 0.0907 | 105.4 | 331.22 | 70.4%  |
| Pruning 30% | 0.2406 | 127.6 | 10.46 | 21.5%  |
| Pruning 50% | 0.0429 | 127.6 | 11.49 | 86.0%  |
| Pruning 70% | 0.0298 | 127.6 | 11.89 | 90.3%  |
| **Distilled (3-layer)** | **0.3009** | **127.6** | **3.88** | **1.8% ** |

<img width="994" height="533" alt="image" src="https://github.com/user-attachments/assets/1bda1b82-db45-451a-98ec-06c9dfabe8a0" />

## What We Found

1. In pruning, the compression cliff lies between 30% and 50%. The F1 drops from 0.2406 to 0.0429, which is a huge collapse. This is the very first time this cliff has been documented specifically for Hindi NLP. We hypothesize it reflects Hindi's higher morphological complexity requiring greated model capacity than English.
2. Knowledge distillation is the best strategy for Hindi NLP deployment. The 3 layer student model model retained 98.2% of baseline accuracy (F1=0.3009) while running 58% faster (3.88ms vs 9.23ms). It is the clear recommendation for low-resource Hindi NLP deployment.
3. INT8 quantization makes latency worse in raw PyTorch. Latency increased from 9.23ms to 331ms because INT8 shifts computation from GPU to CPU. Real-world deployment of quantized Hindi models requires CPU-optimized runtimes like ONNX Runtime — not raw PyTorch inference. This is an important and underreported finding.

## How to Reproduce
 
1. Open each notebook in [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU** (Runtime → Change runtime type → T4 GPU)
3. Run notebooks **in order**: `01` → `02` → `03` → `04`
4. Each notebook saves results to `results/` and downloads a zip automatically
 
**Estimated time per notebook:**
- Notebook 01: ~20 min (fine-tuning)
- Notebook 02: ~10 min (quantization)
- Notebook 03: ~15 min (pruning × 3 levels)
- Notebook 04: ~30 min (distillation training)
 
**Dependencies** (auto-installed in each notebook):
```
transformers datasets sentencepiece psutil scikit-learn
accelerate bitsandbytes matplotlib pandas torch
```
## Model Weights
 
Model weights are not included in this repository due to file size limits.
 
To obtain the trained baseline model, run `01_baseline_training.ipynb` on Google Colab with a T4 GPU — training completes in approximately 20 minutes.

## Tools & Frameworks
 
| Tool | Purpose |
|---|---|
| `ai4bharat/indic-bert` | Base model (ALBERT pretrained on Indian languages) |
| PyTorch | Training, pruning, quantization |
| HuggingFace Transformers | Model loading and tokenization |
| `torch.quantization` | INT8 post-training dynamic quantization |
| `torch.nn.utils.prune` | Global unstructured magnitude pruning |
| `bitsandbytes` | INT4 NF4 quantization |
| `psutil` | RAM profiling during inference |
| `scikit-learn` | Macro F1 evaluation |
| Google Colab T4 GPU | Training and benchmarking environment |
| `matplotlib` / `pandas` | Results visualization and export |

