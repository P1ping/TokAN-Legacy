# TokAN: Accent Normalization Using Self-Supervised Discrete Tokens

> [!IMPORTANT]
> This is the **legacy** repository for the Interspeech 2025 conference paper.
> An extended version with a cleaner, updated implementation is now
> available at **[P1ping/TokAN](https://github.com/P1ping/TokAN)** — please head
> there for the latest code and models.

This repository contains the official implementation and pretrained models for the Interspeech 2025 paper: **"Accent Normalization Using Self-Supervised Discrete Tokens with Non-Parallel Data"**.

**📄 [Paper](https://arxiv.org/abs/2507.17735) | 🎵 [Demo](https://p1ping.github.io/TokAN-Demo/)**

## Installation

We strongly recommend using a `conda` or `venv` virtual environment with Python 3.8+.

### Step 1: Clone Repository and Setup Environment

First, clone the repository, making sure to initialize the `fairseq` submodule.

```bash
git clone --recurse-submodules https://github.com/P1ping/TokAN-Legacy.git
cd TokAN-Legacy
```

If you forgot the `--recurse-submodules` flag, run `git submodule update --init --recursive` after cloning.

Then, create and activate your virtual environment:
```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate
# Or you can create a new environment via conda
```

### Step 2: Install PyTorch and (optional) eSpeak

For successful load of fairseq models, we recommend `torch<=2.5.1`.

> For other CUDA versions or CPU-only installation of PyTorch, please visit the [official PyTorch website](https://pytorch.org/get-started/locally/) for the correct command.

**Install eSpeak (required for training):**
```bash
# On Ubuntu/Debian
sudo apt-get install espeak espeak-data
# On other systems, visit: http://espeak.sourceforge.net/

# Please ensure the output version is latest (1.48)
espeak --version
```

> **Note:** eSpeak (version 1.48) is required if you plan to run the training pipeline, as it's used for phoneme processing in the text-to-speech components.

### Step 3: Install the Custom Fairseq Dependency

The `token-to-token` component of TokAN requires the fairseq toolkit, which is included in the `third_party` directory.
This **should be installed before** the the main project. Fairseq requires a lower version of `hydra-core`, which should be upgraded during installation of the main project.

> **💡 Tip:** You probably need `pip<=24.0` and `gcc 9 or later` for successful fairseq installation.

```bash
# Navigate to the submodule directory
cd third_party/fairseq

# Install this specific version of fairseq in editable mode
pip install -e .

# Return to the project root directory
cd ../..
```

### Step 4: Install Project Dependencies and TokAN

Now that the critical dependencies are in place, you can install the remaining packages from `requirements.txt` and then install the TokAN source code itself.

```bash
# Install all other required packages
pip install -r requirements.txt

# Install the project's core source code in editable mode
pip install -e .
```

After these steps, the environment is fully configured to run inference.

## Running Inference

```bash
python inference.py --input_path /path/to/input.wav --output_path /path/to/output.wav --download_models
```

This will automatically download all required models and perform accent conversion on your audio file. Specifically, it will split the input audio (when it is long) into chunks and convert them one by one. The chunking is based on `silero_vad`, please check the arguments for detailed configurations. When `--preserve_duration` is given, the script will select the model with flow-matching-based duration prediction for total duration preservation.

> **⚠️ Note:** If the download is interrupted, please delete the partially downloaded file so that it can be downloaded again.

## Training Steps

### Data Preparation

TokAN requires two main datasets for training:

#### 1. LibriTTS-R Dataset
LibriTTS-R is used for training the text-to-speech components and token-to-token pre-training. The dataset should be organized as follows:

```
LibriTTS_R/
├── train-clean-100/
├── train-clean-360/
├── train-other-500/
├── dev-clean/
├── dev-other/
├── test-clean/
└── test-other/
```

**Download Instructions:**
1. Visit: https://www.openslr.org/60/ (LibriTTS) or https://www.openslr.org/141/ (LibriTTS-R)
2. Download all required subsets (train-clean-100, train-clean-360, train-other-500, dev-clean, dev-other, test-clean, test-other)
3. Extract each subset to your LibriTTS-R directory
4. Update the `libritts_root` path in `run.sh`

#### 2. L2ARCTIC Dataset + ARCTIC Native Speakers
L2ARCTIC contains non-native English speech, while ARCTIC provides native English speakers for comparison. The combined dataset should include:

**L2ARCTIC Speakers (by accent):**
- Arabic (`<ar>`): ABA, YBAA, ZHAA, SKA
- Chinese (`<zh>`): BWC, LXC, NCC, TXHC  
- Hindi (`<hi>`): ASI, RRBI, SVBI, TNI
- Korean (`<ko>`): HJK, YDCK, YKWK, HKK
- Spanish (`<es>`): EBVS, ERMS, NJS, MBMPS
- Vietnamese (`<vi>`): HQTV, PNV, THV, TLV

**ARCTIC Native Speakers (`<us>`):** BDL, RMS, SLT, CLB

**Expected Directory Structure:**
```
l2arctic/
├ # L2ARCTIC speakers (direct extraction)
├── YBAA/
├── BWC/
├── ...
├ # ARCTIC native speakers
├── BDL/
├── RMS/
├── SLT/
└── CLB/
```

**Download Instructions:**

*For L2ARCTIC:*
1. Visit: https://psi.engr.tamu.edu/l2-arctic-corpus/
2. Download and extract to your L2ARCTIC root directory

*For ARCTIC Native Speakers:*
1. Visit: http://festvox.org/cmu_arctic/
2. Download: `cmu_us_bdl_arctic.tar.bz2`, `cmu_us_rms_arctic.tar.bz2`, `cmu_us_slt_arctic.tar.bz2`, `cmu_us_clb_arctic.tar.bz2`
3. Extract to L2ARCTIC's directory with the same meaning pattern (speaker's tag in the upper case)

#### Data Verification
Before training, verify your datasets are properly prepared:

```bash
# Verify both datasets
bash run.sh --stage 0 --stop_stage 0

# Or run the verification script directly
python scripts/verify_data_preparation.py --libritts_root /path/to/LibriTTS_R --l2arctic_root /path/to/l2arctic
```

The verification script will:
- Check all required LibriTTS-R subsets
- Verify all L2ARCTIC and ARCTIC speakers are present
- Provide detailed download instructions for missing data
- Display dataset statistics

### Training Stages

TokAN training consists of multiple stages that build upon each other. We recommend running stages sequentially using `bash run.sh --stage [STAGE] --stop_stage [STAGE]`.

#### Stage 0: Data Verification
```bash
bash run.sh --stage 0 --stop_stage 0
```
Validates that all required datasets are properly prepared and accessible.

#### Stage 1: Tokenization-related Check
```bash
bash run.sh --stage 1 --stop_stage 1
```
Checks for pre-trained HuBERT and k-means models required for speech tokenization. We recommend *specify the related paths directly in the script*. If you use HuBERT other than `hubert_large_ll60k` or our pre-trained K-means model, please also update `hubert_path`, `hubert_layer`, and `n_clusters` correspondingly.

**Options:**
1. **Use Meta's official HuBERT-large and our K-means model**: Download via the `### Pre-trained Models` section below
2. **Train your own k-means**: Follow instructions in `third_party/fairseq/examples/hubert/simple_kmeans/README.md`

> **💡 Note:** K-means clustering is storage-intensive. We recommend using the one we provide, which was clustered on LibriTTS-R. If you have changed the number of clusters, please also update n_vocab in components/token_to_mel/configs/model/yirga.yaml

#### Stage 2: Matcha-TTS Training
```bash
bash run.sh --stage 2 --stop_stage 2
```
Trains the text-to-speech model on LibriTTS-R. This model generates synthetic target data for accent conversion training.

**What happens:**
- Generates manifests for LibriTTS-R
- Extracts speaker embeddings  
- Trains Matcha-TTS model
- Checkpoints saved to `components/text_to_mel/experiments/`

#### Stage 3: Token-to-Mel Synthesizer Training
```bash
bash run.sh --stage 3 --stop_stage 3
```
Trains the token-to-Mel synthesizer (Yirga) on LibriTTS-R using HuBERT tokens.

**What happens:**
- Extracts HuBERT tokens and speaker embeddings
- Trains token-to-Mel model
- Checkpoints saved to `components/token_to_mel/experiments/`

#### Stage 4: Token-to-Token Pre-training
```bash
bash run.sh --stage 4 --stop_stage 4
```
Pre-trains the accent conversion model on LibriTTS-R using self-supervised objectives.

**What happens:**
- Generates training manifests and dictionaries to `components/token_to_token/data/libritts/`
- Extracts HuBERT tokens for pre-training
- Trains token-to-token model with denoising objectives
- Checkpoints saved to `components/token_to_token/experiments/libritts_pretrain/`

#### Stage 5: Token-to-Token Fine-tuning
```bash
# Specify text-to-Mel checkpoint before running
bash run.sh --stage 5 --stop_stage 5 --text_to_mel_ckpt /path/to/matcha/checkpoint.ckpt
```
Fine-tunes the accent conversion model on L2ARCTIC for accent-specific adaptation.

**What happens:**
- Synthesizes L1-accented target data using Matcha-TTS to `components/token_to_token/data/l2arctic/synthetic_target/`
- Extract accent embeddings from the source audio to `components/token_to_token/data/l2arctic/accent_embeddings/`
- Generates L2ARCTIC training manifests to `components/token_to_token/data/l2arctic/`
- The token dictionaries are copied from the pre-training data directory
- Extracts features for fine-tuning
- Fine-tunes pre-trained model on accent conversion task
- Checkpoints saved to `components/token_to_token/experiments/l2arctic_finetune/`

#### Stage 6: Generation and Testing
```bash
# Specify both checkpoints before running  
bash run.sh --stage 6 --stop_stage 6 \
    --token_to_token_ckpt /path/to/t2t/checkpoint.pt \
    --token_to_mel_ckpt /path/to/t2m/checkpoint.ckpt
```
Generates accent-converted speech on the test set.

**What happens:**
- Decodes tokens using trained token-to-token model
- Generates audio from tokens using token-to-Mel model
- Outputs saved to `experiments/l2arctic_finetune/test_output/`

#### Stage 7: Evaluation
```bash
bash run.sh --stage 7 --stop_stage 7
```
Evaluates the generated speech using multiple metrics.

**What happens:**
- Computes Word Error Rate (WER) using a native-only ASR
- Measures PPG distance with synthetic targets
- Calculates speaker similarity with source audio
- Results saved to `experiments/l2arctic_finetune/test_output/`

### Configuration and Customization

#### Model Checkpoints
You can specify custom checkpoint paths:
```bash
bash run.sh --stage [STAGE] --stop_stage [STAGE] \
    --text_to_mel_ckpt /path/to/matcha.ckpt \
    --token_to_token_ckpt /path/to/t2t_model.pt \
    --token_to_mel_ckpt /path/to/t2m_model.ckpt
```

#### Training Configuration
- Modify batch sizes and other hyperparameters in `components/*/configs/`
- Change parallel data-processing jobs with `--nj` parameter

#### Resuming Training
After data preprocessing is complete, you can comment out the preprocessing sections in `run.sh` to avoid repeated computation.

### Pre-trained Models

Download pre-trained models to skip training stages:

```bash
python tokan/utils/model_utils.py
```

Models will be downloaded to `pretrained_models/`. You can use these directly in your pipeline as long as the k-means model is consistent.

### Pre-trained Models

To accelerate training or skip certain stages, you can download our pre-trained models:

```bash
python tokan/utils/model_utils.py
```

This will download models to the `pretrained_models/` directory with the following structure:
```
pretrained_models/
├── hubert_km/
│   └── hubert_km_libritts_l17_1000.pt    # K-means model
├── token_to_mel/
│   ├── tokan-t2m-v1-paper/               # Token-to-Mel model with regression duration predictor
│   │   └ model.ckpt
│   └── tokan-t2m-v2-paper/               # Token-to-Mel model with flow-matching duration predictor
│       └ model.ckpt
└── token_to_token/
    └── tokan-t2t-base-paper/             # Token-to-token model
        ├─ model.pt                       # Fairseq checkpoint
        ├─ dict.src.txt                   # Input token dictionary
        ├─ dict.tgt.txt                   # Input token dictionary
        └─ dict.aux.txt                   # Intermediate phone dictionary
```

**Using Pre-trained Models:**

You can reference these models in your training pipeline. Update the paths in `run.sh`:
```bash
# Use pre-trained k-means model
km_path=$(pwd)/pretrained_models/hubert_km/hubert_km_libritts_l17_1000.pt

# Use pre-trained token-to-Mel model  
token_to_mel_ckpt=$(pwd)/pretrained_models/token_to_mel/tokan-t2m-v2-paper/model.ckpt

# Use pre-trained token-to-token model
token_to_token_ckpt=$(pwd)/pretrained_models/token_to_token/tokan-t2t-base-paper/model.pt
```

> **⚠️ Important:** Ensure k-means model consistency across all components. All models should use the same HuBERT layer (17) and cluster count (1000) for compatibility.

### Paper Training Configuration

**Hardware Requirements:**
- We used 4 V100 GPUs (32GB each) for all model training
- Training batch sizes in the default configuration are optimized for this setup
- Adjust batch sizes in `components/*/configs/` or in `fairseq-train` arguments if using different hardware

**Notes on Paper's Implementation:**
- **Mel synthesizer training**: Only samples shorter than 15 seconds were used
- **LibriTTS-R partitioning**: We used random partitioning across the entire dataset, whereas the current practice is subset-wise partitioning
- **Flow-matching duration predictor**: Trained separately with other modules frozen, initialized from a well-trained regression-based model

> **💡 Note:** These implementation choices may affect training dynamics and final model performance, but the impact should be small. The current codebase discards these practices for cleaness and clarity.

## Acknowledgements

This work was conducted at Tencent TEA-Lab.

This project leverages several excellent open-source projects:

- [Fairseq](https://github.com/facebookresearch/fairseq): For training the accent-removal model
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS): For text-to-speech synthesis to generate synthetic data
- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template): For speech synthesis training
- [textlesslib](https://github.com/facebookresearch/textlesslib): For extract HuBERT tokens
- [BigVGAN](https://github.com/NVIDIA/BigVGAN): For high-quality neural vocoding

We express our gratitude to the authors and contributors of these projects for making their work available to the community.

For detailed information about our modifications to these codebases, please see the [Notes](tokan/README.md).

## Citation

If you find our work useful in your research, please cite our paper:

```bibtex
@inproceedings{bai2025accent,
  title={Accent Normalization Using Self-Supervised Discrete Tokens with Non-Parallel Data},
  author={Bai, Qibing and Inoue, Sho and Wang, Shuai and Jiang, Zhongjie and Wang, Yannan and Li, Haizhou},
  booktitle={Proceedings of Interspeech},
  year={2025}
}