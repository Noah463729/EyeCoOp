# EyeCoOp
**Knowledge Guided Prompt Learning for Generalizable Fundus Disease Classification**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-orange.svg)]()

> EyeCoOp is a knowledge guided prompt learning framework for **CFP-only retinal disease classification**.  
> It explicitly organizes fine-grained retinal disease concepts on the text side, aligns learnable prompts with class-level concept centers, and transfers stable semantic supervision from a stronger retinal teacher model.

---

## Highlights

- For practical retinal screening and diagnosis
- **Prompt learning which is based on Fine-grained diagnostic clues** with structured disease concepts
- **Category-aware Concept Semantic Alignment (CCSA)** to reduce semantic drift
- **Teacher Guided Reordering Knowledge Distillation** from a stronger retinal teacher model
- Supports **few-shot adaptation**, **base-to-novel generalization**, and standard classification settings
- Includes implementations based on **FLAIR** and **BiomedCLIP-style** vision-language backbones

---

## Datasets

The experiments in this project are conducted on retinal disease classification benchmarks.  
Please place the processed data under your local dataset directory and replace the following links with the official dataset pages or download links.

- **Dataset 1:** [MultiEYE](https://huggingface.co/datasets/Luxuriant16/MultiEYE)
- **Dataset 2:** [FIVES](https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169/1)
- **Dataset 3:** [ODIR-5K](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)

---

## Implementation

### 1. Environment

Create a new environment and install the requirements:

```bash
conda create -n eyecoop python=3.11
conda activate eyecoop
pip install -r requirements.txt

```

Check Dependencies:

```
numpy==1.24.4
opencv-python==4.8.1.78
scikit-learn==1.2.2
scipy==1.11.4
torch==1.13.1
torchaudio==0.13.1
torchcam==0.3.2
torchvision==0.14.1
transformers==4.27.4
```

### 2. Training

```bash
python main.py \
--modality "fundus" \
--data_path "YOUR_DATA_PATH" \
--concept_path "YOUR_CONCEPT_PATH" \
--retfound_ckpt "YOUR_RETFOUND_CHECKPOINT" \
--batch_size 2 \
--n_classes 9 \
--epochs 60 \
--lr 5e-5 \
--lambda_sccm 0.1 \
--lambda_kdsp 0.1 \
--temperature 8.0 \
--output_dir "checkpoints/eyecoop_checkpoint" \
--device_id [SELECT_GPU_ID]
```
### 3. inference
```bash
python infere.py \
--modality "fundus" \
--data_path "YOUR_DATA_PATH" \
--concept_path "YOUR_CONCEPT_PATH" \
--ckpt_path "YOUR_CHECKPOINT_PATH" \
--split "test" \
--device_id [SELECT_GPU_ID]
```
## Citation

If this code is useful for your research, please citing:

@article{eyecoop2026,
  title   = {EyeCoOp: Knowledge Guided Prompt Learning for Generalizable Fundus Disease Classification},
  author  = {Wang JinFeng,Xie JiaQi},
  journal = {arXiv preprint arXiv:},
  year    = {2026}
}