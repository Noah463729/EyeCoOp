# EyeCoOp
**Knowledge-Guided Prompt Learning for CFP-Only Retinal Disease Classification**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-orange.svg)]()

> EyeCoOp is a knowledge-guided prompt learning framework for **CFP-only retinal disease classification**.  
> It explicitly organizes fine-grained retinal disease concepts on the text side, aligns learnable prompts with class-level concept centers, and transfers stable semantic supervision from a stronger retinal teacher model.

---

## Highlights

- **CFP-only setting** for practical retinal screening and diagnosis
- **Knowledge-guided prompt learning** with structured disease concepts
- **Category-aware Concept Semantic Alignment (CCSA)** to reduce semantic drift
- **Concept-based semantic distillation** from a stronger retinal teacher model
- Supports **few-shot adaptation**, **base-to-novel generalization**, and standard classification settings
- Includes implementations based on **FLAIR** and **BiomedCLIP-style** vision-language backbones

---

## Overview

Retinal disease classification often suffers from two limitations:

1. Class names alone provide only coarse semantic supervision.
2. Learnable prompts can drift away from medically meaningful disease knowledge during optimization.

To address this, **EyeCoOp** introduces a concept-guided text-side learning framework for retinal vision-language models.  
Instead of relying only on generic prompt templates, EyeCoOp builds structured disease concepts and uses them to guide prompt optimization and semantic distillation.

The current implementation focuses on **color fundus photography (CFP)** and supports a **RetFound-based teacher** together with lightweight prompt adaptation on the student side.

---

## Method

EyeCoOp mainly includes the following components:

### 1. Structured Prompt Construction
Disease descriptions are organized into structured text concepts so that the prompt contains richer retinal semantic clues than class names alone.

### 2. Attribute-Augmented Prompt Learning
Learnable prompt tokens are combined with shared fine-grained diagnostic clues, such as:
- location
- morphology
- other disease-relevant retinal semantics

### 3. Category-aware Concept Semantic Alignment (CCSA)
A class-level concept center is used as a semantic anchor to keep the optimized prompt close to disease knowledge space.

### 4. Concept-based Knowledge Distillation
A stronger teacher model provides stable supervision to improve semantic robustness and generalization.

---
