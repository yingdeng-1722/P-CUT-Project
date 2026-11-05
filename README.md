# P-CUT: Physics-Informed Contrastive Unpaired Translation for Virtual H&E Staining

This repository contains the core inference code and representative data for our paper: 
**"Physics-informed virtual H&E staining from optical coherence tomography for murine colitis assessment"** 

## 📖 Overview

Virtual staining has emerged as a promising computational alternative to traditional histology. However, conventional unpaired generative models frequently exhibit structural distortion due to the absence of physical constraints. 

To address this, we propose a **Physics-Informed Contrastive Unpaired Translation (P-CUT)** framework. This framework explicitly integrates depth-resolved optical attenuation coefficient (\(\mu_t\)) maps and tissue-layer masks as multi-channel physical priors into the generative process. By doing so, the model synthesizes high-fidelity virtual H&E (VHE) images from label-free optical coherence tomography (OCT) while preserving critical histopathological semantics.

## ✨ Key Features

*   **Physics-Informed Generation:** Incorporates depth-resolved \(\mu_t\) maps and tissue segmentation masks to guide the generative process and suppress structural artifacts.
*   **High Structural Fidelity:** Achieves lower FID and KID scores compared to purely data-driven baselines (e.g., standard CUT).
*   **Pathological Interpretability:** Generated VHE images retain macroscopic pathological structures, enabling reliable downstream classification and Grad-CAM-based visual interpretability.

## 🛠️ Repository Structure

*   `inference.py`: Core script for generating VHE images from input OCT data.
*   `models/`: Network architecture definitions for the generator and discriminator.
*   `sample_data/`: Representative pairs of OCT and generated VHE images.
*   `weights/`: Instructions for downloading the pre-trained model weights.
*   `requirements.txt`: Python environment dependencies.

## 🚀 Quick Start

*(Instructions on how to install dependencies and run the inference script will go here)*
