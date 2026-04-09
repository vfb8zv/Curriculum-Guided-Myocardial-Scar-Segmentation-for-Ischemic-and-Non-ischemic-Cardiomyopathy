# Curriculum-Guided-Myocardial-Scar-Segmentation-for-Ischemic-and-Non-ischemic-Cardiomyopathy

This is the official repository for the paper "Curriculum-Guided Myocardial Scar Segmentation for Ischemic and Non-ischemic Cardiomyopathy"

### Abstract

Identification and quantification of myocardial scar is important for diagnosis and prognosis of cardiovascular diseases. However, reliable scar segmentation from Late Gadolinium Enhancement Cardiac Magnetic Resonance (LGE-CMR) images remains a challenge due to variations in contrast enhancement across patients, suboptimal imaging conditions such as post-contrast washout, and inconsistencies in ground-truth annotations on diffuse scars caused by inter-observer variability. In this work, we propose a curriculum learning–based framework designed to improve segmentation performance under these challenging conditions. The method introduces a progressive training strategy that guides the model from high-confidence, clearly defined scar regions to low-confidence or visually ambiguous samples with limited scar burden. By structuring the learning process in this manner, the network develops robustness to uncertain labels and subtle scar appearances that are often underrepresented in conventional training pipelines. Experimental results show that the proposed approach enhances segmentation accuracy and consistency, particularly for cases with minimal or diffuse scar, outperforming standard training baselines. This strategy provides a principled way to leverage imperfect data for improved myocardial scar quantification in clinical applications. 

![Curriculum-guided training framework for LGE segmentation, using an arbitrary segmentation backbone, integrating
implicit and explicit difficulty-aware labels](arch_CL_github.jpg)

### Data Structure

dataset
----train
--------0
------------subject1_raw_1.pt
------------subject1_lge_1.pt
------------subject1_raw_2.pt
------------subject1_lge_2.pt
------------subject1_raw_3.pt
------------subject1_lge_3.pt
------------subject2_raw_1.pt
------------subject2_lge_1.pt
------------subject2_raw_2.pt
------------subject2_lge_2.pt
...
------------subjectn_raw_m.pt
------------subjectn_lge_m.pt
--------1
--------2
----test


### Training 

Run the following command to train the segmentation backbone for different stages. Set the --model_type to the backbone's name and --stage to the data folder corresponding to each curriculum stage.

        python train.py

### Testing 

Run the following command to obtain the predictions, dice score and error in scar burden.

        python test.py
