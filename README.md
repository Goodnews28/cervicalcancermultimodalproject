# Multimodal AI Model for Early Cervical Cancer Detection Using Non-Invasive Biomarkers and Cytology Images

## Overview
Cervical cancer is a highly preventable disease, yet it remains a leading cause of cancer-related deaths in underserved populations. This project proposes a **multimodal AI-based diagnostic system** that integrates non-invasive molecular biomarkers, cytology images, and clinical data to improve the **early detection and accessibility** of cervical cancer screening tools.

---

##  Project Premise & Novelty

This research explores the development of a **multimodal machine learning model** that combines:
- **Genomic biomarkers** from non-invasive sources (blood, urine, self-swabs)
- **Pap smear cytology images**
- **Clinical and symptomatic data**

By fusing tabular, visual, and genomic data types, the model aims to:
- Improve early-stage diagnosis accuracy
- Increase access to screening in resource-limited settings
- Demonstrate a scalable, cost-effective AI pipeline suitable for global health equity

This comprehensive diagnostic approach is rarely explored in high school-level research and holds strong potential for recognition at science fairs like ISEF.

---

## Selected Biomarkers & Justification

| Biomarker    | Source             | Reason for Selection |
|--------------|--------------------|-----------------------|
| **miRNA-21** | Blood plasma       | Consistently upregulated in cervical cancer across studies |
| **p16INK4a** | Cervical cells     | Overexpressed in cells with integrated high-risk HPV |
| **HPV DNA**  | Urine, swabs       | Present in >99% of cervical cancer cases |
| **Ki-67**    | Cervical cytology  | Marker of cell proliferation; elevated in precancerous/cancerous lesions |

---

## Biomarker Extraction Methods

### Non-Invasive
- **Blood samples** (liquid biopsy for miRNA-21)
- **Urine samples** (for HPV DNA)
- **Self-collected vaginal swabs**

### Minimally Invasive
- **Pap smear** (for cytology and p16INK4a / Ki-67 expression)
- **Cervical tissue biopsy** (research-only validation)

---

## Image Modality

- **Pap smear cytology images** from publicly available datasets
- Used to detect morphological abnormalities via convolutional neural networks (CNNs)

---

## 8-Week Research Roadmap

| Week | Goals |
|------|-------|
| 1-2  | Literature review, biomarker validation, dataset sourcing |
| 3-4  | Data preprocessing: tabular biomarker data + cytology images |
| 5-6  | Model architecture: design and train multimodal AI model |
| 7    | Evaluation: accuracy, sensitivity, specificity, interpretability |
| 8    | Documentation, visuals, poster creation, feedback integration |

---

## Tools & Technologies

- **Languages**: Python
- **Libraries**: scikit-learn, TensorFlow/Keras, pandas, OpenCV
- **Data**: GEO microarray datasets, TCGA (if applicable), public cytology image sets

---


---

## Broader Impact

This model has the potential to:
- Enable **cost-effective, non-invasive screening**
- Assist **early intervention** in high-risk populations
- Promote **equity in cervical cancer diagnostics**, especially in underserved regions

---

## Citation (if applicable)

If you use public datasets or models from this work, please cite the original sources appropriately. This project is intended for **educational and research purposes only**.

---

## Contact
For questions or collaboration opportunities, contact:
**Goodnews Babade**  
Email: *[goodnewsababade@gmail.com]*  


