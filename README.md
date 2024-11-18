# CosmoGemma: Enhancing Research Productivity in Cosmology using Gemma

This repository contains all codes used to develop CosmoGemma,which is a project as a part of KaggleX Fellowship Program (Cohort 4): https://www.kaggle.com/kagglex.

## What is CosmoGemma?

Gemma_2b_en fine-tuned model on QA pairs (3.5k) generated from Cosmology and Nongalactic Astrophysics articles (arXiv astro-ph.CO) from 2018-2022 and tested on QA pairs (1k) generated from 2023 articles, scoring over 75% accuracy. The model is already deployed in Hugging Face 🤗 space X Gradio, check it out https://huggingface.co/spaces/sultan-hassan/cosmology-expert.


## Repository Structure

 - **QA_dataset_generation.ipynb** a notebook shows all steps to read arxiv data from Kaggle (download the data first from https://www.kaggle.com/datasets/Cornell-University/arxiv), filtering data to select Cosmology and Nongalactic Astrophysics articles (arXiv astro-ph.CO) from 2018-2022 for fine-tuning and from 2023 for testing. Once the dataset is selected, the notebook uses a combination of langchain/langchain_community, and Ollama (both must be first installed) to run llama3.1:8b-instruct-fp16 model to generate QA pair from a given abstract. The full prompt is provided in the notebook. To install langchain/langchain_community, run ```$ pip install langchain langchain_community```, and refer to https://github.com/ollama/ollama to install Ollama.

 - **fine_tuning_gemma.py** A python script to finetune Gemma model using LoRA: Low-Rank Adaptation of Large Language Model (see https://arxiv.org/abs/2106.09685). This scripts computes and compare BiLingual Evaluation Understudy (BLEU) score, using sacrebleu package (https://github.com/mjpost/sacrebleu) before and after each fine-tuning step. Credit: The code is a modified version of this notebook: https://www.kaggle.com/code/nilaychauhan/fine-tune-gemma-models-in-keras-using-lora. An ouput of the training loop for testing BLEU score before/after one finetuning step for two random examples reads:

 ```
################
BEFORE FINETUNING 0
################
############### EXAMPLE INDEX 2209
Question:
 What is the main limitation of current semi-analytical schemes to simulate the displacement of CDM?
REF  ANSWER:
 Their inability to model the evolution of overdensities in the initial density field.
PRED ANSWER:
 The main limitation of current semi-analytical schemes to simulate the displacement of CDM is that they are not able to capture the full range of displacement behavior. This is because they are based on a simplified model of the CDM, which does not take into account the full range of displacement behavior.

The full range of displacement behavior includes both the displacement of the CDM and the displacement of the surrounding material. The displacement of the CDM is determined by the material properties of the CDM and the surrounding material, while the displacement of the surrounding material is determined by the material properties of the surrounding material and the displacement of the CDM.

The full range of displacement behavior is important because it allows for a more accurate simulation of the displacement of CDM. This is because the full range of displacement behavior includes both the displacement of the CDM and the displacement of the surrounding material.

The full range of displacement behavior is also important because it allows for a more accurate simulation of the displacement of CDM. This is because the full range of displacement behavior includes both the displacement of the CDM and the displacement of the surrounding material.

The full range of displacement behavior is also important because it allows
BLEU SCORE BEFORE FINETUNING= 0.1889678722649679
############### EXAMPLE INDEX 1953
Question:
 Can using multiple statistical measures simultaneously reduce systematic errors in cosmological parameter estimation?
REF  ANSWER:
 Yes, it can be very effective in mitigating these systematic errors.
PRED ANSWER:
 The answer is yes.

The answer is no.

The answer is maybe.

The answer is yes, but only if you use the right statistical measures.

The answer is no, but only if you use the wrong statistical measures.

The answer is maybe, but only if you use the wrong statistical measures.

The answer is yes, but only if you use the right statistical measures.

The answer is no, but only if you use the wrong statistical measures.

The answer is maybe, but only if you use the wrong statistical measures.

The answer is yes, but only if you use the right statistical measures.

The answer is no, but only if you use the wrong statistical measures.

The answer is maybe, but only if you use the wrong statistical measures.

The answer is yes, but only if you use the right statistical measures.

The answer is no, but only if you use the wrong statistical measures.

The answer is maybe, but only if you use the wrong statistical measures.

The answer is yes, but only if you use the right statistical measures.

The answer is no, but only if
BLEU SCORE BEFORE FINETUNING= 0.19420499496159066
################
FINETUNIGN STEP 0
################
875/875 ━━━━━━━━━━━━━━━━━━━━ 218s 164ms/step - loss: 0.8358 - sparse_categorical_accuracy: 0.5251
################
AFTER FINETUNING 0
################
Question:
 What is the main limitation of current semi-analytical schemes to simulate the displacement of CDM?
REF ANSWER:
 Their inability to model the evolution of overdensities in the initial density field.
PRED ANSWER:
 The main limitation is the lack of a proper treatment of the displacement of CDM.
BLEU SCORE AFTER FINETUNING= 2.908317710573757
Question:
 Can using multiple statistical measures simultaneously reduce systematic errors in cosmological parameter estimation?
REF ANSWER:
 Yes, it can be very effective in mitigating these systematic errors.
PRED ANSWER:
 Yes, it can reduce systematic errors in cosmological parameter estimation.
BLEU SCORE AFTER FINETUNING= 4.02724819242185
 ```

 - **evaluation_LLM_as_judge.ipynb** a notebook shows how to evaulate CosmoGemma performance on the testing sample using LLMs as a judge.
 
 - **arxiv_filtered_astrophco-18-22.json** contains filtered astro-ph.CO abstracts from Kaggle Arxiv dataset from 2018-2022, about 3,497 abstracts for fine-tuning.
 - **arxiv_filtered_astrophco-23.json** contains filtered astro-ph.CO abstracts from Kaggle Arxiv dataset from 2023, about 1,055 abstracts for testing.

 - **arxiv_astrophco_qa_pairs_2018_2022_finetuning.jso** contains the generated Questions (['Question']) and the generated (reference) answer (['REF_ANS']) from abstracts using llama3.1 for the finetuning sample (3,497 from astro-ph.CO 2018-2022)

   ``` {'Question': 'What type of astrophysical objects are known to have the highest mass-to-light ratios in the Universe?', 'REF_ANS': 'Dwarf spheroidal galaxies'} ```

 - **arxiv_astrophco_qa_pairs_2023_testing.json** contains the generated Questions (['Question']) and the generated (reference) answer (['REF_ANS']) from abstracts using llama3.1 as well as the predicted answer from the CosmoGemma after fine-tuning (['PRED_ANS']) for the testing sample (1,055 from astro-ph.CO 2023). Here's a random example:

   ```{'Question': 'Can photometric redshift errors impact constraints on a primordial non-Gaussianity parameter?', 'REF_ANS': 'Yes, they can increase the error by up to 18%.', 'PRED_ANS': 'Yes, large photometric redshift errors can degrade the constraints on this parameter.'}```

