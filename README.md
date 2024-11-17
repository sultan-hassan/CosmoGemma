# CosmoGemma: Enhancing Research Productivity in Cosmology using Gemma

This repository contains all codes used to develop CosmoGemma,which is a project as a part of KaggleX Fellowship Program (Cohort 4): https://www.kaggle.com/kagglex.

## What is CosmoGemma?

Gemma_2b_en fine-tuned model on QA pairs (3.5k) generated from Cosmology and Nongalactic Astrophysics articles (arXiv astro-ph.CO) from 2018-2022 and tested on QA pairs (1k) generated from 2023 articles, scoring over 75% accuracy.

## Repository Structure

 - **QA_dataset_generation.ipynb** a notebook shows all steps to read arxiv data from Kaggle (download the data first from https://www.kaggle.com/datasets/Cornell-University/arxiv), filtering data to select Cosmology and Nongalactic Astrophysics articles (arXiv astro-ph.CO) from 2018-2022 for fine-tuning and from 2023 for testing. Once the dataset is selected, the notebook uses a combination of langchain/langchain_community, and Ollama (both must be first installed) to run llama3.1:8b-instruct-fp16 model to generate QA pair from a given abstract. The full prompt is provided in the notebook. To install langchain/langchain_community, run ```$ pip install langchain langchain_community```, and refer to https://github.com/ollama/ollama to install Ollama.

 - **evaluation_LLM_as_judge.ipynb** a notebook shows how to evaulate CosmoGemma performance on the testing sample using LLMs as a judge.
 
 - **arxiv_filtered_astrophco-18-22.json** contains filtered astro-ph.CO abstracts from Kaggle Arxiv dataset from 2018-2022, about 3,497 abstracts for fine-tuning.
 - **arxiv_filtered_astrophco-23.json** contains filtered astro-ph.CO abstracts from Kaggle Arxiv dataset from 2023, about 1,055 abstracts for testing.

 - **arxiv_astrophco_qa_pairs_2018_2022_finetuning.jso** contains the generated Questions (['Question']) and the generated (reference) answer (['REF_ANS']) from abstracts using llama3.1 for the finetuning sample (3,497 from astro-ph.CO 2018-2022)

   ``` {'Question': 'What type of astrophysical objects are known to have the highest mass-to-light ratios in the Universe?', 'REF_ANS': 'Dwarf spheroidal galaxies'} ```

 - **arxiv_astrophco_qa_pairs_2023_testing.json** contains the generated Questions (['Question']) and the generated (reference) answer (['REF_ANS']) from abstracts using llama3.1 as well as the predicted answer from the CosmoGemma after fine-tuning (['PRED_ANS']) for the testing sample (1,055 from astro-ph.CO 2023). Here's a random example:

   ```{'Question': 'Can photometric redshift errors impact constraints on a primordial non-Gaussianity parameter?', 'REF_ANS': 'Yes, they can increase the error by up to 18%.', 'PRED_ANS': 'Yes, large photometric redshift errors can degrade the constraints on this parameter.'}```

