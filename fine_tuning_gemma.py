######################################################################################################################################################
# Credit: this code is a modified version of the following notebook: https://www.kaggle.com/code/nilaychauhan/fine-tune-gemma-models-in-keras-using-lora
######################################################################################################################################################                                      

import os
os.environ["KERAS_BACKEND"] = "jax"  # Or "torch" or "tensorflow".
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import keras_nlp
import json
import numpy as np
import time
import sacrebleu


# Hyper parameters
token_limit =128
lora_rank = 8
lr_value = 1e-4
train_epoch = 10
batch_size = 4
weight_decay = 0.01
model_id = "gemma2_2b_en" 


# helper functions to get answer from model
def get_answer_from_model(llm,prompt):
    input = f"Instruction:\n{prompt}\n\nResponse:\n"
    output = llm.generate(input, max_length=256)
    index  = output.index('\n\nResponse:\n') + len('\n\nResponse:\n')
    return output[index:]

# helper functions to get answer from full prompt 
def get_answer_from_data(prompt):
    index  = prompt.index('\n\nResponse:\n') + len('\n\nResponse:\n')
    return prompt[index:]




######################### READ DATASET ###########################
with open('arxiv_astrophco_qa_pairs_2018_2022_finetuning.json', 'r') as f:
  data_arxiv = json.load(f)

data_arxiv_list = []
for i in range(len(data_arxiv)):
    instruction   = data_arxiv[i]['Question']
    response      = data_arxiv[i]['REF_ANS']
    tmp = f"Instruction:\n{instruction}\n\nResponse:\n{response}"
    data_arxiv_list.append(tmp)

######################### LOAD MODEL ###########################

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(model_id)
gemma_lm.backbone.enable_lora(rank=lora_rank)
gemma_lm.summary()
# Limit the input sequence length (to control memory usage).                                                                                              
gemma_lm.preprocessor.sequence_length = token_limit
# Use AdamW (a common optimizer for transformer models).                                                                                                  
optimizer = keras.optimizers.AdamW(
    learning_rate=lr_value,
    weight_decay=weight_decay,
)
# Exclude layernorm and bias terms from decay.                                                                                                            
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

######################### FINETUNING LOOP###########################                                                                                                                      


# select few random indicies for testing
ind = np.random.randint(1, len(data_arxiv),2) 

##### Compare BLEU score before and after finetuning steps #########
for i in range(10):

    print ("################")
    print ("BEFORE FINETUNING",i)
    print ("################")
    
    for j in ind:
        print ("############### EXAMPLE INDEX",j)
        pred_ans = get_answer_from_model(gemma_lm,data_arxiv[j]['Question'])
        score    = sacrebleu.corpus_bleu([pred_ans],  data_arxiv[j]['REF_ANS']).score
        print ("Question:\n",    data_arxiv[j]['Question'])
        print ("REF  ANSWER:\n", data_arxiv[j]['REF_ANS'])
        print ("PRED ANSWER:\n", pred_ans)
        print ("BLEU SCORE BEFORE FINETUNING=", score)
        

    print ("################")
    print ("FINETUNING STEP", i)
    print ("################")
    
    gemma_lm.fit(data_arxiv_list, epochs=1, batch_size=batch_size)
    
    print ("################")
    print ("AFTER FINETUNING", i)
    print ("################")
    
    for j in ind:
        pred_ans = get_answer_from_model(gemma_lm,data_arxiv[j]['Question'])
        score    = sacrebleu.corpus_bleu([pred_ans],  data_arxiv[j]['REF_ANS']).score
        print ("Question:\n", data_arxiv[j]['Question'])
        print ("REF  ANSWER:\n", data_arxiv[j]['REF_ANS'])
        print ("PRED ANSWER:\n", pred_ans)
        print ("BLEU SCORE AFTER FINETUNING=", score)


# SAVING THE MODEL TO DISK

#gemma_lm.save_to_preset('./CosmoGemma')
    
