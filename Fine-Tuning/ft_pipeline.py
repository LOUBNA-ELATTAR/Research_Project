# initialization
# import necessary packages and libraries
import os
import torch
from datasets import load_dataset  # Import dataset loading function
from transformers import (  
    AutoModelForCausalLM,  # pre-trained model for causal language modeling if the task is causal LM
    AutoTokenizer,  # pre-trained tokenizer
    BitsAndBytesConfig,  # configuration for quantization
    HfArgumentParser,  # parser for Hugging Face arguments
    TrainingArguments,  # arguments for training the model
    pipeline,  # pipeline for processing data
)
from peft import LoraConfig, PeftModel  # LoRA related modules
from trl import SFTTrainer  # Supervised Fine-Tuning Trainer

# task and their corresponding pretrained models:
# Sequence-to-sequence language modeling: AutoModelForSequenceClassification
# CAUSAL_LM: Causal language modeling: AutoModelForCausalLM
# TOKEN_CLS: Token classification: AutoModelForTokenClassification 
# QUESTION_ANS: Question answering: AutoModelForQuestionAnswering



# data loading
# load dataset for fine-tuning
dataset_name = "educational_dataset"
dataset = load_dataset(dataset_name, split="train")  



# model
# define model configuration
model_name = "pretrained_model"  
# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)  # load tokenizer
tokenizer.pad_token = tokenizer.eos_token  # set padding token
tokenizer.padding_side = "right"  # specify padding side
# quantization for bitsandbytes
use_4bit = True  # 4-bit quantization
bnb_4bit_compute_dtype = "float16"  # dtype for 4-bit base models
bnb_4bit_quant_type = "nf4"  # quantization type (fp4 or nf4)
use_nested_quant = False  # nested quantization for 4-bit base models (double quantization)
# configure LoRA
lora_r = 64  # LoRA attention dimension
lora_alpha = 16  # alpha parameter for LoRA scaling
lora_dropout = 0.1  # dropout probability for LoRA layers
# BitsAndBytesConfig for quantization
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)
# load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)
# configure PeftModel for LoRA
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",  # task type accordingly
)
# overview of the supported task types:
# SEQ_CLS: Text classification
# SEQ_2_SEQ_LM: Sequence-to-sequence language modeling
# CAUSAL_LM: Causal language modeling
# TOKEN_CLS: Token classification
# QUESTION_ANS: Question answering
# FEATURE_EXTRACTION: Feature extraction (Provides the hidden states for downstream tasks)





# training
# training parameters
output_dir = "./results"  # output directory for trained model
num_train_epochs = 10  # number of training epochs
per_device_train_batch_size = 4  # batch size per GPU
gradient_accumulation_steps = 1  # number of gradient accumulation steps
optim = "paged_adamw_32bit"  # optimizer type
learning_rate = 2e-4  # initial learning rate
weight_decay = 0.001  # weight decay
fp16 = False  # FP16 training
bf16 = False  # BF16 training
max_grad_norm = 0.3  # max gradient norm
max_steps = -1  # maximum number of training steps
warmup_ratio = 0.03  # warmup ratio
group_by_length = True  # group batches by length
lr_scheduler_type = "cosine"  # learning rate scheduler type
save_steps = 0  # save model checkpoints every X steps
logging_steps = 25  # log training information every X steps
max_seq_length = None  # maximum sequence length
packing = False  # whether to use packing
device_map = {"": 0}  # device map
# TrainingArguments
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
)
# SFTTrainer for supervised fine-tuning
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",  # Change accordingly based on dataset structure
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)
# start training
trainer.train()
# save the fine-tuned model
new_model = "finetunedmodel" 
trainer.model.save_pretrained(new_model)


#inference
# example prompt for text generation
prompt = "prompt"  
# text generation pipeline
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
# generate text based on prompt
result = pipe(f"<s>[INST] {prompt} [/INST]")  
print(result[0]['generated_text'])  