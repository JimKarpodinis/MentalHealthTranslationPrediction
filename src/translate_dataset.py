import json
import numpy as np
import logging
import ray
import os
import s3fs
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import PartialState
from accelerate.utils import gather_object
from datasets import Dataset, load_dataset
import datasets
import torch


@ray.remote(num_gpus=2)
def main(model_name: str, data_dir: str,
         target_language: str, temperature: float,
         batch_size: int) -> None:

    job_params = locals()
    log_job_params(job_params)

    hf_token = os.getenv("HF_TOKEN")

    dataset = load_dataset("csv", data_dir=data_dir, split="train")

    quantization_config = BitsAndBytesConfig(load_in_8_bit=True)

    model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", token=hf_token,
             quantization_config=quantization_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    dataset = load_dataset("csv", data_dir=data_dir, split="train")

    dataset = dataset.map(lambda example: create_instruction_prompt(
        example, target_language))

    dataset = dataset.map(lambda batch: translate_sentences(
        batch, model, tokenizer, temperature), batched=True, batch_size=batch_size)

    save_dataset(dataset, data_dir, model_name, temperature)


def log_job_params(job_params: dict):

    logger = logging.getLogger("ray")
    logger.info(f"job Parameters: {job_params}")


def create_instruction_prompt(example: dict, target_language: str) -> dict:

    messages = [
        {"role": "system",
        "content": f"""Translate the following sentences to {target_language}.
            Translate only to this language during your response."""},
        {"role": "user", "content": example["text"]},
    ]

    example["instruction_prompt"] = messages

    return example


def translate_sentences(examples: dict,
                        model: AutoModelForCausalLM, tokenizer: AutoTokenizer, temperature: float) -> dict:

    sentence_lengths = tokenizer(
            examples["text"], padding=False,
            truncation=False, return_length=True)["length"]

    max_batch_sentence_length = max(sentence_lengths)

    if tokenizer.chat_template is None:

        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    if tokenizer.pad_token is None:

        tokenizer.pad_token = tokenizer.eos_token

    tokenized_examples = tokenizer.apply_chat_template(
            examples["instruction_prompt"],
            tokenize=True,
            add_generation_prompt=True, return_dict=True,
            return_tensors="pt", 
            truncation=False, padding="longest").to(model.device)

    model_outputs = model.generate(
            **tokenized_examples,
            max_new_tokens=max_batch_sentence_length,
            return_dict_in_generate=True, temperature=temperature, do_sample=True)

    prompt_length = tokenized_examples["input_ids"].shape[1]
    translations = tokenizer.batch_decode(model_outputs.sequences[:, prompt_length+1:])

    examples["translated_text"] = translations

    return examples


def translate_sentences_accelerate(examples: dict, model: AutoModelForCausalLM,
                                   tokenizer: AutoTokenizer, temperature: float) -> dict:

    partial_state = PartialState()
    sentence_lengths = tokenizer(
            examples["text"], padding=False,
            truncation=False, return_length=True)["length"]

    max_batch_sentence_length = max(sentence_lengths)

    if tokenizer.chat_template is None:

        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    if tokenizer.pad_token is None:

        tokenizer.pad_token = tokenizer.eos_token

    with partial_state.split_between_processes(
            examples["instruction_prompt"],
            apply_padding=True) as batched_prompts:

        tokenized_examples = tokenizer.apply_chat_template(
                batched_prompts,
                tokenize=True,
                add_generation_prompt=True, return_dict=True,
                return_tensors="pt", 
                truncation=False, padding="longest").to(model.device)

        prompt_length = tokenized_examples["input_ids"].shape[1]

        model_outputs_ = model.generate(
                **tokenized_examples,
                max_new_tokens=max_batch_sentence_length,
                return_dict_in_generate=True,
                temperature=temperature, do_sample=True)

        model_outputs_ = np.unique(
                model_outputs_.sequences[:, prompt_length+1:])

    partial_state.wait_for_everyone()
    model_outputs = gather_object(model_outputs_)

    if partial_state.is_main_process:

        translations = tokenizer.batch_decode(model_outputs)

        examples["translated_text"] = translations

    return examples


def save_dataset(dataset: Dataset, data_dir: str, model_name: str, temperature: float): 

    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_session_token = os.getenv("AWS_SESSION_TOKEN")

    dataset_name = os.path.basename(data_dir)
    model_name = model_name.split("/")[-1]

    s3 = s3fs.S3FileSystem(client_kwargs={
        "aws_access_key_id":aws_access_key_id,
        "aws_secret_access_key":aws_secret_access_key,
        "aws_session_token":aws_session_token})

    # dataset_name += f"_model_name={model_name}_translated"
    project_name= "MentalHealthTranslationPrediction"

    dataset.to_csv(
            f"s3://dimitris-bucket/project_name={project_name}/data/translations/dataset={dataset_name}/model={model_name}/temperature={temperature}/translation.csv",
            storage_options=s3.storage_options)


if __name__ == "__main__":
    
  with open("json/translate_args.json", "rb") as f:
      args = json.load(f)
      # dict of argument

  ray.init(
          logging_config=ray.LoggingConfig(
              log_level="INFO"))

  ray.get(main.remote(**args))
