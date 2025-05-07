import argparse
import os
from transformers import AutoModel, pipeline
from accelerate import PartialState
from accelerate.utils import gather_object
from datasets import Dataset, load_dataset
import torch


def main(model_name: str, data_dir: str, target_language: str) -> None:

    hf_token = os.getenv("HF_TOKEN")

    model = pipeline(
            "text-generation", model=model_name, device_map="auto",
            token=hf_token, return_full_text=False)

    dataset = load_dataset("csv", data_dir=data_dir, split="train")

    dataset = dataset.map(lambda example: create_instruction_prompt(
        example, target_language))

    dataset = dataset.map(lambda batch: translate_sentences(
        batch, model), batched=True, batch_size=64)

    save_dataset(dataset, data_dir, model_name)


def create_instruction_prompt(example: dict, target_language: str) -> dict:

    messages = [
        {"role": "system",
        "content": f"""Translate the following sentences to {target_language}.
            Translate only to this language during your response."""},
        {"role": "user", "content": example["text"]},
    ]

    example["instruction_prompt"] = messages

    return example


def translate_sentences(examples: dict, model: AutoModel) -> dict:

    sentence_lengths = model.tokenizer(
            examples["text"], padding=False,
            truncation=False, return_length=True)["length"]

    max_batch_sentence_length = max(sentence_lengths)

    model_outputs = model(
            examples["instruction_prompt"],
            max_new_tokens=max_batch_sentence_length)

    translations = [output[0]["generated_text"] for output in model_outputs]

    examples["translated_text"] = translations

    return examples


def translate_sentences_accelerate(examples: dict, model: AutoModel) -> dict:

    partial_state = PartialState()

    sentence_lengths = model.tokenizer(
            examples["text"], padding=False,
            truncation=False, return_length=True)["length"]

    max_batch_sentence_length = max(sentence_lengths)

    with partial_state.split_between_processes(
            examples["instruction_prompt"]) as batched_prompts:

        model_outputs_ = model(
                batched_prompts,
                max_new_tokens=max_batch_sentence_length)

    partial_state.wait_for_everyone()
    model_outputs = gather_object(model_outputs_)

    if partial_state.is_main_process:

        translations = [output[0]["generated_text"] for output in model_outputs]

        examples["translated_text"] = translations

    return examples


def save_dataset(dataset: Dataset, data_dir: str, model_name: str): 

    dataset_name = os.path.basename(data_dir)
    model_name = model_name.split("/")[-1]

    dataset_name += f"_{model_name}_translated"

    dataset.to_csv(f"data/processed/{dataset_name}")


if __name__ == "__main__":
    
  parser = argparse.ArgumentParser()

  parser.add_argument("--model_name", help= "The hf model name", type=str)
  parser.add_argument("--data_dir", help= "The path where the data reside", type=str)

  parser.add_argument("--target_language",
                      help= "The language to which the text should be translated",
                      type=str)

  args = vars(parser.parse_args())
  # dict of argument

  main(**args)
