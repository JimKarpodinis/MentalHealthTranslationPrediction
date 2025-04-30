import argparse
import os
from transformers import AutoModel, pipeline
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import Dataset, load_dataset
import torch


def main(model_name: str, data_dir: str, target_language: str) -> None:

    accelerator = Accelerator()

    hf_token = os.getenv("HF_TOKEN")

    model = pipeline(
            "text-generation", model=model_name, device_map="auto",
            token=hf_token, return_full_text=False)

    dataset = load_dataset("csv", data_dir=data_dir)

    dataset = dataset.map(lambda example: create_instruction_prompt(
        example, target_language))

    dataset = dataset.map(lambda examples: translate_sentences(
        examples, model=model,
        target_language=target_language, accelerator=accelerator),
        batched=True, remove_columns=["text"])

    save_dataset(dataset, data_dir)


def create_instruction_prompt(example: dict, target_language: str) -> dict:

    messages = [
        {"role": "system",
        "content": f"""Translate the following sentences to {target_language}.
            Translate only to this language during your response."""},
        {"role": "user", "content": example["text"]},
    ]

    example["instruction_prompt"] = messages

    return example


def translate_sentences(examples: list, model: AutoModel,
        target_language: str, accelerator: Accelerator) -> str:

    split_prompts = accelerator.split_between_processes(
    examples["instruction_prompt"], apply_padding=True)

    with split_prompts as batched_prompts:
        for prompt in batched_prompts:
            model_output = model(prompt,
                max_new_tokens=10000)

    all_model_outputs = gather_object(model_outputs)
    all_translated_texts = [output["generated_text"] for output in all_model_outputs]

    examples["translated_texts"] = all_translated_texts

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
