import argparse
import os
from transformers import AutoModel, pipeline
from datasets import Dataset, load_dataset
import torch

def main(model_name: str, data_dir: str, target_language: str) -> None:

    hf_token = os.getenv("HF_TOKEN")

    
    model = pipeline(
            "text-generation", model=model_name, device_map="auto", token=hf_token)

    dataset = load_dataset("csv", data_dir=data_dir)

    breakpoint()

    dataset = dataset.map(lambda batch: translate_sentences(
        batch, model=model, target_language=target_language, batched=True))

    save_dataset(dataset, data_dir)


def save_dataset(dataset: Dataset, data_dir: str): 

    dataset_name = os.path.basename(data_dir)
    dataset.to_csv(f"data/processed/{dataset_name}")


def translate_sentences(examples: list, model: AutoModel, target_language: str) -> list:
    
    prompt = f"""
    Please translate the following sentences to {target_language}:
    {examples}
    """

    return model(prompt)

if __name__ == "__main__":
    
  parser = argparse.ArgumentParser()

  parser.add_argument("--model_name", help= "The hf model name", type=str)
  parser.add_argument("--data_dir", help= "The path where the data reside", type=str)

  parser.add_argument("--target_language",
                      help= "The language to which thew data should be translated",
                      type=str)

  args = vars(parser.parse_args())
  # dict of argument

  main(**args)
