"""
Detect duplicate rule content using sentence embeddings and Faiss similarity search.
"""

import argparse
import json
import os
import torch
import numpy as np
import faiss
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def parse_args():
    """
    Define and parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Similarity Detection for Duplicate Rule Content"
    )
    parser.add_argument(
        "--sentence_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
    )
    parser.add_argument("--input_file", type=str, default="./data/step2/2.3_rules.json")
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/step2/deduplication",
    )
    parser.add_argument(
        "--encoding_batch_size",
        type=int,
        default=65536,
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--search_space_size",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--search_batch_size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--device", type=int, default=0, help="ID of the GPU device to use (-1 for CPU)"
    )
    parser.add_argument(
        "--save_faiss_index", action="store_true", help="Save the Faiss index"
    )
    parser.add_argument(
        "--no_save_faiss_index",
        action="store_false",
        dest="save_faiss_index",
        help="Do not save the Faiss index",
    )
    parser.set_defaults(save_faiss_index=True)
    return parser.parse_args()


def main(args):
    """
    Execute duplicate detection pipeline using embeddings and Faiss.
    """
    model = SentenceTransformer(args.sentence_model)
    if args.device >= 0 and torch.cuda.is_available():
        device_name = f"cuda:{args.device}"
    else:
        device_name = "cpu"
        if args.device >= 0 and not torch.cuda.is_available():
            print(
                f"Info: CUDA device {args.device} requested but unavailable. Using CPU."
            )
        elif args.device < 0:
            print("Info: CPU explicitly selected.")
        else:
            print("Info: CUDA not available. Using CPU.")

    model.to(device=device_name, dtype=torch.float32)
    print(f"Using device: {device_name} for SentenceTransformer model.")

    dataset_dict = load_dataset("json", data_files=args.input_file, keep_in_memory=True)
    dataset = dataset_dict["train"]

    with open(args.input_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    if len(records) != len(dataset):
        print(
            f"Warning: Number of records from json.load ({len(records)}) does not match dataset size ({len(dataset)})."
        )

    texts = [
        "\n".join(row["rule_content"]) for row in tqdm(dataset, desc="Preparing texts")
    ]

    if not texts:
        print("No texts were extracted. Exiting.")
        return

    embeddings_list = []
    for i in tqdm(
        range(0, len(texts), args.encoding_batch_size), desc="Encoding batches"
    ):
        batch_texts = texts[i : i + args.encoding_batch_size]
        emb_batch_tensor = model.encode(
            batch_texts, convert_to_tensor=True, show_progress_bar=False
        )
        embeddings_list.append(emb_batch_tensor.cpu().numpy())

    if not embeddings_list:
        print("No embeddings were generated. Exiting.")
        return
    embeddings = np.concatenate(embeddings_list, axis=0)

    dataset = dataset.add_column("embeddings", embeddings.tolist())

    faiss_creation_device = args.device if device_name.startswith("cuda") else -1
    dataset.add_faiss_index(column="embeddings", device=faiss_creation_device)

    if args.save_faiss_index:
        original_faiss_index = dataset.get_index("embeddings").faiss_index
        index_to_save = original_faiss_index
        if device_name.startswith("cuda"):
            print("Converting Faiss index from GPU to CPU for saving...")
            try:
                index_to_save = faiss.index_gpu_to_cpu(original_faiss_index)
            except Exception as e:
                print(f"Error converting Faiss index: {e}")
        os.makedirs(args.output_path, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        index_filepath = os.path.join(args.output_path, f"{base_name}.faiss")
        try:
            faiss.write_index(index_to_save, index_filepath)
            print(f"Successfully saved Faiss index to {index_filepath}")
        except Exception as e:
            print(f"Failed to save Faiss index: {e}")

    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    output_duplicates_file = os.path.join(
        args.output_path, f"{base_name}_duplicates.jsonl"
    )
    print(f"Writing duplicate detection results to {output_duplicates_file}")

    num_search_batches = (
        len(dataset) + args.search_batch_size - 1
    ) // args.search_batch_size
    with open(output_duplicates_file, "w", encoding="utf-8") as out_f:
        for batch_idx in tqdm(
            range(num_search_batches), desc="Searching batches for duplicates"
        ):
            start_index = batch_idx * args.search_batch_size
            end_index = min((batch_idx + 1) * args.search_batch_size, len(dataset))
            query_embeddings_batch = embeddings[start_index:end_index]
            scores_batch, indices_batch = dataset.search_batch(
                index_name="embeddings",
                queries=query_embeddings_batch,
                k=args.search_space_size,
            )
            for i in range(len(scores_batch)):
                idx_global = start_index + i
                scores = scores_batch[i]
                indices = indices_batch[i]
                min_dist = float("inf")
                if len(indices) > 0:
                    if indices[0] == idx_global and len(scores) > 1:
                        min_dist = float(scores[1])
                    else:
                        min_dist = float(scores[0])
                neighbors = [
                    int(idx)
                    for idx, sc in zip(indices, scores)
                    if sc < args.distance_threshold and idx != idx_global
                ]
                record = records[idx_global].copy()
                record["min_neighbor_distance"] = (
                    min_dist if min_dist != float("inf") else -1.0
                )
                record["repeat_count"] = len(neighbors)
                if neighbors:
                    rec_idx = min(neighbors)
                    try:
                        record["min_similar_rule_id"] = records[rec_idx].get("id")
                    except (KeyError, IndexError):
                        record["min_similar_rule_id"] = None
                else:
                    record["min_similar_rule_id"] = None
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Duplicate detection completed.")


if __name__ == "__main__":
    args = parse_args()
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file not found at {args.input_file}")
    else:
        os.makedirs(args.output_path, exist_ok=True)
        main(args)
