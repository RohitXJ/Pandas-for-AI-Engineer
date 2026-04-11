import sys, os, torch, warnings, tqdm
import regex as re_adv
import pandas as pd # Use standard pandas for initial CPU text work
import cudf        # Use cuDF for the final GPU-accelerated save
import contractions
from itertools import islice
from sentence_transformers import SentenceTransformer

# 1. Pre-compile Regex Globally (Huge speed boost)
EMOJI_PATTERN = re_adv.compile(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+", flags=re_adv.UNICODE)
CLEAN_PATTERN = re_adv.compile(r'[^\p{L}\p{N}\s.,!?]', flags=re_adv.V1)
URL_PATTERN = re_adv.compile(r'http\S+|https\S+|www\S+')

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('all-mpnet-base-v2', device=device)
warnings.filterwarnings("ignore")

def text_process(text):
    if not isinstance(text, str): return ""
    text = contractions.fix(text)
    text = URL_PATTERN.sub('', text)
    text = EMOJI_PATTERN.sub('', text)
    return CLEAN_PATTERN.sub('', text).strip()

def main(file_name, chunk_size=10000):
    out_dir = os.path.basename(file_name).replace('.ft.txt', '') + '_embeddings_data'
    os.makedirs(out_dir, exist_ok=True)

    # 2. Pre-calculate total chunks for the "Overall" Progress Bar
    print(f"Scanning {file_name} for total line count...")
    with open(file_name, 'rb') as f:
        total_lines = sum(1 for _ in f)
    total_chunks = (total_lines + chunk_size - 1) // chunk_size

    with open(file_name, 'r', encoding='utf-8') as f:
        pbar = tqdm.tqdm(total=total_chunks, desc="Overall Progress", unit="chunk")

        for i in range(total_chunks):
            lines = list(islice(f, chunk_size))
            if not lines: break

            # 3. Efficient Parsing (Split once)
            labels = [1 if l.startswith('__label__2') else 0 for l in lines]
            texts = [l.split(' ', 1)[1] if ' ' in l else "" for l in lines]

            # 4. CPU-bound cleaning (Mapped for speed)
            processed_texts = list(map(text_process, texts))

            # 5. Maximize GPU Throughput (Increased Batch Size)
            # Higher batch_size (128-256) reduces GPU idle time
            embeddings = model.encode(
                processed_texts,
                batch_size=256,
                show_progress_bar=True,
                convert_to_numpy=True
            )

            # 6. Final Data Assembly on GPU (cuDF)
            # We move data to GPU only once at the end for the Parquet write
            gdf = cudf.DataFrame({
                'label': labels,
                'embedding': list(embeddings)
            })

            gdf.to_parquet(os.path.join(out_dir, f"chunk_{i+1}.parquet"))
            pbar.update(1)

        pbar.close()
    print(f"\nSuccess! Saved to {out_dir}/")

if __name__ == '__main__':
    # Using your existing logic for Colab/CMD compatibility
    #input_file = '/content/test.ft.txt'
    #c_size = 5000
    if len(sys.argv) > 1 and sys.argv[1] != '-f':
        input_file = sys.argv[1]
        if len(sys.argv) > 2:
            try: c_size = int(sys.argv[2])
            except: pass
    main(input_file, c_size)