import sys, os, re, torch, warnings
import regex as re_adv
import pandas as pd
from sentence_transformers import SentenceTransformer
import contractions
from itertools import islice
from tqdm import tqdm

# Pre-compile regex globally (Done once, not 10,000 times per chunk)
EMOJI_REGEX = re_adv.compile(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+", flags=re.UNICODE)
CLEAN_REGEX = re_adv.compile(r'[^\p{L}\p{N}\s.,!?]', flags=re_adv.V1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('all-mpnet-base-v2', device=device)
warnings.filterwarnings("ignore")

def text_process(text):
    if not isinstance(text, str): return ""
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = EMOJI_REGEX.sub('', text)
    return CLEAN_REGEX.sub('', text).strip()

def main(file_name, chunk_size=10000):
    out_dir = os.path.basename(file_name).replace('.ft.txt', '') + '_embeddings'
    os.makedirs(out_dir, exist_ok=True)

    # Calculate total lines to determine the "finish line" for the progress bar
    print(f"Counting lines in {file_name}...")
    with open(file_name, 'rb') as f:
        total_lines = sum(1 for _ in f)
    total_chunks = (total_lines + chunk_size - 1) // chunk_size

    with open(file_name, 'r', encoding='utf-8') as f:
        # Now tqdm knows the 'total'
        pbar = tqdm(total=total_chunks, desc="Overall Progress", unit="chunk")
        i = 0
        while True:
            lines = list(islice(f, chunk_size))
            if not lines:
                break

            data = [line.strip().split(' ', 1) for line in lines if line.strip()]
            df = pd.DataFrame(data, columns=['label', 'text'])
            df['label'] = df['label'].map(lambda x: 1 if x == '__label__2' else 0)

            processed_text = df['text'].map(text_process).tolist()

            # encode progress hidden to keep the CMD clean
            embeddings = model.encode(processed_text, batch_size=256, show_progress_bar=True)

            output_df = pd.DataFrame({'label': df['label'], 'embedding': list(embeddings)})
            output_df.to_parquet(os.path.join(out_dir, f"chunk_{i}.parquet"))

            i += 1
            pbar.update(1)
        pbar.close()

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