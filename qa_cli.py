# qa_cli.py
import argparse
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
import os
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False


def simple_cli(query, index_dir='index'):
    with open(os.path.join(index_dir, 'chunks.pkl'), 'rb') as f:
        obj = pickle.load(f)
    chunks = obj['chunks']
    meta = obj['metadata']
    model = SentenceTransformer('all-MiniLM-L6-v2')
    q_emb = model.encode([query], convert_to_numpy=True).astype('float32')
    if HAS_FAISS:
        idx = faiss.read_index(os.path.join(index_dir, 'faiss_index.bin'))
        D, I = idx.search(q_emb, 3)
        for d,i in zip(D[0], I[0]):
            print('SCORE', d, 'SOURCE', meta[i])
            print(chunks[i][:800])
    else:
        print('No FAISS available. Use streamlit UI or fallback index')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--q', required=True)
    args = p.parse_args()
    simple_cli(args.q)
