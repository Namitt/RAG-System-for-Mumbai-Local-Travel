# RAG System for Mumbai Local Travel
 
A Retrieval-Augmented Generation (RAG) pipeline that answers natural-language questions about Mumbai's local railway stations. It retrieves the most relevant station information from a custom dataset using semantic search, then feeds that context to a language model to generate a response.
 
This is a learning project built to understand the end-to-end mechanics of a RAG system: embedding, vector indexing, similarity search, and grounded generation.
 
## How it works
 
The pipeline has five stages:
 
1. **Data loading & preprocessing** — A CSV of Mumbai local-train stations is loaded with Pandas. The `About` and `Nearby attractions` fields are combined into a single text field per station, with missing values filled.
2. **Text chunking** — Each station's combined text is split into overlapping chunks (512 words, 50-word overlap) so no single input exceeds the embedding model's limit while preserving context across boundaries.
3. **Embedding** — Each chunk is encoded into a dense vector using the `all-MiniLM-L6-v2` Sentence-Transformer, chosen as a balance between speed and semantic quality. Embeddings are cached to JSON to avoid recomputation on re-runs.
4. **Retrieval (FAISS)** — Embeddings are L2-normalised and stored in a FAISS `IndexFlatL2` index. At query time, the user's question is embedded with the same model and the index returns the nearest chunk(s) by Euclidean distance.
5. **Generation (GPT-2)** — The retrieved chunk is injected into a prompt and passed to GPT-2, which generates a free-text answer grounded in the retrieved context.
```
Query → embed → FAISS nearest-neighbour search → retrieved chunk → prompt → GPT-2 → response
```
 
## Tech stack
 
| Component | Choice |
|---|---|
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector search | FAISS (`IndexFlatL2`, L2-normalised) |
| Generation | GPT-2 (Hugging Face Transformers) |
| Data handling | Pandas, NumPy |
| Environment | Python 3.10, PyTorch, Google Colab |
 
## Dataset
 
A custom CSV (`Mumbai Local Train Dataset.csv`) describing local railway stations across the Mumbai Suburban Railway network — each row covers a station's history, layout, connectivity and nearby attractions.
 
## Running it
 
The project runs as a single notebook (`5568424.ipynb`), built in Google Colab.
 
```bash
pip install faiss-cpu sentence_transformers transformers torch pandas
```
 
Open the notebook and run the cells in order. The dataset is pulled directly from this repo, so no manual download is needed. (Use `faiss-gpu` instead of `faiss-cpu` if running on a GPU.)
 
## Example
 
**Query:** "What is the significance of Thane station?"
 
The system retrieves the Thane station chunk and generates a response opening with grounded facts — Thane as an A1-category station, the first passenger railway service in India (16 April 1853, Bori Bunder to Thane), passenger volumes, and platform layout.
 
## Limitations & what I'd improve
 
This is a prototype, and the generation stage is its weakest link — worth being explicit about:
 
- **GPT-2 hallucinates beyond the retrieved context.** Responses tend to start accurate (echoing the retrieved chunk) but drift into invented detail once the model runs past the grounded text. GPT-2 is small and not instruction-tuned, so it doesn't reliably stay anchored to the provided context.
- **Sampling parameters were inert.** `temperature` and `top_p` were set without `do_sample=True`, so generation ran greedily and those knobs had no effect — a bug to fix.
- **Retrieval is exhaustive.** `IndexFlatL2` does a brute-force search, which is fine at this dataset size but wouldn't scale.
Concrete next steps: swap GPT-2 for an instruction-tuned model (e.g. Flan-T5 or a small Llama variant) to keep answers grounded; constrain generation to extract-and-summarise rather than open-ended continuation; enable proper sampling; and evaluate retrieval quality (e.g. recall@k) rather than judging on generation alone.
 
## What this project demonstrates
 
The full RAG loop assembled from scratch — semantic embedding, vector indexing, nearest-neighbour retrieval, and context-grounded generation — plus an honest read of where a naive generator breaks down and how to fix it.
 





