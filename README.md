# Searching Technical Documents with Text Vectors and Cosine Similarity

**EGR 444 Peer-Led ML Workshop** · 45 minutes · Google Colab, CPU only, no training

## 1. The project need
A technician has a DC motor that won't start and a long equipment manual. Instead of scrolling, they type
**"Why isn't my DC motor turning on?"** and get the most relevant troubleshooting section back.

**Decision this workshop answers:** should a team build a document lookup tool with **keyword search (TF-IDF)**
or **meaning search (sentence embeddings)**?

## 2. What you will produce
A working search in a Colab notebook that ranks 30 manual sections by **cosine similarity** to a question and
returns the **top 3 matches with scores**. You'll build it twice (TF-IDF, then embeddings), run your own question
through both, and explain why the results differ.

## 3. Files
| File | What it is |
|---|---|
| `workshop_notebook.ipynb` | **Start here.** The tutorial notebook, with expected outputs saved in it |
| `data/dc_motor_manual.csv` | 30 short manual sections (`section_id`, `title`, `text`) |
| `README.md` | This guide: setup, steps, expected results, troubleshooting |

## 4. Setup (about 3 minutes)
1. Open `workshop_notebook.ipynb` in Google Colab (**File → Upload notebook**, or use the GitHub link).
2. The runtime can stay on **CPU**. You don't need a GPU.
3. Run the **Step 0** cell. It installs `sentence-transformers` (red "dependency" warnings from pip are normal).
4. Run the **Step 1** cell. It downloads the CSV from `DATA_URL`. If it can't, upload `dc_motor_manual.csv` with the
   📁 icon on the left and run the cell again.

Requirements (Colab already has these except sentence-transformers): `pandas`, `numpy`, `scikit-learn`, `sentence-transformers`.

## 5. Tutorial steps and checkpoints
| Step | What you do | ✅ Expected result |
|---|---|---|
| 1 | Load the CSV | `df.shape` → `(30, 3)` |
| 2 | TF-IDF vectors | `X_tfidf.shape` → `(30, 315)` (sections × vocabulary words) |
| 3 | Vectorize the question, compute cosine similarity | question `(1, 315)`, scores `(30,)` |
| 3 | TF-IDF top 3 for *"Why isn't my DC motor turning on?"* | `0.340 Turning the shaft during installation` · `0.171 Motor runs in the wrong direction` · `0.122 Insulation resistance test` (**wrong section on top!**) |
| 3 | Look at the words TF-IDF sees | `['isn', 'dc', 'motor', 'turning']`: it matched on "turning" and "dc", not on meaning |
| 4 | Embeddings with `all-MiniLM-L6-v2` | `X_emb.shape` → `(30, 384)` (sections × embedding size) |
| 4 | Embedding top 3, same question | **"Motor fails to start"** at or near #1 |
| 5 | `compare()` both methods side by side | Two ranked lists |
| 6 | **Your turn:** your own question + explanation | See section 6 |

**Reasonable score ranges** (the two methods use different scales, so compare rankings, not raw numbers):

| | Strong match | Weak or unrelated |
|---|---|---|
| TF-IDF | about 0.3 and up | exactly 0.0 when no words are shared |
| Embeddings | about 0.5 and up | about 0.0 to 0.2, rarely exactly 0 |

Embedding scores can differ in the 3rd decimal place across library versions. That's fine.

## 6. Participant task (about 5 minutes)
1. Set `MY_QUESTION` to your own troubleshooting question. Try describing a symptom **without** the manual's words
   (*"it smells like something is cooking"*, *"the motor keeps speeding up and slowing down"*).
2. Run `compare(MY_QUESTION)`.
3. In the notebook, write down: Did both methods pick the same #1? If not, **which words** explain the TF-IDF result? Which would you trust?

**How to check your answer:** a good explanation points to a specific word match or mismatch, such as *"TF-IDF matched 'turning' in
the installation section, while the embeddings understood that 'turning on' means 'start'."*
**Bonus:** find a question where TF-IDF wins. Exact terms like *megohmmeter*, *growler*, or *encoder* are good tries.

## 7. Session plan (40 minutes + 5 minute buffer)
| Time | Activity | Notebook |
|---|---|---|
| 0–5 | Project need; show the finished side-by-side result | Step 5 output |
| 5–10 | Key idea: text → vectors → cosine angle. Everyone runs Steps 0–1 | Checkpoint 1 |
| 10–20 | TF-IDF build and search. Discover the wrong #1 result and why | Checkpoints 2–4 |
| 20–30 | Embeddings build and search, then side-by-side comparison | Checkpoints 5–6 |
| 30–35 | Participant task | Step 6 |
| 35–40 | Handoff: when to use it, limitation, next steps, questions | Step 7 |
| 40–45 | Buffer and troubleshooting | |

While one person leads, the other team members walk the room and check that everyone matches each checkpoint.

## 8. Troubleshooting
| Symptom / error | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: dc_motor_manual.csv` | CSV not downloaded or uploaded | Upload it with the 📁 icon (it must sit in `/content`, not a subfolder), then rerun Step 1 |
| CSV loads but `df.shape` is wrong or it's full of HTML | `DATA_URL` points to the GitHub *page*, not the *raw* file | Use the URL from the **Raw** button (`raw.githubusercontent.com/...`). Delete the bad file with `!rm dc_motor_manual.csv` and rerun |
| `ModuleNotFoundError: No module named 'sentence_transformers'` | Step 0 not run, or the runtime restarted | Rerun Step 0 |
| `NameError: name 'tfidf' / 'model' / 'X_emb' is not defined` | Cells run out of order, or the runtime reset | **Runtime → Run before** on the current cell |
| `ValueError: Incompatible dimension for X and Y matrices` | Question and sections are in different vector spaces (e.g. `fit_transform` on the question, or a TF-IDF question vs. `X_emb`) | Use `tfidf.transform([q])` with `X_tfidf`, and `model.encode([q])` with `X_emb` |
| TF-IDF top 3 are all `0.000` | None of your question's words are in the manual's vocabulary | That's the expected keyword-search weakness, so explain it in your answer. Check with `analyzer(MY_QUESTION)` |
| Model download hangs or shows an HTTP/Hugging Face error | Network hiccup or rate limit | Wait 30 seconds and rerun the cell. If it still fails, **Runtime → Restart session** and rerun from the top |
| "Runtime disconnected" | Colab idle timeout | Reconnect, then **Runtime → Run all** (about 1 minute) |
| Red pip warnings about dependency conflicts in Step 0 | Colab's preinstalled packages | Safe to ignore if it prints `Setup done` |

## 9. Handoff: when to use this
- **Useful when:** any team has a pile of documents (manuals, SOPs, tickets, papers) and people need the right passage quickly.
- **TF-IDF:** instant, no download, explainable, and strong on exact codes and terms. Weak on synonyms.
- **Embeddings:** handles paraphrases and synonyms. Needs a small model download and can blur exact identifiers. Many systems combine both ("hybrid search").
- **Key limitation:** it **finds** relevant text but does **not answer** the question. In a full RAG system, the top-k sections go to an LLM that writes the answer.
- **Adapt it:** swap in your own CSV of text chunks (one paragraph per row) and keep the code as is.

**Documentation**
- scikit-learn TfidfVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- scikit-learn cosine_similarity: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
- sentence-transformers: https://sbert.net
- all-MiniLM-L6-v2: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

**Credits:** The manual text is a simplified, workshop-written example and is not from a real product manual. The model is `all-MiniLM-L6-v2` by the sentence-transformers project (Apache 2.0).

## 10. Contribution note (for instructor)
| Member | Preparation | Workshop role |
|---|---|---|
| Nicholas Doran | dataset, notebook TF-IDF section | presents 10–20 min, floats 20–40 |
| Connor Valdez | Presentation slides, Code, Cosine similarity | Presents 10-20 minutes, floats 20-40 | 

