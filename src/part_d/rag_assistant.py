import os
import sys
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

sys.path.insert(0, PROJECT_ROOT)

from mock_llm import chat

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POLICY_DIR = "data/policies"


def chunk_text(text, chunk_size=40, overlap=10):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunks.append(
            " ".join(
                words[start:end]
            )
        )

        if end >= len(words):
            break

        start += (
            chunk_size
            - overlap
        )

    return chunks


def build_chunks(overlap=10):

    all_chunks = []

    for fname in os.listdir(
        POLICY_DIR
    ):

        if not fname.endswith(
            ".txt"
        ):
            continue

        path = os.path.join(
            POLICY_DIR,
            fname
        )

        with open(
            path,
            encoding="utf-8"
        ) as f:

            text = f.read()

        chunks = chunk_text(
            text,
            40,
            overlap
        )

        for chunk in chunks:

            all_chunks.append({
                "source": fname,
                "text": chunk
            })

    return all_chunks


def build_index(overlap=10):

    chunks = build_chunks(overlap)

    texts = [
        c["text"]
        for c in chunks
    ]

    vectorizer = TfidfVectorizer()

    matrix = vectorizer.fit_transform(
        texts
    )

    return (
        chunks,
        vectorizer,
        matrix
    )


def retrieve(
    query,
    k=3,
    overlap=10
):

    chunks, vectorizer, matrix = (
        build_index(overlap)
    )

    q = vectorizer.transform(
        [query]
    )

    scores = cosine_similarity(
        q,
        matrix
    )[0]

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for chunk, score in ranked[:k]:

        results.append({
            "source": chunk["source"],
            "score": float(score),
            "text": chunk["text"]
        })

    return results

def answer(question, overlap=10):

    # Retrieve top 3 relevant chunks
    top3 = retrieve(
        question,
        k=3,
        overlap=overlap
    )

    # Build all chunks to identify neighbouring chunks
    all_chunks = build_chunks(overlap)

    # Identify selected chunks
    selected_indices = set()

    for result in top3:

        for i, chunk in enumerate(all_chunks):

            if (
                chunk["source"] == result["source"]
                and chunk["text"] == result["text"]
            ):

                # Include previous, current and next chunk
                for j in range(
                    max(0, i - 1),
                    min(len(all_chunks), i + 2)
                ):

                    if all_chunks[j]["source"] == result["source"]:
                        selected_indices.add(j)

                break

    # Arrange selected chunks in document order
    context_chunks = [
        all_chunks[i]
        for i in sorted(selected_indices)
    ]

    # Construct context
    context = ""

    for chunk in context_chunks:

        context += (
            f"[{chunk['source']}]\n"
            f"{chunk['text']}\n\n"
        )

    messages = [

        {
            "role": "system",
            "content": """
            POLICY ASSISTANT

            Answer only using the supplied policy context.

            Read all relevant context carefully.

            If a policy statement continues into
            another chunk, combine the information
            before answering.

            Do not invent information.

            If the information cannot be found,
            say exactly:

            I could not find that information
            in the policy documents.
            """
        },

        {
            "role": "user",
            "content": (
                "CONTEXT:\n"
                + context
                + "\nQUESTION:\n"
                + question
            )
        }
    ]

    response = chat(messages)

    return {

        "answer": (
            response
            .choices[0]
            .message
            .content
        ),

        "sources": list(dict.fromkeys(
            c["source"] for c in top3
        ))
    }

def evaluate(overlap=10):

    eval_df = pd.read_csv(
        "data/raw/policy_eval.csv"
    )

    retrieval_hits = 0
    answer_hits = 0
    fallback_hits = 0

    answerable = 0
    unanswerable = 0

    for _, row in eval_df.iterrows():

        result = answer(row["question"], overlap=overlap)

        sources = result["sources"]
        ans = str(result["answer"])

        expected_doc = str(row["expected_doc"]).strip()
        expected_fact = row["expected_fact"]

        if expected_doc.upper() != "NONE":

            answerable += 1

            if expected_doc in sources:
                retrieval_hits += 1

            if pd.notna(expected_fact) and str(expected_fact) in ans:
                answer_hits += 1
            else:
                print("FAIL (overlap=%d):" % overlap, row["question"],
                      "| expected:", expected_fact)

        else:

            unanswerable += 1

            if ans.startswith("I could not find"):
                fallback_hits += 1

    return {
        "overlap": overlap,
        "retrieval": f"{retrieval_hits}/{answerable}",
        "grounding": f"{answer_hits}/{answerable}",
        "fallback": f"{fallback_hits}/{unanswerable}"
    }

if __name__ == "__main__":
    from collections import Counter

    print("\nNORMAL OVERLAP EVALUATION")
    print(evaluate(overlap=10))

    print("\nZERO OVERLAP EVALUATION")
    print(evaluate(overlap=0))

    print("\nCHUNK COUNTS (overlap=10)")
    c10 = build_chunks(10)
    print(len(c10), dict(Counter(x["source"] for x in c10)))

    print("\nZERO-OVERLAP CHUNKS (refund and payment)")
    for c in build_chunks(0):
        if c["source"] in ("refund_policy.txt", "payment_policy.txt"):
            print(c["source"], "|", c["text"], "\n")