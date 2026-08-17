import os
import math
import re
from collections import Counter

# ============================================================
# INFORMATION RETRIEVAL SYSTEM
# TF-IDF + COSINE SIMILARITY DOCUMENT RANKING
# ============================================================

FOLDER = "documents"
MIN_WORDS = 100

# Common English words removed during preprocessing.
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "this", "to", "was", "were", "will", "with", "can",
    "also", "been", "being", "but", "into", "may", "more", "such", "than",
    "these", "those", "through", "using", "used", "use", "which", "while",
    "where", "when", "how", "about", "over", "under", "between", "during",
    "after", "before", "both", "each", "other", "should", "would", "could",
    "very", "only", "their", "them", "they", "we", "you", "your"
}


def preprocess(text):
    """
    Convert text into normalized tokens.
    1. Convert to lowercase.
    2. Extract alphabetic words.
    3. Remove common stop words.
    """
    words = re.findall(r"[a-z]+", text.lower())

    return [
        word for word in words
        if word not in STOP_WORDS and len(word) > 1
    ]


def load_documents():
    """
    Load exactly 10 TXT documents and verify
    that every document contains at least 100 words.
    """
    if not os.path.isdir(FOLDER):
        raise FileNotFoundError(
            f'\nERROR: "{FOLDER}" folder was not found.\n'
            f'Create the "documents" folder beside ranking.py.'
        )

    files = [
        file for file in os.listdir(FOLDER)
        if file.lower().endswith(".txt")
    ]

    # Natural numeric order: D1, D2, ..., D9, D10.
    files.sort(
        key=lambda x: int(re.search(r"\d+", x).group())
    )

    if len(files) != 10:
        raise ValueError(
            f"\nERROR: Expected exactly 10 TXT documents, "
            f"but found {len(files)}."
        )

    documents = []

    print("\nChecking documents...")
    print("-" * 78)

    for file in files:
        path = os.path.join(FOLDER, file)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Count original words for the professor's 100-word requirement.
        word_count = len(
            re.findall(r"\b[a-zA-Z]+\b", text)
        )

        if word_count < MIN_WORDS:
            raise ValueError(
                f'ERROR: "{file}" contains only {word_count} words. '
                f"Every document must contain at least {MIN_WORDS} words."
            )

        tokens = preprocess(text)

        if not tokens:
            raise ValueError(
                f'ERROR: "{file}" contains no usable words.'
            )

        documents.append({
            "name": file,
            "text": text,
            "word_count": word_count,
            "tokens": tokens
        })

        print(f"{file:<50}{word_count:>5} words")

    print("-" * 78)
    print("All 10 documents passed the minimum 100-word requirement.")

    return documents


def calculate_idf(documents):
    """
    Smoothed IDF:
    IDF(t) = log((N + 1) / (DF(t) + 1)) + 1

    N  = total number of documents
    DF = number of documents containing the term
    """
    total_documents = len(documents)
    document_frequency = Counter()

    for document in documents:
        unique_terms = set(document["tokens"])

        for term in unique_terms:
            document_frequency[term] += 1

    idf = {}

    for term, df in document_frequency.items():
        idf[term] = math.log(
            (total_documents + 1) / (df + 1)
        ) + 1

    return idf


def calculate_tf(tokens):
    """
    Normalized Term Frequency:
    TF(t,d) = count(t,d) / total_terms_in_document
    """
    counts = Counter(tokens)
    total_terms = len(tokens)

    return {
        term: count / total_terms
        for term, count in counts.items()
    }


def calculate_tfidf(tokens, idf):
    """Create a TF-IDF vector from document/query tokens."""
    tf = calculate_tf(tokens)

    return {
        term: tf_value * idf.get(term, 0.0)
        for term, tf_value in tf.items()
    }


def cosine_similarity(vector_a, vector_b):
    """
    Cosine Similarity:
    cos(theta) = (A . B) / (|A| * |B|)
    """
    common_terms = set(vector_a) & set(vector_b)

    dot_product = sum(
        vector_a[term] * vector_b[term]
        for term in common_terms
    )

    magnitude_a = math.sqrt(
        sum(value * value for value in vector_a.values())
    )

    magnitude_b = math.sqrt(
        sum(value * value for value in vector_b.values())
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (
        magnitude_a * magnitude_b
    )


def rank_documents(query, documents, idf):
    """Calculate similarity between query and all documents."""
    query_tokens = preprocess(query)

    if not query_tokens:
        return [], query_tokens

    query_vector = calculate_tfidf(
        query_tokens,
        idf
    )

    results = []

    for index, document in enumerate(documents):
        document_vector = calculate_tfidf(
            document["tokens"],
            idf
        )

        score = cosine_similarity(
            query_vector,
            document_vector
        )

        results.append({
            "rank": 0,
            "document": document["name"],
            "score": score,
            "index": index
        })

    # Highest similarity = Rank 1.
    # Original D1-D10 order breaks ties consistently.
    results.sort(
        key=lambda result: (
            -result["score"],
            result["index"]
        )
    )

    for rank, result in enumerate(results, start=1):
        result["rank"] = rank

    return results, query_tokens


def display_results(query, query_tokens, results):
    """Display the final ranked results."""
    print("\n")
    print("=" * 82)
    print("                 INFORMATION RETRIEVAL SYSTEM")
    print("=" * 82)

    print(f"\nQuery: {query}")
    print(f"Processed Query: {' '.join(query_tokens)}")

    print("\n" + "-" * 82)
    print(
        f"{'RANK':<8}"
        f"{'DOCUMENT':<52}"
        f"{'COSINE SIMILARITY':>22}"
    )
    print("-" * 82)

    for result in results:
        print(
            f"{result['rank']:<8}"
            f"{result['document']:<52}"
            f"{result['score']:>22.6f}"
        )

    print("-" * 82)

    if results:
        best = results[0]

        print(
            f"\nTop Result: {best['document']}"
            f" | Score: {best['score']:.6f}"
        )

    print("=" * 82)


def main():
    print("\n" + "=" * 82)
    print("              TF-IDF DOCUMENT RANKING SYSTEM")
    print("=" * 82)

    try:
        documents = load_documents()
        idf = calculate_idf(documents)

        print("\nSystem ready.")
        print("Enter a search query related to the documents.")
        print("Example: marine ecosystem")
        print('Type "exit" to close the program.')

        while True:
            query = input("\nEnter search query: ").strip()

            if query.lower() == "exit":
                print("\nIR system closed.")
                break

            if not query:
                print("Please enter a search query.")
                continue

            results, query_tokens = rank_documents(
                query,
                documents,
                idf
            )

            if not query_tokens:
                print(
                    "The query contains only common words. "
                    "Please enter more meaningful terms."
                )
                continue

            display_results(
                query,
                query_tokens,
                results
            )

    except (FileNotFoundError, ValueError) as error:
        print(error)


if __name__ == "__main__":
    main()
