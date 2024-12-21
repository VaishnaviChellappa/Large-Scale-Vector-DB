import pytrec_eval
from pyserini.search import SimpleSearcher

# Paths to queries and qrels
QUERIES_FILE = "queries.txt"  # Format: qid \t query_text
QRELS_FILE = "qrels.txt"      # Format: qid 0 docid relevance
INDEX_DIR = "msmarco_index"   # Path to your BM25 index
TOP_K = 10

def load_queries(queries_file):
    """
    Load queries from a file of the form: qid \t query_text
    """
    queries = {}
    with open(queries_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, qtext = line.split('\t', 1)
            queries[qid] = qtext
    return queries

def load_qrels(qrels_file):
    """
    Load qrels in TREC format: qid 0 docid relevance
    Returns a dict suitable for pytrec_eval: { qid: { docid: rel, ...}, ... }
    """
    qrels = {}
    with open(qrels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, _, docid, rel = line.split()
            rel = int(rel)
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = rel
    return qrels

def evaluate(qrels, run):
    """
    Evaluate the run using pytrec_eval, computing NDCG, MRR, and Recall@10.
    """
    metrics = {
        "ndcg",
        "recip_rank",   # MRR
        "recall_10"     # recall at cutoff 10
    }

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    results = evaluator.evaluate(run)

    ndcg_values = []
    mrr_values = []
    recall_values = []

    for res in results.values():
        if 'ndcg' in res:
            ndcg_values.append(res['ndcg'])
        if 'recip_rank' in res:
            mrr_values.append(res['recip_rank'])
        if 'recall_10' in res:
            recall_values.append(res['recall_10'])

    avg_ndcg = sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0.0
    avg_mrr = sum(mrr_values) / len(mrr_values) if mrr_values else 0.0
    avg_recall = sum(recall_values) / len(recall_values) if recall_values else 0.0

    return avg_ndcg, avg_mrr, avg_recall

def main():
    # Load queries and qrels
    queries = load_queries(QUERIES_FILE)
    qrels = load_qrels(QRELS_FILE)

    # Initialize a Pyserini searcher on the BM25 index
    searcher = SimpleSearcher(INDEX_DIR)
    # BM25 parameters for better performance
    searcher.set_bm25(k1=0.9, b=0.4)

    # Build a run dictionary: run[qid][docid] = score
    run = {}
    for qid, qtext in queries.items():
        hits = searcher.search(qtext, k=TOP_K)
        run[qid] = {}
        for i, hit in enumerate(hits):
            # Use hit.score as the retrieval score
            run[qid][hit.docid] = float(hit.score)

    # Evaluate using pytrec_eval
    avg_ndcg, avg_mrr, avg_recall = evaluate(qrels, run)
    print(f"Average NDCG: {avg_ndcg:.4f}")
    print(f"Average MRR: {avg_mrr:.4f}")
    print(f"Average Recall@10: {avg_recall:.4f}")

if __name__ == "__main__":
    main()
