import json
import requests
import pytrec_eval

# Paths to your queries and qrels file
QUERIES_FILE = "queries.txt"
QRELS_FILE = "qrels.txt"
EC2_ENDPOINT = "http://ec2-endpoint>:5000/search"
TOP_K = 10  # number of results to retrieve per query

def load_queries(queries_file):
    """
    Load queries from a tab-separated file: query_id\tquery_text
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
    Load qrels from a TREC-format qrels file: qid 0 docid relevance
    Returns a dict suitable for pytrec_eval:
    {
      'qid': {
          'docid': relevance_score,
          ...
      },
      ...
    }
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

def query_ec2_endpoint(query_text):
    """
    Given a query text, post it to the EC2 endpoint and return retrieved passages.
    Expecting response as JSON array: [{"id": "X", "passage": "..."}, ...]
    """
    payload = {"query": query_text}
    response = requests.post(EC2_ENDPOINT, json=payload)
    response.raise_for_status()
    passages = response.json()
    return passages

def build_run(queries):
    """
    Build a run dictionary:
    run[qid][docid] = score
    This represents the system's retrieval results for evaluation.
    """
    run = {}
    for qid, qtext in queries.items():
        # Query the EC2 endpoint
        try:
            results = query_ec2_endpoint(qtext)
        except requests.RequestException as e:
            print(f"Error querying {qid}: {e}")
            results = []

        # Assign a score for ranking. Assume returned results are in ranked order.
        run[qid] = {}
        for rank, passage in enumerate(results, start=1):
            docid = passage['id']
            # Using a simple scoring scheme: higher rank gets higher score
            score = TOP_K - rank + 1
            run[qid][docid] = score
    return run

def evaluate(qrels, run):
    """
    Evaluate using pytrec_eval, compute NDCG, MRR, and Recall@10.
    """
    metrics = {
        'ndcg',
        'recip_rank',   # for MRR
        'recall_10'     # recall at cutoff 10
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

    # Build the run by querying the EC2 endpoint
    run = build_run(queries)

    # Evaluate using pytrec_eval
    avg_ndcg, avg_mrr, avg_recall = evaluate(qrels, run)
    print(f"Average NDCG: {avg_ndcg:.4f}")
    print(f"Average MRR: {avg_mrr:.4f}")
    print(f"Average Recall@10: {avg_recall:.4f}")

if __name__ == "__main__":
    main()
