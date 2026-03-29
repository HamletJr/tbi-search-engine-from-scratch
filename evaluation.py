import re
import math
import argparse
from bsbi import BSBIIndex
from spimi import SPIMIIndex
from lsi import LSIIndex
from compression import StandardPostings, VBEPostings, OptPForDeltaPostings, BP128Postings

######## >>>>> sebuah IR metric: RBP p = 0.8

def rbp(ranking, p = 0.8):
  """ menghitung search effectiveness metric score dengan 
      Rank Biased Precision (RBP)

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score RBP
  """
  score = 0.
  for i in range(1, len(ranking)):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score

def dcg(ranking):
  """
  Menghitung search effectiveness metric score dengan 
  Discounted Cumulative Gain (DCG)

  Parameters
  ----------
  ranking: List[int]
      Vektor biner yang merepresentasikan relevansi dari dokumen di rank 1, 2, 3, dst.
    
  Returns
  -------
  Float
    Score DCG
  """
  score = ranking[0]
  for i in range(2, len(ranking) + 1):
    pos = i - 1
    score += ranking[pos] / (math.log2(i))
  return score

def ndcg(ranking):
  """
  Menghitung search effectiveness metric score dengan 
  Normalized Discounted Cumulative Gain (NDCG)

  Parameters
  ----------
  ranking: List[int]
    Vektor biner yang merepresentasikan relevansi dari dokumen di rank 1, 2, 3, dst.
  
  Returns
  -------
  Float
    Score NDCG
  """
  ideal_ranking = sorted(ranking, reverse=True)
  ideal_dcg = dcg(ideal_ranking)
  if ideal_dcg == 0:
    return 0.0
  return dcg(ranking) / ideal_dcg

def ap(ranking, total_relevant_docs = 0):
  """
  Menghitung search effectiveness metric score dengan 
  Average Precision (AP)

  Parameters
  ----------
  ranking: List[int]
      Vektor biner yang merepresentasikan relevansi dari dokumen di rank 1, 2, 3, dst.
  total_relevant_docs: int
      Jumlah dokumen yang relevan dengan query. Jika tidak diberikan, akan menggunakan jumlah dokumen relevan yang ditemukan dalam ranking.

  Returns
  -------
  Float
    Score AP
  """
  score = 0.0
  num_relevant = 0
  for i in range(1, len(ranking) + 1):
    pos = i - 1
    if ranking[pos] == 1:
      num_relevant += 1
      score += num_relevant / i
  if num_relevant == 0:
    return 0.0
  if total_relevant_docs == 0:
    total_relevant_docs = num_relevant
  return score / total_relevant_docs

######## >>>>> memuat qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ memuat query relevance judgment (qrels) 
      dalam format dictionary of dictionary
      qrels[query id][document id]

      dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
      relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
      Doc 10 tidak relevan dengan Q3.

  """
  qrels = {"Q" + str(i) : {i:0 for i in range(1, max_doc_id + 1)} \
                 for i in range(1, max_q_id + 1)}
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1
  return qrels

######## >>>>> EVALUASI !

def eval(qrels, query_file = "queries.txt", k = 1000, eval_metric = "rbp", scoring_method = "tfidf", index_instance = None):
  """ 
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    untuk setiap query, kembalikan top-1000 documents
  """
  with open(query_file) as file:
    eval_scores = []
    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])

      # HATI-HATI, doc id saat indexing bisa jadi berbeda dengan doc id
      # yang tertera di qrels
      ranking = []
      if scoring_method == "tf-idf":
        for (score, doc) in index_instance.retrieve_tfidf(query, k = k):
            did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
            ranking.append(qrels[qid][did])
      elif scoring_method == "bm25":
        for (score, doc) in index_instance.retrieve_bm25(query, k = k):
          did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
          ranking.append(qrels[qid][did])
      elif scoring_method == "bm25-wand":
        for (score, doc) in index_instance.retrieve_bm25_wand(query, k = k):
          did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
          ranking.append(qrels[qid][did])
      elif scoring_method == "lsi":
        for (score, doc) in index_instance.retrieve(query, k = k):
          did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
          ranking.append(qrels[qid][did])

      if eval_metric == "rbp":
        eval_scores.append(rbp(ranking))
      elif eval_metric == "dcg":
        eval_scores.append(dcg(ranking))
      elif eval_metric == "ndcg":
        eval_scores.append(ndcg(ranking))
      elif eval_metric == "ap":
        eval_scores.append(ap(ranking))

  print(f"Hasil evaluasi {scoring_method.upper()} terhadap 30 queries")
  print(f"{eval_metric.upper()} score =", sum(eval_scores) / len(eval_scores))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Evaluasi IR system')
  parser.add_argument('--eval', action='store', help='Pilih metric evaluasi: rbp, dcg, ndcg, ap', default='rbp')
  parser.add_argument('--scoring', type=str, choices=['tf-idf', 'bm25', 'bm25-wand', 'lsi'], default='tf-idf',
                      help='Pilih metode scoring (tf-idf, bm25, bm25-wand, atau lsi)')
  parser.add_argument('--compression', type=str, default='vbe', choices=['standard', 'vbe', 'optpfor', 'bp128'], help='Metode compression untuk postings list')
  parser.add_argument('--spimi', action='store_true', help='Gunakan SPIMI untuk indexing')
  args = parser.parse_args()

  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "qrels salah"
  assert qrels["Q1"][300] == 0, "qrels salah"

  if args.scoring == 'lsi':
      index_instance = LSIIndex(data_dir='collection', output_dir='index', latent_dim=100)
      index_instance.load()
  elif args.spimi:
    match args.compression:
      case 'standard':
          index_instance = SPIMIIndex(data_dir = 'collection', \
                                    postings_encoding = StandardPostings, \
                                    output_dir = 'index')
      case 'vbe':
          index_instance = SPIMIIndex(data_dir = 'collection', \
                                    postings_encoding = VBEPostings, \
                                    output_dir = 'index')
      case 'optpfor':
          index_instance = SPIMIIndex(data_dir = 'collection', \
                                    postings_encoding = OptPForDeltaPostings, \
                                    output_dir = 'index')
      case 'bp128':
          index_instance = SPIMIIndex(data_dir = 'collection', \
                                    postings_encoding = BP128Postings, \
                                    output_dir = 'index')
  else:
    match args.compression:
      case 'standard':
          index_instance = BSBIIndex(data_dir = 'collection', \
                                  postings_encoding = StandardPostings, \
                                  output_dir = 'index')
      case 'vbe':
          index_instance = BSBIIndex(data_dir = 'collection', \
                                  postings_encoding = VBEPostings, \
                                  output_dir = 'index')
      case 'optpfor':
          index_instance = BSBIIndex(data_dir = 'collection', \
                                  postings_encoding = OptPForDeltaPostings, \
                                  output_dir = 'index')
      case 'bp128':
          index_instance = BSBIIndex(data_dir = 'collection', \
                                  postings_encoding = BP128Postings, \
                                  output_dir = 'index')

  eval(qrels, eval_metric = args.eval, scoring_method = args.scoring, index_instance = index_instance)