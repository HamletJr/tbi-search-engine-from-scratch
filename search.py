import argparse
from bsbi import BSBIIndex
from compression import VBEPostings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieve dokumen menggunakan TF-IDF atau BM25.')
    parser.add_argument('--scoring', type=str, choices=['tf-idf', 'bm25'], default='tf-idf',
                        help='Pilih metode scoring (tf-idf atau bm25)')
    args = parser.parse_args()

    # sebelumnya sudah dilakukan indexing
    # BSBIIndex hanya sebagai abstraksi untuk index tersebut
    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = VBEPostings, \
                              output_dir = 'index')

    queries = ["alkylated with radioactive iodoacetate", \
               "psychodrama for disturbed children", \
               "lipid metabolism in toxemia and normal pregnancy"]
               
    for query in queries:
        print("Query  : ", query)
        print(f"Results ({args.scoring.upper()}):")
        
        if args.scoring == 'bm25':
            results = BSBI_instance.retrieve_bm25(query, k = 10)
        elif args.scoring == 'tf-idf':
            results = BSBI_instance.retrieve_tfidf(query, k = 10)
            
        for (score, doc) in results:
            print(f"{doc:30} {score:>.3f}")
        print()