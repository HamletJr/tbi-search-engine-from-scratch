import argparse
import time
from bsbi import BSBIIndex
from spimi import SPIMIIndex
from compression import StandardPostings, VBEPostings, OptPForDeltaPostings, BP128Postings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieve dokumen menggunakan TF-IDF atau BM25.')
    parser.add_argument('--scoring', type=str, choices=['tf-idf', 'bm25', 'bm25-wand'], default='tf-idf',
                        help='Pilih metode scoring (tf-idf, bm25, atau bm25-wand)')
    parser.add_argument('--verbose', action='store_true', help='Tampilkan informasi jumlah dokumen yang dievaluasi dan waktu proses retrieval')
    parser.add_argument('--compression', type=str, default='vbe', choices=['standard', 'vbe', 'optpfor', 'bp128'], help='Metode compression untuk postings list')
    parser.add_argument('--spimi', action='store_true', help='Gunakan indexing SPIMI untuk retrieval')
    parser.add_argument('--query', type=str, help='Query yang ingin diretrieve')
    args = parser.parse_args()

    # sebelumnya sudah dilakukan indexing
    # BSBIIndex hanya sebagai abstraksi untuk index tersebut
    if args.spimi:
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
            
    queries = ["alkylated with radioactive iodoacetate", \
               "psychodrama for disturbed children", \
               "lipid metabolism in toxemia and normal pregnancy"]
               
    if args.query:
        queries = [args.query]

    for query in queries:
        print("Query  : ", query)
        
        if args.scoring == 'bm25':
            if args.verbose:
                start_time = time.time()
                results = index_instance.retrieve_bm25(query, k = 10, verbose=True)
                end_time = time.time()
                print("Waktu proses retrieval: {:.6f} detik".format(end_time - start_time))
            else:
                results = index_instance.retrieve_bm25(query, k = 10)
        elif args.scoring == 'tf-idf':
            if args.verbose:
                start_time = time.time()
                results = index_instance.retrieve_tfidf(query, k = 10)
                end_time = time.time()
                print("Waktu proses retrieval: {:.6f} detik".format(end_time - start_time))
            results = index_instance.retrieve_tfidf(query, k = 10)
        elif args.scoring == 'bm25-wand':
            if args.verbose:
                start_time = time.time()
                results = index_instance.retrieve_bm25_wand(query, k = 10, verbose=True)
                end_time = time.time()
                print("Waktu proses retrieval: {:.6f} detik".format(end_time - start_time))
            else:
                results = index_instance.retrieve_bm25_wand(query, k = 10)

        print(f"Results ({args.scoring.upper()}):")
        for (score, doc) in results:
            print(f"{doc:30} {score:>.3f}")
        print()