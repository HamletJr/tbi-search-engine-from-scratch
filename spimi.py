import os
import pickle
import contextlib
import heapq
import math
import nltk

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings, OptPForDeltaPostings, BP128Postings
from tqdm import tqdm

import argparse
import shutil

class SPIMIIndex:
    """
    SPIMI (Single-Pass In-Memory Indexing) scheme implementation.

    Attributes
    ----------
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    max_docs_per_block(int): Number of documents processed before writing an intermediate block to disk.
    """
    def __init__(self, data_dir, output_dir, postings_encoding, temp_dir="tmp", index_name="main_index", max_docs_per_block=100):
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.temp_dir = temp_dir

        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.max_docs_per_block = max_docs_per_block

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

        self.stemmer = nltk.stem.PorterStemmer()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
    
    def _preprocess_text(self, text):
        """
        Melakukan preprocessing terhadap text, termasuk case folding, stopword removal, dan stemming.
        """
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        processed_tokens = []
        for token in tokens:
            processed_token = self._preprocess_word(token)
            if processed_token is not None:
                processed_tokens.append(processed_token)
        return processed_tokens
        
    def _preprocess_word(self, word):
        if word not in self.stopwords and word.isalnum():
            return self.stemmer.stem(word)
        return None

    def _preprocess_query(self, query):
        """
        Melakukan preprocessing terhadap query, termasuk case folding, stopword removal, dan stemming.
        """
        query = query.lower()
        tokens = nltk.word_tokenize(query)
        processed_tokens = []
        for token in tokens:
            processed_token = self._preprocess_word(token)
            if processed_token is not None:
                processed_tokens.append(processed_token)
        return processed_tokens

    def save(self):
        """Menyimpan doc_id_map ke output directory via pickle"""
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)
    
    def load(self):
        """Memuat doc_id_map dari output directory via pickle"""
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def invert_write(self, term_dict, term_tf, block_id):
        """
        Menulis block (intermediate index) ke disk.
        """
        index_id = f'intermediate_index_{block_id}'
        self.intermediate_indices.append(index_id)
        
        with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.temp_dir) as index:
            for term in sorted(term_dict.keys()):
                sorted_doc_id = sorted(list(term_dict[term]))
                assoc_tf = [term_tf[term][doc_id] for doc_id in sorted_doc_id]
                index.append(term, sorted_doc_id, assoc_tf)

    def merge(self, indices, merged_index, N=None, avg_doc_length=None, merged_doc_length=None):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.
        """
        def find_max_bm25(postings, tf_list, k1=1.6, b=0.75):
            if N is None or avg_doc_length is None or merged_doc_length is None:
                return 0.0
            df = len(postings)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            max_score = 0.0
            for doc_id, tf in zip(postings, tf_list):
                dl = merged_doc_length[doc_id]
                score = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avg_doc_length))))
                if score > max_score:
                    max_score = score
            return max_score

        merged_iter = heapq.merge(*indices, key=lambda x: x[0])

        # Menambahkan exception handling jika tidak ada term
        try:
            curr, postings, tf_list = next(merged_iter)
        except StopIteration:
            return

        for t, postings_, tf_list_ in merged_iter:
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.term_max_score[curr] = find_max_bm25(postings, tf_list)
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
                
        merged_index.term_max_score[curr] = find_max_bm25(postings, tf_list)
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query: str, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_list pada merged index
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        if len(self.doc_id_map) == 0:
            self.load()
        
        terms = self._preprocess_query(query)
        
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            docs_evaluated = 0

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    N = len(merged_index.doc_length)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        docs_evaluated += 1
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        if tf > 0:
                            scores[doc_id] += math.log(N / df) * (1 + math.log(tf))

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: (-x[0], x[1]))[:k]

    def retrieve_bm25(self, query, k=10, k1=1.6, b=0.75, verbose=False):
        """
        Melakukan Boolean Retrieval untuk retrieve-k dokumen teratas
        berdasarkan nilai skor BM25.
        """
        if len(self.doc_id_map) == 0:
            self.load()
        
        terms = self._preprocess_query(query)

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            docs_evaluated = 0

            scores = {}
            N = len(merged_index.doc_length)
            
            # Mendapatkan precomputed average document length (dengan fallback jika index dibuat sebelum perubahan)
            avgdl = getattr(merged_index, 'avg_doc_length', 0)
            if avgdl == 0:
                avgdl = sum(merged_index.doc_length.values()) / N if N > 0 else 1.0

            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    postings, tf_list = merged_index.get_postings_list(term)
                    
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)  # IDF dengan smoothing untuk BM25
                    for i in range(len(postings)):
                        docs_evaluated += 1
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0.0
                        
                        dl = merged_index.doc_length[doc_id]
                        score_component = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl))))
                        scores[doc_id] += score_component

            if verbose:
                print("Jumlah dokumen yang dievaluasi (BM25):", docs_evaluated)

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: (-x[0], x[1]))[:k]

    def retrieve_bm25_wand(self, query, k=10, k1=1.6, b=0.75, verbose=False):
        """
        Melakukan Boolean Retrieval untuk retrieve-k dokumen teratas
        berdasarkan nilai skor BM25 menggunakan optimasi WAND (Weak AND).
        """
        if len(self.doc_id_map) == 0:
            self.load()
        
        terms = self._preprocess_query(query)

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            docs_evaluated = 0

            N = len(merged_index.doc_length)
            avgdl = getattr(merged_index, 'avg_doc_length', 0)
            if avgdl == 0:
                avgdl = sum(merged_index.doc_length.values()) / N if N > 0 else 1.0

            # Menyusun struktur data untuk DaaT dengan WAND
            term_iters = []
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    postings, tf_list = merged_index.get_postings_list(term)
                    max_score = getattr(merged_index, 'term_max_score', {}).get(term, 0)
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1) # IDF dengan smoothing untuk BM25
                    
                    if max_score == 0:
                        max_score = idf * (k1 + 1)
                        
                    term_iters.append({
                        'term': term,
                        'postings': postings,
                        'tf_list': tf_list,
                        'idx': 0,
                        'max_score': max_score,
                        'idf': idf,
                        'len': len(postings)
                    })
            
            top_k_heap = []
            threshold = 0.0
            
            while True:
                # Ambil iterator yang masih menunjuk ke dokumen valid
                term_iters = [t for t in term_iters if t['idx'] < t['len']]
                if not term_iters:
                    break
                
                # Sort iterator berdasarkan doc_id secara non-decreasing
                term_iters.sort(key=lambda x: x['postings'][x['idx']])
                
                # Tentukan pivot term dengan mengakumulasi max_score
                upper_bound = 0.0
                pivot_idx = -1
                for i, t in enumerate(term_iters):
                    upper_bound += t['max_score']
                    if upper_bound > threshold:
                        pivot_idx = i
                        break
                
                # Kalau tidak ada term yang bisa melebihi threshold, berarti sudah tidak ada lagi kandidat valid
                if pivot_idx == -1:
                    break

                # Tentukan dokumen yang menjadi pivot 
                pivot_doc_id = term_iters[pivot_idx]['postings'][term_iters[pivot_idx]['idx']]
                
                # Periksa apakah term pertama menunjuk ke dokumen yang sama dengan pivot
                if term_iters[0]['postings'][term_iters[0]['idx']] == pivot_doc_id:
                    # Kalau sama, berarti sukses, semua term menunjuk ke dokumen pivot. Lakukan evaluasi
                    docs_evaluated += 1
                    doc_id = pivot_doc_id
                    
                    score = 0.0
                    dl = merged_index.doc_length[doc_id]
                    doc_K = k1 * (1 - b + b * (dl / avgdl))
                    
                    # Modifikasi: hitung score BM25 sekaligus majukan semua iterator yang menunjuk ke doc_id pivot
                    for t in term_iters:
                        if t['postings'][t['idx']] == doc_id:
                            tf = t['tf_list'][t['idx']]
                            score += t['idf'] * ((tf * (k1 + 1)) / (tf + doc_K))
                            t['idx'] += 1
                            
                    # Periksa apakah masih ada tempat kosong di top-k heap
                    if len(top_k_heap) < k:
                        # Kalau masih ada, tidak perlu bandingkan, langsung aja tambahkan
                        heapq.heappush(top_k_heap, (score, doc_id))
                        # Update threshold kalau heap sudah penuh; ambil score terkecil di heap sebagai threshold
                        if len(top_k_heap) == k:
                            threshold = top_k_heap[0][0]
                    else:
                        # Kalau sudah penuh, harus bandingkan dengan threshold dulu
                        if score > threshold:
                            heapq.heappushpop(top_k_heap, (score, doc_id))
                            threshold = top_k_heap[0][0]
                else:
                    # Pivot belum memiliki mass yang cukup, majukan term sebelumnya.
                    # Modifikasi: karena term_iters sudah diurutkan berdasarkan doc_id,
                    # langsung aja majukan semua iterator sebelumnya
                    target_doc_id = pivot_doc_id
                    for term_idx in range(pivot_idx):
                        idx = term_iters[term_idx]['idx']
                        postings = term_iters[term_idx]['postings']
                        
                        while idx < term_iters[term_idx]['len'] and postings[idx] < target_doc_id:
                            idx += 1
                        term_iters[term_idx]['idx'] = idx
            
            if verbose:
                print("Jumlah dokumen yang dievaluasi (BM25-WAND):", docs_evaluated)

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (score, doc_id) in top_k_heap]
            return sorted(docs, key = lambda x: (-x[0], x[1]))[:k]

    def index(self):
        """
        Melakukan indexing dengan pendekatan SPIMI.
        """
        # Dictionary in-memory untuk SPIMI
        term_dict = {}
        term_tf = {}
        
        doc_count = 0
        block_count = 0
        
        # Berbeda dengan BSBI yang perlu melakukan segmentasi dokumen menjadi blok,
        # SPIMI akan langsung memproses dokumen satu per satu sampai mencapai batas 
        # memori (dalam kasus ini, saya membatasinya menjadi max_docs_per_block agar
        # tidak benar-benar menghabiskan RAM, bisa dinaikkan jika perlu tergantung
        # memori yang dimiliki sistem).

        # Membuat daftar semua file doc path di collection
        all_docs = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                # Membuat format file seperti: ./collection/1/1.txt
                doc_path = os.path.join(root, file).replace('\\', '/')  # Windows menggunakan \ untuk separator
                if not doc_path.startswith('./'):
                    doc_path = './' + doc_path
                all_docs.append(doc_path)

        for docname in tqdm(all_docs):
            doc_id = self.doc_id_map[docname]
            
            with open(docname, "r", encoding="utf8", errors="surrogateescape") as f:
                text = f.read()
            
            tokens = self._preprocess_text(text)
            
            # Untuk SPIMI, langsung masukkan term dalam bentuk string tanpa perlu mapping ke term_id
            for processed_token in tokens:
                if processed_token not in term_dict:
                    term_dict[processed_token] = set()
                    term_tf[processed_token] = {}
                
                term_dict[processed_token].add(doc_id)
                if doc_id not in term_tf[processed_token]:
                    term_tf[processed_token][doc_id] = 0
                term_tf[processed_token][doc_id] += 1
                
            doc_count += 1
            
            # Kalau sudah mencapai batas dokumen untuk satu block, langsung tulis 
            # ke disk sebagai intermediate index dan flush term dict dan term tf 
            # untuk block berikutnya (masing-masing block tidak perlu berbagi state)
            if doc_count >= self.max_docs_per_block:
                self.invert_write(term_dict, term_tf, block_count)
                term_dict = {}
                term_tf = {}
                doc_count = 0
                block_count += 1
        
        # Di akhir loop, pastikan untuk menulis sisa data yang belum sempat ditulis
        if doc_count > 0:
            self.invert_write(term_dict, term_tf, block_count)
            term_dict = {}
            term_tf = {}
            
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.temp_dir))
                               for index_id in self.intermediate_indices]
                
                merged_doc_length = {}
                for idx in indices:
                    for doc_id, length in idx.doc_length.items():
                        if doc_id not in merged_doc_length:
                            merged_doc_length[doc_id] = 0
                        merged_doc_length[doc_id] += length
                
                if len(merged_doc_length) > 0:
                    merged_index.avg_doc_length = sum(merged_doc_length.values()) / len(merged_doc_length)
                
                N = len(merged_doc_length)
                self.merge(indices, merged_index, 
                           N=N, avg_doc_length=merged_index.avg_doc_length, merged_doc_length=merged_doc_length)

if __name__ == "__main__":
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    parser = argparse.ArgumentParser(description='SPIMI Indexing')
    parser.add_argument('--compression', type=str, default='vbe', choices=['standard', 'vbe', 'optpfor', 'bp128'], help='Metode compression untuk postings list')
    parser.add_argument('--compare', action='store_true', help='Bandingkan semua metode compression')
    parser.add_argument('--max-docs', type=int, default=100, help='Batas dokumen untuk setiap block memori')

    args = parser.parse_args()

    if args.compare:
        compression_methods = {
            'Standard': StandardPostings,
            'VBE': VBEPostings,
            'OptPForDelta': OptPForDeltaPostings,
            'BP128': BP128Postings
        }
        size_results = {}
        print("Membuat direktori sementara untuk menyimpan index di tmp/tmp_index...")
        os.makedirs('tmp/tmp_index', exist_ok=True)
        for name, encoding in compression_methods.items():
            print(f"Indexing dengan metode {name}...")
            SPIMI_instance = SPIMIIndex(data_dir='collection', postings_encoding=encoding, output_dir='tmp/tmp_index', max_docs_per_block=args.max_docs)
            SPIMI_instance.index()
            size_results[name] = os.path.getsize(os.path.join('tmp/tmp_index', 'main_index.index'))
            print()
        shutil.rmtree('tmp/tmp_index')
        print("\nHasil ukuran index untuk setiap metode compression:")
        for name, size in size_results.items():
            print(f"{name}: {size} bytes")
            
    else:
        match args.compression:
            case 'standard':
                SPIMI_instance = SPIMIIndex(data_dir='collection', \
                                        postings_encoding=StandardPostings, \
                                        output_dir='index', max_docs_per_block=args.max_docs)
            case 'vbe':
                SPIMI_instance = SPIMIIndex(data_dir='collection', \
                                        postings_encoding=VBEPostings, \
                                        output_dir='index', max_docs_per_block=args.max_docs)
            case 'optpfor':
                SPIMI_instance = SPIMIIndex(data_dir='collection', \
                                        postings_encoding=OptPForDeltaPostings, \
                                        output_dir='index', max_docs_per_block=args.max_docs)
            case 'bp128':
                SPIMI_instance = SPIMIIndex(data_dir='collection', \
                                        postings_encoding=BP128Postings, \
                                        output_dir='index', max_docs_per_block=args.max_docs)
                
        SPIMI_instance.index() # memulai indexing!