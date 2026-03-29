import os
import pickle
import numpy as np
import faiss
from tqdm import tqdm
from util import IdMap
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import argparse

class LSIIndex:
    """
    Attributes
    ----------
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen ke docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    latent_dim(int): Jumlah dimensi laten untuk SVD
    index_name(str): Nama dari file yang berisi index faiss
    """
    def __init__(self, data_dir, output_dir, latent_dim=100, index_name="lsi"):
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.latent_dim = latent_dim
        self.index_name = index_name

        self.stemmer = nltk.stem.PorterStemmer()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        
        self.vectorizer = None
        self.svd = None
        self.faiss_index = None

    def _preprocess_word(self, word):
        if word not in self.stopwords and word.isalnum():
            return self.stemmer.stem(word)
        return None

    def _preprocess_text(self, text):
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        processed_tokens = []
        for token in tokens:
            processed_token = self._preprocess_word(token)
            if processed_token is not None:
                processed_tokens.append(processed_token)
        return processed_tokens

    def build_index(self):
        """Membangun LSI dan FAISS Index dari dataset collection.
        Fungsi ini akan membaca semua dokumen (dengan asumsi muat di memori), 
        kemudian membangun TF-IDF matrix menggunakan TfidfVectorizer dan
        pipeline preprocessing yang sudah dibuat sebelumnya, lalu menghitung 
        SVD menggunakan TruncatedSVD agar bisa handle matriks sparse yang besar, 
        lalu membangun FAISS index untuk retrieval."""
        docs_text = []
        
        # Membaca semua dokumen dan menyimpan konten ke dalam list
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            dir_path = "./" + self.data_dir + "/" + block_dir_relative
            for filename in next(os.walk(dir_path))[2]:
                docname = dir_path + "/" + filename
                _ = self.doc_id_map[docname]
                with open(docname, "r", encoding="utf8", errors="surrogateescape") as f:
                    docs_text.append(f.read())

        # Membangun TF-IDF matrix dengan fungsi preprocessing yang sudah dibuat
        # Menggunakan sublinear_tf untuk mengurangi dampak dari term frequency 
        # yang sangat tinggi, dan min_df dan max_df untuk mengabaikan term yang 
        # terlalu jarang atau terlalu sering muncul 
        self.vectorizer = TfidfVectorizer(tokenizer=self._preprocess_text, 
                                          token_pattern=None, 
                                          sublinear_tf=True,
                                          min_df=2,
                                          max_df=0.95)
        X = self.vectorizer.fit_transform(docs_text)
        
        # Melakukan TruncatedSVD dengan k dimensi
        k = min(self.latent_dim, X.shape[1] - 1, X.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=k, random_state=42)
        doc_vectors = self.svd.fit_transform(X)
        
        doc_vectors = np.ascontiguousarray(doc_vectors, dtype=np.float32)
        
        # Normalisasi vektor dokumen sebelum membangun index FAISS agar cosine similarity bisa dihitung dengan inner product
        faiss.normalize_L2(doc_vectors)

        # Menggunakan IndexFlatIP untuk menghitung inner product (yang setara dengan cosine similarity setelah normalisasi)
        self.faiss_index = faiss.IndexFlatIP(k)
        self.faiss_index.add(doc_vectors)
        
    def save(self):
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, 'lsi_docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)
        
        # Simpan vectorizer, SVD model, dan FAISS index agar bisa digunakan untuk retrieval nanti
        with open(os.path.join(self.output_dir, 'lsi_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(self.output_dir, 'lsi_svd.pkl'), 'wb') as f:
            pickle.dump(self.svd, f)
        
        faiss.write_index(self.faiss_index, os.path.join(self.output_dir, self.index_name + ".faiss"))

    def load(self):
        # Muat doc_id_map, vectorizer, SVD model, dan FAISS index dari file yang sudah disimpan
        with open(os.path.join(self.output_dir, 'lsi_docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'lsi_vectorizer.pkl'), 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(os.path.join(self.output_dir, 'lsi_svd.pkl'), 'rb') as f:
            self.svd = pickle.load(f)
            
        self.faiss_index = faiss.read_index(os.path.join(self.output_dir, self.index_name + ".faiss"))

    def retrieve(self, query, k=10):
        # Ubah query ke vektor TF-IDF menggunakan vectorizer yang sama dengan indexing
        X_q = self.vectorizer.transform([query])

        # Proyeksikan vektor query ke ruang laten yang sama dengan SVD sebelumnya
        q_latent = self.svd.transform(X_q)
        q_latent = np.ascontiguousarray(q_latent, dtype=np.float32)
        
        # Normalisasi lagi
        faiss.normalize_L2(q_latent)
        
        # Lakukan pencarian di index FAISS untuk mendapatkan top-k dokumen
        scores, I = self.faiss_index.search(q_latent, k)
        
        results = []
        for idx in range(k):
            doc_id = int(I[0][idx])
            if doc_id == -1:    # -1 artinya FAISS tidak menemukan dokumen yang relevan, jadi skip saja
                continue
            doc_name = self.doc_id_map[doc_id]
            score = float(scores[0][idx])
            results.append((score, doc_name))
            
        return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bangun LSI Index')
    parser.add_argument('--k', type=int, default=100, help='Jumlah latent dimensions untuk LSI')
    args = parser.parse_args()

    index = LSIIndex(data_dir='collection', output_dir='index', latent_dim=args.k)
    index.build_index()
    index.save()
