from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import numpy as np

class LexicalRetriever:
    def __init__(self, analyzer='char_wb', ngram_range=(3, 5), max_features=None):
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer, 
            ngram_range=ngram_range,
            max_features=max_features,
            lowercase=True
        )
        self.s2_s3_matrix = None
        self.s2_s3_ids = []
        
    def fit(self, s2_data, s3_data, text_extract_func):
        corpus = []
        for row in s2_data:
            corpus.append(text_extract_func(row))
            self.s2_s3_ids.append(row['entity_id'])
            
        for row in s3_data:
            corpus.append(text_extract_func(row))
            self.s2_s3_ids.append(row['entity_id'])
            
        self.s2_s3_matrix = self.vectorizer.fit_transform(corpus)
        
    def retrieve_batch(self, s1_data, text_extract_func, top_k=50, threshold=0.0, batch_size=100):
        queries = [text_extract_func(row) for row in s1_data]
        s1_ids = [row['entity_id'] for row in s1_data]
        
        query_matrix = self.vectorizer.transform(queries)
        
        results = {}
        num_queries = query_matrix.shape[0]
        
        for start_idx in range(0, num_queries, batch_size):
            end_idx = min(start_idx + batch_size, num_queries)
            q_batch = query_matrix[start_idx:end_idx]
            
            # similarity: (batch_size, N_s2_s3)
            sim_batch = q_batch.dot(self.s2_s3_matrix.T)
            
            for i in range(end_idx - start_idx):
                s1_id = s1_ids[start_idx + i]
                row_sims = sim_batch.getrow(i)
                
                valid_mask = row_sims.data >= threshold
                valid_scores = row_sims.data[valid_mask]
                valid_indices = row_sims.indices[valid_mask]
                
                if len(valid_scores) == 0:
                    results[s1_id] = []
                    continue
                    
                if len(valid_scores) > top_k:
                    top_k_idx = np.argpartition(-valid_scores, top_k)[:top_k]
                    top_k_sorted_idx = top_k_idx[np.argsort(-valid_scores[top_k_idx])]
                    
                    final_scores = valid_scores[top_k_sorted_idx]
                    final_indices = valid_indices[top_k_sorted_idx]
                else:
                    sort_idx = np.argsort(-valid_scores)
                    final_scores = valid_scores[sort_idx]
                    final_indices = valid_indices[sort_idx]
                    
                candidates = [
                    (self.s2_s3_ids[idx], float(score)) 
                    for idx, score in zip(final_indices, final_scores)
                ]
                results[s1_id] = candidates
                
        return results

def extract_normalized_name_address(row):
    from src.preprocessing.normalization import normalize_text
    name = normalize_text(row.get("business_name", ""))
    address = normalize_text(row.get("business_address", ""))
    return f"{name} {address}".strip()
