import collections

class ExactBlocker:
    """
    Implements deterministic / structured blocking (Channel A).
    """
    def __init__(self, key_funcs=None):
        """
        Args:
            key_funcs (list): List of functions that take a row dict and return a blocking key.
        """
        self.key_funcs = key_funcs if key_funcs else []
        self.blocks = collections.defaultdict(list)
        
    def fit(self, s2_data, s3_data):
        """
        Builds the blocking index over S2 and S3 data.
        
        Args:
            s2_data (list of dict): S2 records.
            s3_data (list of dict): S3 records.
        """
        for source_data in [s2_data, s3_data]:
            for row in source_data:
                entity_id = row['entity_id']
                for i, kf in enumerate(self.key_funcs):
                    key = kf(row)
                    if key:
                        self.blocks[(i, key)].append(entity_id)
                        
    def retrieve(self, s1_row):
        """
        Retrieves candidates for a single S1 row.
        
        Args:
            s1_row (dict): S1 record.
            
        Returns:
            list: List of candidate entity_ids.
        """
        candidates = set()
        for i, kf in enumerate(self.key_funcs):
            key = kf(s1_row)
            if key and (i, key) in self.blocks:
                candidates.update(self.blocks[(i, key)])
        return list(candidates)

# Example key functions
def exact_normalized_name_key(row):
    from src.preprocessing.normalization import normalize_text
    name = row.get("business_name", "")
    return normalize_text(name) if name else None

def alphanumeric_compact_name_key(row):
    from src.preprocessing.normalization import extract_name_views
    name = row.get("business_name", "")
    if not name: return None
    views = extract_name_views(name)
    return views.get('alphanumeric_compact')
