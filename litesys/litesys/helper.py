import datasets
from datasets import Dataset
def generate_requests(dataset: Dataset, field_name: str, data_format: str, trial: int = 1, rank: int = 0, world_size: int = 1):
    
    requests= []
    total = len(dataset)
    per_proc = total // world_size
    remainder = total % world_size
    start = rank * per_proc + min(rank, remainder)
    end = start + per_proc + (1 if rank < remainder else 0)  
    subset = dataset.select(list(range(start, end)))
   
    for data in subset:
            coversations = [
                        {"role": "user", "content":data_format.format(Question=data[field_name])}
            ]
            
            data["conversations"] = coversations
            requests.append(data)
    
    return requests
        
        
    