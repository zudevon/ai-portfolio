from transformers import AutoTokenizer
import shutil
import os

# Get the cache directory
cache_dir = AutoTokenizer.from_pretrained("gpt2", cache_dir=None).cache_dir
parent_cache = os.path.dirname(cache_dir)

# Delete the entire transformers cache
if os.path.exists(parent_cache):
    shutil.rmtree(parent_cache)
    print("Deleted entire transformers cache")