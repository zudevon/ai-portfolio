import os
import shutil
from transformers import AutoTokenizer

def list_cached_models():
    """List all cached models"""
    try:
        # Get cache directory using a different approach
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        
        if os.path.exists(cache_dir):
            models = [d for d in os.listdir(cache_dir) if d.startswith("models--")]
            print("Cached models:")
            for model in models:
                print(f"  - {model.replace('models--', '').replace('--', '/')}")
        else:
            print("No cache directory found")
    except Exception as e:
        print(f"Error: {e}")

def delete_cached_model(model_name):
    """Delete a specific model from cache"""
    try:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        
        model_cache_path = os.path.join(cache_dir, "models--" + model_name.replace("/", "--"))
        if os.path.exists(model_cache_path):
            shutil.rmtree(model_cache_path)
            print(f"Successfully deleted {model_name} from cache")
        else:
            print(f"Model {model_name} not found in cache")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
if __name__ == "__main__":
    print("=== Cached Models ===")
    list_cached_models()
    
    print("\n=== Delete Model ===")
    # Uncomment the line below to delete a specific model
    # delete_cached_model("gpt2")
    # delete_cached_model("meta-llama/Meta-Llama-3-8B-Instruct")
    delete_cached_model("EleutherAI/gpt-neo-125M")
    delete_cached_model("gpt2-medium")
    delete_cached_model("microsoft/Phi-3-mini-4k-instruct")