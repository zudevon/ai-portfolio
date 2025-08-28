from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Choose a compatible model
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Requires special access
# model_name = "microsoft/Phi-3-mini-4k-instruct"      # Has compatibility issues
model_name = "gpt2"                                  # Basic GPT-2 - more reliable
# model_name = "gpt2-medium"                           # Medium GPT-2
# model_name = "EleutherAI/gpt-neo-125M"                # Small GPT-Neo

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device if device == "cuda" else None,  # Only use device_map for CUDA
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Format the prompt properly for the model
prompt = "What is the capital of France?"

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

# Move inputs to the same device as the model
if device == "cuda":
    inputs = {k: v.to(device) for k, v in inputs.items()}

print("inputs: ", inputs)

# Generate text
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,  # Short and focused
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.9,
        top_k=50,           # Limit vocabulary choices
        top_p=0.9,          # Nucleus sampling
    )

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Generated text: {generated_text}")

