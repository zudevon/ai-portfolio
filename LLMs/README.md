# LLMs (Large Language Models) Portfolio

This folder contains examples and implementations of various Large Language Models using the Hugging Face Transformers library.

## Available Models

### Small/Medium Models (Good for testing, lower resource usage)
- `"gpt2"` - Basic GPT-2 model, good for learning
- `"gpt2-medium"` - Slightly larger version of GPT-2
- `"distilgpt2"` - Distilled version, faster inference
- `"EleutherAI/gpt-neo-125M"` - Small but modern architecture
- `"microsoft/DialoGPT-small"` - Good for conversations

### Better Quality Models (Medium resource usage)
- `"EleutherAI/gpt-neo-1.3B"` - Much better quality than GPT-2
- `"microsoft/DialoGPT-medium"` - Better conversation model
- `"facebook/opt-350m"` - Meta's open model
- `"tiiuae/falcon-7b"` - High quality, but larger

### Large Models (High resource usage, better quality)
- `"tiiuae/falcon-7b"` - Very good quality
- `"microsoft/DialoGPT-large"` - Best conversation model
- `"EleutherAI/gpt-neo-2.7B"` - High quality text generation

### Specialized Models
- `"microsoft/DialoGPT-medium"` - Great for chat/conversation
- `"gpt2-xl"` - Larger GPT-2 variant
- `"EleutherAI/gpt-neo-125M"` - Good for creative writing

## Current Implementation: ezLLM.py

The `ezLLM.py` file demonstrates:
- Loading models with automatic device detection (CUDA/CPU)
- Proper tokenization and input handling
- Text generation with configurable parameters
- Error handling and compatibility fixes

## Usage

### Basic Usage
```bash
cd ai-portfolio/LLMs
python ezLLM.py
```

### Changing Models
Edit the `model_name` variable in `ezLLM.py`:
```python
# Choose a compatible model
model_name = "gpt2"  # Current default
# model_name = "EleutherAI/gpt-neo-125M"  # Alternative
# model_name = "microsoft/DialoGPT-medium"  # For conversations
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers library
- Accelerate (for CUDA support)

## Installation

```bash
pip install torch transformers accelerate
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Use smaller models or reduce `max_new_tokens`
2. **Model Compatibility**: Some models require specific transformers versions
3. **Repetitive Output**: Adjust `temperature`, `do_sample`, or use `repetition_penalty`

### Performance Tips

- Use `device_map="auto"` for automatic device placement
- Enable `torch.compile()` for faster inference (PyTorch 2.0+)
- Use quantization for memory efficiency

## Model Selection Guide

### For Learning/Testing
- **GPT-2**: Good starting point, well-documented
- **DistilGPT-2**: Faster, good for quick experiments

### For Better Quality
- **GPT-Neo 1.3B**: Significant improvement over GPT-2
- **DialoGPT-medium**: Great for conversational AI

### For Production
- **Falcon-7B**: High quality, good for serious applications
- **GPT-Neo 2.7B**: Best balance of quality and resource usage

## License Considerations

- **GPT-2**: MIT License - Free to use
- **GPT-Neo**: Apache 2.0 - Free to use
- **DialoGPT**: MIT License - Free to use
- **Falcon**: Apache 2.0 - Free to use

## Future Improvements

- Add support for more recent models
- Implement streaming generation
- Add model comparison benchmarks
- Support for fine-tuning workflows
