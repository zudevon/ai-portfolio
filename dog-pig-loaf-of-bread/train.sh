#!/bin/bash
# Train the image classifier
echo "🚀 Starting model training..."
python main.py train --epochs 50 --batch-size 32
