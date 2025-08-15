#!/usr/bin/env python3
"""
Convenience CLI for the project.
Usage:
  python -m src.main download
  python -m src.main preprocess
  python -m src.main train
  python -m src.main detect
"""
import argparse
from src.data.download import download_and_extract
from src.data.preprocess import preprocess
from src.train import main as train_main
from src.detect import main as detect_main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["download", "preprocess", "train", "detect"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    if args.command == "download":
        download_and_extract(args.config)
    elif args.command == "preprocess":
        preprocess(args.config)
    elif args.command == "train":
        train_main(args.config)
    elif args.command == "detect":
        detect_main(args.config)

if __name__ == "__main__":
    main()
