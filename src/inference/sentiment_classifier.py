import torch
import numpy as np
from scipy.special import softmax

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig


class SentimentClassifier:

    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    CACHE_DIR = "/scratch/users/k23035472/hf_cache/"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL,
            cache_dir=self.CACHE_DIR
        )

        self.config = AutoConfig.from_pretrained(
            self.MODEL,
            cache_dir=self.CACHE_DIR
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL,
            cache_dir=self.CACHE_DIR
        ).to(self.DEVICE)

    @torch.no_grad()
    def inference(self, input: str):
        encoded_input = self.tokenizer(input, return_tensors='pt').to(self.DEVICE)
        output = self.model(**encoded_input)
        scores = output[0][0].cpu()
        scores[1] = torch.tensor(-10000)
        scores = scores.argmax().item()
        label = self.config.id2label[scores]
        return label
