import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class DeBERTaNLI():

    CACHE_DIR = '/scratch/users/k23035472/hf_cache'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    def init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "cross-encoder/nli-deberta-v3-large",
            cache_dir=self.CACHE_DIR,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "cross-encoder/nli-deberta-v3-large",
            cache_dir=self.CACHE_DIR,
        ).to(self.DEVICE)


    @torch.no_grad()
    def inference(self, premise: str, hypothesis: list) -> list:
        # duplicate premise for each hypothesis
        hypothesis = [premise] * len(hypothesis)

        input = self.tokenizer(premise, hypothesis, padding=True, truncation=True, return_tensors="pt").to(self.DEVICE)

        scores = self.model(**input).logits
        return scores


    def select_intention(self, premise: str, hypothesis: list) -> str: 

        print(hypothesis)
        print(premise)

        scores = self.inference(premise, hypothesis)
        print(scores)
        best_intention_idx = torch.softmax(scores, dim=1)[:, -1].argmax().cpu().item()
        return premise[best_intention_idx]

