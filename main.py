import torch

from KGQA.model import RelationExtractor
from kge.model import KgeModel
from kge.util.io import load_checkpoint

kg_name = "ComplEx_fbwq_full"
pretrained_model_path = ""
attention_mask = 0


class QuestionAnswering:
    def __init__(self):
        self.model = RelationExtractor()
        self.model.load_state_dict(torch.load(pretrained_model_path))
        print("Model loaded.")

    def predict(self, question):
        heads = self.extract_head(question)
        relations = self.extract_relation(question)
        if heads is None:
            return
        candidates = []
        self.model.eval()
        for head in heads:
            for relation in relations:
                tails = self.model.get_top_k_tail(head, relation, 3)
                candidates = candidates.extend(tails)
        print(candidates)

        return candidates

    def extract_head(self, question):
        pass

    def extract_relation(self, question):
        question_embedding = self.model.getQuestionEmbedding(question.unsqueez(0), attention_mask)
        prediction = self.model.applyNonLinear(question_embedding)
        prediction = torch.sigmoid(prediction).squeeze()
        scores, candidates = torch.topk(prediction, k=5, largest=True, sorted=True)

        return candidates


if __name__ == '__main__':
    qa = QuestionAnswering()
