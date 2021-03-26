import torch

from KGQA.model import RelationExtractor
from kge.model import KgeModel
from kge.util.io import load_checkpoint

kg_name = "ComplEx_fbwq_full"
pretrained_model_path = ""


class QuestionAnswering:
    def __init__(self, hops, gpu, use_cuda):
        checkpoint_file = 'pretrained_models/embeddings/' + kg_name + '/checkpoint_best.pt'
        kge_checkpoint = load_checkpoint(checkpoint_file)
        kge_model = KgeModel.create_from(kge_checkpoint)
        kge_model.eval()
        e = getEntityEmbeddings(kge_model, hops)
        print('Loaded entities and relations')
        entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)

        self.model = RelationExtractor()
        self.model.load_state_dict(torch.load(pretrained_model_path))
        print("Model loaded.")

    def predict(self, question, attention_mask):
        head = self.extract_head(question)
        if head is None:
            return

        question = self.model.getQuestionEmbedding(question.unsqueeze(0), attention_mask.unsqueeze(0))
        self.model.eval()
        candidates = self.model.get_score_ranked(head, question, attention_mask)

        return candidates

    def extract_head(self, question):
        # Todo: 提取问题主体的函数
        head = question

        return head


def prepare_embeddings(embedding_dict):
    entity2idx = {}
    idx2entity = {}
    i = 0
    embedding_matrix = []
    for key, entity in embedding_dict.items():
        entity2idx[key] = i
        idx2entity[i] = key
        i += 1
        embedding_matrix.append(entity)
    return entity2idx, idx2entity, embedding_matrix


def getEntityEmbeddings(kge_model, hops):
    e = {}
    entity_dict = '../../pretrained_models/embeddings/ComplEx_fbwq_full/entity_ids.del'
    if 'half' in hops:
        entity_dict = '../../pretrained_models/embeddings/ComplEx_fbwq_half/entity_ids.del'
        print('Loading half entity_ids.del')
    embedder = kge_model._entity_embedder
    f = open(entity_dict, 'r', encoding='utf-8')
    for line in f:
        line = line[:-1].split('\t')
        ent_id = int(line[0])
        ent_name = line[1]
        e[ent_name] = embedder._embeddings(torch.LongTensor([ent_id]))[0]
    f.close()
    return e


if __name__ == '__main__':
    qa = QuestionAnswering(gpu=0, use_cuda=True, hops='1')
