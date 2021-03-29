import torch
import re

from KGQA.model import RelationExtractor
from kge.model import KgeModel
from kge.util.io import load_checkpoint
from nltk.tokenize import MWETokenizer, word_tokenize

kg_name = "ComplEx_fbwq_full"
pretrained_model_path = ""
attention_mask = 0

entity_path = 'data/EntityDictionary/mid2name.txt'


class QuestionAnswering:
    def __init__(self):
        # self.model = RelationExtractor()
        # self.model.load_state_dict(torch.load(pretrained_model_path))
        print("Model loaded.")

    def __get_dictionary(self):
        f = open(entity_path)
        word_dict = []
        index_dict = []
        while True:
            line = f.readline()
            if not line:
                break
            indexed_token = line.lower().split()
            token = indexed_token
            index = token.pop(0)
            token = ' '.join(token)

            index_dict.append(index)
            word_dict.append(token)

        f.close()
        dict = [tuple(index_dict), tuple(word_dict)]
        return dict

    def __get_tokenizer(self):
        f = open(entity_path)
        mwetokenizer = MWETokenizer([], separator=' ')
        i = 30
        while True:
            i = i - 1
            if i <= 0:
                break
            line = f.readline()
            if not line:
                break
            indexed_token = line.lower().split()
            token = indexed_token
            token.pop(0)
            token = tuple(token)
            # print(token)
            mwetokenizer.add_mwe(token)

        f.close()
        return mwetokenizer

    def __clean_punct(self, sentence):
        p1 = re.compile(r'-\{.*?(zh-hans|zh-cn):([^;]*?)(;.*?)?\}-')
        p2 = re.compile(r'[(][: @ . , ？！\s][)]')
        p3 = re.compile(r'[「『]')
        p4 = re.compile(r'[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）0-9 , : ; \-\ \[\ \]\ ]')
        sentence = p1.sub(r' ', sentence)
        sentence = p2.sub(r' ', sentence)
        sentence = p3.sub(r' ', sentence)
        sentence = p4.sub(r' ', sentence)
        return sentence

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
        """
        功能：将给定问题对应的头实体从知识库中找出来
        :param question: str类型，问题，不分大小写；
        :return: 包含头实体索引以及头实体名称的字典。
        """
        entity_dictionaty = self.__get_dictionary()
        index_dict = entity_dictionaty[0]
        entity_dict = entity_dictionaty[1]
        mwetokenizer = self.__get_tokenizer()

        q = self.__clean_punct(question)
        q_lower = q.lower()

        q_tokenized = mwetokenizer.tokenize(word_tokenize(q_lower))
        # print(q_tokenized)
        # 停用词 这部分可能不需要用，因为本身就不会在头实体知识库中出现停用词。
        # stop_words = set(stopwords.words('english'))
        # 从词典里面找
        entities = [token for token in q_tokenized if token in entity_dict]
        # q_filtered = [token for token in q_tokenized if (token not in stop_words) and (token in entity_dict)]
        # print(entities)
        index_entities = dict()
        for token in entities:
            index = entity_dict.index(token)
            index_entity = {str(index_dict[index]): token}
            index_entities.update(index_entity)
        print(index_entities)
        return index_entities

    def extract_relation(self, question):
        question_embedding = self.model.getQuestionEmbedding(question.unsqueez(0), attention_mask)
        prediction = self.model.applyNonLinear(question_embedding)
        prediction = torch.sigmoid(prediction).squeeze()
        scores, candidates = torch.topk(prediction, k=5, largest=True, sorted=True)

        return candidates


if __name__ == '__main__':
    qa = QuestionAnswering()
    q = 'what disease did abraham lincoln had'
    qa.extract_head(q)
