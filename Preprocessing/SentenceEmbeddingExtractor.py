import torch
from transformers import BertTokenizer, BertModel


class SentenceEmbeddingExtractor:

    def __init__(self, device=torch.device("cpu")):
        #TODO: Do we need to handle unsupported languages?
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.device = device
       
        
    def extract_sentence_embedding(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt').to(self.device)
        cls = self.bert(**encoded_input)['last_hidden_state'][0][0] # [batch][tokens]
        #print(self.bert(**encoded_input)["pooler_output"].shape)
        return cls


if __name__ == '__main__':
    text = "Hello World, this is a test."
    ext = SentenceEmbeddingExtractor(language="en")
    cls = ext.extract_sentence_embedding(text)
    print(cls.shape)
