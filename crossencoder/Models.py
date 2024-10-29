import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
class CrossEncoder(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.1):
        super(CrossEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_tokens(['[E]','[/E]'])
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        for param in self.bert.parameters():
            param.requires_grad = True
        self.linear = nn.Linear(self.bert.config.hidden_size, 32)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(32,num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        #cls_vector = pooled_output[:, 0, :]
        logits = torch.relu(self.linear(pooled_output))
        logits = self.dropout(logits)
        logits = self.final_linear(logits)
        final_logit = torch.flatten(logits)
        return torch.sigmoid(final_logit)





class BiEncoder(nn.Module):
    def __init__(self, model_name, num_labels,):
        super(BiEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_tokens(['[E]','[/E]'])
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        for param in self.bert.parameters():
            param.requires_grad = True


    def forward(self, query_ids, candidate_ids, query_attention_mask=None, candidate_attention_mask=None):

        query_outputs = self.bert(input_ids=query_ids, attention_mask=query_attention_mask)
        query_pooled_output = query_outputs[1]
        candidate_outputs = self.bert(input_ids=candidate_ids, attention_mask=candidate_attention_mask)
        candidate_pooled_output = candidate_outputs[1]
        dot_prod = (query_pooled_output * candidate_pooled_output).sum(axis=1).reshape(-1,1)
        return dot_prod.reshape(-1)

class Bider(nn.Module):
    def __init__(self, model_name, num_labels,):
        super(Bider, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_tokens(['[E]','[/E]'])
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        for param in self.bert.parameters():
            param.requires_grad = True


    def forward(self, query_ids, candidate_ids, query_attention_mask=None, candidate_attention_mask=None):

        query_outputs = self.bert(input_ids=query_ids, attention_mask=query_attention_mask)
        query_pooled_output = query_outputs[1]
        query_pooled_output = query_pooled_output.squeeze(dim=0)
        candidate_outputs = self.bert(input_ids=candidate_ids, attention_mask=candidate_attention_mask)
        candidate_pooled_output = candidate_outputs[1]
        candidate_pooled_output = candidate_pooled_output.squeeze(dim=0)
        #dot_prod = (query_pooled_output * candidate_pooled_output).sum(axis=1).reshape(-1,1)
        #return dot_prod.reshape(-1)
        return query_pooled_output,candidate_pooled_output