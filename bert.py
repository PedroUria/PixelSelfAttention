import torch
from pytorch_transformers import BertForSequenceClassification, BertTokenizer
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def tokenization_step(input_seq, tok=tokenizer, pad=True):
    tokenized_mapped = tok.convert_tokens_to_ids(tok.tokenize(input_seq))
    essay_size = len(tokenized_mapped)
    if pad:
        return (torch.LongTensor(np.array([101] + tokenized_mapped + [102] + [0] * (510 - essay_size)).reshape(1, -1)),
                torch.LongTensor(np.array([1]*(essay_size + 2) + [0]*(510-essay_size)).reshape(1, -1)))
    else:
        return (torch.LongTensor(np.array([101] + tokenized_mapped + [102]).reshape(-1, 1)),
                torch.LongTensor(np.array([1] * (essay_size + 2)).reshape(1, -1)))


x, mask = tokenization_step("Hello World", pad=True)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
y = model(x, attention_mask=mask)
