from transformers import BertTokenizer

# load tokenization model
tokenizer = BertTokenizer.from_pretrained("bert-base-cased", use_fast=True) # 可以使用use fast加速

# testing sentence
sequence = "Using a Transformer network is simple"

# sentence to ids
ids_obj = tokenizer(sequence)
print(ids_obj)
# {'input_ids': [101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}


input_ids = tokenizer.encode(sequence)
print(input_ids)
# [101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102]

# sentence to tokens
tokens = tokenizer.tokenize(sequence)
print(tokens)
# ['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']

print(tokenizer.decode([101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102]))
# [CLS] Using a Transformer network is simple [SEP]

