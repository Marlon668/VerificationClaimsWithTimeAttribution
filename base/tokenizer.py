import os

import spacy
import torch
from torch import nn
#from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertTokenizer, AutoTokenizer

import Claim
from datasetOld import dump_load, NUS, dump_write

from tokenizers.implementations import BertWordPieceTokenizer


def makeDataset(mode,path):
    nlp = spacy.load("en_core_web_sm")
    predictorOIE = "None"
    predictorNER = "None"
    coreference = "None"
    basepath = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    if mode != 'Test':
        with open(path, 'r', encoding='utf-8') as file:
            for claim in file:
                elements = claim.split('\t')
                claim = Claim.claim(elements[0], elements[1], elements[2], elements[3], elements[4], elements[5],
                                    elements[6],
                                    elements[7], elements[8], elements[9], elements[10], elements[11], elements[12],
                                    "snippets/", predictorOIE, predictorNER, nlp, coreference)
                with open(basepath + "/text" + "/" + claim.claimID + "/" + "claim", "r", encoding="utf-8") as f:
                    doc = f.read()
                    words = nlp(doc)
                    for token in words:
                        with open("words/words.txt", 'a', encoding='utf-8') as fp:
                            fp.write(str(token))
                            fp.write('\n')
                            fp.close()
                    f.close()
                for snippet in claim.getSnippets():
                    with open(basepath +"/text" + "/" + claim.claimID + "/" + snippet.number, "r", encoding="utf-8") as f:
                        doc = f.read()
                        words = nlp(doc)
                        for token in words:
                            with open("words/words.txt", 'a', encoding='utf-8') as fp:
                                fp.write(str(token))
                                fp.write('\n')
                                fp.close()
                        f.close()
    else:
        with open(path, 'r', encoding='utf-8') as file:
            for claim in file:
                elements = claim.split('\t')
                claim = Claim.claim(elements[0], elements[1],"None", elements[2], elements[3], elements[4], elements[5],
                                    elements[6],
                                    elements[7], elements[8], elements[9], elements[10], elements[11],
                                    "snippets/", predictorOIE, predictorNER, nlp, coreference)
                with open(basepath + "/text" + "/" + claim.claimID + "/" + "claim", "r", encoding="utf-8") as f:
                    doc = f.read()
                    words = nlp(doc)
                    for word in words:
                        with open("words/words.txt", 'a', encoding='utf-8') as fp:
                            fp.write(str(word))
                            fp.write('\n')
                            fp.close()
                    f.close()
                for snippet in claim.getSnippets():
                    with open(basepath + "/text" + "/" + claim.claimID + "/" + snippet.number, "r", encoding="utf-8") as f:
                        doc = f.read()
                        words = nlp(doc)
                        for word in words:
                            with open("words/words.txt", 'a', encoding='utf-8') as fp:
                                fp.write(str(word))
                                fp.write('\n')
                                fp.close()
                        f.close()


tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=False
)
'''
makeDataset("Dev","dev/dev.tsv")
makeDataset("Test","test/test.tsv")
makeDataset("Train","train/train.tsv")
'''
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=False
)
'''
tokenizer.train(files=["words/words.txt"], vocab_size=50000, min_frequency=2,
                limit_alphabet=1000, wordpieces_prefix='##',
                special_tokens=[
                    '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
                )

#os.mkdir('base/bert-it')

tokenizer.save_model('bert-it')
'''
#tokenizer = BertTokenizer.from_pretrained('bert-it',do_lower_case=True,add_special_tokens=False)
#print(tokenizer.all_special_tokens)
#print(tokenizer.all_special_ids)
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')

'''
with open('bert-it/vocab.txt', 'r',encoding="utf8") as fp:
    vocab = fp.read().split('\n')
'''
input_ids = tokenizer(text="The evidence with the title 'New tattoos change color based on your health - INSIDER' says", padding=True, truncation=False)['input_ids']
print(tokenizer.all_special_ids)
print(tokenizer.all_special_tokens)
'''
print(tokenizer.decode(input_ids))
print(input_ids)
print(tokenizer.get_vocab())
key_list = list(tokenizer.get_vocab().keys())
for id in input_ids:
    postition = list(tokenizer.get_vocab().values()).index(id)
    print(key_list[postition])
input_ids = [i for i in tokenizer.encode(text="The evidence with the title 'New tattoos change color based on your health - INSIDER' says") if i not in {0,2,4}]
for id in input_ids:
    print(vocab[id])
'''
decoding = []
for i in input_ids:
    if i not in {0,2,3,1,50264}:
        decoding.append(str(tokenizer.decode([i])))
print(decoding)

