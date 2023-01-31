import os

import spacy
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertTokenizer

import Claim
from datasetOld import dump_load, NUS, dump_write

from tokenizers.implementations import BertWordPieceTokenizer


def makeDataset(mode,path):
    nlp = spacy.load("en_core_web_sm")
    predictorOIE = "None"
    predictorNER = "None"
    coreference = "None"
    if mode != 'Test':
        with open(path, 'r', encoding='utf-8') as file:
            for claim in file:
                elements = claim.split('\t')
                claim = Claim.claim(elements[0], elements[1], elements[2], elements[3], elements[4], elements[5],
                                    elements[6],
                                    elements[7], elements[8], elements[9], elements[10], elements[11], elements[12],
                                    "snippets/", predictorOIE, predictorNER, nlp, coreference)
                with open("text" + "/" + claim.claimID + "/" + "claim", "r", encoding="utf-8") as f:
                    doc = f.read()
                    words = nlp(doc)
                    for token in words:
                        with open("baseModel/words/words.txt", 'a', encoding='utf-8') as fp:
                            fp.write(str(token))
                            fp.write('\n')
                            fp.close()
                    f.close()
                for snippet in claim.getSnippets():
                    with open("text" + "/" + claim.claimID + "/" + snippet.number, "r", encoding="utf-8") as f:
                        doc = f.read()
                        words = nlp(doc)
                        for token in words:
                            with open("baseModel/words/words.txt", 'a', encoding='utf-8') as fp:
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
                with open("text" + "/" + claim.claimID + "/" + "claim", "r", encoding="utf-8") as f:
                    doc = f.read()
                    words = nlp(doc)
                    for word in words:
                        with open("baseModel/words/words.txt", 'a', encoding='utf-8') as fp:
                            fp.write(str(word))
                            fp.write('\n')
                            fp.close()
                    f.close()
                for snippet in claim.getSnippets():
                    with open("text" + "/" + claim.claimID + "/" + snippet.number, "r", encoding="utf-8") as f:
                        doc = f.read()
                        words = nlp(doc)
                        for word in words:
                            with open("baseModel/words/words.txt", 'a', encoding='utf-8') as fp:
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
makeDataset("Dev","dev/dev.tsv")
makeDataset("Test","test/test.tsv")

tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=False
)

tokenizer.train(files=["baseModel/words/words.txt"], vocab_size=100000, min_frequency=2,
                limit_alphabet=1000, wordpieces_prefix='##',
                special_tokens=[
                    '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
                )

#os.mkdir('baseModel/bert-it')

tokenizer.save_model('baseModel/bert-it')
tokenizer = BertTokenizer.from_pretrained('baseModel/bert-it',do_lower_case=False,add_special_tokens=False)
print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)

with open('baseModel/bert-it/vocab.txt', 'r',encoding="utf8") as fp:
    vocab = fp.read().split('\n')

input_ids = [i for i in tokenizer.encode(text='Is this an image of subtropical storm Alberto approaching Pensacola beach?') if i not in {0,2,4}]
for id in input_ids:
    print(vocab[id])

