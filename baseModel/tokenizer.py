import os

import spacy
import torch
from torch import nn
#from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertTokenizer, AutoTokenizer

import Claim

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

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')


input_ids = tokenizer(text="The evidence with the title 'New tattoos change color based on your health - INSIDER' says", padding=True, truncation=False)['input_ids']
print(tokenizer.all_special_ids)
print(tokenizer.all_special_tokens)

decoding = []
for i in input_ids:
    if i not in {0,2,3,1,50264}:
        decoding.append(str(tokenizer.decode([i])))
print(decoding)

