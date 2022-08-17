import json
import os
import re
import datetime
import xml.etree.ElementTree as ET

class snippet:

    def __init__(self, claimID, number, title, article, url,OIEpredictor,NERpredictor,spacy,coreference):
        self.coreference = coreference
        self.spacy = spacy
        self.predictorOIE = OIEpredictor
        self.predictorNER = NERpredictor
        self.claimID = claimID
        self.number = number
        self.title = title
        self.article = article
        self.parts = self.readParts(article)
        self.publishTime = self.parts[0]
        '''
        try:
            datetime.datetime.strptime(snippet.publishTime[:-1], '%b %d, %Y')
        except:
            self.processDate("processedSnippets" + "/" + self.claimID+"/"+self.number)
        '''
        self.url = url
        self.optional = None

        for part in self.parts:
            index = part.find("Published")
            if index!=(-1):
                self.optional = part[index:]
            else:
                index = part.find("Posted")
                if index!= (-1):
                    self.optional = part[index:]

    def getSnippetDate(self):
        try:
            self.publishTime =  datetime.datetime.strptime(self.publishTime[:-1], '%b %d, %Y')
        except:
            self.publishTime = None

    def readParts(self, article):
        return article.split("...")

    def processDate(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        if len(root) == 1:
            if root[0].attrib['value'].find(':') == -1:
                if len(root[0].attrib['value']) == 7:
                    # print('Claim ' + self.claimID)
                    self.claimDate = datetime.datetime.strptime(root[0].attrib['value'], '%Y-%m')
                else:
                    if len(root[0].attrib['value']) == 4:
                        # print('Claim ' + self.claimID)
                        self.claimDate = datetime.datetime.strptime(root[0].attrib['value'], '%Y')
                    else:
                        self.claimDate = datetime.datetime.strptime(root[0].attrib['value'], '%Y-%m-%d')
            else:
                if root[0].attrib['value'][root[0].attrib['value'].find(':') + 1:].find(':') == -1:
                    self.claimDate = datetime.datetime.strptime(root[0].attrib['value'].replace('24:', '12:'),
                                                                '%Y-%m-%dT%H:%M')
                else:
                    self.claimDate = datetime.datetime.strptime(root[0].attrib['value'].replace('24:', '12:'),
                                                                '%Y-%m-%dT%H:%M:%S')
        else:
            if root[0].attrib['value'].find('X') != -1:
                # correct fault of assigning 12pm to 24 in HeidelTime
                self.claimDate = datetime.datetime.strptime(
                    root[1].attrib['value'] + root[0].attrib['value'][root[0].attrib['value'].find('T'):].replace('24:',
                                                                                                                  '12:'),
                    '%Y-%m-%dT%H:%M')
            else:
                if root[0].attrib['value'].find('T') == -1 and root[1].attrib['value'].find('T') != -1:
                    # correct fault of assigning 12pm to 24 in HeidelTime
                    self.claimDate = datetime.datetime.strptime(root[1].attrib['value'].replace('24:', '12:'),
                                                                '%Y-%m-%dT%H:%M')
                else:
                    # we choose publication date as claim date not last updated date
                    if root[0].attrib['value'].find('T') != -1:
                        self.claimDate = datetime.datetime.strptime(root[0].attrib['value'], '%Y-%m-%dT%H:%M')
                    else:
                        self.claimDate = datetime.datetime.strptime(root[0].attrib['value'], '%Y-%m-%d')
        # print(self.claimDate)

    def processOpenInformation(self):
        document = self.parts
        if self.title != "None":
            document.insert(0, self.title)
        openInformation = []
        for sen in document:
            sentences = self.spacy(sen)
            for sentence in sentences.sents:
                sent = self.processSentence(sentence)
                if sent is not None:
                    openInformation.append(self.predictorOIE.predict(sent))

        f = open("OpenInformation" + "/" + self.claimID + "/" + self.number, "w", encoding="utf-8")
        json.dump(openInformation, f)
        f.close()

    def readOpenInformationExtraction(self):
        path = "OpenInformation" + "/" + self.claimID + "/" + self.number
        f = open(path, "r", encoding="utf-8")
        openInfo = json.load(f)
        return openInfo


    def processSentence(self,sentence):
        sentence = str(sentence)
        if sentence.strip():
            ner = self.predictorNER.predict(sentence)
            tags = ner.get('tags')
            words = ner.get('words')
            for i in range(len(tags)):
                if i == 0:
                    words[i] = words[i][0].upper() + words[i][1:]
                else:
                    if tags[i] != "O":
                        if words[i] != "'s":
                            words[i] = words[i][0].upper() + words[i][1:]
                    else:
                        words[i] = words[i].lower()
            sent = ""
            index = 0
            parts = sentence.split(' ')
            for j in range(len(parts)):
                # print(segmentation)
                indexNew = index
                lastWord = ""
                for i in range(index, len(words)):
                    if words[i].lower() in parts[j].lower():
                        # print(words[i])
                        indexNew += 1
                        sent += words[i]
                        lastWord += words[i]
                        if lastWord.lower() == parts[j].lower():
                            index = indexNew
                            sent += " "
                            break
            return sent



    def getSnippetText(self):
        document = self.parts
        snippet = []
        title = ""
        if self.title != "None":
            sentences = self.spacy(self.title)
            for sentence in sentences.sents:
                title += self.processSentence(sentence)
        for sen in document:
            sentences = self.spacy(sen)
            part = []
            for sentence in sentences.sents:
                processed = self.processSentence(sentence)
                if processed is not None:
                    part.append(processed)
            snippet.append(part)
        string ="The evidence "
        if len(title)>0:
            string += "with the title '"
            string += title
            string += "' "
        string += "says "
        for part in snippet:
            for sen in part:
                string += sen
            string += " ... "
        string = string[:-5]
        return string
