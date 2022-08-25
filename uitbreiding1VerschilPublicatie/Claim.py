import json
import re
import xml.etree.ElementTree as ET
import datetime
import os
import calendar
from lxml import etree
from transformers import AutoTokenizer

from uitbreiding1VerschilPublicatie import Snippet


class claim:

    def __init__(self, claimID, claim, label, claimURL, reason, categories, speaker, checker, tags, articleTitle,
                 publishDate, claimDate, entities, directorySnippets,OIEpredictor,NERpredictor,spacy,coreference):
        self.coreference = coreference
        self.spacy = spacy
        self.claimID = claimID
        self.predictorOIE = OIEpredictor
        self.predictorNER = NERpredictor
        self.claim = claim
        self.label = label
        self.claimURl = claimURL
        self.reason = reason
        self.speaker = speaker
        self.checker = checker
        self.articleTitle = articleTitle
        self.publishDate = publishDate
        self.claimDate = claimDate
        if (self.claimDate == "None"):
            self.claimDate = publishDate
        self.readTags(tags)
        self.readEntities(entities)
        self.readCategories(categories)
        self.readSnippets(directorySnippets + "/" + claimID)

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
                indexNew = index
                lastWord = ""
                for i in range(index, len(words)):
                    if words[i].lower() in parts[j].lower():
                        indexNew += 1
                        sent += words[i]
                        lastWord += words[i]
                        if lastWord.lower() == parts[j].lower():
                            index = indexNew
                            sent += " "
                            break
            return sent

    '''
    Construct the document by a title and articletext (where the correct predicted capital letters are set)
    '''
    def processDocument(self):
        claim = []
        document = self.claim.split('...')
        title = ""
        if self.articleTitle != "None":
            sentences = self.spacy(self.articleTitle)
            for sentence in sentences.sents:
                title += self.processSentence(sentence)
        for sen in document:
            sentences = self.spacy(sen)
            part = []
            for sentence in sentences.sents:
                part.append(self.processSentence(sentence))
            claim.append(part)
        document = self.getClaimText(title, claim)
        return document

    '''
    Constuct the document by a title and articletext
    '''
    def processDocument2(self):
        claim = []
        document = self.claim.split('...')
        title = ""
        if self.articleTitle != "None":
            sentences = self.spacy(self.articleTitle)
            for sentence in sentences.sents:
                title += str(sentence)
        for sen in document:
            sentences = self.spacy(sen)
            part = []
            for sentence in sentences.sents:
                part.append(str(sentence))
            claim.append(part)
        document = self.getClaimText(title, claim)
        return document

    def processCoreference(self):
        claim = []
        document = self.claim.split('...')

        # article title
        title = ""
        if self.articleTitle != "None":
            sentences = self.spacy(self.articleTitle)
            for sentence in sentences.sents:
                title += self.processSentence(sentence)
        for sen in document:
            sentences = self.spacy(sen)
            part = []
            for sentence in sentences.sents:
                part.append(self.processSentence(sentence))
            claim.append(part)
        document = self.getClaimText(title,claim)
        for ev in self.snippets:
            document += ev.getSnippetText()
        f = open("Coreference" + "/" + "Versie1" + "/" + self.claimID, "a", encoding="utf-8")

        perform = self.coreference.predict(document)

    def processCoreference2(self):
        claim = []
        document = self.claim.split('...')

        # article title
        title = ""
        if self.articleTitle != "None":
            sentences = self.spacy(self.articleTitle)
            for sentence in sentences.sents:
                title += self.processSentence(sentence)
        for sen in document:
            sentences = self.spacy(sen)
            part = []
            for sentence in sentences.sents:
                part.append(self.processSentence(sentence))
            claim.append(part)
        document = self.getClaimText(title,claim)
        for ev in self.snippets:
            document += ev.getSnippetText()
        f = open("Coreference" + "/" "Versie2" + "/" + self.claimID, "a", encoding="utf-8")
        perform = self.coreference.perform_coreference(document)
        Wordtokens = perform['tokenized_doc']['orig_tokens']
        subWordTokens = perform['tokenized_doc']['subtoken_map']
        coreferenceOutput = dict()
        coreferenceOutput['Wordtokens'] = Wordtokens
        coreferenceOutput['SubWordTokens'] = subWordTokens
        coreferenceOutput['Clusters'] = perform['clusters']
        json.dump(coreferenceOutput,f)

    def readCoreference(self):
        path = "Coreference" + "/" + self.claimID
        f = open(path, "r", encoding="utf-8")
        coreferenceOutput = json.load(f)
        wordTokens = coreferenceOutput['Wordtokens']
        subWordTokens =  coreferenceOutput['SubWordTokens']
        clusters = coreferenceOutput['Clusters']
        return wordTokens,subWordTokens,clusters


    def processOpenInformation(self):
        document = self.claim.split('...')
        if not (os.path.exists(os.pardir+"/OpenInformation" + "/" + self.claimID)):
            os.mkdir(os.pardir+"/OpenInformation" + "/" + self.claimID)
        if self.articleTitle != "None":
            document.insert(0,self.articleTitle)
        openInformation = []
        for sen in document:
            sentences = self.spacy(sen)
            for sentence in sentences.sents:
                sent = self.processSentence(sentence)
                if sent is not None:
                    openInformation.append(self.predictorOIE.predict(sent))
        f = open(os.pardir+"/OpenInformation" + "/" + self.claimID + "/" + "claim", "w", encoding="utf-8")
        json.dump(openInformation, f)
        f.close()

    def readOpenInformationExtraction(self):
        path = os.pardir+"/OpenInformation" + "/" + self.claimID + "/" + "claim"
        f = open(path, "r", encoding="utf-8")
        openInfo = json.load(f)
        return openInfo

    def readPublicationDate(self):
        path = os.pardir+'/ProcessedDates'+'/'+self.claimID+'.xml'
        print(path)
        if os.path.exists(path):
            tree = ET.parse(path)

            root = tree.getroot()
            if len(root) == 1:
                if root[0].attrib['value'].find(':') == -1:
                    if len(root[0].attrib['value']) == 7:
                        dateB = \
                            root[0].attrib['value'].split('-')[
                                0] + '-' + root[0].attrib['value'].split('-')[
                                1] + '-01'
                        dateE = root[0].attrib['value'].split('-')[0] + '-' + root[0].attrib['value'].split('-')[
                            1] + '-' + str(calendar.monthrange(int(root[0].attrib['value'].split('-')[0]),
                                                               int(root[0].attrib['value'].split('-')[1]))[1])
                        self.claimDate = tuple(
                            [datetime.datetime.strptime(dateB,
                                                        '%Y-%m-%d'),
                             datetime.datetime.strptime(dateE,
                                                        '%Y-%m-%d')])
                    else:
                        if len(root[0].attrib['value']) == 4:
                            dateB = \
                                root[0].attrib['value'].split('-')[
                                    0] + '-' + '01' + '-01'
                            dateE = \
                                root[0].attrib['value'].split('-')[
                                    0] + '-' + '12' + '-31'
                            self.claimDate = tuple(
                                [datetime.datetime.strptime(dateB,
                                                            '%Y-%m-%d'),
                                 datetime.datetime.strptime(dateE,
                                                            '%Y-%m-%d')])
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
                            if len(root[0].attrib['value']) == 7:
                                dateB = \
                                    root[0].attrib['value'].split('-')[
                                        0] + '-' + root[0].attrib['value'].split('-')[
                                        1] + '-01'
                                dateE = root[0].attrib['value'].split('-')[0] + '-' + \
                                        root[0].attrib['value'].split('-')[
                                            1] + '-' + str(
                                    calendar.monthrange(int(root[0].attrib['value'].split('-')[0]),
                                                        int(root[0].attrib['value'].split('-')[1]))[1])
                                self.claimDate = tuple(
                                    [datetime.datetime.strptime(dateB,
                                                                '%Y-%m-%d'),
                                     datetime.datetime.strptime(dateE,
                                                                '%Y-%m-%d')])
                            else:
                                if len(root[0].attrib['value']) == 4:
                                    # print('Claim ' + self.claimID)
                                    dateB = \
                                        root[0].attrib['value'].split('-')[
                                            0] + '-' + '01' + '-01'
                                    dateE = \
                                        root[0].attrib['value'].split('-')[
                                            0] + '-' + '12' + '-31'
                                    self.claimDate = tuple(
                                        [datetime.datetime.strptime(dateB,
                                                                    '%Y-%m-%d'),
                                         datetime.datetime.strptime(dateE,
                                                                    '%Y-%m-%d')])
                                else:
                                    self.claimDate = datetime.datetime.strptime(root[0].attrib['value'], '%Y-%m-%d')
        else:
            print('Not found -' +  path)
            self.claimDate = None

    def readTime(self):
        path = os.pardir+"/ProcessedTimes" + "/" + self.claimID + "/" + "claim"+".xml"
        if os.path.exists(path):
            f = open(path, "r", encoding="utf-8")
            lines = f.readlines()
            oldSplits = lines[1].split('<TIMEX3')
            newSplits = []
            for line in oldSplits:
                split = line.split('</TIMEX3>')
                for part in split:
                    newSplits.append(part.replace('\n', ''))
            duren = set()
            refs = set()
            sets = set()
            datas = []
            indices = []
            for index in range(len(newSplits)):
                if newSplits[index].find('tid=') != -1:
                    datas.append(0)
                    indices.append(index)
                else:
                    datas.append(newSplits[index])
            file = open(path, encoding="utf-8-sig")
            parser = etree.XMLParser(recover=True)
            tree = etree.parse(path, parser)
            root = tree.getroot()
            for depth in range(len(root)):
                if(str(root[depth]).find('TIMEX3')) !=-1:
                    if root[depth].attrib['type'] == "DURATION":
                        duren.add(root[depth].attrib['value'])
                        datas[indices[depth]] = [root[depth].attrib['value'],root[depth].text.strip()]
                    else:
                        if root[depth].attrib['value'].find('REF') != -1:
                            refs.add(root[depth].attrib['value'])
                            datas[indices[depth]] = [root[depth].attrib['value'],root[depth].text.strip()]
                        else:
                            if root[depth].attrib['type'] == 'SET':
                                sets.add(root[depth].attrib['value'])
                                datas[indices[depth]] = [root[depth].attrib['value'],root[depth].text.strip()]
                            else:
                                if root[depth].attrib['value'].find('W') != -1 and root[depth].attrib['value'].find('WI') == -1:
                                    if root[depth].attrib['value'].split('-')[0] != 'XXXX':
                                        if root[depth].attrib['value'].split('-')[-1] =='WE':
                                            datas[indices[depth]] = [tuple(
                                                [datetime.datetime.strptime(
                                                    root[depth].attrib['value'].replace('-WE', '') + '-5',
                                                    "%Y-W%W-%w"),
                                                 datetime.datetime.strptime(
                                                     root[depth].attrib['value'].replace('-WE', '') + '-6',
                                                     "%Y-W%W-%w")]),root[depth].text.strip()]
                                        else:
                                            datas[indices[depth]] = [tuple(
                                                [datetime.datetime.strptime(root[depth].attrib['value'] + '-1',
                                                                            "%Y-W%W-%w"),
                                                 datetime.datetime.strptime(root[depth].attrib['value'] + '-6',
                                                                            "%Y-W%W-%w")]),root[depth].text.strip()]
                                else:
                                    if root[depth].attrib['value'].find('SU') != -1:
                                        if root[depth].attrib['value'].split('-')[0] != 'XXXX':
                                            dateB = root[depth].attrib['value'].split('-')[0] + '-' + '06-21'
                                            dateE = root[depth].attrib['value'].split('-')[0] + '-' + '09-20'
                                            datas[indices[depth]] = [tuple(
                                                [datetime.datetime.strptime(dateB, '%Y-%m-%d'),
                                                 datetime.datetime.strptime(dateE, '%Y-%m-%d')]),root[depth].text.strip()]
                                    else:
                                        if root[depth].attrib['value'].find('WI') != -1:
                                            if root[depth].attrib['value'].split('-')[0] != 'XXXX':
                                                dateB = root[depth].attrib['value'].split('-')[0] + '-' + '12-21'
                                                dateE = str(int(root[depth].attrib['value'].split('-')[0])+1) + '-' + '03-20'
                                                datas[indices[depth]] = [tuple(
                                                    [datetime.datetime.strptime(dateB, '%Y-%m-%d'),
                                                     datetime.datetime.strptime(dateE, '%Y-%m-%d')]),
                                                    root[depth].text.strip()]
                                        else:
                                            if root[depth].attrib['value'].find('FA') != -1:
                                                if root[depth].attrib['value'].split('-')[0] != 'XXXX':
                                                    dateB = root[depth].attrib['value'].split('-')[0] + '-' + '09-21'
                                                    dateE = str(
                                                        int(root[depth].attrib['value'].split('-')[0]) + 1) + '-' + '12-20'
                                                    datas[indices[depth]] = [tuple(
                                                        [datetime.datetime.strptime(dateB, '%Y-%m-%d'),
                                                         datetime.datetime.strptime(dateE, '%Y-%m-%d')]),
                                                        root[depth].text.strip()]
                                            else:
                                                if root[depth].attrib['value'].find('SP') != -1:
                                                    if root[depth].attrib['value'].split('-')[0] != 'XXXX':
                                                        dateB = root[depth].attrib['value'].split('-')[0] + '-' + '03-21'
                                                        dateE = str(
                                                            int(root[depth].attrib['value'].split('-')[
                                                                    0]) + 1) + '-' + '06-20'
                                                        datas[indices[depth]] = [tuple(
                                                            [datetime.datetime.strptime(dateB, '%Y-%m-%d'),
                                                             datetime.datetime.strptime(dateE, '%Y-%m-%d')]),
                                                            root[depth].text.strip()]
                                                else:
                                                    if root[depth].attrib['value'].find(':') == -1:
                                                        if len(root[depth].attrib['value']) == 7:
                                                            # print('Claim ' + self.claimID)
                                                            if root[depth].attrib['value'].split('-')[0] != 'XXXX':
                                                                if root[depth].attrib['value'].split('-')[1]=='H1':
                                                                    dateB = root[depth].attrib['value'].split('-')[0] +'-01-01'
                                                                    dateE = root[depth].attrib['value'].split('-')[0] + '-06-30'
                                                                    datas[indices[depth]] = [tuple(
                                                                        [datetime.datetime.strptime(dateB, '%Y-%m-%d'),
                                                                         datetime.datetime.strptime(dateE, '%Y-%m-%d')]),
                                                                        root[depth].text.strip()]
                                                                else:
                                                                    if root[depth].attrib['value'].split('-')[1]=='H2':
                                                                        dateB = root[depth].attrib['value'].split('-')[0]+'-07-01'
                                                                        dateE = root[depth].attrib['value'].split('-')[0]+'-12-31'
                                                                        datas[indices[depth]] = [tuple(
                                                                            [datetime.datetime.strptime(dateB, '%Y-%m-%d'),
                                                                             datetime.datetime.strptime(dateE, '%Y-%m-%d')]),
                                                                            root[depth].text.strip()]
                                                                    else:
                                                                        if root[depth].attrib['value'].split('-')[
                                                                            1] == 'Q1':
                                                                            dateB = \
                                                                            root[depth].attrib['value'].split('-')[
                                                                                0] + '-01-01'
                                                                            dateE = \
                                                                            root[depth].attrib['value'].split('-')[
                                                                                0] + '-03-31'
                                                                            datas[indices[depth]] = [tuple(
                                                                                [datetime.datetime.strptime(dateB,
                                                                                                            '%Y-%m-%d'),
                                                                                 datetime.datetime.strptime(dateE,
                                                                                                            '%Y-%m-%d')]),
                                                                                root[depth].text.strip()]
                                                                        else:
                                                                            if root[depth].attrib['value'].split('-')[
                                                                                1] == 'Q2':
                                                                                dateB = \
                                                                                    root[depth].attrib['value'].split(
                                                                                        '-')[
                                                                                        0] + '-04-01'
                                                                                dateE = \
                                                                                    root[depth].attrib['value'].split(
                                                                                        '-')[
                                                                                        0] + '-06-30'
                                                                                datas[indices[depth]] = [tuple(
                                                                                    [datetime.datetime.strptime(dateB,
                                                                                                                '%Y-%m-%d'),
                                                                                     datetime.datetime.strptime(dateE,
                                                                                                                '%Y-%m-%d')]),
                                                                                    root[depth].text.strip()]
                                                                            else:
                                                                                if \
                                                                                root[depth].attrib['value'].split('-')[
                                                                                    1] == 'Q3':
                                                                                    dateB = \
                                                                                        root[depth].attrib[
                                                                                            'value'].split(
                                                                                            '-')[
                                                                                            0] + '-07-01'
                                                                                    dateE = \
                                                                                        root[depth].attrib[
                                                                                            'value'].split(
                                                                                            '-')[
                                                                                            0] + '-09-30'
                                                                                    datas[indices[depth]] = [tuple(
                                                                                        [datetime.datetime.strptime(
                                                                                            dateB,
                                                                                            '%Y-%m-%d'),
                                                                                         datetime.datetime.strptime(
                                                                                             dateE,
                                                                                             '%Y-%m-%d')]),
                                                                                        root[depth].text.strip()]
                                                                                else:
                                                                                    if root[depth].attrib['value'].split('-')[
                                                                                    1] == 'Q4':
                                                                                        dateB = \
                                                                                            root[depth].attrib[
                                                                                                'value'].split(
                                                                                                '-')[
                                                                                                0] + '-10-01'
                                                                                        dateE = \
                                                                                            root[depth].attrib[
                                                                                                'value'].split(
                                                                                                '-')[
                                                                                                0] + '-12-31'
                                                                                        datas[indices[depth]] = [tuple(
                                                                                            [datetime.datetime.strptime(
                                                                                                dateB,
                                                                                                '%Y-%m-%d'),
                                                                                             datetime.datetime.strptime(
                                                                                                 dateE,
                                                                                                 '%Y-%m-%d')]),
                                                                                            root[depth].text.strip()]
                                                                                    else:
                                                                                        datas[indices[depth]] = [datetime.datetime.strptime(
                                                                                            root[depth].attrib['value'], '%Y-%m'),
                                                                                            root[depth].text.strip()]
                                                        else:
                                                            if len(root[depth].attrib['value'][1: ].split('-')) == 1:
                                                                # print('Claim ' + self.claimID)
                                                                if len(root[depth].attrib['value'])<4 and(root[depth].attrib['value'][0:2] == '19' or root[depth].attrib['value'][0:2]=='20' or root[depth].attrib['value'][0:2]=='18') :
                                                                    date = root[depth].attrib['value']
                                                                    while len(date)!=4:
                                                                        date = date + '0'
                                                                    datas[indices[depth]] = [datetime.datetime.strptime(str(max(int(date),int(datetime.MINYEAR))), '%Y'),root[depth].text.strip()]
                                                                else:
                                                                    if root[depth].attrib['value'].split('-')[0].find('XX') == -1:
                                                                        if int(datetime.MINYEAR)>int(root[depth].attrib['value']):
                                                                            datas[indices[depth]] = [datetime.datetime.strptime(str(1000)
                                                                                , '%Y').replace(year=1),root[depth].text.strip()]
                                                                        else:
                                                                            if root[depth].attrib['value'] == '16':
                                                                                dateB = '1600-01-01'
                                                                                dateE = '1699-12-31'
                                                                                datas[indices[depth]] = [tuple(
                                                                                    [datetime.datetime.strptime(
                                                                                        dateB,
                                                                                        '%Y-%m-%d'),
                                                                                        datetime.datetime.strptime(
                                                                                            dateE,
                                                                                            '%Y-%m-%d')]),
                                                                                    root[depth].text.strip()]
                                                                            else:
                                                                                if root[depth].attrib['value'] == '17':
                                                                                    dateB = '1700-01-01'
                                                                                    dateE = '1799-12-31'
                                                                                    datas[indices[depth]] = [tuple(
                                                                                        [datetime.datetime.strptime(
                                                                                            dateB,
                                                                                            '%Y-%m-%d'),
                                                                                            datetime.datetime.strptime(
                                                                                                dateE,
                                                                                                '%Y-%m-%d')]),
                                                                                        root[depth].text.strip()]
                                                                                else:
                                                                                    if root[depth].attrib['value'] == '15':
                                                                                        dateB = '1500-01-01'
                                                                                        dateE = '1599-12-31'
                                                                                        datas[indices[depth]] = [tuple(
                                                                                            [datetime.datetime.strptime(
                                                                                                dateB,
                                                                                                '%Y-%m-%d'),
                                                                                                datetime.datetime.strptime(
                                                                                                    dateE,
                                                                                                    '%Y-%m-%d')]),
                                                                                            root[depth].text.strip()]
                                                                                    else:
                                                                                        if root[depth].attrib[
                                                                                            'value'] == '18':
                                                                                            dateB = '1800-01-01'
                                                                                            dateE = '1899-12-31'
                                                                                            datas[indices[depth]] = [tuple(
                                                                                                [datetime.datetime.strptime(
                                                                                                    dateB,
                                                                                                    '%Y-%m-%d'),
                                                                                                    datetime.datetime.strptime(
                                                                                                        dateE,
                                                                                                        '%Y-%m-%d')]),
                                                                                                root[depth].text.strip()]
                                                                                        else:
                                                                                            if root[depth].attrib[
                                                                                                'value'] == '13':
                                                                                                dateB = '1300-01-01'
                                                                                                dateE = '1399-12-31'
                                                                                                datas[
                                                                                                    indices[depth]] = [tuple(
                                                                                                    [
                                                                                                        datetime.datetime.strptime(
                                                                                                            dateB,
                                                                                                            '%Y-%m-%d'),
                                                                                                        datetime.datetime.strptime(
                                                                                                            dateE,
                                                                                                            '%Y-%m-%d')]),
                                                                                                    root[
                                                                                                        depth].text.strip()]
                                                                                            else:
                                                                                                if root[depth].attrib[
                                                                                                    'value'] == '15':
                                                                                                    dateB = '1500-01-01'
                                                                                                    dateE = '1599-12-31'
                                                                                                    datas[
                                                                                                        indices[
                                                                                                            depth]] = [tuple(
                                                                                                        [
                                                                                                            datetime.datetime.strptime(
                                                                                                                dateB,
                                                                                                                '%Y-%m-%d'),
                                                                                                            datetime.datetime.strptime(
                                                                                                                dateE,
                                                                                                                '%Y-%m-%d')]),
                                                                                                        root[
                                                                                                            depth].text.strip()]
                                                                                                else:
                                                                                                    if root[depth].attrib[
                                                                                                        'value'] == '03':
                                                                                                        dateB = '3000-01-01'
                                                                                                        dateE = '3999-12-31'
                                                                                                        datas[indices[depth]] = [tuple(
                                                                                                            [datetime.datetime.strptime(
                                                                                                                dateB,
                                                                                                                '%Y-%m-%d').replace(year=300),
                                                                                                                datetime.datetime.strptime(
                                                                                                                    dateE,
                                                                                                                    '%Y-%m-%d').replace(year=399)]),
                                                                                                            root[
                                                                                                                depth].text.strip()]
                                                                                                    else:
                                                                                                        if \
                                                                                                        root[depth].attrib[
                                                                                                            'value'] == '01':
                                                                                                            dateB = '3000-01-01'
                                                                                                            dateE = '3999-12-31'
                                                                                                            datas[indices[
                                                                                                                depth]] = [tuple(
                                                                                                                [
                                                                                                                    datetime.datetime.strptime(
                                                                                                                        dateB,
                                                                                                                        '%Y-%m-%d').replace(
                                                                                                                        year=100),
                                                                                                                    datetime.datetime.strptime(
                                                                                                                        dateE,
                                                                                                                        '%Y-%m-%d').replace(
                                                                                                                        year=199)]),root[depth].text.strip()]
                                                                                                        else:
                                                                                                            if \
                                                                                                                    root[
                                                                                                                        depth].attrib[
                                                                                                                        'value'] == '06':
                                                                                                                dateB = '6000-01-01'
                                                                                                                dateE = '6999-12-31'
                                                                                                                datas[
                                                                                                                    indices[
                                                                                                                        depth]] = [tuple(
                                                                                                                    [
                                                                                                                        datetime.datetime.strptime(
                                                                                                                            dateB,
                                                                                                                            '%Y-%m-%d').replace(
                                                                                                                            year=600),
                                                                                                                        datetime.datetime.strptime(
                                                                                                                            dateE,
                                                                                                                            '%Y-%m-%d').replace(
                                                                                                                            year=699)]),root[depth].text.strip()]
                                                                                                            else:
                                                                                                                if \
                                                                                                                        root[
                                                                                                                            depth].attrib[
                                                                                                                            'value'] == '21':
                                                                                                                    dateB = '2100-01-01'
                                                                                                                    dateE = '2199-12-31'
                                                                                                                    datas[
                                                                                                                        indices[
                                                                                                                            depth]] = [tuple(
                                                                                                                        [
                                                                                                                            datetime.datetime.strptime(
                                                                                                                                dateB,
                                                                                                                                '%Y-%m-%d'),
                                                                                                                            datetime.datetime.strptime(
                                                                                                                                dateE,
                                                                                                                                '%Y-%m-%d')]),
                                                                                                                        root[
                                                                                                                            depth].text.strip()]
                                                                                                                else:
                                                                                                                    if \
                                                                                                                            root[
                                                                                                                                depth].attrib[
                                                                                                                                'value'] == '02':
                                                                                                                        dateB = '2100-01-01'
                                                                                                                        dateE = '2199-12-31'
                                                                                                                        datas[
                                                                                                                            indices[
                                                                                                                                depth]] = [tuple(
                                                                                                                            [
                                                                                                                                datetime.datetime.strptime(
                                                                                                                                    dateB,
                                                                                                                                    '%Y-%m-%d').replace(
                                                                                                                            year=200),
                                                                                                                                datetime.datetime.strptime(
                                                                                                                                    dateE,
                                                                                                                                    '%Y-%m-%d').replace(
                                                                                                                            year=299)]),
                                                                                                                            root[
                                                                                                                                depth].text.strip()]
                                                                                                                    else:
                                                                                                                        dateB = \
                                                                                                                        root[depth].attrib['value'].split('-')[
                                                                                                                            0] + '-01-01'
                                                                                                                        dateE = \
                                                                                                                        root[depth].attrib['value'].split('-')[
                                                                                                                            0] + '-12-31'
                                                                                                                        datas[indices[depth]] = [tuple(
                                                                                                                            [datetime.datetime.strptime(dateB,
                                                                                                                                                        '%Y-%m-%d'),
                                                                                                                             datetime.datetime.strptime(dateE,
                                                                                                                                                        '%Y-%m-%d')]),
                                                                                                                            root[
                                                                                                                                depth].text.strip()]
                                                            else:
                                                                if root[depth].attrib['value'].split('-')[0].find('XX') ==-1:
                                                                    if root[depth].attrib['value'].find('UNDEF-this') == -1:
                                                                        datas[indices[depth]] = [datetime.datetime.strptime(
                                                                            root[depth].attrib['value'].replace('TNI','').replace('TMO','').replace('TEV','').replace('TAF',''),
                                                                            '%Y-%m-%d'),root[depth].text.strip()]
                                                    else:
                                                        if root[depth].attrib['value'][root[depth].attrib['value'].find(':') + 1:].find(
                                                                ':') == -1:
                                                            if root[depth].attrib['value'].split('-')[0] != 'XXXX':
                                                                datas[indices[depth]] = [datetime.datetime.strptime(
                                                                    root[depth].attrib['value'].replace('24:', '12:'),
                                                                    '%Y-%m-%dT%H:%M'),root[depth].text.strip()]
                                                        else:
                                                            datas[indices[depth]] = [datetime.datetime.strptime(
                                                                root[depth].attrib['value'].replace('24:', '12:'),
                                                                '%Y-%m-%dT%H:%M:%S'),root[depth].text.strip()]
            return datas,duren,refs,sets

    def readSnippets(self, path):
        self.snippets = []
        try:
            with open(path, 'r', encoding='utf-8') as file:
                for snippet in file:
                    parts = snippet.split("\t")
                    self.snippets.append(Snippet.snippet(self.claimID, parts[0], parts[1], parts[2], parts[3],self.predictorOIE,self.predictorNER,self.spacy,self.coreference))
        except:
            print(path + " not exist")



    def readEntities(self, entities):
        self.entities = re.findall("'([^']*)'", entities)
    def getSnippets(self):
        return self.snippets

    def readTags(self, tags):
        if (len(tags) == 0):
            self.tags = ['None']
        else:
            if (tags.lstrip()[0] == "["):
                self.tags = re.findall("'([^']*)'", tags)
            else:
                self.tags = tags.split(",")

    def readCategories(self, categories):
        if (categories.lstrip()[0] == "["):
            self.categories = re.findall("'([^']*)'", categories)
        else:
            self.categories = categories.split()

    def getDomain(self):
        return self.claimID.split('-')[1]

    def getClaimId(self):
        return self.claimID

    def getClaim(self):
        return self.label

    def getClaimURL(self):
        return self.claimURl

    def getReason(self):
        return self.reason

    def getCategories(self):
        return self.categories

    def getSpeaker(self):
        return self.speaker

    def getChecker(self):
        return self.checker

    def getTags(self):
        return self.tags

    def getArticle(self):
        return self.articleTitle

    def getPublishDate(self):
        return self.publishDate

    def getClaimDate(self):
        return self.claimDate

    def getEntities(self):
        return self.entities

    def getIndex(self):
        title = ""
        if self.articleTitle != "None":
            sentences = self.spacy(self.articleTitle)
            for sentence in sentences.sents:
                title += self.processSentence(sentence)
        if len(title)>0:
            return 7
        else:
            return 4

    def getIndexHeidel(self):
        title = ""
        if self.articleTitle != "None":
            sentences = self.spacy(self.articleTitle)
            for sentence in sentences.sents:
                title += self.processSentence(sentence)
        if len(title)>0:
            return 6
        else:
            return 3



    def getClaimText(self,title,parts):
        string = ""
        string += "The "
        string += "claim "
        if len(title)>0:
            string += "with the title '"
            string += title
            string += "' "
        string += "says "
        for part in parts:
            for sen in part:
                if sen is not None:
                    string += sen
            string += " ... "
        string = string[:-5]
        return string

    def deriveOpenInformation(self):
        self.openInformation = []

    def getOpenInformation(self):
        return self.openInformation
