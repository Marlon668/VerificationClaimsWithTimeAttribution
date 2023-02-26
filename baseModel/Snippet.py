import json
import os
import re
import calendar
import datetime
from lxml import etree

class snippet:

    '''
    Snippets of an evidence
    '''
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

    def readParts(self, article):
        return article.split("...")

    def readOpenInformationExtraction(self):
        path = "OpenInformation" + "/" + self.claimID + "/" + self.number
        f = open(path, "r", encoding="utf-8")
        openInfo = json.load(f)
        return openInfo

    '''
        Read the timexpressions out of the timeml file belonging to this evidence
        Divide them in durations, references, sets and datas
        Convert the datas to python datetime
    '''
    def readTime(self):
        path = "ProcessedTimes" + "/" + self.claimID + "/" + self.number+".xml"
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
                    datas.append(newSplits[index].split('>')[1])
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
                                                                        if not root[depth].attrib['value'].isdigit():
                                                                            dateB = root[depth].attrib['value'][0:3]+'0-01-01'
                                                                            dateE = root[depth].attrib['value'][0:3]+'9-12-31'
                                                                            datas[indices[depth]] = [tuple(
                                                                                [datetime.datetime.strptime(
                                                                                    dateB,
                                                                                    '%Y-%m-%d'),
                                                                                    datetime.datetime.strptime(
                                                                                        dateE,
                                                                                        '%Y-%m-%d')]),
                                                                                root[depth].text.strip()]
                                                                        else:
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

            datas = [x for x in datas if not isinstance(x, int)]
            return datas,duren,refs,sets

    '''
        Extract publication datum out of text of evidence
    '''
    def processPublicationDate(self):
        try:
            datetime.datetime.strptime(self.publishTime[:-1], '%b %d, %Y')
        except:
            if self.optional != None:
                if not (os.path.exists("snippetDates" + "/" + self.claimID)):
                    os.mkdir("snippetDates" + "/" + self.claimID)
                f = open("snippetDates" + "/" + self.claimID + "/" + self.number, "w", encoding="utf-8")
                f.write(self.optional + "\n")
                f.close()
            else:
                word_list = self.publishTime.split()
                if len(word_list) <= 6:
                    if not (os.path.exists("snippetDates" + "/" + self.claimID)):
                        os.mkdir("snippetDates" + "/" + self.claimID)
                    f = open("snippetDates" + "/" + self.claimID + "/" + self.number, "w", encoding="utf-8")
                    f.write(self.publishTime.replace("Nob", "November").replace("Se", "September") + "\n")
                    f.close()
                else:
                    word_list = self.publishTime.split("|")[0].split()
                    if len(word_list) <= 6:
                        if not (os.path.exists("snippetDates" + "/" + self.claimID)):
                            os.mkdir("snippetDates" + "/" + self.claimID)
                        f = open("snippetDates" + "/" + self.claimID + "/" + self.number, "w",
                                 encoding="utf-8")
                        f.write(self.publishTime.split("|")[0] + "\n")
                        f.close()
                    else:
                        word_list = self.publishTime.split(u"\u2022")[0].split()  # bullet
                        if len(word_list) <= 6:
                            if not (os.path.exists("snippetDates" + "/" + self.claimID)):
                                os.mkdir("snippetDates" + "/" + self.claimID)
                            f = open("snippetDates" + "/" + self.claimID + "/" + self.number, "w",
                                     encoding="utf-8")
                            f.write(self.publishTime.split(u"\u2022")[0] + "\n")
                            f.close()
                        else:
                            word_list = self.publishTime.split("-")[0].split()  # bullet
                            if len(word_list) <= 6:
                                if not (os.path.exists("snippetDates" + "/" + self.claimID)):
                                    os.mkdir("snippetDates" + "/" + self.claimID)
                                f = open("snippetDates" + "/" + self.claimID + "/" + self.number, "w",
                                         encoding="utf-8")
                                f.write(self.publishTime.split("-")[0] + "\n")
                                f.close()

                            else:
                                word_list = self.publishTime.split(":")[0].split()  # bullet
                                if len(word_list) <= 6:
                                    if not (os.path.exists("snippetDates" + "/" + self.claimID)):
                                        os.mkdir("snippetDates" + "/" + self.claimID)
                                    f = open("snippetDates" + "/" + self.claimID + "/" + self.number,
                                             "w", encoding="utf-8")
                                    f.write(
                                        self.publishTime.split(":")[0] + "\n")
                                    f.close()

    '''
        Read the publication file out of the timeml file belonging to this evidence
    '''
    def readPublicationDate(self):
        try:
            self.publishTime =  datetime.datetime.strptime(self.publishTime[:-1], '%b %d, %Y')
        except:
            path = 'SnippetDates' + '/' + self.claimID+'/'+self.number + '.xml'
            if os.path.exists(path):
                parser = etree.XMLParser(recover=True)
                tree = etree.parse(path, parser)
                root = tree.getroot()
                if len(root) == 1:
                    if root[0].attrib['value'].find('FA') != -1:
                        dateB = root[0].attrib['value'].split('-')[0] + '-' + '09-21'
                        dateE = str(
                            int(root[0].attrib['value'].split('-')[0]) + 1) + '-' + '12-20'
                        self.publishTime = tuple(
                            [datetime.datetime.strptime(dateB, '%Y-%m-%d'),
                             datetime.datetime.strptime(dateE, '%Y-%m-%d')])
                    else:
                        if root[0].attrib['value'].find('SP') != -1:
                            dateB = root[0].attrib['value'].split('-')[0] + '-' + '03-21'
                            dateE = str(
                                int(root[0].attrib['value'].split('-')[
                                        0]) + 1) + '-' + '06-20'
                            self.publishTime = tuple(
                                [datetime.datetime.strptime(dateB, '%Y-%m-%d'),
                                 datetime.datetime.strptime(dateE, '%Y-%m-%d')])
                        else:
                            if root[0].attrib['value'].find('SU') != -1:
                                if root[0].attrib['value'].split('-')[0] != 'XXXX':
                                    dateB = root[0].attrib['value'].split('-')[0] + '-' + '06-21'
                                    dateE = root[0].attrib['value'].split('-')[0] + '-' + '09-20'
                                    self.publishTime = tuple(
                                        [datetime.datetime.strptime(dateB, '%Y-%m-%d'),
                                         datetime.datetime.strptime(dateE, '%Y-%m-%d')])
                            else:
                                if root[0].attrib['type'] == "DURATION":
                                    self.publishTime = None
                                else:
                                    if root[0].attrib['value'].find('UNDEF')!=-1:
                                        self.publishTime = None
                                    else:
                                        if root[0].attrib['value'].find('REF') != -1:
                                            self.publishTime = None
                                        else:
                                            if root[0].attrib['type'] == 'SET':
                                                self.publishTime = None
                                            else:
                                                if root[0].attrib['value'].split('-')[0] != "XXXX":
                                                    if root[0].attrib['value'].find(':') == -1:
                                                        if len(root[0].attrib['value']) == 7:
                                                            dateB = \
                                                                root[0].attrib['value'].split('-')[
                                                                    0] + '-' + root[0].attrib['value'].split('-')[
                                                                    1]+'-01'
                                                            dateE = root[0].attrib['value'].split('-')[0] + '-' + root[0].attrib['value'].split('-')[1] +'-'+ str(calendar.monthrange(int(root[0].attrib['value'].split('-')[0]), int(root[0].attrib['value'].split('-')[1]))[1])
                                                            self.publishTime = tuple(
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
                                                                self.publishTime = tuple(
                                                                    [datetime.datetime.strptime(dateB,
                                                                                                '%Y-%m-%d'),
                                                                     datetime.datetime.strptime(dateE,
                                                                                                '%Y-%m-%d')])
                                                            else:
                                                                if root[0].attrib['value'].find('UNDEF-this-hour')==-1:
                                                                    self.publishTime = datetime.datetime.strptime(root[0].attrib['value'], '%Y-%m-%d')
                                                                else:
                                                                    self.publishTime = None
                                                    else:
                                                        if root[0].attrib['value'][root[0].attrib['value'].find(':') + 1:].find(':') == -1:
                                                            self.publishTime = datetime.datetime.strptime(root[0].attrib['value'].replace('24:', '12:'),
                                                                                                        '%Y-%m-%dT%H:%M')
                                                        else:
                                                            self.publishTime = datetime.datetime.strptime(root[0].attrib['value'].replace('24:', '12:'),
                                                                                                        '%Y-%m-%dT%H:%M:%S')
                                                else:
                                                    self.publishTime=None
                else:
                    if len(root)>1:
                        if root[0].attrib['value'].find('X') != -1:
                            parts = root[0].attrib['value'].split('-')
                            parts2 = root[1].attrib['value'].split('-')
                            i = 0
                            j = 0
                            date = ""
                            while i < min(len(parts), 3) or j < min(len(parts2), 3):
                                if parts[i].find('X') == -1:
                                    date += parts[i]
                                    date += "-"
                                    i += 1
                                    j += 1
                                elif parts2[j].find('X') == -1:
                                    date += parts2[j]
                                    date += "-"
                                    i += 1
                                    j += 1
                                else:
                                    i += 1
                                    j += 1
                            date = date[:-1]
                            if i == 1:
                                self.claimDate = datetime.datetime.strptime(
                                    date,
                                    '%Y')
                            elif i == 2:
                                self.claimDate = datetime.datetime.strptime(
                                    date,
                                    '%Y-%m')
                            elif i == 3:
                                self.claimDate = datetime.datetime.strptime(
                                    date,
                                    '%Y-%m-%d')
                            else:
                                self.claimDate = None
                        else:
                            if root[0].attrib['value'].find('T') == -1 and root[1].attrib['value'].find('T') != -1:
                                # correct fault of assigning 12pm to 24 in HeidelTime
                                self.publishTime= datetime.datetime.strptime(root[1].attrib['value'].replace('24:', '12:'),
                                                                            '%Y-%m-%dT%H:%M')
                            else:
                                # we choose publication date as claim date not last updated date
                                if root[0].attrib['value'].find('T') != -1:
                                    self.publishTime = datetime.datetime.strptime(root[0].attrib['value'], '%Y-%m-%dT%H:%M')
                                else:
                                    if len(root[0].attrib['value']) == 4:
                                        # print('Claim ' + self.claimID)
                                        dateB = \
                                            root[0].attrib['value'].split('-')[
                                                0] + '-' + '01' + '-01'
                                        dateE = \
                                            root[0].attrib['value'].split('-')[
                                                0] + '-' + '12' + '-31'
                                        self.publishTime = tuple(
                                            [datetime.datetime.strptime(dateB,
                                                                        '%Y-%m-%d'),
                                             datetime.datetime.strptime(dateE,
                                                                        '%Y-%m-%d')])
                                    else:
                                        self.publishTime = datetime.datetime.strptime(root[0].attrib['value'],
                                                                                      '%Y-%m-%d')
                if len(root)>0 and self.publishTime!=None:
                    #correct years 0-19 to 2020-2019
                    if (not type(self.publishTime) is tuple) and self.publishTime.year<20 and (self.publishTime.month!=1 or self.publishTime.day!=1):
                        self.publishTime = self.publishTime.replace(self.publishTime.year+2000,self.publishTime.month,self.publishTime.day,self.publishTime.hour,self.publishTime.minute,self.publishTime.second)
                    else:
                        #delete impossible publication time
                        if (not type(self.publishTime) is tuple) and self.publishTime.year < 1500:
                            self.publishTime=None
                        if (not type(self.publishTime) is tuple) and self.publishTime.year>2020:
                            self.publishTime = None
                else:
                    self.publishTime = None
            else:
                self.publishTime = None

    def getIndex(self):
        title = ""
        if self.title != "None":
            sentences = self.spacy(self.title)
            for sentence in sentences.sents:
                title += self.processSentence(sentence)
        if len(title)>0:
            return 7
        else:
            return 4

    def getIndexHeidel(self):
        title = ""
        if self.title != "None":
            sentences = self.spacy(self.title)
            for sentence in sentences.sents:
                title += self.processSentence(sentence)
        if len(title)>0:
            return 6
        else:
            return 3

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

    """
    Process the text of the snippet with or without title
    Do segmentation of the text in sentences with Spacy
    Place uppercase letters according to the NER of the sentence
    @param withTitle: with or without title
    @param withPreText: with or without pretext
    """
    def getSnippetText(self, withTitle=True, withPreText=True):
        document = self.parts
        snippet = []
        title = ""
        if self.title != "None" and withTitle:
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
        string =""
        if withPreText:
            string +="The evidence "
            if len(title)>0:
                string += "with the title '"
                string += title
                string += "' "
            string += "says "
        else:
            if len(title)>0:
                string += title
                string += " "

        for part in snippet:
            for sen in part:
                string += sen
            string += " ... "
        string = string[:-5]
        return string

    def getPretext(self, withUpperCaseEditting=True):
        title = ""
        if self.title != "None":
            sentences = self.spacy(self.title)
            for sentence in sentences.sents:
                if withUpperCaseEditting:
                    title += self.processSentence(sentence)
                else:
                    title += str(sentence)
        string = ""
        string += "The evidence "
        if len(title) > 0:
            string += "with the title '"
            string += title
            string += "' "
        string += "says "
        return string

    """
    Process the text of the snippet with or without title
    Do segmentation of the text in sentences with Spacy
    @param withTitle: with or without title
    @param withPreText: with or without pretext
    """
    def getSnippetText2(self, withTitle=True, withPreText=True):
        document = self.parts
        snippet = []
        title = ""
        if self.title != "None" and withTitle:
            sentences = self.spacy(self.title)
            for sentence in sentences.sents:
                title += str(sentence)
        for sen in document:
            sentences = self.spacy(sen)
            part = []
            for sentence in sentences.sents:
                part.append(str(sentence))
            snippet.append(part)
        string = ""
        if withPreText:
            string += "The evidence "
            if len(title) > 0:
                string += "with the title '"
                string += title
                string += "' "
            string += "says "
        else:
            if len(title) > 0:
                string += title
                string += " "
        for part in snippet:
            for sen in part:
                string += sen
            string += " ... "
        string = string[:-5]
        return string