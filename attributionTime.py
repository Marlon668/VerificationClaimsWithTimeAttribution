import io
import math
import os
import statistics
import sys
import random

import matplotlib
import scipy.stats as ss
import numpy
import numpy as np
import torch.nn.functional as F
from fontTools.merge import cmap
from matplotlib import pyplot as plt, colors

from division1DifferencePublication.encoderGlobal import encoder as encoderPublication
from division1DifferencePublication.verificationModelGlobal import verifactionModel as verificationPublication
from division1DifferencePublication import OneHotEncoder, labelEmbeddingLayer, encoderMetadata, \
    instanceEncoder, evidence_ranker, labelMaskDomain
from dataset import NUS
import torch
from torch.utils.data import DataLoader

'''
Calculate attribution for text and time for division1DifferencePublication
code based on https://github.com/ankurtaly/Integrated-Gradients
'''
def getLabelIndicesDomain(domainPath,labelPath,weightsPath):
    domainsIndices = dict()
    domainsLabels = dict()
    domainLabelIndices = dict()
    domainWeights = dict()
    labelSequence = []
    file = open(labelPath,'r')
    lines = file.readlines()
    for line in lines:
        labelSequence.append(line.replace('\n',''))
    file = open(domainPath,'r')
    lines = file.readlines()
    for line in lines:
        parts = line.split("\t")
        labelsDomain = parts[1].split(",")
        labelsDomain[-1] = labelsDomain[-1].replace('\n','')
        labelIndices = []
        for label in labelsDomain:
            labelIndices.append(labelSequence.index(label.replace('\n','')))
        labelIndicesDomainM = sorted(labelIndices)
        labelIndicesDomain = []
        for index in labelIndices:
            labelIndicesDomain.append(labelIndicesDomainM.index(index))
        domainsIndices[parts[0]] = labelIndices
        domainsLabels[parts[0]] = labelsDomain
        domainLabelIndices[parts[0]] = labelIndicesDomain
    file = open(weightsPath, 'r')
    lines = file.readlines()
    for line in lines:

        parts = line.split("\t")
        weightsDomainNormal = parts[1:]
        weightsDomainNormal[-1] = weightsDomainNormal[-1].replace('\n','')
        domainWeights[parts[0]] = torch.zeros(len(weightsDomainNormal))
        for i in range(len(weightsDomainNormal)):
            domainWeights[parts[0]][domainLabelIndices[parts[0]][i]] = float(weightsDomainNormal[i])
    return domainsIndices,domainsLabels,domainLabelIndices,domainWeights

def calculate_outputs_and_gradients(data, model):
    # do the pre-processing
    predict_idx = None
    gradientsEncoding = []
    gradientsTime = []
    for i in range(len(data[0])):
        print(data[1][i])
        print(data[2][i])
        metaDataClaim = oneHotEncoder.encode(data[3][i], device)
        metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
        domain = data[0][0].split('-')[0]
        output,claim_encodingFull,claim_EncodingWithoutTime,time,evidenceEncodings = model.forwardAttribution(data[1][i], data[2][i],metadata_encoding, domain,data[5][i],data[6][i])
        output = F.softmax(output, dim=0)
        target_label_idx = torch.argmax(output).item()
        index = np.ones((1)) * target_label_idx
        index = torch.tensor(index, dtype=torch.int64)
        output = output.gather(0, index)
        # clear grad
        model.zero_grad()
        claim_encodingFull.retain_grad()
        claim_EncodingWithoutTime.retain_grad()
        time.retain_grad()
        for evidenceEncoding in evidenceEncodings:
            print("evidence")
            print(evidenceEncoding)
            evidenceEncoding[0].retain_grad()
            evidenceEncoding[1].retain_grad()
            evidenceEncoding[2].retain_grad()
        output.backward()
        gradient = claim_encodingFull.grad.detach().cpu().numpy()[0]
        gradientsEncoding.append(gradient)
        claim_EncodingWithoutTime.retain_grad()
        gradient = claim_EncodingWithoutTime.grad.detach().cpu().numpy()[0]
        gradientsEncoding.append(gradient)
        gradient = time.grad.detach().cpu().numpy()[0]
        gradientsTime.append(gradient)
        for evidenceEncoding in evidenceEncodings:
            gradient = evidenceEncoding[0].grad.detach().cpu().numpy()[0]
            gradientsEncoding.append(gradient)
            gradient = evidenceEncoding[1].grad.detach().cpu().numpy()[0]
            gradientsEncoding.append(gradient)
            gradient = evidenceEncoding[2].grad.detach().cpu().numpy()[0]
            gradientsTime.append(gradient)
        invoer = [claim_encodingFull,claim_EncodingWithoutTime,time,evidenceEncodings]
    gradientsEncoding = np.array(gradientsEncoding)
    gradientsTime = np.array(gradientsTime)
    return gradientsEncoding,gradientsTime, target_label_idx,invoer,metadata_encoding

def getInputs(data, model):
    for i in range(len(data[0])):
        metaDataClaim = oneHotEncoder.encode(data[3][i], device)
        metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
        domain = data[0][0].split('-')[0]
        output,claim_encodingFull,claim_EncodingWithoutTime,time,evidenceEncodings = model.forwardAttribution(data[1][i], data[2][i],metadata_encoding, domain,data[5][i],data[6][i])
        invoer = [claim_encodingFull,claim_EncodingWithoutTime,time,evidenceEncodings]
    return invoer,metadata_encoding

def calculate_outputs_and_gradientsIntegrated(inputs,metadata_encoding, model,target_label_idx):
    # do the pre-processing
    predict_idx = None
    gradientsEncoding = []
    gradientsTime = []
    for input in inputs:
        gradientsEncodingEntry = []
        gradientsTimeEntry = []

        input[0].requires_grad = True
        input[0].retain_grad()
        input[1].requires_grad = True
        input[1].retain_grad()
        for i in range(len(input[2])):
            input[2][i][0].requires_grad = True
            input[2][i][0].retain_grad()
            input[2][i][1].requires_grad = True
            input[2][i][1].retain_grad()

        output = model.forwardIntegrated(input,metadata_encoding)
        output = F.softmax(output, dim=0)
        index = np.ones((1)) * target_label_idx
        index = torch.tensor(index, dtype=torch.int64)
        output = output.gather(0, index)
        # clear grad
        model.zero_grad()
        output.backward()
        gradient = input[0].grad.detach().cpu().numpy()[0]
        gradientsEncodingEntry.append(gradient)
        gradient = input[1].grad.detach().cpu().numpy()[0]
        gradientsTimeEntry.append(gradient)
        for i in range(len(input[2])):
            gradient = input[2][i][0].grad.detach().cpu().numpy()[0]
            gradientsEncodingEntry.append(gradient)
            gradient = input[2][i][1].grad.detach().cpu().numpy()[0]
            gradientsTimeEntry.append(gradient)
        gradientsTime.append(gradientsTimeEntry)
        gradientsEncoding.append(gradientsEncodingEntry)
    gradientsEncoding = np.array(gradientsEncoding)
    gradientsTime = np.array(gradientsTime)
    return gradientsEncoding,gradientsTime

# integrated gradients
def integrated_gradients(inputs,metadata_encoding, model, target_label_idx, predict_and_gradients, baseline, steps=50, cuda=False):
    # scale inputs and compute gradients
    baselineClaim = torch.zeros([256])
    baselineEvidence = torch.zeros([256])
    baselineClaim, baselineEvidence = model.getBaseLine(baselineClaim, baselineEvidence, metadata_encoding.squeeze(0))
    scaled_inputs = []
    for i in range(0, steps + 1):
        entry = []
        entry.append(baselineClaim + (float(i)/ steps) * (inputs[1]-baselineClaim))
        entry.append(baselineClaim + (float(i) / steps) * (inputs[2]-baselineClaim))
        evidences = []
        for j in range(len(inputs[3])):
            evidences.append((baselineEvidence + (float(i) / steps) * (inputs[3][j][1]-baselineEvidence),baselineEvidence + (float(i) / steps) * (inputs[3][j][2]-baselineEvidence)))
        entry.append(evidences)
        scaled_inputs.append(entry)

    gradientsEncoding,gradientsTime = predict_and_gradients(scaled_inputs,metadata_encoding, model, target_label_idx)

    sumGradsEncoding = [np.zeros(len(inputs[3])+1)]
    sumGradsTime = [np.zeros(len(inputs[3])+1)]
    for i in range(0,steps +1):
        sumGradsEncoding += gradientsEncoding[i]
        sumGradsTime += gradientsTime[i]
    avg_gradientsEncoding = [element/(steps+1) for element in sumGradsEncoding]
    avg_gradientsTime = [element / (steps + 1) for element in sumGradsTime]
    integrated_gradEncoding = []
    integrated_gradTime = []
    integrated_gradEncoding.append((inputs[1].numpy()-baselineClaim.numpy())*avg_gradientsEncoding[0][0])
    integrated_gradTime.append((inputs[2].numpy()-baselineClaim.numpy()) * avg_gradientsTime[0][0])
    for i in range(len(inputs[3])):
        integrated_gradEncoding.append((inputs[3][i][1].numpy()-baselineEvidence.numpy())*avg_gradientsEncoding[0][i+1])
        integrated_gradTime.append((inputs[3][i][2].numpy()-baselineEvidence.numpy()) * avg_gradientsTime[0][i + 1])
    return integrated_gradEncoding,integrated_gradTime

def random_baseline_integrated_gradients(inputs,metadata_encoding, model, target_label_idx, predict_and_gradients, steps, num_random_trials):
    all_intgradsEncoding = []
    all_intgradsTime = []
    integrated_gradEncoding,integrated_gradTime = integrated_gradients(inputs,metadata_encoding, model, target_label_idx, predict_and_gradients, \
                                            baseline=torch.zeros(256), steps=steps)
    all_intgradsEncoding.append(integrated_gradEncoding)
    all_intgradsTime.append(integrated_gradTime)
    return integrated_gradEncoding,integrated_gradTime

if __name__ == '__main__':
    '''
    argument 1 path of model
    argument 2 name of domain to take examples from to calculate attribution
    argument 3 alpha
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    domainIndices, domainLabels, domainLabelIndices, domainWeights = getLabelIndicesDomain(
        'labels/labels.tsv', 'labels/labelSequence', 'labels/weights.tsv')
    domains = {sys.argv[2]}
    datas = []
    for domain in domains:
        test_set = NUS(mode='Test', path='test/test-' + domain + '.tsv',
                       pathToSave="test/time/dataset2/",
                       domain=domain)
        dev_loader = DataLoader(test_set,
                                batch_size=1,
                                shuffle=False)
        datas.append([dev_loader, domain])

    for data in datas:
        oneHotEncoder = OneHotEncoder.oneHotEncoder('Metadata_sequence/metadata')
        labelEmbeddingLayerM = labelEmbeddingLayer.labelEmbeddingLayer(772, domainIndices)
        encoderM = encoderPublication(300, 128, sys.argv[3]).to(device)
        encoderMetadataM = encoderMetadata.encoderMetadata(3, 3, oneHotEncoder).to(device)
        instanceEncoderM = instanceEncoder.instanceEncoder().to(device)
        evidenceRankerM = evidence_ranker.evidenceRanker(772, 100).to(device)
        labelMaskDomainM = labelMaskDomain.labelMaskDomain(772, domainIndices, data[1],
                                                           len(domainIndices[data[1]])).to(device)
        verificationModelTime90A = verificationPublication(encoderM, encoderMetadataM, instanceEncoderM,
                                                                         evidenceRankerM,
                                                                         labelEmbeddingLayerM, labelMaskDomainM,
                                                                         domainIndices, domainWeights,
                                                                         data[1]).to(device)
        verificationModelTime90A.loading_NeuralNetwork(sys.argv[1])
        for entry in data[0]:
            print(entry)
            gradientsEncoding,gradientsTime, predictedLabel,_,_ = calculate_outputs_and_gradients(entry, verificationModelTime90A)
            with torch.no_grad():
                inputs, metadata_encoding = getInputs(entry, verificationModelTime90A)
                label_index = domainLabelIndices[data[1]][domainLabels[data[1]].index(entry[4][0])]
            if predictedLabel != label_index:
                attributionsEncoding,attributionsTime = random_baseline_integrated_gradients(inputs,metadata_encoding, verificationModelTime90A, label_index,
                                                                    calculate_outputs_and_gradientsIntegrated, \
                                                                    steps=1000, num_random_trials=1)
                print("Summation of the attributions encoding")
                print(numpy.sum(attributionsEncoding,axis=1))
                print("Summation of the attributions time")
                print(numpy.sum(attributionsTime,axis=1))
                print("Summation of absolute text encoding")
                print(numpy.sum(np.abs(attributionsEncoding), axis=1))
                print("Summation of absolute time ")
                print(numpy.sum(np.abs(attributionsTime), axis=1))
                print("Mediaan attributions encoding")
                print(numpy.median(attributionsEncoding, axis=1))
                print("Mediaan attributions time")
                print(numpy.median(attributionsTime, axis=1))
