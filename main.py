import argparse
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import os
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.util import bigrams
from collections import Counter
import random

#extract dat from URLS in txt file into individual txt files
def directoryExtract(directories_list, directories_loc):
    with open(directories_list, 'r') as directoryFile:
        for directoryURL in directoryFile:
            try:
                r = requests.get(directoryURL)
                html = r.content
            except Exception as e:
                print(e)

            try:
                soup = BeautifulSoup(html, 'html.parser')
                cleanURL = re.sub('[\W_]+', '', directoryURL)
                fileLocation = directories_loc + cleanURL
                text = soup.get_text()
                text = text.replace('\r', ' ').replace('\n', ' ')
                print(fileLocation)
                with open(fileLocation, "w+") as newDir:
                    newDir.write(text)
            except Exception as e:
                print(e)

#extract dat from URLS in txt file into individual txt files     
def universityExtract(universities_list, universities_loc):
    with open(universities_list, 'r') as universityFile:
        for universityURL in universityFile:
            try:
                universityURL = universityURL.replace('\r', '').replace('\n', '')
                print(universityURL)
                r = requests.get(universityURL)
                print(r)
                html = r.content
            except Exception as e:
                print(e)
                continue
            try:
                soup = BeautifulSoup(html, 'html.parser')
                cleanURL = re.sub('[\W_]+', '', universityURL)
                fileLocation = universities_loc + cleanURL
                text = soup.get_text()
                text = text.replace('\r', ' ').replace('\n', ' ')
                print(fileLocation)
                with open(fileLocation, "w+") as newDir:
                    newDir.write(text)
            except Exception as e:
                print(e)

#initalize words for classifier
def initWords(stemming, lowerCase, directoryFolder, universityFolder, testModel):
    trainDirectory = []
    testDirectory = []
    for file in os.listdir(directoryFolder):
        page = []
        with open(directoryFolder + file, "r") as text:
            data = text.read()
            page += RegexpTokenizer('\w+').tokenize(data)
            if lowerCase:
                for i, word in enumerate(page):
                    page[i] = word.lower()
            if stemming:
                for i, word in enumerate(page):
                    page[i] = PorterStemmer().stem(word)
        if testModel:
            trainBool = random.randint(0, 1)
            if (trainBool):
                trainDirectory.append(page)
            else:
                testDirectory.append(page)
        else:
            trainDirectory.append(page)


    trainUniversity = []
    testUniversity = []
    for file in os.listdir(universityFolder):
        page = []
        with open(universityFolder + file, "r") as text:
            data = text.read()
            page += RegexpTokenizer('\w+').tokenize(data)
            if lowerCase:
                for i, word in enumerate(page):
                    page[i] = word.lower()
            if stemming:
                for i, word in enumerate(page):
                    page[i] = PorterStemmer().stem(word)
        if testModel:
            trainBool = random.randint(0, 1)
            if (trainBool):
                trainUniversity.append(page)
            else:
                testUniversity.append(page)
        else:
            trainUniversity.append(page)

    return trainDirectory, testDirectory, trainUniversity, testUniversity

#train bigram bag of words model
def modelTrain(trainDirectory, trainUniversity):
    dirBigramCount = Counter()
    bigramDirectoryNum = 0
    for document in trainDirectory:
        docBigrams = bigrams(document)
        docBigramCount = Counter(docBigrams)
        dirBigramCount += docBigramCount
        bigramDirectoryNum += docBigramCount.total()

    uniBigramCount = Counter()
    bigramUniversityNum = 0
    for document in trainUniversity:
        docBigrams = bigrams(document)
        docBigramCount = Counter(docBigrams)
        uniBigramCount += docBigramCount
        bigramUniversityNum += docBigramCount.total()

    print(dirBigramCount)
    print(bigramDirectoryNum)
    print(uniBigramCount)
    print(bigramUniversityNum)
    
def modelTest(testDirectory, testUniversity):
    print("TestModel")

#work alongside options to call correct functions
def main(testPage, testDir, stemming, lowerCase, createModel, testModel, extractDirectory, extractUniversity, universityURLs, directoryURLs, universityFolder, directoryFolder):
    if (extractDirectory):
        directoryExtract(directoryURLs, directoryFolder)
    if (extractUniversity):
        universityExtract(universityURLs, universityFolder)
    if (createModel):
        trainDirectory, testDirectory, trainUniversity, testUniversity = initWords(stemming, lowerCase, directoryFolder, universityFolder, testModel)
        modelTrain(trainDirectory, trainUniversity)
        if (testModel):
            modelTest(testDirectory, testUniversity)
        
    if (testDir):
        modelTest(testPage, [])
    


#option set
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('testPage', metavar='testPage', type=str, nargs='?', default='google.com', help='page to test against the model')
    parser.add_argument('--testDir', action="store_false", help = 'do not test a directory page against the model')
    parser.add_argument('--stemming', action="store_false", help = 'word stemming')
    parser.add_argument('--lowerCase', action="store_false", help = 'convert all words to lower case')
    parser.add_argument('--createModel', action="store_true", help = 'create classification model for directory pages')
    parser.add_argument('--testModel', action="store_true", help = 'rnadomely split data into a test and train group to test the models accuracy')
    parser.add_argument('--extractDirectory', action="store_true", help = 'extract directory data for use with training/testing')
    parser.add_argument('--extractUniversity', action="store_true", help = 'extract misc. university page data for use with training/testing')
    parser.add_argument('-universityURLs', type=str, default='university_pages.txt', help='Text file to extract university urls from')
    parser.add_argument('-directoryURLs', type=str, default='faculty_directories.txt', help='Text file to extract directory urls from')
    parser.add_argument('-universityFolder', type=str, default='university_files/', help='Folder to save extracted university pages to')
    parser.add_argument('-directoryFolder', type=str, default='directory_files/', help='Folder to save extracted directory pages to')
    args = parser.parse_args()
    main(args.testPage, args.testDir, args.stemming, args.lowerCase, args.createModel, args.testModel, args.extractDirectory, args.extractUniversity, args.universityURLs, args.directoryURLs, args.universityFolder, args.directoryFolder)