# -*- coding: utf-8 -*-

import numpy as np
import Settings
import os
import sys
import numpy as np
from bert_serving.client import BertClient

class DataSet:
    def __init__(self):
        self.config=Settings.Config()
        self.traindata,self.testdata=self.readData()
        self.saveWordEmb()

    def readData(self):
        traindata={'Q':[],'A':[],'L':[]}
        testdata={'Q':[],'A':[],'L':[]}
        for file in self.config.filelist:
            with open(file,'r+') as fin:
                data=fin.readlines()
                for i in range(len(data)):

                    q=data[i].split('<question>')[1].split('<\question>')[0]
                    a=data[i].split('<question>')[1].split('<\question>')[1].split('<answer>')[1].split('<\\answer>')[0]
                    l=data[i].split('<question>')[1].split('<\question>')[1].split('<answer>')[1].split('<\\answer>')[1].split('<label>')[1].split('<\label>')[0]
                    if i<=int(len(data)*self.config.data2train):
                        traindata['Q'].append(q)
                        traindata['A'].append(a)
                        traindata['L'].append(l)
                    else:
                        testdata['Q'].append(q)
                        testdata['A'].append(a)
                        testdata['L'].append(l)

        return traindata,testdata

    def saveWordEmb(self):
        wordlist=[]
        wordembeds=[]
        #with open(self.config.allsenfile,'w+') as fout:
        #    for sen in self.traindata['Q']:
        #        fout.write(sen)
        #    for sen in self.traindata['A']:
        #        fout.write(sen)
        #    for sen in self.testdata['Q']:
        #        fout.write(sen)
        #    for sen in self.testdata['A']:
        #        fout.write(sen)
        #self.parseChineseWords()
        with open(self.config.tmpwordfile,'r+') as fin:
            line=fin.read()
            words=line.strip().split(' ')
            for word in words:
                if word not in wordlist:
                    wordlist.append(word)
        wordlist.insert(0,'NA')
        #with open(self.config.word2idfile,'w+') as fout:
        #    for i in range(len(wordlist)):
        #        fout.write(wordlist[i]+' '+str(i)+'\n')
        bc = BertClient()
        for word in wordlist:
            wordembed=bc.encode([word])
            wordembeds.append(wordembed[0])
        wordembeds=np.array(wordembeds)
        np.save(self.config.wordembedfile,wordembeds)

    def parseChineseWords(self):
        cmd='java -Xmx1024m -Dfile.encoding=UTF-8 -classpath "/data/fnlp/fnlp-core/target/fnlp-core-2.1-SNAPSHOT.jar:libs/trove4j-3.0.3.jar:libs/commons-cli-1.2.jar" org.fnlp.nlp.cn.tag.CWSTagger -f models/seg.m ' + self.config.allsenfile + ' ' + self.config.tmpwordfile
        os.system(cmd)

if __name__=='__main__':
    data=DataSet()



