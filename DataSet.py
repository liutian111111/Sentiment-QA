# -*- coding: utf-8 -*-

import numpy as np
import Settings
import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from bert_serving.client import BertClient

class DataSet:
    def __init__(self):
        self.config=Settings.Config()
        self.alldata=self.parseFile()
        #self.traindata,self.testdata=self.readData()
        #self.saveWordEmb()
        self.calculateQALength()
        self.wordembed=self.loadWordEmbed()
        self.word2id,self.id2word=self.loadWord2Id()
        if not os.path.exists(self.config.trainfile):
            self.traindata,self.testdata=self.readSegData()
        else:
            self.traindata=pickle.load(self.config.trainfile)
            self.testdata=pickle.load(self.config.testfile)
        self.iter=0
        self.label_set=['neutral','conflict','negative','positive']

    def parseFile(self):
        alldata={'Q':[],'A':[],'L':[]}
        for file in self.config.filelist:
            with open(file,'r+') as fin:
                data=fin.readlines()
                for i in range(len(data)):
                    q=data[i].split('<question>')[1].split('<\question>')[0]
                    a=data[i].split('<question>')[1].split('<\question>')[1].split('<answer>')[1].split('<\\answer>')[0]
                    l=data[i].split('<question>')[1].split('<\question>')[1].split('<answer>')[1].split('<\\answer>')[1].split('<label>')[1].split('<\label>')[0]
                    alldata['Q'].append(q)
                    alldata['A'].append(a)
                    alldata['L'].append(l)
        if not os.path.exists(self.config.questionfile):
            with open(self.config.questionfile,'w+') as f_q, \
                open(self.config.answerfile,'w+') as f_a, \
                open(self.config.labelfile,'w+') as f_l:
                for i in range(len(alldata['Q'])):
                    f_q.write(alldata['Q'][i]+'\n')
                    f_a.write(alldata['A'][i]+'\n')
                    f_l.write(alldata['L'][i]+'\n')
        return alldata

    def readData(self):
        traindata={'Q':[],'A':[],'L':[]}
        testdata={'Q':[],'A':[],'L':[]}
        for file in self.config.filelist:
            with open(file,'r+') as fin:
                data=fin.readlines()
                temp_order=range(len(data))
                np.random.shuffle(temp_order)
                for i in range(len(temp_order)):
                    q=data[temp_order[i]].split('<question>')[1].split('<\question>')[0]
                    a=data[temp_order[i]].split('<question>')[1].split('<\question>')[1].split('<answer>')[1].split('<\\answer>')[0]
                    l=data[temp_order[i]].split('<question>')[1].split('<\question>')[1].split('<answer>')[1].split('<\\answer>')[1].split('<label>')[1].split('<\label>')[0]
                    if i<int(self.config.train_size/3):
                        traindata['Q'].append(q)
                        traindata['A'].append(a)
                        traindata['L'].append(l)
                    else:
                        testdata['Q'].append(q)
                        testdata['A'].append(a)
                        testdata['L'].append(l)

        return traindata,testdata

    def readSegData(self):
        traindata={'Q':[],'A':[],'L':[]}
        testdata={'Q':[],'A':[],'L':[]}
        with open(self.config.question_segfile,'r+') as fin_q, open(self.config.answer_segfile,'r+') as fin_a, \
        open(self.config.labelfile,'r+') as fin_l:
            lines_q=fin_q.readlines()
            lines_a=fin_a.readlines()
            lines_l=fin_l.readlines()
            temp_order=range(len(self.alldata['Q']))
            np.random.shuffle(temp_order)
            for i in range(len(temp_order)):
                if i<self.config.train_size:
                    traindata['Q'].append(lines_q[temp_order[i]].strip().split(' '))
                    traindata['A'].append(lines_a[temp_order[i]].strip().split(' '))
                    traindata['L'].append(lines_l[temp_order[i]].strip())
                else:
                    testdata['Q'].append(lines_q[temp_order[i]].strip().split(' '))
                    testdata['A'].append(lines_a[temp_order[i]].strip().split(' '))
                    testdata['L'].append(lines_l[temp_order[i]].strip())
        f_train=open(self.config.trainfile,'w+')
        pickle.dump(traindata,f_train)
        f_train.close()
        f_test=open(self.config.testfile,'w+')
        pickle.dump(testdata,f_test)
        f_test.close()
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

        with open(self.config.questionfile,'w+') as fout:
            for sen in self.traindata['Q']:
                fout.write(sen+'\n')
            for sen in self.testdata['Q']:
                fout.write(sen+'\n')

        with open(self.config.answerfile,'w+') as fout:
            for sen in self.traindata['A']:
                fout.write(sen+'\n')
            for sen in self.testdata['A']:
                fout.write(sen+'\n')

        #self.parseChineseWords()
        #with open(self.config.tmpwordfile,'r+') as fin:
        #    line=fin.read()
        #    words=line.strip().split(' ')
        #    for word in words:
        #        if word not in wordlist:
        #            wordlist.append(word)
        #wordlist.insert(0,'NA')
        #with open(self.config.word2idfile,'w+') as fout:
        #    for i in range(len(wordlist)):
        #        fout.write(wordlist[i]+' '+str(i)+'\n')
        #bc = BertClient()
        #for word in wordlist:
        #    wordembed=bc.encode([word])
        #    wordembeds.append(wordembed[0])
        #wordembeds=np.array(wordembeds)
        #np.save(self.config.wordembedfile,wordembeds)

    def parseChineseWords(self):
        cmd='java -Xmx1024m -Dfile.encoding=UTF-8 -classpath "/data/fnlp/fnlp-core/target/fnlp-core-2.1-SNAPSHOT.jar:libs/trove4j-3.0.3.jar:libs/commons-cli-1.2.jar" org.fnlp.nlp.cn.tag.CWSTagger -f models/seg.m ' + self.config.allsenfile + ' ' + self.config.tmpwordfile
        os.system(cmd)

    def calculateQALength(self):
        max_question_length=-1
        max_answer_length=-1
        total_ans_length=[]
        total_ans_num=[]
        total_que_length=[]
        total_que_num=[]
        with open(self.config.question_segfile,'r+') as fin:
            lines=fin.readlines()
            for line in lines:
                question_line_len=len(line.strip().split(' '))
                if question_line_len not in total_que_length:
                    total_que_length.append(question_line_len)
                    total_que_num.append(1)
                else:
                    total_que_num[total_que_length.index(question_line_len)]+=1
                if question_line_len>max_question_length:
                    max_question_length=question_line_len
        with open(self.config.answer_segfile,'r+') as fin:
            lines=fin.readlines()
            for line in lines:
                answer_line_len=len(line.strip().split(' '))
                if answer_line_len not in total_ans_length:
                    total_ans_length.append(answer_line_len)
                    total_ans_num.append(1)
                else:
                    total_ans_num[total_ans_length.index(answer_line_len)]+=1
                if answer_line_len>max_answer_length:
                    max_answer_length=answer_line_len
        print 'max_question_length:',max_question_length
        print 'max_answer_length:',max_answer_length
        self.max_quesiton_len=self.config.max_question_length
        self.max_answer_len=self.config.max_answer_length

        sort_q_index=np.argsort(total_que_length)
        total_q_num=[]
        for index in sort_q_index:
            total_q_num.append(total_que_num[index])
        total_q_length=np.sort(total_que_length)
        self.plotfig(total_q_length,total_q_num,self.config.ques_out_fig)

        sort_a_index=np.argsort(total_ans_length)
        total_a_num=[]
        for index in sort_a_index:
            total_a_num.append(total_ans_num[index])
        total_a_length=np.sort(total_ans_length)
        self.plotfig(total_a_length,total_a_num,self.config.ans_out_fig)

    def plotfig(self,x,y,filename):
        plt.clf()
        plt.plot(x, y, lw=1.5)
        plt.xlabel('Length',fontsize=15)
        plt.ylabel('Num',fontsize=15)
        plt.savefig(filename)

    def loadWordEmbed(self):
        wordembed=np.load(self.config.wordembedfile)
        return wordembed

    def loadWord2Id(self):
        word2id={}
        id2word={}
        with open(self.config.word2idfile,'r+') as fin:
            lines=fin.readlines()
            for line in lines:
                item=line.strip().split(' ')
                word2id[item[0]]=int(item[1])
                id2word[int(item[1])]=item[0]
        return word2id,id2word

    def nextBatch(self,is_training=True):
        nextAnswerBatch=[]
        nextQuestionBatch=[]
        nextLabelBatch=[]

        if is_training:
            if (self.iter+1)*self.config.batch_size>len(self.traindata['Q']):
                self.iter=0

            if self.iter==0:
                self.temp_order=range(len(self.traindata['Q']))
                np.random.shuffle(self.temp_order)
            temp_order=self.temp_order[self.iter*self.config.batch_size:(self.iter+1)*self.config.batch_size]
        else:
            if (self.iter+1)*self.config.batch_size>len(self.testdata['Q']):
                self.iter=0

            if self.iter==0:
                self.temp_order=range(len(self.testdata['Q']))
            temp_order=self.temp_order[self.iter*self.config.batch_size:(self.iter+1)*self.config.batch_size]

        for it in temp_order:
            answer=[]
            question=[]
            if is_training:
                temp_question=self.traindata['Q'][it]
                temp_answer=self.traindata['A'][it]
            else:
                temp_question=self.testdata['Q'][it]
                temp_answer=self.testdata['A'][it]

            for i in range(self.max_quesiton_len):
                question.append(0)
            for i in range(min(len(temp_question),self.max_quesiton_len)):
                question[i]=self.word2id[temp_question[i]]

            question=np.array(question)
            nextQuestionBatch.append(question)

            for i in range(self.max_answer_len):
                answer.append(0)
            for i in range(min(len(temp_answer),self.max_answer_len)):
                answer[i]=self.word2id[temp_answer[i]]

            answer=np.array(answer)
            nextAnswerBatch.append(answer)
            if is_training:
                nextLabelBatch.append(self.label_set.index(self.traindata['L'][it]))
            else:
                nextLabelBatch.append(self.label_set.index(self.testdata['L'][it]))

        self.iter+=1
        nextAnswerBatch=np.array(nextAnswerBatch)
        nextQuestionBatch=np.array(nextQuestionBatch)
        nextLabelBatch=np.array(nextLabelBatch)

        return nextQuestionBatch,nextAnswerBatch,nextLabelBatch

if __name__=='__main__':
    data=DataSet()
    #data.calculateQALength()
    #for i in range(50):
    next_q,next_a,next_l=data.nextBatch(True)
    #print 'question:',next_q.shape
    #print 'answer:',next_a.shape
    #print 'label:',next_l.shape
#
    for i in range(len(next_q)):
        question=''
        answer=''
        label=''
        for q in next_q[i]:
            question+=data.id2word[int(q)]
        for a in next_a[i]:
            answer+=data.id2word[int(a)]
        label=data.label_set[int(next_l[i])]
        print 'Question:',question
        print 'Answer:',answer
        print 'label:',label
    #print next_q



