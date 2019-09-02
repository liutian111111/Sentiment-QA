

class Config(object):
    def __init__(self):
        self.data2train=0.8
        self.filelist=['./data/QASC/Beauty_domain.txt','./data/QASC/Electronic_domain.txt','./data/QASC/Shoe_domain.txt']
        self.allsenfile='./data/fnlp/allsens.txt'
        self.tmpwordfile='./data/FNLP/fnlp/tmpwords.txt'
        self.word2idfile='./data/word2id.txt'
        self.wordembedfile='./data/wordembed.npy'