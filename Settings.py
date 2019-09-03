

class Config(object):
    def __init__(self):
        self.data2train=0.8
        self.train_size=24000
        self.test_size=6000
        self.batch_size=50

        self.filelist=['./data/QASC/Beauty_domain.txt','./data/QASC/Electronic_domain.txt','./data/QASC/Shoe_domain.txt']
        self.allsenfile='./data/fnlp/allsens.txt'
        self.questionfile='./data/fnlp/question.txt'
        self.answerfile='./data/fnlp/answer.txt'
        self.question_segfile='./data/fnlp/question.output'
        self.answer_segfile='./data/fnlp/answer.output'
        self.tmpwordfile='./data/fnlp/tmpwords.txt'
        self.word2idfile='./data/word2id.txt'
        self.wordembedfile='./data/wordembed.npy'

        self.learning_rate=0.001
        self.dropout_rate=0.1
        self.head_num=4
        self.hidden_size=768
        self.max_question_length=33
        self.max_answer_length=172
        self.class_num=4
        self.epoch_num=10
        self.q_num_blocks=1
        self.a_num_blocks=1
