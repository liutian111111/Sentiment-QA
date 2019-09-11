

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
        self.labelfile='./data/fnlp/label.txt'
        self.question_segfile='./data/fnlp/question.output'
        self.answer_segfile='./data/fnlp/answer.output'
        self.tmpwordfile='./data/fnlp/tmpwords.txt'
        self.word2idfile='./data/word2id.txt'
        self.wordembedfile='./data/wordembed_128.npy'
        self.modelsavepath='./model/'
        self.bestwriter='./out/best_result.txt'
        self.temp_result='./out/temp_result.txt'
        self.ques_out_fig='./fig/question_dis.pdf'
        self.ans_out_fig='./fig/answer_dis.pdf'
        self.trainfile='./data/train.pkl'
        self.testfile='./data/test.pkl'

        self.learning_rate=0.001
        self.dropout_rate=0.9
        self.head_num=4
        self.hidden_size=128
        self.max_question_length=16
        self.max_answer_length=64
        self.class_num=4
        self.epoch_num=100
        self.q_num_blocks=1
        self.a_num_blocks=1
