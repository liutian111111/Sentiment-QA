import tensorflow as tf
from modules import ff, multihead_attention, ln, positional_encoding
from Settings import Config
import sys


class Attention:
    def __init__(self,is_training,wordembedding):
        self.config=Config()
        self.input_question=tf.placeholder(dtype=tf.int32,shape=[None,self.config.max_question_length],name='input_question')
        self.input_answer=tf.placeholder(dtype=tf.int32,shape=[None,self.config.max_answer_length],name='input_answer')
        self.input_label=tf.placeholder(dtype=tf.int32,shape=[self.config.batch_size],name='input_label')

        wordembedding=tf.get_variable(initializer=wordembedding,name='word_embedding')

        ###
        #question encoder
        ###

        enc_q=tf.cast(tf.nn.embedding_lookup(wordembedding,self.input_question),tf.float32)
        enc_q+=positional_encoding(enc_q,self.config.max_question_length,masking=False)
        if is_training:
            enc_q=tf.layers.dropout(enc_q,rate=self.config.dropout_rate)
        for i in range(self.config.q_num_blocks):
            with tf.variable_scope("q_num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                # self-attention
                enc_q = multihead_attention(queries=enc_q,
                                          keys=enc_q,
                                          values=enc_q,
                                          num_heads=self.config.head_num,
                                          dropout_rate=self.config.dropout_rate,
                                          training=is_training,
                                          causality=False)
                # feed forward
                enc_q = ff(enc_q, num_units=[4*self.config.hidden_size, self.config.hidden_size])
        enc_q_out=enc_q

        ###
        #answer encoder
        ###

        enc_a=tf.cast(tf.nn.embedding_lookup(wordembedding,self.input_answer),tf.float32)
        enc_a+=positional_encoding(enc_a,self.config.max_answer_length,masking=False)
        if is_training:
            enc_a=tf.layers.dropout(enc_a,rate=self.config.dropout_rate)
        for i in range(self.config.a_num_blocks):
            with tf.variable_scope("a_num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                # self-attention
                enc_a = multihead_attention(queries=enc_a,
                                            keys=enc_a,
                                            values=enc_a,
                                            num_heads=self.config.head_num,
                                            dropout_rate=self.config.dropout_rate,
                                            training=is_training,
                                            causality=False)
                # feed forward
                enc_a = ff(enc_a, num_units=[4*self.config.hidden_size, self.config.hidden_size])

        enc_a_out=enc_a

        ###
        #match attention
        ###
        #question2answer
        with tf.variable_scope("match_num_blocks_qa", reuse=tf.AUTO_REUSE):
            # self-attention
            enc_qa = multihead_attention(queries=enc_q_out,
                                        keys=enc_a_out,
                                        values=enc_a_out,
                                        num_heads=self.config.head_num,
                                        dropout_rate=self.config.dropout_rate,
                                        training=is_training,
                                        causality=False)
            # feed forward
            enc_qa = ff(enc_qa, num_units=[4*self.config.hidden_size, self.config.hidden_size])

        enc_qa_out=enc_qa

        #answer2question
        with tf.variable_scope("match_num_blocks_aq", reuse=tf.AUTO_REUSE):
            # self-attention
            enc_aq = multihead_attention(queries=enc_a_out,
                                         keys=enc_q_out,
                                         values=enc_q_out,
                                         num_heads=self.config.head_num,
                                         dropout_rate=self.config.dropout_rate,
                                         training=is_training,
                                         causality=False)
            # feed forward
            enc_aq = ff(enc_aq, num_units=[4*self.config.hidden_size, self.config.hidden_size])

        enc_aq_out=enc_aq

        enc_out=tf.concat([enc_qa_out,enc_aq_out],axis=1)

        ###
        #word-level attention
        ###
        attention_wm_r=tf.get_variable('attention_wm_r',[self.config.hidden_size,1])
        attention_wm_w=tf.get_variable('attention_wm_w',[self.config.hidden_size,self.config.hidden_size])
        attention_wm_u=tf.get_variable('attention_wm_u',[self.config.hidden_size,self.config.hidden_size])
        enc_out_w=tf.reshape(tf.matmul(tf.nn.softmax(tf.reshape(tf.matmul(tf.multiply(tf.matmul(enc_out,attention_wm_w) \
            ,tf.nn.sigmoid(tf.matmul(enc_out,attention_wm_u))),attention_wm_r),\
            [self.config.batch_size,1,self.config.max_question_length+self.config.max_answer_length])) \
            ,enc_out),[self.config.batch_size,self.config.hidden_size])

        ###
        #output layer
        ###
        output_label=tf.one_hot(self.input_label,self.config.class_num)
        answer_embedding=tf.get_variable('answer_embedding',[self.config.hidden_size,self.config.class_num])
        sen_a=tf.get_variable('sen_a',[self.config.class_num])
        sen_a_out=tf.add(tf.matmul(tf.reshape(tf.tanh(enc_out_w),[self.config.batch_size,self.config.hidden_size]),answer_embedding), sen_a)
        self.prob= tf.nn.softmax(sen_a_out)
        self.loss= tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=sen_a_out, labels=output_label))
        self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                                              weights_list=tf.trainable_variables())
        self.total_loss=self.loss+self.l2_loss




