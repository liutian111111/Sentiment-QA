import tensorflow as tf
import os
import sys
import datetime
import numpy as np
from Settings import Config
from DataSet import DataSet
from network import Attention
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

os.environ['CUDA_VISIBLE_DEVICES']='6,7'

FLAGS=tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('train', True, 'set True to train')

def evaluation(y_pred,y_true):
    f1_s=f1_score(y_true,y_pred,average='macro')
    accuracy_s=accuracy_score(y_true,y_pred)
    return f1_s,accuracy_s

def train(sess,setting):
    dataset=DataSet()
    wordembedding=np.load(setting.wordembedfile).astype(np.float32)
    with sess.as_default():
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = Attention(is_training=FLAGS.train, wordembedding=wordembedding)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(setting.learning_rate)
        train_op = optimizer.minimize(m.total_loss, global_step=global_step)
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(max_to_keep=None)
        for epoch in range(setting.epoch_num):
            for i in range(int(setting.train_size/setting.batch_size)):
                question_batch,answer_batch,label_batch=dataset.nextBatch(FLAGS.train)
                feed_dict={}
                feed_dict[m.input_answer]=answer_batch
                feed_dict[m.input_question]=question_batch
                feed_dict[m.input_label]=label_batch
                temp,step,loss_=sess.run([train_op,global_step,m.loss],feed_dict)

                if step%100==0:
                    time_str = datetime.datetime.now().isoformat()
                    tempstr = "{}: step {}, softmax_loss {:g}".format(time_str, step, loss_)
                    print tempstr
                    path = saver.save(sess, setting.modelsavepath + 'MT_ATT_model', global_step=step)

def test(sess,setting):
    dataset=DataSet()
    wordembedding=np.load(setting.wordembedfile).astype(np.float32)
    with sess.as_default():
        with tf.variable_scope("model"):
            mtest = Attention(is_training=FLAGS.train, wordembedding=wordembedding)
        saver = tf.train.Saver()
        testlist=range(100,4800,100)
        best_model_iter=-1
        best_model_f1=-1
        best_model_acc=-1
        for model_iter in testlist:
            try:
                saver.restore(sess, setting.modelsavepath+'MT_ATT_model-' + str(model_iter))
            except Exception:
                continue
            total_pred=[]
            total_y=[]
            for i in range(int(setting.test_size/setting.batch_size)):
                question_batch,answer_batch,label_batch=dataset.nextBatch(FLAGS.train)
                feed_dict={}
                feed_dict[mtest.input_answer]=answer_batch
                feed_dict[mtest.input_question]=question_batch
                feed_dict[mtest.input_label]=label_batch
                prob=sess.run([mtest.prob],feed_dict)
                for j in range(len(prob[0])):
                    total_pred.append(np.argmax(prob[0][j],-1))
                for j in range(len(label_batch)):
                    total_y.append(label_batch[j])
            with open(setting.temp_result,'w+') as fout:
                for i in range(len(total_pred)):
                    fout.write(str(total_y[i])+' '+str(total_pred[i])+'\n')
            f1,accuracy=evaluation(total_pred,total_y)
            if f1>best_model_f1:
                best_model_f1=f1
                best_model_acc=accuracy
                best_model_iter=model_iter
            print 'model_iter:',model_iter
            print 'f1 score:',f1
            print 'accuracy score:',accuracy
        with open(setting.bestwriter,'w+') as fout:
            fout.write('best_model_iter:'+str(best_model_iter)+'\n')
            fout.write('best_model_f1:'+str(best_model_f1)+'\n')
            fout.write('best_model_acc:'+str(best_model_acc))


def main(_):
    setting=Config()
    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        if FLAGS.train==True:
            train(sess,setting)
        else:
            test(sess,setting)

if __name__=='__main__':
    tf.app.run()