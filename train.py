import tensorflow as tf
import random
import math
import os
import datetime
import numpy as np

from mdata import DataUtil
from model import Seq2Seq

batch_size = 100
epoch = 100
saveloc = './state'
savename = 'conversation.ckpt'


def main(_):
    mdata = DataUtil()
    model = Seq2Seq(mdata.vocab_size)
    mdata.load_example()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(saveloc)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("새로운 모델을 생성하는 중 입니다.")
            sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(saveloc, sess.graph)

        total_batch = int(math.ceil(len(mdata.examples) / float(batch_size)))
        print("총 ", total_batch * epoch, "스탭")
        for step in range(total_batch * epoch):
            enc_input, targets = mdata.next_batch(batch_size)
            _, loss = model.train(sess, enc_input, targets)

            if (step + 1) % 100 == 0:
                model.write_logs(sess, writer, enc_input, targets)
                print(str(datetime.datetime.now()), ' Step:', '%06d' % model.global_step.eval(),
                      'cost =', '{:.6f}'.format(loss))
            if (step + 1) % 5000 == 0:
                print(str(datetime.datetime.now()) + "중간 저장")
                checkpoint_path = os.path.join(saveloc, savename)
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)

        checkpoint_path = os.path.join(saveloc, savename)
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

    print('최적화 완료!')


if __name__ == "__main__":
    tf.app.run()
