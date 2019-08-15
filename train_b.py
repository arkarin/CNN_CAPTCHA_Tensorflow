import utils
import CNN_model
import collections
import tensorflow as tf
import numpy as np

baiduiter = utils.BaiduIterator(data_dir="../dataset/baidu/train/", batch_size=50)

vad_iter = utils.BaiduIterator(data_dir="../dataset/baidu/test/", batch_size=50)
vad_imgs = np.asarray(vad_iter.images)
vad_lbs = np.asarray(vad_iter.labels)

DataParams = collections.namedtuple('DataParams', [
        'image_height', 'image_width', 'image_channel',
        'char_length', 'num_char_class', 'batch_size'])

data_params = DataParams(image_height=40,
                          image_width=100,
                          image_channel=1,
                          char_length = 4,
                          num_char_class=25,
                          batch_size=50)

tf.reset_default_graph()

mymodel = CNN_model.ConvModel(data_params=data_params)
mymodel.build_model()

saver = tf.train.Saver()

max_epoch = 10
num_batch = baiduiter.get_num_batch()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./log/train', sess.graph)
    
    for i in range(max_epoch):
        baiduiter.shuffle()
        for j in range(num_batch):
            imgs, lbs = baiduiter.next_batch()
            myfeed_dict = {mymodel.inputs: imgs, mymodel.labels: lbs}
            _, loss, char_acc, str_acc, summary = sess.run([mymodel.train_op, mymodel.loss, mymodel.char_acc, mymodel.str_acc, merged], feed_dict=myfeed_dict)
            train_writer.add_summary(summary, i)

            print(loss, "at epoch: {0}".format(i))
            print(char_acc, str_acc)
        """check the accuracy for vadset
        """
        vad_feed = {mymodel.inputs: vad_imgs, mymodel.labels: vad_lbs}
        loss, char_acc, str_acc = sess.run([mymodel.loss, mymodel.char_acc, mymodel.str_acc], feed_dict=vad_feed)
        print("loss for vadset at epoch{0}: {1}".format(i, loss))
        print(char_acc, str_acc)

    save_path = saver.save(sess, './log/ckpt/model.ckpt')
    print("saved at", save_path)

    
