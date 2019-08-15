import utils
import CNN_model
import collections
import tensorflow as tf

giter = utils.GenIterator(50, ['./fonts/font1.ttf'])
imgs, lbs = giter.next_batch()

DataParams = collections.namedtuple('DataParams', [
        'image_height', 'image_width', 'image_channel',
        'char_length', 'num_char_class', 'batch_size'])

data_params = DataParams(image_height=60,
                          image_width=160,
                          image_channel=1,
                          char_length = 5,
                          num_char_class=10,
                          batch_size=50)

tf.reset_default_graph()

mymodel = CNN_model.ConvModel(data_params=data_params)
mymodel.build_model()

saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./log/train', sess.graph)
    
    for i in range(500):

        imgs, lbs = giter.next_batch()
        myfeed_dict = {mymodel.inputs: imgs, mymodel.labels: lbs}
        _, loss, char_acc, str_acc, summary = sess.run([mymodel.train_op, mymodel.loss, mymodel.char_acc, mymodel.str_acc, merged], feed_dict=myfeed_dict)
        train_writer.add_summary(summary, i)

        print(loss, "at epoch: {0}".format(i))
        print(char_acc, str_acc)

    save_path = saver.save(sess, '/tmp/model.ckpt')
    print("saved at", save_path)

    
    """
    saver.restore(sess, '/tmp/model.ckpt')
    print('restored')
    imgs, lbs = giter.next_batch()
    myfeed_dict = {mymodel.inputs: imgs, mymodel.labels: lbs}
    loss, char_acc, str_acc, preds = sess.run([mymodel.loss, mymodel.char_acc, mymodel.str_acc, mymodel.preds], feed_dict=myfeed_dict)
    print(loss, "at restored model")
    print(char_acc, str_acc)
    """