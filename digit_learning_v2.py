import tensorflow as tf
import input_data

def wight_var(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_var(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

mnist = input_data.read_data_sets('data', one_hot = True)
x = tf.placeholder("float", [None, 784])
x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = wight_var([5,5,1,32])
b_conv1 = bias_var([32])
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides = [1,1,1,1], padding="SAME") + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

W_conv2 = wight_var([5,5,32,64])
b_conv2 = bias_var([64])
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides = [1,1,1,1], padding="SAME") + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

W_fc1 = wight_var([7*7*64, 1024])
b_fc1 = bias_var([1024])
h_fc1 = tf.nn.relu(tf.matmul(tf.reshape(h_pool2, [-1, 7*7*64]), W_fc1) + b_fc1)

#防止过拟合，添加一层dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = wight_var([1024, 10])
b_fc2 = bias_var([10])

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

y_ = tf.placeholder("float", [None, 10])

#定义一个交叉熵用于判断学习到的模型的准确性，使用梯度下降法进行优化
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#计算学习结果的准确性
correct_result = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_result, "float"))

#创建一个Saver保存学习到的变量结果
saver = tf.train.Saver()

#运行
with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    for i in range(3000):
        batch_x, batch_y = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict = {x: batch_x, y_ : batch_y, keep_prob: 1.0})
            print("step %d, training accuracy:[%g]" % (i, train_accuracy))
        s.run(train_step, feed_dict={x : batch_x, y_ : batch_y, keep_prob:0.5})

    #将模型计算出来的结果W，b储存起来
    save_path = saver.save(s, "checkpoint\\variable_v2.ckpt")
    print("Model variable saved in file: "+save_path)

    print(s.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))