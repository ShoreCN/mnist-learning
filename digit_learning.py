import tensorflow as tf
import input_data

mnist = input_data.read_data_sets('data', one_hot = True)

#读入的输入数据存放在x变量中
x = tf.placeholder("float", [None, 784])

#预测模型的结果存放在变量y中，W代表每个数字的加权值，b作为偏置量
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

#实际标记数据的真实值存放在y_中
y_ = tf.placeholder("float", [None, 10])

#定义一个交叉熵用于判断学习到的模型的准确性，使用梯度下降法进行优化
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#创建一个Saver保存学习到的变量结果
saver = tf.train.Saver()

#计算学习结果的准确性
correct_result = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_result, "float"))

#运行
init = tf.initialize_all_variables()
with tf.Session() as s:
    s.run(init)
    for i in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        s.run(train_step, feed_dict={x : batch_x, y_ : batch_y})

    #将模型计算出来的结果W，b储存起来
    save_path = saver.save(s, "checkpoint\\variable.ckpt")
    print("Model variable saved in file: "+save_path)

    print(s.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))