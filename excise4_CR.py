#coding:utf-8
import tensorflow as tf
import numpy as np
import os
import sys
from six.moves import urllib
import csv
import cifar10_input
import tarfile
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 超参数
learning_rate_init = 0.001 # 学习率
training_epochs = 1000 # 训练轮数
batch_size = 100    # 一批次样本数量
display_step = 100   # 显示损失
conv1_kernel_num = 32   # 第一卷积层卷积核的数量
conv2_kernel_num = 32   # 第二卷积层卷积核的数量
fc1_units_num = 192     # 第一全连接层，特征维度数量
fc2_units_num = 96      # 第二全连接层，特征维度数量

#数据集中输入图像的参数
dataset_dir = './cifar10_data'
num_examples_per_epoch_for_train = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
num_examples_per_epoch_for_eval = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

image_size = cifar10_input.IMAGE_SIZE # 图形大小 24*24*3
image_channel = 3  # 通道数
n_classes = cifar10_input.NUM_CLASSES

# 从网上下载 cifar10 数据集,放到指定目录下，
def maybe_download_and_extract(data_dir):
    "下载并解压数据集，from Alex's website."
    dest_directory  = data_dir
    DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory,filename)
    if not os.path.exists(filepath):
        def _progress(count,block_size,total_size):
            sys.stdout.write('\r>>Downloading %s %.1f%%'%(filename,float(count*batch_size)/float(total_size)*100.0))
            sys.stdout.flush()
        filepath,_ = urllib.request.urlretrieve(DATA_URL,filepath,_progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully down loaded',filename,statinfo.st_size,'bytes.')
    tarfile.open(filepath,'r:gz').extractall(dest_directory)

def get_distorted_train_batch(data_dir, batch_size):
    if not data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                    batch_size=batch_size)
    return images, labels

# 对测试数据进行增强
def get_undistorted_eval_batch(data_dir,eval_data,batch_size):
    if not data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(data_dir,'cifar-10-batches-bin')
    images,labels = cifar10_input.inputs(eval_data=eval_data,
                                         data_dir=data_dir,
                                         batch_size=batch_size)
    return images,labels

# W,根据指定的维数返回初始化好的指定名称的权重，
def WeightsVariable(shape,name_str,stddev=0.1):
    initial = tf.truncated_normal(shape=shape,stddev=stddev,dtype=tf.float32)#截断正太分布
    return tf.Variable(initial,dtype=tf.float32,name=name_str)

# b,根据指定的维数返回初始化好的指定名称的偏置，
def BiasesVariable(shape,name_str,init_value=0.0):
    initial = tf.constant(init_value,shape=shape)# 常量节点
    return tf.Variable(initial,dtype=tf.float32,name=name_str)

# 2维卷积层 activation (conv2d+bias) 的封装
def Conv2d(x,W,b,stride=1,padding='SAME',activation=tf.nn.relu,act_name='relu'):
    with tf.name_scope('conv2d_bias'):
        y = tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding=padding)
        y = tf.nn.bias_add(y,b)
    with tf.name_scope(act_name):
        y = activation(y)
    return y

# 2维池化层pool的封装
def Pool2d(x,pool=tf.nn.max_pool,k=2,stride=2,padding='SAME'):
    return pool(x,ksize=[1,k,k,1],strides=[1,stride,stride,1],padding=padding)

# 全连接层 activation (wx+b) 的封装
def FullyConnected(x,W,b,activate=tf.nn.relu,act_name='relu'):
    with tf.name_scope('Wx_b'):
        y = tf.matmul(x,W)
        y = tf.add(y,b)
    with tf.name_scope(act_name):
        y = activate(y)
    return y

# 为每一层的激活输出添加汇总节点
def AddActivationSummary(x):
    tf.summary.histogram('/activations',x)
    tf.summary.scalar('/sparsity',tf.nn.zero_fraction(x))

# 为所有损失节点添加（滑动平均）标量汇总操作
def AddLossesSummary(losses):
    # 计算所有（individual loss）和 (total loss) 的滑动平均
    loss_averages = tf.train.ExponentialMovingAverage(0.9,name='avg')
    loss_averages_op = loss_averages.apply(losses)
    # 为所有 （individual loss）和 (total loss)  绑定标量汇总节点
    # 为所有平滑处理过的（individual loss）和 (total loss) 也绑定标量汇总节点
    for loss in losses:
        # 没有平滑过的 loss 名字后面加上（raw），
        # 平滑过的 loss 使用其原来的名字
        tf.summary.scalar(loss.op.name + '(raw)', loss)
        tf.summary.scalar(loss.op.name + '(avg)',loss_averages.average(loss))
    return loss_averages_op

# 前向推断过程，Inference
def Inference(images_holder):
    "此处设置前向推断层，包括两个卷积层，两个池化层，一个多维转一维层，三个全连接层，"
    # 第一个卷积层，activate,(conv2d+bias)
    with tf.name_scope('Conv2d_1'):
        weights = WeightsVariable(shape=[5,5,image_channel,conv1_kernel_num],
                                  name_str='weights',
                                  stddev=5e-2)
        biases = BiasesVariable(shape=[conv1_kernel_num],
                                  name_str='biases',
                                init_value=0.0)
        con1_out = Conv2d(images_holder,weights,biases,stride=1,padding='SAME')
        AddActivationSummary(con1_out)
    # 第一个池化层
    with tf.name_scope('Pool2d_1'):
        pool1_out = Pool2d(con1_out,pool=tf.nn.max_pool,k=3,stride=2,padding='SAME')
    # 第二个卷积层，activate,(conv2d+bias)
    with tf.name_scope('Conv2d_2'):
        weights = WeightsVariable(shape=[5, 5, conv1_kernel_num, conv2_kernel_num],
                                  name_str='weights',
                                  stddev=5e-2)
        biases = BiasesVariable(shape=[conv2_kernel_num],
                                name_str='biases',
                                init_value=0.0)
        con2_out = Conv2d(pool1_out, weights, biases, stride=1, padding='SAME')
        AddActivationSummary(con2_out)
    # 第二个池化层
    with tf.name_scope('Pool2d_2'):
        pool2_out = Pool2d(con2_out,pool=tf.nn.max_pool,k=3,stride=2,padding='SAME')
    # 将二维特征图转换为一维特征向量
    with tf.name_scope('FeatReshape'):
        features = tf.reshape(pool2_out,[batch_size,-1]) # [batch_size,6*6*64]
        feats_dim = features.get_shape()[1].value   # 6*6*64
    # 第一个全连接层，nonlinear
    with tf.name_scope('FC1_nonlinear'):
        weights = WeightsVariable(shape=[feats_dim, fc1_units_num],
                                  name_str='weights',
                                  stddev=4e-2)
        biases = BiasesVariable(shape=[fc1_units_num],
                                name_str='biases',
                                init_value=0.1)
        fc1_out = FullyConnected(features, weights, biases,
                                  activate=tf.nn.relu,
                                  act_name='relu')
        AddActivationSummary(fc1_out)
    # 第二个全连接层，nonlinear
    with tf.name_scope('FC2_nonlinear'):
        weights = WeightsVariable(shape=[fc1_units_num, fc2_units_num],
                                  name_str='weights',
                                  stddev=4e-2)
        biases = BiasesVariable(shape=[fc2_units_num],
                                name_str='biases',
                                init_value=0.1)
        fc2_out = FullyConnected(fc1_out, weights, biases,
                                 activate=tf.nn.relu,
                                 act_name='relu')
        AddActivationSummary(fc2_out)
    # 第三个全连接层，linear
    with tf.name_scope('FC3_linear'):
        fc3_units_num = n_classes
        weights = WeightsVariable(shape=[fc2_units_num, fc3_units_num],
                                  name_str='weights',
                                  stddev=1.0/fc2_units_num)
        biases = BiasesVariable(shape=[fc3_units_num],
                                name_str='biases',
                                init_value=0.0)
        logits = FullyConnected(fc2_out, weights, biases,
                                 activate=tf.identity,
                                 act_name='linear')
        AddActivationSummary(logits)
    return logits


def TrainModel():
    # 调用上面写的函数构造计算图
    with tf.Graph().as_default():
        # 计算图输入层，Inputs
        with tf.name_scope('Inputs'):
            images_holder = tf.placeholder(tf.float32,
                                           [batch_size,image_size,image_size,image_channel],
                                           name='images')
            labels_holder = tf.placeholder(tf.int32,
                                           [batch_size],
                                           name='labels')
        # 计算图前向推断过程 Inference
        with tf.name_scope('Inference'):
            logits = Inference(images_holder)
        # 计算图，损失层 loss
        with tf.name_scope('Loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_holder,
                                                                           logits=logits)
            cross_entropy_loss = tf.reduce_mean(cross_entropy, name='xentropy_loss')
            total_loss = cross_entropy_loss
            average_losses = AddLossesSummary([total_loss])
        # 计算图 ，优化训练层，train
        with tf.name_scope('Train'):
            learning_rate = tf.placeholder(tf.float32)
            global_step = tf.Variable(0, name='global_step',
                                      trainable=False,
                                      dtype=tf.int64)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(total_loss,global_step=global_step)
        # 评估层，Evalute
        with tf.name_scope('Evaluate'):
            top_K_op = tf.nn.in_top_k(predictions=logits,
                                      targets=labels_holder,
                                      k=1)

        # 获取数据
        # 定义获取训练样本批次的计算节点（有数据增强distorted）
        with tf.name_scope('GetTrainBatch'):
            images_train,labels_train = get_distorted_train_batch(data_dir=dataset_dir,
                                                                  batch_size=batch_size)

        # 定义获取测试样本批次的节点
        with tf.name_scope('GetTestBatch'):
            images_test,labels_test = get_undistorted_eval_batch(eval_data=True,
                                                                 data_dir=dataset_dir,
                                                                 batch_size=batch_size)
        # 收集所有汇总节点
        merged_summaries = tf.summary.merge_all()

        # 添加初始化节点
        init_op = tf.global_variables_initializer()

        print('把计算图写入事件文件，在TensorBoard里面查看')
        summary_writer = tf.summary.FileWriter(logdir='logs/excise411')
        summary_writer.add_graph(graph=tf.get_default_graph())
        summary_writer.flush()

        # 将评估结果保存到csv文件中
        results_list = list()
        # 写入参数配置
        results_list.append(['learning_rate',learning_rate_init,
                             'training_epoches',training_epochs,
                             'batch_size',batch_size,
                             'display_step',display_step,
                             'conv1_kernel_num',conv1_kernel_num,
                             'conv2_kernel_num',conv2_kernel_num,
                             'fc1_units_num',fc1_units_num,
                             'fc2_units_num',fc2_units_num])
        results_list.append(['train_step','train_loss','train_step','train_accuracy'])

        # 开始训练 ，Session
        with tf.Session() as sess:
            sess.run(init_op)
            print('====>>>>>>===开始在训练集上训练模型===<<<<<<===')
            total_batches = int(num_examples_per_epoch_for_train/batch_size)
            print('Total epoch Count:',training_epochs)
            print('Per batch Size:',batch_size)
            print('Train sample Count Per Epoch:',num_examples_per_epoch_for_train)
            print('Total batch Count Per Epoch:',total_batches)

            # 启动数据读取对列
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            # 记录模型被训练的步数
            training_step = 0
            # 训练指定轮数， 每一轮的训练样本总数为：num_examples_per_epoch_for_train
            for epoch in range(training_epochs):
                # 每一轮都要把所有的batch 跑一边
                for batch_idx in range(total_batches):
                    # 运行获取训练数据的计算图，取出一个批次数量
                    images_batch,labels_batch = sess.run([images_train,labels_train])
                    # 运行优化器训练节点
                    _,loss_value = sess.run([train_op,total_loss],
                                            feed_dict={images_holder:images_batch,
                                                       labels_holder:labels_batch,
                                                       learning_rate:learning_rate_init})
                    # 每调用一次训练节点， training_step就加1，最终==training_epochs*total_batch
                    training_step = sess.run(global_step)
                    # 每训练display_step 次， 计算当前模型的损失和分类准确率
                    if training_step % display_step==0:
                        # 运行 accuracy 节点， 计算当前批次的训练样本的准确率
                        predictions = sess.run([top_K_op],
                                               feed_dict={images_holder:images_batch,
                                                          labels_holder :labels_batch})
                        # 当前每个批次上的预测正确的样本量
                        batch_accuracy = np.sum(predictions)*1.0/batch_size
                        results_list.append([training_step,loss_value,training_step,batch_accuracy])
                        print('Training Step:',str(training_step),
                              'Training Loss = {:.6f}'.format(loss_value),
                              'Training Accuracy = {:.5f}'.format(batch_accuracy))
                        # 运行汇总节点
                        summary_str = sess.run(merged_summaries,
                                               feed_dict={images_holder:images_batch,
                                                          labels_holder:labels_batch})
                        summary_writer.add_summary(summary=summary_str, global_step=training_step)
                        summary_writer.flush()
            summary_writer.close()
            print('训练完毕！')

            print('====>>>>>>===开始在测试集上训练模型===<<<<<<===')
            total_batches = int(num_examples_per_epoch_for_eval / batch_size)
            total_examples = total_batches*batch_size
            print('Per batch Size:', batch_size)
            print('Test sample Count Per Epoch:', total_examples)
            print('Total batch Count Per Epoch:', total_batches)
            correct_predicted = 0
            for test_step in range(total_batches):
                # 运行获取测试数据的计算图， 取出一个批次测试数据，
                images_batch,labels_batch = sess.run([images_test,labels_test])
                # 运行 accuracy 节点，计算当前批次的测试样本的准确率
                predictions = sess.run([top_K_op],
                                       feed_dict={images_holder: images_batch,
                                                  labels_holder: labels_batch})
                # 累积每个批次上的预测正确的样本量
                correct_predicted += np.sum(predictions)
            accuracy_score = correct_predicted*1.0/total_examples
            print('------------>Accuracy on Test Examples:',accuracy_score)
            results_list.append(['Accuracy on Test Examples:',accuracy_score])

        # 将评估结果保存到文件中
        results_file = open('evaluate_results.csv','w')
        csv_writer = csv.writer(results_file,dialect='excel')
        for row in results_list:
            csv_writer.writerow(row)

def main(argv=None):
    maybe_download_and_extract(data_dir=dataset_dir)
    train_dir = 'logs/'
    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(train_dir)
    TrainModel()

if __name__ == '__main__':
    start_time = time.time()
    tf.app.run()
    stop_time = time.time()
    print('Time:',(stop_time-start_time))








