#coding:utf-8
import tensorflow as tf
import numpy as np
BATCH_SIZE=8
seed=23455

#基于seed产生随机数,生成训练集
rnd=np.random.RandomState(seed)
X=rnd.rand(32,2)
#生成标签
Y=[[int(x0+x1<1)]for(x0,x1) in X]
print "X:\n",X
print "Y:\n",Y

#定义神经网络的输入、参数和输出，定义前向传播过程
x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1))

w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

#定义损失函数及反响传播算法
loss=tf.reduce_mean(tf.square(y-y_))
train_step=tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#train_step=tf.train.MonentumOptimizer(0.001,0.9).minimize(loss)


#生成对话，训练Step轮
with tf.Session() as sess:
	init_op=tf.global_variables_initializer()
	sess.run(init_op)
	print "w1:\n",sess.run(w1)
	print "w2:\n",sess.run(w2)
	print "\n"

	#训练模型
	STEPS=30000
	for i in range(STEPS):
		start=(i*BATCH_SIZE)%32
		end=start+BATCH_SIZE
		sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
		if i%500==0:
			total_loss=sess.run(loss,feed_dict={x:X,y_:Y})
			print "after %d training steps,loss on all data is %g"%(i,total_loss)
	print "------------------------------"
	print "w1:\t",sess.run(w1)
	print "w2:\t",sess.run(w2)
