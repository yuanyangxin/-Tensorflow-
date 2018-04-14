#coding:utf-8
import tensorflow as tf

x=tf.placeholder(tf.float32,shape=(1,2))
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

a=tf.matmul(x,w1)
b=tf.matmul(a,w2)

#用会话计算结果
with tf.Session() as sess:
	init_op=tf.global_variables_initializer()
	sess.run(init_op)
	print "y in this file is %n",sess.run(b,feed_dict={x:[[0.7,0.5]]})

