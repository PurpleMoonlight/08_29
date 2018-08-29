#-*- coding:utf-8 _*-  
'''
Created on 2018/08/22 

Author Zhengyuanhe

'''
import tensorflow as tf
a=tf.Variable([1.0,1.3,2.1,3.41,4.51])
b=tf.cast(a>3,dtype=tf.bool)
c=tf.cast(a>3,dtype=tf.int8)
e=tf.cast(a<2,dtype=tf.float32)
d=tf.cast(a,dtype=tf.int8)
session=tf.Session()
update=tf.global_variables_initializer()
session.run(update)
print('.............a....................')
print(session.run(a))
print('............b..................')
print(session.run(b))
print('..............c...............')
print(session.run(c))
print('............e..................')
print(session.run(e))
print('..........d......................')
print(session.run(d))

