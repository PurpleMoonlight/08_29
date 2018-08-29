#-*- coding:utf-8 _*-  
'''
Created on 2018/08/22 

Author Zhengyuanhe

'''
import tensorflow as tf
# a1=tf.get_variable(name='a',shape=[3,4],dtype=tf.float32,
#                 initializer=tf.random_uniform_initializer(minval=-10,maxval=10))
a1=tf.Variable(
    [[-7.81154871,4.04034424,-6.7139864,7.21160698],
    [-5.61189413,-5.3594923,-8.48080444,8.50246811],
    [-9.35132217,8.83945084,9.03738022,-8.31165791]],dtype=tf.float32)
a2=tf.Variable(
    [[2,4,6,7],
    [3,8,5,10],
    [1,2,4,2]],dtype=tf.float32)
a3=tf.Variable(
    [[-7.81154871,4.04034424,-6.7139864,7.21160698],
    [-5.61189413,-5.3594923,-8.48080444,8.50246811],
    [-9.35132217,8.83945084,9.03738022,-8.31165791]],dtype=tf.float32)
#dimension=0返回张量中的列索引
#dimension=1返回张量中的行索引
b=tf.argmax(input=a1,dimension=0)
c=tf.argmax(input=a1,dimension=1)
d=tf.argmax(input=a2,dimension=0)
e=tf.argmax(input=a2,dimension=1)
session=tf.Session()
update=tf.global_variables_initializer()
session.run(update)
# print('.............a1.shape.as_list()......................')
# print(a1.shape.as_list())
# print('...............a1...............................')
# print(session.run(a1))
# print(session.run(b))
# print(session.run(c))
# print(session.run(d))
# print(session.run(e))
correct_pred = tf.equal(tf.argmax(a1,1), tf.argmax(a3,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
print('......tf.equal(tf.argmax(a1,1), tf.argmax(a3,1))..................')
print(correct_pred)
print(session.run(correct_pred))
print('.............tf.cast(correct_pred, tf.float32).............')
print(tf.cast(correct_pred, tf.float32))
print(session.run(tf.cast(correct_pred, tf.float32)))
print('......tf.reduce_mean(tf.cast(correct_pred, tf.float32))...........')
print(accuracy)
print(session.run(accuracy))

'''

[[2,4,6,7],
[3,8,5,10],
[1,2,4,2]]
  
[1 1 0 1]
[3 3 2]

'''


'''

a1=tf.get_variable(name='a',shape=[3,4],dtype=tf.float32,
                initializer=tf.random_uniform_initializer(minval=-10,maxval=10))
print(session.run(a1))
print(session.run(b))
print(session.run(c))
结果
[[-7.81154871  4.04034424 -6.7139864   7.21160698]
[-5.61189413  -5.3594923 -8.48080444  8.50246811]
[-9.35132217  8.83945084 9.03738022 -8.31165791]]

[1 2 2 1]
[3 3 2]

'''



'''

a1=tf.get_variable(name='a',shape=[3,4],dtype=tf.float32,
initializer=tf.random_uniform_initializer(minval=-1,maxval=1))
全部位于-1到1之间随机生成的数
第一次生成
[[-0.64629865 -0.62985063 -0.84223628 -0.03107142]
 [ 0.18117023 -0.23141241  0.2572577   0.61768174]
 [ 0.36026883 -0.24973226 -0.80855989 -0.02996159]]
 第二次生成
 [[-0.6281445  -0.77764392 -0.74373055  0.64064384]
 [ 0.18991041  0.28260612  0.72941136  0.6017952 ]
 [-0.04939055 -0.07170749 -0.31147695 -0.19619346]]
 第三次生成
 [[-0.11486506 -0.36281228  0.09078979 -0.93681812]
 [ 0.41888571  0.99265337  0.10560274 -0.76232338]
 [ 0.3802743  -0.83609724 -0.15122199 -0.08644629]]
 
'''


'''

第一次生成
[[-6.49774313 -6.71185017  1.12407207 -5.38413525]
 [ 2.75819588  6.48773193  7.25658417  7.00149155]
 [ 2.46011543 -4.67055082 -0.61751842  3.71996403]]
第二次生成
 [[ 5.34835339  0.63426495  5.13254642 -3.69078398]
 [-4.32682514 -9.58306503 -0.33465385  0.97388744]
 [-0.17994881 -0.71148682  6.5662117  -3.54984283]]
第三次生成
 [[ 3.8856554   1.97188377  3.93816948 -3.34539413]
 [-2.27354288 -4.52176809 -2.47798443  8.46334839]
 [-1.39390182  0.78590584 -5.91628313 -0.89788246]]
 
'''







































































