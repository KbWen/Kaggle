import numpy as np
import pandas as pd
import tensorflow as tf
# from sklearn import linear_model


data = pd.read_csv('train.csv')
pm25 = data[data['item']=='PM2.5'].drop(['item','data'],axis=1)
data = data.drop(['data','item'],axis=1).replace('NR',0)

for j in range(0,4320,18):
    for i in range(15):
        train_data0 = np.array(data.iloc[(0+j):(18+j),(0+i):(9+i)],dtype='float32').reshape(-1)[np.newaxis,:]
        if i == 0 and j == 0:
            train_data = train_data0
            continue
        train_data = np.concatenate((train_data,train_data0),axis=0)


train_target = np.array(pm25.iloc[:,9:],dtype='float32').reshape(-1)[:,np.newaxis]
print(train_data.shape)
print(train_target.shape)

# tensorflow
# x = tf.placeholder(tf.float32,[None,162])
# y_ = tf.placeholder(tf.float32,[None,1])
# keep_prob = tf.placeholder(tf.float32)
#
# def add_layer(inputs, isize, osize):
#     W = tf.Variable(tf.random_normal(shape=[isize,osize]))
#     b = tf.Variable(tf.zeros(shape=[osize])+0.1)
#     Y = tf.matmul(inputs,W) + b
#     return Y
# hidden_layer1 = tf.nn.relu(add_layer(x,162,512))
# hidden_layer2 = tf.nn.relu(add_layer(hidden_layer1,512,128))
# hidden_drop = tf.nn.dropout(hidden_layer2, keep_prob)
# y = add_layer(hidden_drop,128,1)
#
# loss = tf.reduce_mean(tf.square(y_-y))
# train_step = tf.train.AdamOptimizer(5e-3).minimize(loss)
#
#
# sess = tf.InteractiveSession()
# tf.global_variables_initializer().run()
#
# for i in range(20000):
#     feed={x:train_data, y_:train_target,keep_prob:0.5}
#     sess.run(train_step, feed_dict=feed)
#     if i % 1000 == 0 or i == 20000-1:
#         print(sess.run(loss,feed_dict=feed))
#
# test_data0 = pd.read_csv("test_X.csv")
# test_data = test_data0.iloc[:,2:].replace('NR',0)
#
# for i in range(0,4320,18):
#     test_data0 = np.array(test_data.iloc[(0+i):(18+i),:],dtype='float32').reshape(-1)[np.newaxis,:]
#     if i == 0:
#         testing_data = test_data0
#         continue
#     testing_data = np.concatenate((testing_data,test_data0),axis=0)
#
# predictions = sess.run(y,feed_dict={x:testing_data,keep_prob:1.0})
# id_num= []
# for i in range(len(predictions)):
#     id_num.append('id_'+ str(i))
# idx_array = np.array(id_num, dtype=np.str) 
#
# submission = pd.DataFrame({'id' : idx_array, 'value': predictions.flatten()})
# submission.to_csv('test_result.csv',index=False)
# # sklearn
reg = linear_model.LinearRegression()
reg.fit(train_data,train_target)

print(reg.coef_)
pre = reg.predict(testing_data)
np.savetxt('pre.csv', pre, delimiter = ',')
