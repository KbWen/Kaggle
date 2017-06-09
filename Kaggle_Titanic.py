import numpy as np
import pandas as pd
import math
import tensorflow as tf

# import data
train_data = pd.read_csv('train.csv')
print(train_data.columns)
print(train_data.head(1))

# print(train_data.columns)
# print(train_data.head())
# print(train_data.describe())
# print(train_data.index)

def preprocessing(train_data):
    train_data_ar = [train_data]
    for dataset in train_data_ar:
        dataset['Age'] = (dataset['Age'].fillna(train_data['Age'].mean()))/max(train_data['Age'])
        dataset['Fare'] = dataset['Fare']/max(train_data['Fare'])
        dataset['Parch'] = dataset['Parch']/max(train_data['Parch'])
        dataset['SibSp'] = dataset['SibSp']/max(train_data['SibSp'])
        #  Cabin
        dataset['Cabin'] = dataset['Cabin'].fillna('U')
        dataset['Cabin'] = dataset.Cabin.str.extract('([A-Za-z])', expand=False)
        dataset['Cabin'] = dataset['Cabin'].map( {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E':0,
                                                'F':0, 'G':0, 'T':0, 'U':1} ).astype(float)
        # obtain title from name
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royalty":5, "Officer": 6}
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        # Covert 'Title' to numbers (Mr->1, Miss->2 ...)
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Dona'],'Royalty')
        dataset['Title'] = dataset['Title'].replace(['Mme'], 'Mrs')
        dataset['Title'] = dataset['Title'].replace(['Mlle','Ms'], 'Miss')
        dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Major','Rev'], 'Officer')
        dataset['Title'] = dataset['Title'].replace(['Jonkheer', 'Don','Sir'], 'Royalty')
        dataset.loc[(dataset.Sex == 'male')   & (dataset.Title == 'Dr'),'Title'] = 'Mr'
        dataset.loc[(dataset.Sex == 'female') & (dataset.Title == 'Dr'),'Title'] = 'Mrs'
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = (dataset['Title'].fillna(0))/len(title_mapping)
        # if age < 16, set 'Sex' to Child
        dataset.loc[(dataset.Age < 16),'Sex'] = 'Child'
        # Covert 'Sex' to numbers (female:1, male:2)
        dataset['Sex'] = (dataset['Sex'].map( {'female': 1, 'male': 0, 'Child': 2} ).astype(float))/2
        # Covert 'Embarked' to numbers
        dataset['Embarked'] = dataset['Embarked'].fillna(train_data.Embarked.dropna().mode()[0])
        dataset['Embarked'] = (dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(float))/2
        dataset['Pclass'] = (dataset['Pclass'])/max(train_data['Pclass'])
    train_data = train_data.drop(['Name', 'PassengerId','Ticket'], axis=1)
    return train_data
train_data = preprocessing(train_data=train_data)

training_data = np.array(train_data.iloc[:,1:], dtype='float32')
target_training_data = np.array(train_data.Survived, dtype='int8').reshape(-1, 1)
print(training_data[1])

train_data_shape = training_data.shape[1]
train_target_shape = target_training_data.shape[1]

x = tf.placeholder(tf.float32,[None,train_data_shape])
y_ = tf.placeholder(tf.float32,[None,train_target_shape])
keep_prob = tf.placeholder(tf.float32)

def add_layer(inputs, isize, osize):
    W = tf.Variable(tf.random_normal(shape=[isize,osize]))
    b = tf.Variable(tf.zeros(shape=[osize])+0.1)
    Y = tf.matmul(inputs,W) + b
    return Y

hidden_layer1 = tf.nn.relu(add_layer(x,train_data_shape,128))
hidden_layer2 = tf.nn.relu(add_layer(hidden_layer1, 128, 64))
hidden_drop = tf.nn.dropout(hidden_layer2, keep_prob)
hidden_layer3 = tf.nn.relu(add_layer(hidden_drop, 64, 32))
hidden_drop2 = tf.nn.dropout(hidden_layer3, keep_prob)
predictions = tf.nn.sigmoid(add_layer(hidden_drop2,32,train_target_shape))

cross_entropy = -tf.reduce_mean(y_*tf.log(tf.maximum(0.00001, predictions)) +
                   (1.0 - y_)*tf.log(tf.maximum(0.00001, 1.0-predictions)))

correct_prediction = tf.equal((predictions > 0.5), (y_ > 0.5))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))



train_step = tf.train.AdamOptimizer(3e-4).minimize(cross_entropy)
# train_step = tf.train.GradientDescentOptimizer(0.06).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(50000):
    feed={x:training_data, y_:target_training_data,keep_prob:0.5}
    sess.run(train_step, feed_dict=feed)
    if i % 1000 == 0 or i == 50000-1:
        print('{} {} {:.2f}%'.format(i, sess.run(cross_entropy, feed_dict=feed), sess.run(accuracy, feed_dict=feed)*100.0))


test_data = pd.read_csv('test.csv')
test_features = preprocessing(test_data)
test_features = np.array(test_features, dtype='float32')
print(test_features[0])
predicted = sess.run((predictions > 0.5), feed_dict={x:test_features,keep_prob:1.0})

# Write data

result = pd.DataFrame()
result['PassengerId'] = test_data['PassengerId']
result['Survived'] = pd.Series(predicted.reshape(-1)).map({True:1, False:0})
print(result)
result.to_csv('tinatic_submission.csv', index=False)
