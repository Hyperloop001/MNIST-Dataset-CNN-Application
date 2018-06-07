### Linear algorithm implementation for mnist training and deploying ###
### Date: 2018-06-07

import tensorflow as tf
from mnistFunc import *
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

##########SWITCHES#########
train = False # True: training model
useHandWrittenDigit = True # True: use hand written digits
###########################

# Load mnist data
mnist = input_data.read_data_sets('./MNIST_data/trainData', one_hot=True)
print(mnist)

# Feed in original data and corresponding labels, and reshape to proper size
x = tf.placeholder(tf.float32,[None,784])
y_expect = tf.placeholder(tf.float32, [None, 10])

# Define weight and bias variables
W = weight_variable([784, 10], 'LINEAR')
b = bias_variable([10], 'LINEAR')

# Output result
y_predict = tf.nn.softmax(tf.matmul(x, W) + b)


# Training Steps
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_expect * tf.log(y_predict), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) ###train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

# Accuracy Calculations
correct_prediction = tf.equal(tf.argmax(y_expect, 1), tf.argmax(y_predict, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))    

# Define saver
saver = tf.train.Saver()

#---------------------------Run Session---------------------------#
# Initialize session as sess
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Training model
if train:
    # Training procedure
    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        
        # Show status and save for every 100 training
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_expect: batch_ys})
            print ("Step %d: Training accuracy: %g" % (i, train_accuracy))
            savepath = saver.save(sess, './MNIST_model/basic/version1/basic_model_v1')
            
        # Training step occur here          
        sess.run(train_step, feed_dict={x: batch_xs, y_expect: batch_ys}) 

    # Test model accuracy after training
    test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_expect: mnist.test.labels})
    print ("Training finished: Test accuracy: %g" % (test_accuracy))       

# Deploying model
else:
    # Load model
    savepath = saver.restore(sess, './MNIST_model/basic/version1/basic_model_v1')   
    
    # Check hand written digits
    if useHandWrittenDigit:
        for i in range(10):
            imgloadPath = './MNIST_data/applicationData/' + (str)(i) + '.jpeg'
            imgData = normalizedReadImg(imgloadPath, invert=1)
            imgPrecision = y_predict.eval(feed_dict={x: imgData.eval().reshape([1,28*28])})
            # Print recognition result
            result = np.argmax(imgPrecision)
            print(result)    
                 #print(imgData.eval())
            # Generate probability plot
            plt.subplot(4,5,i+1)
            pltShowImg(imgData.eval(), to_color = 0)
            plt.title(str(result)+'('+str(imgPrecision[0,result] * 100)+'%)')
            plt.subplot(4,5,i+11)
            pltShowImg(W.eval()[:,i].reshape(28,28))
        plt.show()
