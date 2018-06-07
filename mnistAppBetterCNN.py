### Convolutional Neuron Network algorithm implementation for mnist training and deploying ###
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

# Input Layer
# Feed in original data and corresponding labels, and reshape to proper size
x = tf.placeholder(tf.float32,[None,784]) # Shape: imageNum, 784
y_expect = tf.placeholder(tf.float32, [None, 10]) # Shape: imageNum, 10
x_image = tf.reshape(x, [-1, 28, 28, 1]) # Shape: imageNum, 28, 28, 1 channel

# Convilution Layer 1
W_filter1 = weight_variable([5, 5, 1, 32], 'CNN') # Shape: 5 x 5 kernal, 1 channel, 32 filters
b_filter1 = bias_variable([32], 'CNN')
h_conv1 = tf.nn.relu(conv2d(x_image, W_filter1) + b_filter1) # Shape: imageNum, 28, 28, 32
h_pool1 = max_poop_2x2(h_conv1) # Shape: imageNum, 14, 14, 32

# Convilution Layer 2
W_filter2 = weight_variable([5, 5, 32, 64], 'CNN') # Shape: 5 x 5 kernal, 32 channel, 64 filters 
b_filter2 = bias_variable([64], 'CNN')
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_filter2) + b_filter2) # Shape: imageNum, 14, 14, 64
h_pool2 = max_poop_2x2(h_conv2) # Shape: imageNum, 7, 7, 64

# Dence Layer
# Full connection between neurons
W_fc1 = weight_variable([7 * 7 * 64, 1024], 'CNN') # Shape: 7 x 7 x 64, 1024
b_fc1 = bias_variable([1024], 'CNN')
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64]) # Shape: imageNum, 7 x 7 x 64
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # Shape: imageNum, 1024

# Dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # Shape: imageNum, 1024

# Output Layer
W_fc2 = weight_variable([1024, 10], 'CNN') # Shape: 1024, 10
b_fc2 = bias_variable([10], 'CNN')
y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) # Shape: imageNum, 10

# Training Steps
cross_entropy = -tf.reduce_sum(y_expect * tf.log(y_predict))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

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
    for i in range(20000):
        batch_xs, batch_ys = mnist.train.next_batch(50) 
        
        # Show status and save for every 100 training
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_expect: batch_ys, keep_prob: 1.0})
            print ("Step %d: Training accuracy: %g" % (i, train_accuracy))
            savepath = saver.save(sess, './MNIST_model/better/version2/better_model_v2')
            
        # Training step occur here    
        sess.run(train_step, feed_dict={x: batch_xs, y_expect: batch_ys, keep_prob: 0.5})       

    # Test model accuracy after training
    test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_expect: mnist.test.labels, keep_prob: 1.0})
    print ("Training finished: Test accuracy: %g" % (test_accuracy))       

# Deploying model
else:
    # Load model
    savepath = saver.restore(sess, './MNIST_model/better/version1/better_model_v1')   
    
    # Check hand written digits
    if useHandWrittenDigit:
        for i in range(10):
            imgloadPath = './MNIST_data/applicationData/' + (str)(i) + '.jpeg'
            imgData = normalizedReadImg(imgloadPath, invert=1)
            imgPrecision = y_predict.eval(feed_dict={x: imgData.eval().reshape([1,28*28]), keep_prob: 1.0})
            # Print recognition result
            result = np.argmax(imgPrecision)
            print(result)    
                 #print(imgData.eval())
            # Generate probability plot
            plt.subplot(4,5,i+1)
            pltShowImg(imgData.eval(), to_color = 0)
            plt.title(str(result)+'('+str(imgPrecision[0,result] * 100)+'%)')
            plt.subplot(4,5,i+11)
            pltShowImg(W_fc2.eval()[:,i].reshape(32,32))
        plt.show()
 
    
          