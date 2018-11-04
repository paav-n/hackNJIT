# Acquire the example dataset

from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

tf.__version__

# Create Dataset from a path (folders)
DATASET_PATH = '/dataset/'

# Image Parameters
N_CLASSES = 5
IMG_HEIGHT = 64
IMG_WIDTH = 64
CHANNELS = 3

#Dataset reading
def read_images(dataset_path, batch_size):
    imagepaths, labels = list(), list()
    
    # An ID will be added to folders numerically
    label = 0
    # Listing directory
    try:
        classes = sorted(os.walk(dataset_path).next()[1])
    except Exception:
        classes = sorted(os.walk(dataset_path).__next__()[1])
    # List each sub-directory
    for c in classes:
        c_dir = os.path.join(dataset_path, c)
        try:
            walk = os.walk(c_dir).next()
        except Exception:
            walk = os.walk(c_dir).__next__()
        # Adding image
        if sample.endswith('.jpg') or sample.endswith('.jpeg'):
            imagepaths.append(os.path.join(c_dir, sample))
            labels.append(label)
        label += 1
    
    # Converting this set to Tensor
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    # Build a TF Queue and Shuffle the data
    image, label = tf.train.slice_input_producer([imagepaths, labels], shuffle=True)

    # Read images from the disk
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=CHANNELS)

    # Resize images to a common size
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])

    # Normalize
    image = image * 1.0/127.5 - 1.0

    # Create batches
    X, Y = tf.train.batch([image, label], batch_size=batch_size, capacity=batch_size * 8, num_threads = 4)

    return X, Y

# Parameters
learning_rate = 0.01
num_steps = 10000
batch_size = 128
display_step = 100
dropout = .75      # Dropout, probability to keep units

# Build Data input
X, Y = read_images(/examples/, 1200) # Batch size 400

# Create Model
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out
    
    return out

# Create a graph for training
logits_train = conv_net(X, N_CLASSES, dropout, reuse=False, is_training=True)
# Create another graph for testing that reuse the same weights
logits_test = conv_net(X, N_CLASSES, dropout, reuse=True, is_training=False)

# Define loss and optimizer (with train logits, for dropout to take effect)
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits_train, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.cast(Y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Saver object
saver = tf.train.Saver()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Start the data queue
    tf.train.start_queue_runners()

    # Training cycle
    for step in range(1, num_steps+1):

        if step % display_step == 0:
            # Run optimization and calculate batch loss and accuracy
            _, loss, acc = sess.run([train_op, loss_op, accuracy])
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
        else:
            # Only run the optimization op (backprop)
            sess.run(train_op)

    print("Optimization Finished!")

    # Save your model
    saver.save(sess, 'my_tf_model')

"""
# Make the labels
#  [:num] is the first 'num' examples
train_labels = train_labels[:5].reshape(-1,32*32)/255.0

# Returns a sequential model
def create_model():
    # Build Model
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # Compile Model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy'])
    
    return model

# Create a basic model instance
model = create_model()
# model.summary()

del = create_model()

del.fit(train_images, trainlabels, epochs=10, validation_data = (test_images, test_labels), callbacks = [cp_callback])
 # Passes the callback to training

model.save('candy.h5')
"""
