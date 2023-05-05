# Brown CSCI 1430 assignment

# Data parameters (scene_rec)
img_size = 64
scene_class_count = 15
num_train_per_category = 100
num_test_per_category = 100

# Data parameters (MNIST)
mnist_class_count = 10

# Training parameters

# numEpochs is the number of epochs. If you experiment with more
# complex networks you might need to increase this. Likewise if you add
# regularization that slows training.
num_epochs = 10

# batch_size defines the number of training examples per batch:
# You don't need to modify this.
batch_size = 1

# learning_rate is a critical parameter that can dramatically affect
# whether training succeeds or fails. For most of the experiments in this
# homework the default learning rate is safe.
learning_rate = 0.5

# Momentum on the gradient (if you use a momentum-based optimizer)
momentum = 0.01
