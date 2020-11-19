from uwnet import *
def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
			make_batchnorm_layer(8),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
			make_batchnorm_layer(16),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
			make_batchnorm_layer(32),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)


print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 500
momentum = .9
decay = .005

m = conv_net()
print("training...")
for i in range(10):
	rate = i/100
	train_image_classifier(m, train, batch, iters, rate, momentum, decay)
	print("done")

	print("evaluating model...")
	print("training accuracy: %f, %f", rate, accuracy_net(m, train))
	print("test accuracy:     %f, %f", rate, accuracy_net(m, test))

# 7.6 Question: What do you notice about training the convnet with/without batch normalization? How does it affect convergence? How does it affect what magnitude of learning rate you can use? Write down any observations from your experiments:
# TODO: Your answer
# Without 0.01: 40.3% test. 1.59 Loss
# With    0.01: 53.5% test. 1.28 Loss
# With    0.1:  51.5% test. 1.38 Loss
# With    0.06: 59.4% test. 1.25 loss
# We trained from all learning rates from 0.01 to 0.1 incrementing by 0.01 each time and found that the best result
# was a learning rate of 0.06. We found that 0.06 learning rate netted us the hightest
# test accuracy at 59.4% and a low loss at 1.25 which is significantly higher than the
# 51% and 53% accuracies from 0.1 and 0.01 learning rate from before. The test accuracies
# for the learning rates surrounding 0.06 were all quite similar at around 58.5% to 58.7%
