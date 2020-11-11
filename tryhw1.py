from uwnet import *

def conv_net():
	l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1),
	        make_activation_layer(RELU),
	        make_maxpool_layer(32, 32, 8, 3, 2),
	        make_convolutional_layer(16, 16, 8, 16, 3, 1),
	        make_activation_layer(RELU),
	        make_maxpool_layer(16, 16, 16, 3, 2),
	        make_convolutional_layer(8, 8, 16, 32, 3, 1),
	        make_activation_layer(RELU),
	        make_maxpool_layer(8, 8, 32, 3, 2),
	        make_convolutional_layer(4, 4, 32, 64, 3, 1),
	        make_activation_layer(RELU),
	        make_maxpool_layer(4, 4, 64, 3, 2),
	        make_connected_layer(256, 10),
	        make_activation_layer(SOFTMAX)]
	return make_net(l)

def fully_connected_net():
	l = [	make_connected_layer(3072,300),
			make_activation_layer(RELU),
			make_connected_layer(300,300),
			make_activation_layer(RELU),
			make_connected_layer(300,250),
			make_activation_layer(RELU),
			make_connected_layer(250,84),
			make_activation_layer(RELU),
			make_connected_layer(84,10),
			make_activation_layer(SOFTMAX)]
	return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

"""
Number of Operations:
27*8*1024*128 =  28,311,552 operations +
72*256*16*128 =  37,748,736 operations +
144*64*32*128 =  37,748,736 operations +
288*16*64*128 =  37,748,736 operations +
256*10*128    =     327,680 operations +
Total         = 141,885,440 operations for batch size of 128

How accurate is the fully connected network vs the convnet when they use similar number of operations?
Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
Your answer:

The convnet had 69.15% training accuracy and 64.47% test accuracy whereas the
fully connected network had 55.1% training accuracy and 49.96% test accuracy.
The fact that the convnet had a better score than the fully connected network was
to be expected considering the format of the input. The input is of images which
are spatial in nature; pixels that are far apart in an image do not need to both
have an effect on some neuron in the second layer because they do not form any
meaningful features together. By only connecting neurons that are close together,
we can extract more meaningful features like lines or shapes from the image.

"""
