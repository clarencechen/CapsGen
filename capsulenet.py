"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python capsulenet.py
       python capsulenet.py --epochs 50
       python capsulenet.py --epochs 50 --routings 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import numpy as np
from keras import layers, models, optimizers, losses, initializers, metrics
from keras import backend as K
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from utils import combine_images
from PIL import Image
from math import sqrt, cos, pi
from capsulelayers import CapsuleLayer, PrimaryCap, InvPrimaryCap, Longest, MaskNoise

K.set_image_data_format('channels_first')


def CapsNet(input_shape, n_class, routings, testing):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape, name='x')
    # Conventional stack of Conv2D layers whose ouptut area is [-1, 16, 16]
    conv0 = layers.Conv2D(filters=96, kernel_size=9, strides=1, padding='valid', \
        activation='relu', kernel_initializer='glorot_uniform', name='conv0')(x)
    conv1 = layers.Conv2D(filters=96, kernel_size=9, strides=1, padding='valid', \
        activation='relu', kernel_initializer='glorot_uniform', name='conv1')(conv0)
    # Conv2D layer with `squash` activation whose ouptut shape is [None, channels*dim_capsule, 6, 6]
    # then permuting and reshaping to [None, channels*6*6, dim_capsule]
    primarycaps = PrimaryCap(channels=16, dim_capsule=12, kernel_size=6, strides=2, padding='valid', \
        initializer='glorot_uniform')(conv1)

    # Classification layer. Routing algorithm works here.
    classcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='classcaps')(primarycaps)

    # This is an auxiliary layer to mask (and noise) each capsule with the correct category for training.
    class_length = layers.Lambda(lambda inputs: K.sqrt(K.sum(K.square(inputs), axis=-1)), name='class_length')(classcaps)

    # These are auxillary inputs to manipulate the primary capsule layer for the decoder
    noise = layers.Input(shape=(16*6*6, 12), name='noise')
    noised_prim = layers.Add(name='add_noise')([noise, primarycaps])

    # Inverted primary layer. Reshapes to [None, dim_capsule*channels, input_size, input_size]
    # and then applies conventional DeConv2D.
    invprim = InvPrimaryCap(channels=16, input_size=6, kernel_size=6, strides=2, padding='valid', \
        activation='relu', initializer='glorot_uniform')(noised_prim)
    # Conventional stack of Conv2DTranspose layers whose input area is [-1, 16, 16]
    deconv1 = layers.Conv2DTranspose(filters=96, kernel_size=9, strides=1, padding='valid', \
        activation='relu', kernel_initializer='glorot_uniform', name='deconv1')(invprim)
    deconv0 = layers.Conv2DTranspose(filters=64, kernel_size=9, strides=1, padding='valid', \
        activation='relu', kernel_initializer='glorot_uniform', name='deconv0')(deconv1)

    # Separable 1x1 Conv2D layer that outputs final reconstructed image
    final_image = layers.SeparableConv2D(filters=3, kernel_size=1, strides=1, padding='valid', \
        use_bias=False, activation='sigmoid', kernel_initializer='glorot_uniform', name='final_image')(deconv0)

    # Create model to return
    model = models.Model(inputs=[x, noise], outputs=[class_length, final_image], name='model')
    return model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true*K.square(K.maximum(0., 0.9 -y_pred)) +0.5*(1 -y_true)*K.square(K.maximum(0., y_pred -0.1))
    return K.mean(K.sum(L, 1))

def mse_img(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=None)

def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
     # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs', batch_size=args.batch_size, histogram_freq=args.debug,
        write_graph=True, write_grads=True, write_images=True)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/w_{epoch:02d}.h5', monitor='val_class_length_categorical_accuracy',
                                           save_best_only=True, save_weights_only=True, verbose=1, period=args.debug)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * args.decay**epoch)

    # compile the models
    model.compile(optimizer=optimizers.RMSprop(lr=args.lr), loss=[margin_loss, mse_img],
        loss_weights=[1, args.lam_recon], metrics={'class_length': 'categorical_accuracy'})
    model.fit([x_train, np.zeros((len(x_train), 16*6*6, 12))], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, np.zeros((len(x_test), 16*6*6, 12))], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])

    model.save_weights(args.save_dir + '/model.h5')
    print('Trained model saved to \'%s' % args.save_dir)

    return

def test(model, data, args):
    print('-'*30 + 'Begin Testing' + '-'*30)

    x_test, y_test = data
    y_pred, x_recon = model.predict([x_test, np.zeros((1000, 16*6*6, 12))], batch_size=100)
    
    print('Classifier accuracy:', metrics.categorical_accuracy(y_test, y_pred))
    
    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img*255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End Testing' + '-' * 30)

def manipulate_latent(model, data, args):
    print('-'*30 + 'Begin Manipulation' + '-'*30)

    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.category
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 16*6*6, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:,:,dim] = r
            x_recon = model.predict([x, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img*255
    Image.fromarray(image.astype(np.uint8), mode='RGB').save(args.save_dir + '/manipulate-%d.png' % args.category)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.category))
    print('-' * 30 + 'End: manipulate' + '-' * 30)


def load_cifar10():
    # the data, shuffled and split between train and test sets
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on CIFAR10.")
    parser.add_argument('-e', '--epochs', default=100, type=int)
    parser.add_argument('-b', '--batch_size', default=200, type=int)
    parser.add_argument('-i', '--lr', default=1e-4, type=float,
                        help="Initial learning rate of the classifier")
    parser.add_argument('-d', '--decay', default=0.9, type=float,
                        help="The value multiplied by lr at the beginning of each epoch")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('-l', '--lam_recon', default=1536, type=float,
                        help="Loss weight for decoder")
    parser.add_argument('--debug', default=5,
                        help="Save weights and log details in TensorBoard after this many epochs have elapsed")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-c', '--category', default=8, type=int,
                        help="Image Category to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved model weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    # define model
    model = CapsNet(input_shape=x_train.shape[1:],
            n_class=len(np.unique(np.argmax(y_train, 1))),
            routings=args.routings, testing=args.testing)

    print('-'*30 + 'Summary for Model' + '-'*30)
    model.summary()
    print('-'*30 + 'Summaries Done' + '-'*30)
    # init the model weights with provided one
    if args.weights is not None:
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('Weights are not provided. Will test using random initialized weights.')
        manipulate_latent(model=model, data=(x_test, y_test), args=args)
        test(model=model, data=(x_test, y_test), args=args)
