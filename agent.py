# -*- coding:utf-8 -*-
import warnings

warnings.filterwarnings("ignore")
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from config import Model
import tensorflow as tf
import numpy as np
from tensorflow import keras


class Agent:
    def __init__(self, ds_img, ds_label, path, index, **kwargs):
        self.model_path = os.path.join(path.agent_model,
                                       "agent{}".format(index))
        self.file_path = os.path.join(path.file_folder, "agent{}".format(index))
        self.plot_path = os.path.join(path.figure_folder,
                                      "agent{}".format(index))
        self.server_path = path.server_model
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.file_path, exist_ok=True)
        os.makedirs(self.plot_path, exist_ok=True)

        self.index = index
        self._async = kwargs["async"]
        self.sp = kwargs["sp"]
        self.ds_type = kwargs["ds_type"]
        self.batch = kwargs["batch"]
        self.mr = kwargs["mr"]
        self.epoch = kwargs["epoch"]

        self.train_img = ds_img
        self.train_label = ds_label
        print("[agent {}] train samples {}".format(self.index, len(self.train_label)))
        self.learning_rate = kwargs["lr"]
        self.optimizer = keras.optimizers.SGD(lr=self.learning_rate)
        self.loss_fn = tf.losses.SparseCategoricalCrossentropy()
        self.metric = ["accuracy"]
        self.hist = dict()
        self.hist["acc"] = []
        self.hist["loss"] = []
        self.model = Model().load_model()
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn,
                           metrics=self.metric)
        np.save(os.path.join(self.model_path, "theta.npy"),
                np.array(self.model.get_weights()))
        print("[agent {}] save init weights".format(self.index))

    def process(self, epoch):
        theta = self.param_update(epoch)
        # if epoch % 10 == 0:
        #     self.learning_rate *= 0.9
        #     self.optimizer = keras.optimizers.SGD(lr=self.learning_rate)
        #     self.model.compile(optimizer=self.optimizer, loss=self.loss_fn,
        #     metrics=[self.metric])
        if theta is not None:
            self.model.set_weights(theta)

        print("Training agent {}, epoch {}".format(self.index, epoch))
        self.train(epoch)
        self.compute_gradients(theta, epoch)
        return True

    def compute_gradients(self, theta, epoch):
        update = self.model.get_weights()
        np.save(os.path.join(self.model_path, "theta.npy"), update)
        np.save(os.path.join(self.model_path, "theta_{}.npy".format(epoch)), update)

        if theta is not None:
            delta = np.array(update) - np.array(theta)
        else:
            delta = np.array(update)
        np.save(os.path.join(self.model_path, "delta.npy"), delta)

    def train(self, epoch):
        self.benign_train(epoch)

    def benign_train(self, epoch):
        # print("[agent {}] train label {}".format(self.index, np.unique(self.train_label)))
        buffer_size = len(self.train_label)
        trainset = tf.data.Dataset.from_tensor_slices(
            (self.train_img, self.train_label)).shuffle(
            buffer_size).batch(self.batch)

        steps = buffer_size // self.batch
        hist = self.model.fit(trainset, epochs=1, steps_per_epoch=steps,
                              verbose=2)

        with open(os.path.join(self.file_path, "train.log"), "a+") as f:
            f.write("{},{}\n".format(hist.history["accuracy"][0],
                                     hist.history["loss"][0]))

    def param_update(self, epoch):
        global_weights = None
        theta_pre = None
        try:
            global_weights = \
                np.load(os.path.join(self.server_path,
                                     "global_weights.npy"),
                        allow_pickle=True)
        except FileNotFoundError:
            pass
        try:
            theta_pre = np.load(os.path.join(self.model_path,
                                             "theta.npy"),
                                allow_pickle=True)
        except FileNotFoundError:
            pass

        if self._async == 1:
            if (epoch - 1) % self.sp == 0:
                print("Load global weights for agent ", self.index)
                theta = theta_pre - self.mr * (theta_pre - global_weights)
                return theta
            else:
                print("Load pre weights for agent ", self.index)
                return theta_pre
        else:
            print("Load global weights for agent ", self.index)
            return global_weights
