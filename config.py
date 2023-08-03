# -*- coding:utf-8 -*-
import warnings

from keras.layers import MaxPooling2D
from keras.layers import  Conv2D
from keras.layers import Flatten
from keras.layers import  Dense

warnings.filterwarnings("ignore")
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import datetime
import numpy as np
from tensorflow import keras
import pathlib
import re
import sys
from pathlib import Path
import os

os.chdir(sys.path[0])

args = {
    "gpu": "0",
    "agent_num": 10,
    "attack_type": "",  # "", "GAN", "poison", "random"
    "ratio": 5,  # GAN: fake/normal, random: scale
    "attack_level": 1,  # 0(benign train with attack), 1(attack only)
    "target": 3,
    "mal_idx": 0,
    "iid": 0,  # 0-iid, 1-complete non-overlap, 2-partial overlap
    "ds_type": "mnist",
    "batch": 64,
    "lr": 0.001,  # learning rate
    "async": 0,  # async strategy: 0-sync, 1-secFL, 2-other
    "mr": 0.9,  # moving rate
    "frac": 1.0,
    "epoch": 1,
    "sp": 4,  # syncing period
    "beta": 0.5,
    "secFL": False,
    "krum": False,
    "threhold": 0.1,
    # "cosine_element", "cosine_layer", "pearson_layer"
    "dis_type": [],
}


class Folder:
    def __init__(self):
        global args
        cur_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._dir_name = Path(".").resolve().parent
        folder = "agent{}_lr{}_batch{}_{}_".format(args["agent_num"],
                                                   args["lr"],
                                                   args["batch"],
                                                   args["ds_type"])
        if args["attack_type"] == "random":
            folder += "attack_random_{}".format(args["ratio"])
        elif args["attack_type"] in ["GAN", "poison"]:
            folder += "attack_{}_{}_target{}_".format(args["attack_type"],
                                                      args["attack_level"],
                                                      args["target"])
            if args["attack_level"] == 0 and args["attack_type"] == "GAN":
                folder += "ratio{}_".format(args["ratio"])

        if args["iid"] == 1:
            folder += "noniid1_"
        elif args["iid"] == 2:
            folder += "noniid2_"

        if args["async"]:
            folder += "async{}_mr{}_sp{}_".format(args["async"],
                                                  args["mr"], args["sp"])
        if args["frac"] < 1:
            folder += "frac{}_".format(args["frac"])

        if args["dis_type"]:
            folder += "_".join(args["dis_type"])
            folder += "_thr_{}".format(args["threhold"])
            if args["secFL"]:
                folder += "_secFL_beta{}".format(args["beta"])
        if args["krum"]:
            folder += "_krum"

        self._server_model = self._dir_name / folder / str(
            cur_time) / "server_model"
        self._agent_model = self._dir_name / folder / str(
            cur_time) / "agent_model"
        self._file_folder = self._dir_name / folder / str(cur_time) / "file"
        self._figure_folder = self._dir_name / folder / str(cur_time) / "figure"

    @property
    def server_model(self):
        return self._server_model

    @property
    def agent_model(self):
        return self._agent_model

    @property
    def file_folder(self):
        return self._file_folder

    @property
    def figure_folder(self):
        return self._figure_folder


class Dataset:
    def __init__(self):
        global args
        self.agent_num = args["agent_num"]
        self.attack_type = args["attack_type"]
        self.attack_level = args["attack_level"]
        self.target = args["target"]
        self.mal_idx = args["mal_idx"]
        self.ds_type = args["ds_type"]
        self._iid = args["iid"]
        self._train_data, self._test_data = self.load_dataset()

    @property
    def train_data(self):
        return self._train_data

    @property
    def test_data(self):
        return self._test_data

    @staticmethod
    def read_pgm(pgmf, byteorder=">"):
        buffer = pgmf.read()
        try:
            header, width, height, maxval = re.search(
                b"(^P5\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
        except AttributeError:
            raise ValueError("Not a raw PGM file")
        return np.frombuffer(buffer,
                             dtype="u1" if int(
                                 maxval) < 256 else byteorder + "u2",
                             count=int(width) * int(height),
                             offset=len(header)
                             ).reshape((int(height), int(width)))

    def att_hepler(self, img_path):
        os.chdir(img_path)
        base_dir = os.path.abspath(os.path.dirname("__file__"))
        width = 92
        height = 112
        imgs = np.zeros(400, dtype=[("image", np.uint8, (height, width)),
                                    ("label", int, 1)])

        idx = 0
        for img in pathlib.Path(base_dir).glob("**/*.pgm"):
            data = self.read_pgm(open(str(img), mode="rb"))
            label = int(img.as_posix().split("/")[-2][1:])
            imgs["image"][idx] = data
            imgs["label"][idx] = label
            idx += 1
        os.chdir(sys.path[0])
        np.save("att.npy", imgs)

    def load_dataset(self):
        train_img, train_label, test_img, test_label = None, None, None, None
        if "mnist" == str(self.ds_type).lower():
            (train_img, train_label), (
                test_img, test_label) = keras.datasets.mnist.load_data()

        elif "fmnist" == str(self.ds_type).lower():
            (train_img, train_label), (
                test_img, test_label) = keras.datasets.fashion_mnist.load_data()

        train_img = tf.image.resize(train_img, [224, 224]).numpy()
        test_img = tf.image.resize(test_img, [224, 224]).numpy()
        train_img = np.expand_dims(train_img, axis=3)
        test_img = np.expand_dims(test_img, axis=3)
        train_img = np.array((train_img.astype(np.float32) - 127.5) / 127.5)
        test_img = np.array((test_img.astype(np.float32) - 127.5) / 127.5)

        # if "att" in self.ds_type:
        #     att = np.load("att.npy")
        #     img = att["image"]
        #     train_label = att["label"]
        #     train_img = resize(img, [img.shape[0], 64, 64],
        #     anti_aliasing=True)
        #     train_img = np.expand_dims(train_img, axis=3)

        if train_label is not None and test_label is not None:
            train_label = train_label.reshape(-1, 1)
            test_label = test_label.reshape(-1, 1)
        else:
            return None, None, None, None

        img_list = []
        label_list = []
        idx = np.arange(train_img.shape[0])
        np.random.shuffle(idx)
        if self._iid != 0:
            idx_digit = [[] for _ in range(10)]
            for digit in range(10):
                for i in idx:
                    if train_label[i] == digit:
                        idx_digit[digit].append(i)
            # img_list = [[] for _ in range(self.agent_num)]
            # label_list = [[] for _ in range(self.agent_num)]
            if self._iid == 1:
                for i in range(self.agent_num):
                    img_list.append(
                        np.concatenate([train_img[idx_digit[2 * i]], train_img[idx_digit[2 * i + 1]]], axis=0))
                    label_list.append(
                        np.concatenate([train_label[idx_digit[2 * i]], train_label[idx_digit[2 * i + 1]]], axis=0))
            elif self._iid == 2:
                for i in range(self.agent_num):
                    img_list.append(
                        np.concatenate([train_img[idx_digit[i + 1]], train_img[idx_digit[0]]], axis=0))
                    label_list.append(
                        np.concatenate([train_label[idx_digit[i + 1]], train_label[idx_digit[0]]], axis=0))
        else:
            if self.attack_type and self.attack_level == 1:
                idx_shard = np.array_split(idx, self.agent_num - 1)
                idx_shard.insert(self.mal_idx, [])
            elif self.attack_type == "GAN":
                idx_shard = np.array_split(idx, self.agent_num)
                mal_rm_idx = []
                for i in range(len(idx_shard[self.mal_idx])):
                    if train_label[idx_shard[self.mal_idx][i]] == self.target:
                        mal_rm_idx.append(i)
                idx_shard[0] = np.delete(idx_shard[0], mal_rm_idx)
            else:
                idx_shard = np.array_split(idx, self.agent_num)

            for i in range(self.agent_num):
                img_list.append(train_img[idx_shard[i]])
                label_list.append(train_label[idx_shard[i]])

            if self.attack_type == "poison":
                labels = np.setdiff1d(test_label, [self.target])
                poison_label = np.random.choice(labels)
                poison_img = test_img[0]
                for i in range(test_img.shape[0]):
                    if test_label[i] == self.target:
                        poison_img = test_img[i]
                if self.attack_level == 1:
                    img_list[0] = np.expand_dims(poison_img, 0)
                    label_list[0] = np.array(poison_label).reshape(-1, 1)
                else:
                    img_list[0] = np.insert(img_list[0], 0, poison_img, axis=0)
                    label_list[0] = np.insert(label_list[0], 0, poison_label,
                                              axis=0)
        return (img_list, label_list), (test_img, test_label)


class Model:
    def __init__(self):
        pass

    def load_model(self, attack=False):
        global args
        if args["attack_type"] == "GAN":
            if attack:
                return self.mnist_generator()
            else:
                return self.mnist_discriminator(class_num=11)
        elif args["attack_type"] == "poison":
            return self.mnist_cnn()
        elif args["ds_type"].lower() == "mnist":
            return self.mnist_discriminator()
        else:
            return self.vgg16()

    @staticmethod
    def vgg16(name="vgg16",class_num=10):
        _input = keras.Input((224, 224, 1))

        conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(_input)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(conv1)
        pool1 = MaxPooling2D((2, 2))(conv2)

        conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(pool1)
        conv4 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(conv3)
        pool2 = MaxPooling2D((2, 2))(conv4)

        conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(pool2)
        conv6 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(conv5)
        conv7 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(conv6)
        pool3 = MaxPooling2D((2, 2))(conv7)

        conv8 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(pool3)
        conv9 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv8)
        conv10 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv9)
        pool4 = MaxPooling2D((2, 2))(conv10)

        conv11 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(pool4)
        conv12 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv11)
        conv13 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv12)
        pool5 = MaxPooling2D((2, 2))(conv13)

        flat = Flatten()(pool5)
        dense1 = Dense(4096, activation="relu")(flat)
        dense2 = Dense(4096, activation="relu")(dense1)
        output = Dense(1000, activation="softmax")(dense2)

        vgg16_model = keras.Model(inputs=_input, outputs=output)
        return vgg16_model

    @staticmethod
    def mnist_cnn(name="cnn", class_num=10):
        model = keras.Sequential(name=name)
        model.add(keras.layers.Conv2D(64, (5, 5), padding="valid",
                                      input_shape=(28, 28, 1)))
        model.add(keras.layers.Activation("relu"))

        model.add(keras.layers.Conv2D(64, (5, 5)))
        model.add(keras.layers.Activation("relu"))

        model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128))
        model.add(keras.layers.Activation("relu"))

        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(class_num))
        model.add(keras.layers.Activation("softmax"))

        img = keras.Input(shape=(28, 28, 1))
        label = model(img)
        model.summary()
        return keras.Model(img, label)

    @staticmethod
    def distance_dnn(name="distance", input_shape=(3,)):
        init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        model = keras.Sequential(name=name)
        model.add(
            keras.layers.Dense(input_shape[0], activation="tanh",
                               input_shape=input_shape,
                               kernel_initializer=init))
        model.add(
            keras.layers.Dense(1, activation="tanh", kernel_initializer=init))
        return model

    @staticmethod
    def mnist_discriminator(name="discriminator", class_num=10):
        model = keras.Sequential(name=name)
        model.add(
            keras.layers.Conv2D(64, (5, 5), padding="same",
                                input_shape=(28, 28, 1),
                                activation="tanh"))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(
            keras.layers.Conv2D(128, (5, 5), activation="tanh"))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1024, activation="tanh"))
        model.add(keras.layers.Dense(class_num, activation="softmax"))

        img = keras.Input(shape=(28, 28, 1))
        label = model(img)
        model.summary()
        return keras.Model(img, label)

    @staticmethod
    def mnist_generator(name="generator", class_num=10, noise_dim=100):
        init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        model = keras.Sequential(name=name)
        model.add(
            keras.layers.Dense(input_dim=noise_dim,
                               units=1024,
                               activation="tanh",
                               kernel_initializer=init))
        model.add(keras.layers.Dense(128 * 7 * 7, kernel_initializer=init))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Activation("tanh"))
        model.add(
            keras.layers.Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))

        model.add(keras.layers.UpSampling2D(size=(2, 2)))
        model.add(
            keras.layers.Conv2D(64, (5, 5),
                                padding="same",
                                kernel_initializer=init))
        model.add(keras.layers.Activation("tanh"))

        model.add(keras.layers.UpSampling2D(size=(2, 2)))
        model.add(
            keras.layers.Conv2D(1, (5, 5),
                                padding="same",
                                activation="tanh",
                                kernel_initializer=init))
        model.add(keras.layers.Activation("tanh"))

        noise = keras.Input(shape=(noise_dim,))
        label = keras.Input(shape=(1,), dtype="int32")
        label_embedding = keras.layers.Flatten()(keras.layers.Embedding(
            class_num, noise_dim)(label))
        model_input = keras.layers.multiply([noise, label_embedding])
        img = model(model_input)

        model.summary()
        return keras.Model([noise, label], img)


if __name__ == "__main__":
    Dataset().load_dataset()
