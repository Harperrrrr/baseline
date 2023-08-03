# -*- coding:utf-8 -*-
import warnings

warnings.filterwarnings("ignore")
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import classification_report
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from functools import reduce
from config import Model, Dataset


class Server:
    def __init__(self, path, **kwargs):
        self.server_path = path.server_model
        self.plot_path = os.path.join(path.figure_folder, "server")
        os.makedirs(self.server_path, exist_ok=True)
        os.makedirs(self.plot_path, exist_ok=True)
        self.agent_path = path.agent_model
        self.agent_num = kwargs["agent_num"]
        self.train_num = int(self.agent_num * kwargs["frac"])
        self._secFL = kwargs["secFL"]
        self._async = kwargs["async"]
        self._krum = kwargs["krum"]
        self.alpha = 0.9
        if self._async:
            self._period = kwargs["sp"]
        else:
            self._period = 1

        # distance analysis
        self._shape = []
        self._offset = [0]
        self._distance = {x: [[] for _ in range(self.train_num)] for x in
                          kwargs[
                              "dis_type"]
                          }
        if self._distance:
            print("[server] compute distance: ", self._distance.keys())

        self.beta = kwargs["beta"]
        self._threhold = kwargs["threhold"]
        self.F = None

        self.dnn = Model().load_model()
        self.dnn.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
                         loss=["sparse_categorical_crossentropy"])
        self.dnn.summary()

        self.test_img, self.test_label = Dataset().test_data
        self.test_model = Model().load_model()
        self.loss_fn = ["sparse_categorical_crossentropy"]
        self.metric = ["accuracy"]
        self.optimizer = keras.optimizers.SGD()
        self.test_model.compile(loss=self.loss_fn, metrics=[self.metric],
                                optimizer=self.optimizer)
        self.global_weights = np.array(self.test_model.get_weights())
        np.save(os.path.join(self.server_path, "global_weights.npy"),
                self.global_weights)
        self.global_update = self.global_weights

    @staticmethod
    def exp(x):
        return np.exp(x + 1)

    def flatten(self, param):
        self._shape = []
        self._offset = [0]
        arr_list = []
        for layer in param:
            self._shape.append(layer.shape)
            arr_list.append(layer.reshape(-1, 1))
            self._offset.append(arr_list[-1].shape[0])
        arr = np.concatenate(arr_list).reshape(-1, )
        self._offset = np.cumsum(self._offset)
        return arr

    def reshape(self, arr):
        layer_list = []
        for i, s in enumerate(self._shape):
            layer_list.append(arr[self._offset[i]:
                                  self._offset[i + 1]].reshape(s))
        return layer_list

    # def pearson_cor(self, epoch, delta_mean, delta_list):
    #     # scaler = MinMaxScaler()
    #     print("[server] computing pearson_cor")
    #     C = np.zeros((self.agent_num, len(delta_mean)))
    #     for k in range(len(delta_mean)):
    #         if epoch <= self._period:
    #             layer_global = self.kth_layer(delta_mean, k).flatten()
    #         else:
    #             layer_global = self.kth_layer(self.global_update, k).flatten()
    #         for i in range(self.agent_num):
    #             layer_delta = self.kth_layer(delta_list[i], k).flatten()
    #             C[i, k] = pearsonr(layer_global, layer_delta)[0]
    #     # C = self.exp(C)
    #     # print(C)
    #     # C = tf.nn.softmax(C, axis=0).numpy()
    #     # C = scaler.fit_transform(C.transpose()).transpose()
    #     # print(C)
    #     return C

    # def cosine_sim(self, epoch, delta_mean, delta_list, dis_level="element"):
    #     # scaler = MinMaxScaler()
    #     print("[server] computing cosine_similarity")
    #     C_element = None
    #     C_layer = None
    #     if "element" in dis_level:
    #         # element-wise cosine sim
    #         # C = cos_sim([global_weights[j], delta_i[j]], [global_weights[
    #         # j], delta_mean[j]])
    #         w1 = self.flatten(delta_mean)
    #         if epoch > self._period:
    #             w0 = self.flatten(self.global_update)
    #         else:
    #             w0 = np.random.normal(loc=0.0, size=w1.shape)
    #         C_element = np.zeros([self.agent_num, len(w1)])
    #
    #         offset = 1000
    #         for i in range(self.agent_num):
    #             w2 = self.flatten(delta_list[i])
    #             v1 = np.concatenate([w0, w1], axis=1)
    #             v2 = np.concatenate([w0, w2], axis=1)
    #             start = 0
    #             print("......")
    #             while start < len(v1):
    #                 if start + offset < len(v1):
    #                     stop = start + offset
    #                 else:
    #                     stop = len(v1)
    #                 C_element[i, start:stop] = np.diag(
    #                     cosine_similarity(v1[start:stop], v2[start:stop]))
    #                 start = stop
    #         # C_element = tf.nn.softmax(C_element, axis=0).numpy()
    #         # C_element = scaler.fit_transform(C_element.transpose(
    #         # )).transpose()
    #         # C_element = self.exp(C_element)
    #
    #     if "layer" in dis_level:
    #         if epoch <= self._period:
    #             global_update = delta_mean
    #         else:
    #             global_update = self.global_update
    #         C_layer = np.zeros([self.agent_num, len(global_update)])
    #         for i in range(self.agent_num):
    #             for k in range(len(global_update)):
    #                 cur_layer = self.kth_layer(delta_list[i], k).flatten()
    #                 pre_layer = self.kth_layer(global_update, k).flatten()
    #                 C_layer[i, k] = \
    #                     cosine_similarity([cur_layer], [pre_layer])[0][0]
    #         # C_layer = self.exp(C_layer)
    #         # print(C_layer)
    #         # C_layer = tf.nn.softmax(C_layer, axis=0).numpy()
    #         # C_layer = scaler.fit_transform(C_layer.transpose()).transpose()
    #         # print(C_layer)
    #     return C_element, C_layer

    # def euclidean_dis(self, delta_mean, delta_list):
    #     print("[server] computing euclidean_dis")
    #     C = np.zeros([self.agent_num, len(delta_mean)])
    #     delta_mean = tf.convert_to_tensor(delta_mean)
    # 
    #     for i in range(self.agent_num):
    #         delta = tf.convert_to_tensor(self.flatten(delta_list[i]))
    #         C[i] = tf.norm([delta_mean, delta], axis=0).numpy().flatten()
    #     C = 1.0 - tf.nn.softmax(C, axis=0)
    #     return C
    # 
    # def gradient_dis(self, grad_mean, grad_list):
    #     print("[server] computing gradient_dis")
    #     grad_mean = self.flatten(grad_mean)
    #     C = np.zeros([self.agent_num, len(grad_mean)])
    # 
    #     for i in range(self.agent_num):
    #         grad = self.flatten(grad_list[i])
    #         C[i] = (grad_mean - grad).flatten()
    #     C = tf.nn.softmax(C, axis=0)
    #     return C

    @staticmethod
    def top_n_param(n, param, reverse=False):
        if n > len(param):
            return None
        if not reverse:
            idx = np.argpartition(param, -n, axis=0)[-n:]
        else:
            idx = np.argpartition(param, n, axis=0)[:n]
        # idx.sort()
        return idx

    def select_parameter(self, parameter, offset=(0,)):
        def helper(param):
            if len(param) < 20:
                select_idx = np.arange(start=0, stop=len(param))
                num = len(param)
                length = len(param)
            elif len(offset) == 1:
                length = len(param)
                num = int(self._threhold * length)
                select_idx = self.top_n_param(n=num, param=param)
            else:
                length = abs(offset[1] - offset[0])
                num = int(self._threhold * length)
                select_idx = self.top_n_param(n=num,
                                              param=param[offset[0]:offset[1]])

            with open(os.path.join(self.server_path, "select.log"), "a+") as f:
                f.write("s,{},{}\n".format(num, length))
            return select_idx

        for data in parameter:
            yield helper(data)

    def compute_distance(self, epoch, agent_train):
        delta_list = []
        try:
            for index in agent_train[epoch]:
                delta = np.load(os.path.join(self.agent_path,
                                             "agent{}".format(index),
                                             "delta.npy"),
                                allow_pickle=True)
                delta_list.append(self.flatten(delta))
        except FileNotFoundError:
            return False, []

        element_level_itr = self.select_parameter(delta_list)
        element_level_idx = reduce(np.union1d, element_level_itr)
        with open(os.path.join(self.server_path, "select.log"), "a+") as f:
            f.write("e_u,{},{},{:.3f}\n".format(len(element_level_idx),
                                                len(delta_list[0]),
                                                len(element_level_idx) / len(
                                                    delta_list[0])))

        layer_level_idx = []
        for i, j in zip(self._offset, self._offset[1:]):
            layer_level_itr = self.select_parameter(delta_list, offset=(i, j))
            layer_level_idx.append(reduce(np.union1d, layer_level_itr))
            with open(os.path.join(self.server_path, "select.log"), "a+") as f:
                f.write(
                    "l_u,{},{},{:.3f}\n".format(len(layer_level_idx[-1]), j - i,
                                                len(layer_level_idx[-1]) / (
                                                        j - i)))

        global_update = self.flatten(self.global_update)
        delta_mean = np.median(delta_list, axis=0).flatten()

        for dist in self._distance.keys():
            dis_type, dis_level = str(dist).split('_')
            print(dis_type, dis_level)

            if "cosine" == dis_type and "element" == dis_level:
                w0 = global_update[element_level_idx].reshape(-1, 1)
                w1 = delta_mean[element_level_idx].reshape(-1, 1)
                v0 = np.concatenate([w0, w1], axis=1)

                for (agent, delta) in enumerate(delta_list):
                    w2 = delta[element_level_idx].reshape(-1, 1)
                    v1 = np.concatenate([w0, w2], axis=1)
                    self._distance[dist][agent] = \
                        - tf.keras.losses.cosine_similarity(v0,
                                                            v1,
                                                            axis=1).numpy(

                        ).flatten()
            elif "l2" == dis_type and "element" == dis_level:
                for (index, delta) in enumerate(delta_list):
                    v0 = delta_mean[element_level_idx]
                    v1 = delta[element_level_idx]
                    self._distance[dist][index] = tf.norm([v0, v1],
                                                          axis=0).numpy(

                    ).flatten()
            elif "layer" == dis_level:
                for (agent, delta) in enumerate(delta_list):
                    for layer, i, j in zip(range(len(self._shape)),
                                           self._offset, self._offset[1:]):
                        v0 = delta_mean[i:j][layer_level_idx[layer]]
                        v1 = delta[i:j][layer_level_idx[layer]]
                        if "cosine" == dis_type:
                            value = - tf.keras.losses.cosine_similarity(v0,
                                                                        v1).numpy()
                        elif "l2" == dis_type:
                            value = tf.norm([v0, v1]).numpy()
                        elif "pearson" == dis_type:
                            try:
                                value = pearsonr(v0, v1)[0]
                            except ValueError:
                                value = -1
                        elif "JS" == dis_type:
                            value = jensenshannon(v0, v1, base=2)
                        elif "std" == dis_type:
                            value = np.std(v1, ddof=1)
                        else:
                            return False, element_level_idx
                        value = np.where(np.isnan(value), -1, value)
                        self._distance[dist][agent].append(value)
            else:
                return False, element_level_idx

            print(self._distance[dist])
            self._distance[dist] = tf.nn.softmax(self._distance[dist],
                                                 axis=0).numpy()
            print("[server] computed ", dist)

        return True, element_level_idx

        # self.compute_C(epoch=epoch)
        # if self.F is None:
        #     self.F = self.C
        # else:
        #     self.F = self.beta * self.F + (1.0 - self.beta) * self.C
        # self.F = tf.nn.softmax(self.C, axis=0).numpy()
        # np.save(os.path.join(self.server_path, "E_{}.npy".format(epoch)),
        #         self.F)
        # # if "euclidean" in dis_type:
        #     w0 = self.flatten(delta_mean)
        #     C = self.euclidean_dis(w0, delta_list)
        #     if self.D_euclidean is None:
        #         self.D_euclidean = tf.ones(C.shape, dtype=tf.float64)
        #         self.D_euclidean *= (1 / self.agent_num)
        #     self.D_euclidean = self.beta * C + (1.0 - self.beta) *
        #     self.D_euclidean
        #     np.save(os.path.join(self.server_path, "E_euclidean_{
        #     }.npy".format(epoch)), self.D_euclidean)
        # if "gradient" in dis_type:
        #     if epoch < 2:
        #         grad_list = delta_list
        #         grad_mean = delta_mean
        #     else:
        #         grad_list = []
        #         for index in range(self.agent_num):
        #             delta = np.load(os.path.join(self.agent_path,
        #                                          "agent{}".format(index),
        #                                          "delta_{}.npy".format(
        #                                          epoch - 1)),
        #                             allow_pickle=True)
        #             grad_list.append(delta_list[index] - delta)
        #         grad_mean = np.mean(delta_list, axis=0)
        #
        #     C = self.gradient_dis(grad_mean, grad_list)
        #     if self.E_gradient is None:
        #         self.E_gradient = tf.ones(C.shape, dtype=tf.float64)
        #         self.E_gradient *= (1 / self.agent_num)
        #     self.E_gradient = self.beta * C + (1.0 - self.beta) *
        #     self.E_gradient
        #     np.save(os.path.join(self.server_path, "E_gradient_{
        #     }.npy".format(epoch)), self.E_gradient)
        # else:
        #     return False
        # return True

    def kth_layer(self, param, k=0, flatten=False):
        if not flatten:
            layer = param[k]
            return layer.reshape(-1, 1)
        else:
            layer = param[self._offset[k]:self._offset[k + 1]]
            return layer

    # def compute_progress(self, epoch, dis_type):
    #     if epoch < self._period:
    #         return
    #     G = None
    #     for i in range(self.agent_num):
    #         if self.pre_state[i] is None:
    #             self.pre_state[i] = np.load(os.path.join(self.agent_path,
    #                                                      "agent{}".format(i),
    #                                                      "delta_{
    #                                                      }.npy".format(
    #                                                      epoch)),
    #                                         allow_pickle=True)
    #         cur_state = np.load(os.path.join(self.agent_path,
    #                                          "agent{}".format(i),
    #                                          "delta_{}.npy".format(epoch)),
    #                             allow_pickle=True)
    #         if G is None:
    #             G = np.ones((self.agent_num, len(cur_state)))
    #             G *= (1 / self.agent_num)
    #             # if self.G_euclidean is None:
    #         #     self.G_euclidean = np.zeros((self.agent_num,
    #         len(cur_state)))
    #         for k in range(len(cur_state)):
    #             cur_layer = self.kth_layer(cur_state, k).flatten()
    #             pre_layer = self.kth_layer(self.pre_state[i], k).flatten()
    #             G[i, k] = cosine_similarity([cur_layer], [pre_layer])[0][0]
    #         # self.G_euclidean[i, k] = tf.norm([cur_layer, pre_layer]).numpy()
    #         # self.G_euclidean[i] = (self.G_euclidean[i] - np.min(
    #         self.G_euclidean[i])) / np.ptp(self.G_euclidean[i])
    #
    #         if epoch % self._period:
    #             self.pre_state[i] = np.load(os.path.join(self.agent_path,
    #                                                      "agent{}".format(i),
    #                                                      "delta_{
    #                                                      }.npy".format(
    #                                                      epoch)),
    #                                         allow_pickle=True)
    #     return G

    def compute_F(self, epoch, element_idx):
        dist = list(self._distance.keys())
        # for d in dist:
        #     assert d in self._distance.keys()

        cluster = MeanShift(bandwidth=0.1)
        length = self._offset[-1]
        F_t = np.zeros([self.train_num, length])
        if self.F is None:
            self.F = np.ones([self.train_num, length]) / self.train_num

        values = np.zeros([len(dist), self.train_num, length])
        for i, d in enumerate(dist):
            cluster.fit(self._distance[d])
            centroid = cluster.cluster_centers_[0]
            for agent in range(self.train_num):
                score = 1 - abs(
                    (self._distance[d][agent] - centroid) / centroid)
                if "element" in d:
                    values[i][agent][element_idx] = score
                elif "layer" in d:
                    for layer in range(len(self._offset) - 1):
                        start = self._offset[layer]
                        stop = self._offset[layer + 1]
                        values[i][agent][start:stop] = score[layer]
        F_t = np.median(self.exp(values), axis=0)
        # for agent in range(self.agent_num):
        #     values = np.zeros([len(dist), length])
        #     for i in range(len(dist)):
        #         if "element" in dist[i]:
        #             values[i][element_idx] = self._distance[dist[i]][agent]
        #         elif "layer" in dist[i]:
        #             for layer in range(len(self._offset) - 1):
        #                 start = self._offset[layer]
        #                 stop = self._offset[layer + 1]
        #                 values[i][start:stop] = self._distance[dist[i]][agent][
        #                     layer]
        #     F_t[agent] = np.mean(self.exp(values), axis=0)

        # softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
        F_t = tf.nn.softmax(F_t, axis=0).numpy()
        self.F = (1 - self.beta) * F_t + self.beta * self.F
        # np.save(os.path.join(self.server_path, "F_{}.npy".format(epoch)),
        #         self.F)

    def plot_delta(self, data, epoch):
        label = ["adversary", "normal agent", "mean"]
        for l in range(len(data[0])):
            plt.clf()
            fig, ax = plt.subplots(len(data), 1, figsize=(30, 20))
            for d in range(len(data)):
                layer = data[d][l].reshape(-1, )
                ax[d].scatter(x=np.arange(start=0, stop=len(layer)), y=layer,
                              alpha=0.8)
                ax[d].set_title(label[d])
            fig.tight_layout()
            plt.savefig(os.path.join(self.plot_path,
                                     "delta_{}_{}.png".format(epoch, l)))

    def krum(self, epoch, agent_train):

        delta_list = []
        try:
            for index in agent_train[epoch]:
                delta = np.load(os.path.join(self.agent_path,
                                             "agent{}".format(index),
                                             "delta.npy"),
                                allow_pickle=True)
                delta_list.append(self.flatten(delta))
        except FileNotFoundError:
            return False
        print("[server.krum]")
        # collated_weights = []
        # collated_bias = []
        theta_con = []
        agg_num = int(self.train_num - 1 - 2)
        for k, theta in enumerate(delta_list):
            theta_flat = self.flatten(theta)
            theta_con.append(theta_flat)
            # weights_curr, bias_curr = collate_weights(return_dict[str(curr_agents[k])])
            # weights_curr, bias_curr = collate_weights(return_dict[str(k)])
            # collated_weights.append(weights_curr)
            # collated_bias.append(collated_bias)
        score_array = np.zeros(self.train_num)
        for k in range(self.train_num):
            dists = []
            for i in range(self.train_num):
                if i == k:
                    continue
                else:
                    dists.append(np.linalg.norm(theta_con[k] - theta_con[i]))
            dists = np.sort(np.array(dists))
            dists_subset = dists[:agg_num]
            score_array[k] = np.sum(dists_subset)
        print(score_array)
        krum_index = np.argmin(score_array).tolist()
        print(krum_index)
        self.global_weights += np.mean(delta_list[krum_index], axis=0)
        if krum_index == 0:
            print("[server.krum] epoch {} select mal index".format(epoch))
        with open(os.path.join(self.server_path, "krum.log"), "a+") as f:
            f.write("{}\n".format(score_array))
            f.write("[s]{}\n".format(krum_index))

    def analyze_distance(self, epoch):
        for dist in self._distance.keys():
            # if epoch % 20 == 0:
            # np.save(
            #     os.path.join(self.server_path, "{}_{}.npy".format(dist,
            #                                                       epoch)),
            #     self._distance[dist])
            self._distance[dist] = [[] for _ in range(self.train_num)]

    def scheduler(self, epoch):
        option = list(range(self.agent_num))
        if self._async == 2:
            index = random.sample(option, self.agent_num // 2)
        else:
            index = random.sample(option, self.train_num)
        print("[server] schuduler", index)
        return index

    def updater(self, epoch, agent_train):
        # os.environ["CUDA_VISIBLE_DEVICES"] = "6"
        # gpu = tf.config.experimental.list_physical_devices("GPU")
        # tf.config.experimental.set_memory_growth(gpu[0], True)

        flag = False
        if self._distance:
            flag, element_idx = self.compute_distance(epoch, agent_train)
            if flag:
                self.compute_F(epoch, element_idx)
                self.analyze_distance(epoch)

        if self._async == 1 and epoch % self._period != 0:
            return False
        try:
            theta_list = []
            for index in agent_train[epoch]:
                theta = np.load(os.path.join(self.agent_path,
                                             "agent{}".format(index),
                                             "theta.npy"),
                                allow_pickle=True)
                theta_list.append(theta)
        except FileNotFoundError:
            return False

        if self._async == 0:
            print("[server] sync avg federate")
            self.global_weights = np.mean(theta_list, axis=0)
        elif self._async == 1:
            if self._secFL and flag:
                print("[server] secure federate")
                update_list = []
                for agent in range(self.train_num):
                    w0 = self.flatten(theta_list[agent])
                    w0 *= self.F[agent]
                    update_list.append(self.reshape(w0))
                self.global_weights = np.sum(update_list, axis=0)
            elif self._krum:
                self.krum(epoch, agent_train)
            else:
                print("[server] async avg federate")
                self.global_weights = np.mean(theta_list, axis=0)
        elif self._async == 2:
            while True:
                agent = np.random.randint(low=0, high=4)
                t = agent_train[agent]
                if t > 0 and epoch - t <= self._period:
                    break
            print("[server] async 2")
            alpha_t = self.alpha * pow(epoch - t + 1, -0.5)
            self.global_weights = (1 - alpha_t) * self.global_weights + \
                                  alpha_t * theta_list[agent]
        self.global_test(epoch)

    def global_test(self, epoch):
        # if self.F is None:
        #     print("[update] Invalid E")
        #     return False
        # flag = False
        # if self._distance:
        #     flag, element_idx = self.compute_distance(epoch)
        #     if flag:
        #         self.compute_F(epoch, element_idx)
        #         self.analyze_distance(epoch)
        #
        # try:
        #     theta_list = []
        #     for index in range(self.agent_num):
        #         theta = np.load(os.path.join(self.agent_path,
        #                                      "agent{}".format(index),
        #                                      "theta_{}.npy".format(epoch)),
        #                         allow_pickle=True)
        #         theta_list.append(theta)
        # except FileNotFoundError:
        #     return False
        #
        # if self._async and epoch % self._period != 0:
        #     return False

        # if self._secFL and flag:
        #     update_list = []
        #     for agent in range(self.agent_num):
        #         w0 = self.flatten(theta_list[agent])
        #         w0 *= self.F[agent]
        #         update_list.append(self.reshape(w0))
        #     self.global_weights = np.sum(update_list, axis=0)
        #     # if "element" in dis_level:
        #     #     # element-wise
        #     #     # E = self.compute_E(epoch, dis_type)
        #     #     update_list = []
        #     #     for i in range(self.agent_num):
        #     #         w0 = self.flatten(delta[i]).reshape(1, -1)
        #     #         # E = np.mean([self.D_cosine[i], self.D_euclidean[i]])
        #     #         w0 *= self.C_cosine_element[i]
        #     #         # w0 *= self.E[i, :]
        #     #         update_list.append(self.reshape(np.reshape(w0, (-1, 1))))
        #     #     global_weights = np.sum(update_list, axis=0)
        #     # elif "layer" in dis_level:
        #     #     if "cosine" in dis_type:
        #     #         E = self.C_cosine_layer
        #     #     elif "pearson" in dis_type:
        #     #         E = self.C_pearson
        #     #     else:
        #     #         return False
        #     #     # layer-wise
        #     #     update_list = []
        #     #     for i in range(self.agent_num):
        #     #         for j in range(delta.shape[1]):
        #     #             delta[i, j] *= E[i, j]
        #     #         update_list.append(delta[i])
        #     #     global_weights = np.sum(update_list, axis=0)
        # else:
        #     self.global_weights = np.mean(theta_list, axis=0)

        self.test_model.set_weights(self.global_weights)
        logits = self.test_model.predict(self.test_img, batch_size=64,
                                         verbose=2)
        y_pred = tf.argmax(logits, 1)
        report = classification_report(y_true=self.test_label, y_pred=y_pred,
                                       output_dict=True)
        with open(os.path.join(self.server_path, "test.log"), "a+") as f:
            for label in range(10):
                f.write("{},".format(report[str(label)]["precision"]))
            f.write("\n")
            for label in range(10):
                f.write("{},".format(report[str(label)]["recall"]))
            f.write("\n")
            f.write("{}\n".format(report["accuracy"]))

        self.global_update = self.global_weights - np.load(
            os.path.join(self.server_path, "global_weights.npy"),
            allow_pickle=True)

        np.save(os.path.join(self.server_path, "global_weights.npy"),
                self.global_weights)
        # np.save(os.path.join(self.server_path, "global_weights_{
        # }.npy".format(epoch)), self.global_weights)
        print("[server] save global updates for epoch ", epoch + 1)
