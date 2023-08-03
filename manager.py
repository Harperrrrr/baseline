# -*- coding:utf-8 -*-
import warnings

warnings.filterwarnings("ignore")
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Process, Manager
import numpy as np
import tensorflow as tf
from agent import Agent
# from adversary import AdversaryGAN, AdversaryMP, AdversaryRD
from server import Server
from config import Model, Folder, Dataset, args

os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu"]
gpu = tf.config.experimental.list_physical_devices("GPU")
print(gpu)
# tf.config.experimental.set_memory_growth(gpu[1], True)
tf.config.experimental.set_memory_growth(gpu[0], True)


def run(epochs=100):
    agent_index = np.arange(args["agent_num"])
    agent_list = []

    fd = Folder()
    ds_img, ds_label = Dataset().train_data
    if not args["attack_type"]:
        for index in agent_index:
            agent = Agent(index=index, ds_img=ds_img[index],
                          ds_label=ds_label[index], path=fd,
                          **args)
            agent_list.append(agent)
    # elif args["attack_type"] == "GAN":
    #     for index in agent_index:
    #         if index == args["mal_idx"]:
    #             agent = AdversaryGAN(index=index, ds_img=ds_img[index],
    #                                  ds_label=ds_label[index], path=fd, **args)
    #         else:
    #             agent = Agent(index=index, ds_img=ds_img[index],
    #                           ds_label=ds_label[index], path=fd, **args)
    #         agent_list.append(agent)
    # elif args["attack_type"] == "poison":
    #     for index in agent_index:
    #         if index == args["mal_idx"]:
    #             agent = AdversaryMP(index=index, ds_img=ds_img[index],
    #                                 ds_label=ds_label[index], path=fd, **args)
    #         else:
    #             agent = Agent(index=index, ds_img=ds_img[index],
    #                           ds_label=ds_label[index], path=fd, **args)
    #         agent_list.append(agent)
    # elif args["attack_type"] == "random":
    #     for index in agent_index:
    #         if index == args["mal_idx"]:
    #             agent = AdversaryRD(index=index, ds_img=ds_img[index],
    #                                 ds_label=ds_label[index], path=fd, **args)
    #         else:
    #             agent = Agent(index=index, ds_img=ds_img[index],
    #                           ds_label=ds_label[index], path=fd, **args)
    #         agent_list.append(agent)
    else:
        return False

    server = Server(path=fd, **args)

    agent_train = dict()
    for epoch in range(1, epochs + 1):
        agent_select = server.scheduler(epoch)
        if args["async"] != 1:
            agent_train[epoch] = agent_select
        elif (epoch - 1) % args["sp"] == 0:
            agent_train[epoch] = agent_select
        else:
            agent_train[epoch] = agent_train[epoch - 1]

        with ThreadPoolExecutor(max_workers=20) as executor:
            for idx, agent in enumerate(agent_list):
                if args["secFL"]:
                    executor.submit(agent.process, epoch)
                elif idx in agent_train[epoch]:
                    executor.submit(agent.process, epoch)
        server.updater(epoch, agent_train)
    return True


if __name__ == "__main__":
    for i in range(5):
        run(epochs=200)
