from pyfastbss_core import PyFastbss, pyfbss
from pyfastbss_testbed import pyfbss_tb
import pyfastbss_example as pyfbss_ex
import numpy as np
import os
import sys
import math
import random
import argparse
import progressbar
# 引用一些hop节点
from hop_B_Manager import hop_B_Manager
from hop_MeICA_controller import hop_MeICA_controller
import hop_MeICA_newton_iteration
from hop_Source_controller import hop_Source_controller
# 画图
import matplotlib.pyplot as plt
import matplotlib as mpl


PGB = progressbar.ProgressBar()


# init parameter to generate S
source_number = 5
mixing_type = "normal"
max_min = (1, 0.01)
mu_sigma = (0, 1)
folder_address = 'google_dataset/32000_wav_factory'
duration = 3

# 用于输出结果的参数
tmp_meica = [source_number]
Eve_MeICA = []

#参数 MeICA
max_iter = 100
tol = 1e-04,
break_coef = 0.92
ext_multi_ica = 2


def get_latency_random(mean, std):
    lr = random.gauss(mean, std)
    # 物理延迟，最小延迟
    if lr < (2*10**-5):
        lr = 2*10**-5
    return lr


def init_arg():

    parser = argparse.ArgumentParser(
        description='run MeICA base on Mircosevice in Native & Plot')
    # 设定函数
    parser.set_defaults(func=run_default)
    # 设定默认source个数
    parser.add_argument(
        "--source_num",
        type=int,
        default=5,
        help="setting source number, default: 5 ",
    )
    parser.add_argument(
        "--test_num",
        type=int,
        default=3,
        help="setting the test number for each source number, default: 3 ",
    )
    # 设定是否运算Fast
    parser.add_argument(
        "--run_FastICA",
        action='store_true',
        help="run FastICA",
    )
    # 设定是否运行MeICA
    parser.add_argument(
        "--run_MeICA",
        action='store_true',
        help="run MeICA",
    )
    # 子命令行
    subparser = parser.add_subparsers(title='Option Function')
    # 子命令：节点信息
    parser_iter_time = subparser.add_parser(
        'iter_time', help='get the info of MeICA base on Microservice')
    # 子命令: 节点信息----执行函数
    parser_iter_time.set_defaults(func=run_ms)
    # 子命令：节点信息----是否画图
    parser_iter_time.add_argument(
        "--plot",
        action='store_true',
        help="plot the information in Figure[at now don't work]",
    )
    # 子命令：节点信息----信息源个数
    parser_iter_time.add_argument(
        "--source_num",
        type=int,
        default=5,
        help="setting source number, default: 5 ",
    )
    # 子命令：进行模拟
    parser_simulator = subparser.add_parser(
        'simulator', help='simulate the situation in the Network')
    # 设定默认程序
    parser_simulator.set_defaults(func=run_simulator)
    # 子命令：进行模拟----设定模拟模式
    parser_simulator.add_argument(
        "--mode",
        type=str,
        default='L2FWD',
        choices=['UDP', 'L2FWD', 'ALL', 'None'],
        help='UDP:optimaize the random latency, latency setting change to mean latency. L2FWD: hop run one by one. ALL run & plot all. Default mode is [L2FWD]'
    )
    # 子命令：进行模拟----设定模拟次数
    parser_simulator.add_argument(
        "--test_num",
        type=int,
        default=3,
        help="setting the test number for each source number, default: 3 ",
    )
    # 子命令：进行模拟----设定最大节点个数
    parser_simulator.add_argument(
        "--source_num",
        type=int,
        default=5,
        help="setting source number, default: 5 ",
    )
    # 子命令：进行模拟----设定是否运算Fast
    parser_simulator.add_argument(
        "--run_FastICA",
        action='store_true',
        help="run FastICA",
    )
    # 子命令：进行模拟----设定是否运行MeICA
    parser_simulator.add_argument(
        "--run_MeICA",
        action='store_true',
        help="run MeICA",
    )
    # 子命令：进行模拟----设定远端服务器的延迟
    parser_simulator.add_argument(
        "--service_latency",
        type=float,
        default=150,
        help="setting the latency of service for origin MeICA, default: 150 [ms]"
    )
    # 子命令：进行模拟----设定服务器的性能倍数
    parser_simulator.add_argument(
        "--service_performance",
        type=int,
        default=10,
        help="setting the performance of service compare to mircoservice, default: 10 [times] "
    )
    # 子命令：进行模拟----设定本地micro service的延迟
    parser_simulator.add_argument(
        "--micro_latency",
        type=float,
        default=50,
        help="setting the latency of micro service, default: 50 [ms]"
    )
    args = parser.parse_args()
    return args


def run_ms(args: argparse.ArgumentParser):
    # 设定参数
    source_number = args.source_num

    print(f'Run MeICA base mircoservice, source number:{source_number}')
    Source_controller = hop_Source_controller(
        folder_address, duration, source_number, mixing_type, max_min, mu_sigma)
    B_Manager = hop_B_Manager(Source_controller)
    MeICA_controller = hop_MeICA_controller(
        max_iter, tol, break_coef, ext_multi_ica, Source_controller, B_Manager)
    hat_S = MeICA_controller.get_hat_S()
    runtime_list = MeICA_controller.get_runtime_list()
    grad = MeICA_controller.get_grad()
    iter_list = MeICA_controller.get_iter_list()
    # 写入文件
    ad_work = os.getcwd()
    os.chdir("./test_results/naibao_result/")
    ms_res = open("iter_info.txt", "w")
    # 打印节点信息
    ms_res.write('-------------Hop Usage:--------------\n')
    ms_res.write('hop_MeICA_Controller:          1\n'
                 + 'hop_Source_controller:         1\n'
                 + 'hop_B_Manager:                 1\n'
                 + f'hop_MeICA_newton_iteration:    {grad}\n')
    ms_res.write('-------------------------------------\n')
    ms_res.write(f'Totol Hop:                     {3+grad}\n')
    # 打印iteration迭代信息
    ms_res.write('------------Info Iteration-----------\n')
    ms_res.write('-------------------------------------\n')
    ms_res.write('hop           time[ms]       iter_num\n')
    for hop_i in range(grad):
        ms_res.write(
            f'{hop_i+1:2}            {runtime_list[hop_i]*1000:8.4f}        {iter_list[hop_i]:<}\n')
    ms_res.write('-------------------------------------\n')
    ms_res.write(
        f'Meantime[ms]                 {np.mean(np.array(runtime_list))*1000:8.4f}\n')
    ms_res.close()
    os.chdir(ad_work)
    print('finish!\nplease check ./test_results/naibao_result/iter_info.txt')


def run_default(args: argparse.ArgumentParser):
    if_fast = args.run_FastICA
    if_me = args.run_MeICA
    simu_mode = 'None'
    source_number = args.source_num
    test_number=args.test_num
    core(if_fast, if_me, source_number,test_number, simu_mode)


def run_simulator(args: argparse.ArgumentParser):
    if_fast = args.run_FastICA
    if_me = args.run_MeICA
    simu_mode = args.mode
    source_number = args.source_num
    latency_ms = args.micro_latency
    latency_serv = args.service_latency
    performance = args.service_performance
    test_number = args.test_num
    core(if_fast, if_me, source_number, test_number, simu_mode,
         latency_ms, latency_serv, performance)


def core(if_fast: bool, if_me: bool, source_number: int, test_number: int, simu_mode: str, *args):
    if simu_mode != 'None':
        # 设定simulate的*args
        latency_ms = args[0]
        latency_serv = args[1]
        performance = args[2]

    # 设定评估参数为sdr
    eval_type = 'sdr'

    # 设定x axis
    source_list = np.arange(2,source_number+1,1)
    # 设定 ms_MeICA
    time_list_ms_MeICA = np.tile(np.arange(source_number-1), (test_number, 1))
    time_list_ms_MeICA_L2FWD = np.tile(
        np.arange(source_number-1), (test_number, 1))
    time_list_ms_MeICA_UDP = np.tile(
        np.arange(source_number-1), (test_number, 1))
    snr_list_ms_MeICA = np.tile(np.arange(source_number-1), (test_number, 1))
    # 设定 Me_ICA
    time_list_MeICA = np.tile(np.arange(source_number-1), (test_number, 1))
    time_list_MeICA_Simu = np.tile(
        np.arange(source_number-1), (test_number, 1))
    snr_list_MeICA = np.tile(np.arange(source_number-1), (test_number, 1))
    # 设定 Fast_ICA
    time_list_FastICA = np.tile(np.arange(source_number-1), (test_number, 1))
    time_list_FastICA_Simu = np.tile(
        np.arange(source_number-1), (test_number, 1))
    snr_list_FastICA = np.tile(np.arange(source_number-1), (test_number, 1))
    for t in PGB(range(test_number)):
        for s in range(2, source_number+1, 1):
            print(f'\nstarting for source_number:{s}')
            print('running Ms_MeICA...')
            Source_controller = hop_Source_controller(
                folder_address, duration, s, mixing_type, max_min, mu_sigma)
            B_Manager = hop_B_Manager(Source_controller)
            # use same init B
            B_init = B_Manager.get_B()

            # Part MS_MeICA
            pyfbss_tb.timer_start()
            MeICA_controller = hop_MeICA_controller(
                max_iter, tol, break_coef, ext_multi_ica, Source_controller, B_Manager)
            hat_S_ms = MeICA_controller.get_hat_S()
            time_ms_MeICA = pyfbss_tb.timer_value()
            # Time simulator
            time_list_ms_MeICA[t][s-2] = time_ms_MeICA
            if simu_mode != 'None':
                # simu_mode=='L2FWD':
                '''
                算法：
                client发送接受Source_Controller和B_Manager的信息，latency*4
                client将信息发送MeICA_Controller最后接受S，latency*2
                MeICA_Controller发送B给hop_MeICA_newton_iteration[1]..., latency
                [N-1]到[N],latency*(grad-1)
                MeICA_Controller接受B，算S，latency
                '''
                time_ms_MeICA_L2FWD = time_ms_MeICA + (4+2)*latency_ms
                for i in range(MeICA_controller.get_grad()+1):
                    time_ms_MeICA_L2FWD = time_ms_MeICA_L2FWD + \
                        get_latency_random(latency_ms, latency_ms/5)
                    pass
                time_list_ms_MeICA_L2FWD[t][s-2] = time_ms_MeICA_L2FWD
                #   simu_mode=='UDP':
                '''
                # 算法：
                # client发送接受Source_Controller和B_Manager的信息，latency*4
                # client将信息发送MeICA_Controller最后接受S，latency*2
                # MeICA_Controller接受B，最后更新B，latency*2【程序中设定的是随时与B_Manager进行通信，但这样延迟过于高且暂时无实际意义，那么设定只在开始和结尾进行通信。
                # MeICA_Controller与hop_MeICA_newton_iteration的通信可以取grad个hop中延迟最小的那个
                # 假设有grad个hop，做grad次运算，对每次设定一个lantency list
                '''
                time_ms_MeICA_UDP = time_ms_MeICA + (4+2)*latency_ms
                for i in range(MeICA_controller.get_grad()):
                    latency_temp_list = []
                    for j in range(MeICA_controller.get_grad()):
                        latency_temp_list.append(
                            get_latency_random(latency_ms, latency_ms/5))
                    time_ms_MeICA_UDP = time_ms_MeICA_UDP + \
                        2*min(latency_temp_list)

                time_list_ms_MeICA_UDP[t][s-2] = time_ms_MeICA_UDP
            snr_list_ms_MeICA[t][s-2] = pyfbss_tb.bss_evaluation(
                Source_controller.get_S(), hat_S_ms, eval_type)

            # Part FastICA
            if if_fast:
                print('running FastICA...')
                pyfbss_tb.timer_start()
                X = Source_controller.get_X()
                X, V = pyfbss.whiten(X)
                B1 = B_init
                B2 = pyfbss.newton_iteration(B1, X, max_iter, tol)[0]
                hat_S_Fast = np.dot(B2, X)
                time_Fast = pyfbss_tb.timer_value()

                # Time Simulator
                if simu_mode != 'None':
                    # 算法：
                    # 中间的运算时间根据性能比线性缩短，之后latency*2
                    time_Fast_Simu = time_Fast/performance+latency_serv*2
                    time_list_FastICA_Simu[t][s-2]=time_Fast_Simu
                    pass

                time_list_FastICA[t][s-2] = time_Fast
                snr_list_FastICA[t][s-2]=pyfbss_tb.bss_evaluation(
                    Source_controller.get_S(), hat_S_Fast, eval_type)

            # Part MeICA
            if if_me:
                print('running MeICA...')
                pyfbss_tb.timer_start()
                X = Source_controller.get_X()
                pyfbss.Stack = []
                B1 = B_init
                B2 = pyfbss.multi_level_extraction_newton_iteration(
                    X, B1, max_iter, tol, break_coef, ext_multi_ica)
                hat_S_MeICA = np.dot(B2, X)
                time_MeICA = pyfbss_tb.timer_value()
                # Time Simulator
                if simu_mode != 'None':
                    # 算法和FastICA一样
                    time_MeICA_Simu = time_MeICA/performance+latency_serv*2
                    time_list_MeICA_Simu[t][s-2]=time_MeICA_Simu
                time_list_MeICA[t][s-2]=time_MeICA
                snr_list_MeICA[t][s-2]=pyfbss_tb.bss_evaluation(
                    Source_controller.get_S(), hat_S_MeICA, eval_type)

    # plot
    plt.figure()
    plt.subplot(1, 2, 1)
    if simu_mode == 'None':
        plt.plot(np.array(source_list), np.mean(
            time_list_ms_MeICA,axis=0), 'o-', label='MircoService_MeICA')
    if simu_mode == 'L2FWD':
        plt.plot(np.array(source_list), np.mean(
            time_list_ms_MeICA_L2FWD,axis=0), 'o-', label='MircoService_MeICA_L2FWD')
    if simu_mode == 'UDP':
        plt.plot(np.array(source_list), np.mean(
            time_list_ms_MeICA_UDP,axis=0), 'o-', label='MircoService_MeICA_UDP')
    if simu_mode == 'ALL':
        plt.plot(np.array(source_list), np.mean(
            time_list_ms_MeICA_UDP,axis=0), 'o-', label='MircoService_MeICA_UDP')
        plt.plot(np.array(source_list), np.mean(
            time_list_ms_MeICA_L2FWD,axis=0), 'o-', label='MircoService_MeICA_L2FWD')
    if if_fast:
        # plt.plot(np.array(source_list), np.mean(
        #     time_list_FastICA,axis=0), 'o-', label='FastICA')
        if simu_mode != 'None':
            plt.plot(np.array(source_list), np.mean(
                time_list_FastICA_Simu,axis=0), 'o-', label='FastICA_Simu')
    if if_me:
        # plt.plot(np.array(source_list), np.array(
        #     time_list_MeICA), 'o-', label='MeICA')
        if simu_mode != 'None':
            plt.plot(np.array(source_list), np.mean(
                time_list_MeICA_Simu,axis=0), 'o-', label='MeICA_Simu')
    plt.title(f'Simulator:{simu_mode}')
    plt.xlabel('Source Number')
    plt.ylabel('Time[ms]')
    # 添加图例
    plt.legend(loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(np.array(source_list), np.mean(
        snr_list_ms_MeICA,axis=0), 'o-', label='MircoService_MeICA')
    if if_fast:
        plt.plot(np.array(source_list), np.mean(
            snr_list_FastICA,axis=0), 'o-', label='FastICA')
    # if if_me:
    #     plt.plot(np.array(source_list), np.array(
    #         snr_list_MeICA), 'o-', label='MeICA')
    plt.title(f'SNR')
    plt.xlabel('Source Number')
    plt.ylabel('SNR[dB]')
    # 添加图例
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    # 接受命令
    args = init_arg()
    args.func(args)

    # 获取迭代参数
    # if if_mean:
    #     run_ms()

    # 画出对比图形
    # run_plot(if_fast, if_me, if_simu, latency_ms, latency_serv, performance)
