import argparse
import logging

from dnnweaver2.graph import Graph, get_default_graph
from dnnweaver2.tensorOps.cnn import conv2D, maxPool, flatten, matmul, addBias, batch_norm, reorg, concat, leakyReLU, add
from dnnweaver2 import get_tensor
import logging
from dnnweaver2.scalar.dtypes import FQDtype, FixedPoint
import pandas as pd
import numpy as np
import torch

import os

def fc(tensor_in, output_channels=1024,
        f_dtype=None, w_dtype=None,
        act='linear', act_fname=None, wgt_fname=None, out_fname=None, in_sparsity_util=0, out_sparsity_util=0):  # 有修改
    input_channels = tensor_in.shape[-1]
    weights = get_tensor(shape=(output_channels, input_channels),
            name='weights',
            dtype=w_dtype)
    biases = get_tensor(shape=(output_channels,),
            name='biases',
            dtype=FixedPoint(32,w_dtype.frac_bits + tensor_in.dtype.frac_bits))
    _fc = matmul(tensor_in, weights, biases, dtype=f_dtype, act_fname=act_fname, wgt_fname=wgt_fname, out_fname=out_fname, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)  # 有修改

    if act == 'leakyReLU':
        with get_default_graph().name_scope(act):
            act = leakyReLU(_fc, dtype=_fc.dtype)
    elif act == 'linear':
        with get_default_graph().name_scope(act):
            act = _fc
    else:
        raise ValueError('Unknown activation type {}'.format(act))

    return act

def conv(tensor_in, filters=32, stride=None, kernel_size=3, pad='SAME',
        c_dtype=None, w_dtype=None,
        act='linear', act_fname=None, wgt_fname=None, out_fname=None, in_sparsity_util=0, out_sparsity_util=0):  # 有修改

    if stride is None:
        stride = (1,1,1,1)

    input_channels = tensor_in.shape[-1]

    weights = get_tensor(shape=(filters, kernel_size, kernel_size, input_channels),
                         name='weights',
                         dtype=w_dtype)
    biases = get_tensor(shape=(filters),
                         name='biases',
                         dtype=FixedPoint(32,w_dtype.frac_bits + tensor_in.dtype.frac_bits))
    _conv = conv2D(tensor_in, weights, biases, stride=stride, pad=pad, dtype=c_dtype, act_fname=act_fname, wgt_fname=wgt_fname, out_fname=out_fname, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)  # 有修改

    if act == 'leakyReLU':
        with get_default_graph().name_scope(act):
            act = leakyReLU(_conv, dtype=_conv.dtype)
    elif act == 'linear':
        with get_default_graph().name_scope(act):
            act = _conv
    else:
        raise ValueError('Unknown activation type {}'.format(act))

    return act


benchlist = []
snn_benchlist = ['CIFAR10SNN', 'DVS128Gesture', 'ImageNetSNN']
# snn_benchlist = ['CIFAR10SNN']


def get_bench_nn(bench_name, TW, batch_size, WRPN=False):  # 有修改
    if bench_name == 'CIFAR10SNN':  # 有修改
        return get_cifar10_snn(TW, batch_size)  # 有修改
    elif bench_name == 'DVS128Gesture':  # 有修改
        return get_dvs128gesture_snn(TW, batch_size)  # 有修改
    elif bench_name == 'ImageNetSNN':  # 有修改
        return get_imagenet_snn(TW, batch_size)  # 有修改


def get_bench_numbers(graph, sim_obj, TW, Speed):
    stats = {}
    for opname, op in graph.op_registry.items():  # 遍历神经网络算子的注册表（网络架构）
        out = sim_obj.get_cycles(op, TW, Speed)  # 获取算子对应的时钟周期数
        if out is not None:
            s, l = out
            stats[opname] = s
            stats[opname].in_sparsity_util = op.in_sparsity_util  # 有修改
            stats[opname].out_sparsity_util = op.out_sparsity_util  # 有修改
    return stats


class Singleton(type):  # 有修改
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
class SnnDataLoader(metaclass=Singleton):  # 有修改
    def __init__(self) -> None:
        super().__init__()
        self.data = {}  # 创建空字典self.data用于存储数据

    def LoadInSnnInputData(self, inputFile: str, totalTimesteps):
        if inputFile in self.data.keys():
            return self.data[inputFile]  # 如果已经读取数据到data，直接返回即可
        # 使用 Pandas 库的 read_csv 函数从 inputFile 中读取数据，指定没有标题行，使用逗号作为分隔符，将数据类型转换为 float，并关闭日期时间推断功能
        data = pd.read_csv(inputFile, header=None, delimiter=',', dtype=float, infer_datetime_format=False)
        # 将读取的数据转换为 NumPy 数组，并将其类型转换为布尔类型（脉冲值）
        data = np.array(data).astype(bool)
        # 调整数组的形状，以便使其符合给定的总时间步数，并根据原始数据的最后一个维度大小来确定每个时间步长中的行数
        data = data.reshape(totalTimesteps, -1, data.shape[-1])
        self.data[inputFile] = data  # (timestep, cin*k*k, row numInVector) row numInVector = H*W
        return self.data[inputFile]


def get_input_sparsity_util(input_fn: str, tw, totalTimesteps=100):  # 获取输入稀疏性
    # 读取数据
    dl = SnnDataLoader()
    input = dl.LoadInSnnInputData(input_fn, totalTimesteps)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    input = torch.tensor(input, device=device)  # shape: [timesteps, cin*k*k, H*W]

    # Pads the input tensor along dimension 0
    # 如果输入张量的第一维长度不为1，则进行补零填充操作，使其能够被tw整除
    if input.shape[0] != 1:
        pad_0 = tw - input.shape[0] % tw
        padding = (0, 0, 0, 0, 0, pad_0)
        input = torch.nn.functional.pad(input, padding, 'constant', 0)
    
    # Reshape the tensor to merge every 'tw' rows into one
    new_shape = (-1, tw, input.shape[1], input.shape[2])  # 将张量形状重塑为(-1(100//tw+?1), tw, input.shape[1](cin*k*k), input.shape[2](row number in vector))
    input = input.view(new_shape)
    
    # Sum along the newly created dimension (1) to merge every 'tw' rows
    input = torch.sum(input, dim=1)  # 合并同一时间窗内的数据，处理后形状为(100//tw+?1, cin*k*k, row number in vector)
    
    # Sum along the remaining spatial dimensions
    input = torch.sum(input, dim=(1, 2))  # 沿着剩余的空间维度对张量进行求和，处理后形状为100//tw+?1
    
    # Flatten the tensor to a 1-D tensor
    input = input.flatten()  # 展平成一维向量，长度为tw number
    
    # Compute Sparsity
    in_sparsity = 1 - torch.count_nonzero(input).float() / input.numel()  # 计算稀疏性，稀疏性等于1减去非零元素的数量除以总元素的数量
    
    return in_sparsity.cpu().numpy().item()  # 以浮点数形式返回


def cal_output_sparsity(output_fn, TW, totalTimesteps=100):
    # 读取数据
    dl = SnnDataLoader()
    output = dl.LoadInSnnInputData(output_fn, totalTimesteps)  # (timestep, row numInVector, output_dim)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # print(f"device: {device}")
    output = torch.tensor(output, device=device)  # shape: [timesteps, h*w, cout]

    # 对输出张量进行填充
    # 如果输出张量的第一维长度不为1，则进行补零填充操作，使其能够被tw整除
    if output.shape[0] != 1:
        pad_0 = TW - output.shape[0] % TW if output.shape[0] % TW != 0 else 0
        padding0 = (0, 0, 0, 0, 0, pad_0)
        output = torch.nn.functional.pad(output, padding0, 'constant', 0)

    # 重塑张量
    new_shape = (-1, TW, output.shape[-2], output.shape[
        -1])  # 将张量形状重塑为(-1(100//tw+?1), tw, output.shape[-2](row number in vector), output.shape[-1](output dim))
    output = output.view(new_shape)  # shape: [100//tw?+1, tw, h*w, cout]

    # 按列求和
    output = torch.sum(output, dim=1)  # 对每个时间窗内的数据求和，处理后的数据维度为（TW number, H*W, Cout）

    # 计算稀疏率
    total_output_number = output.numel()
    non_zero_number = (output != 0).sum().cpu().item()
    output_sparsity = 1 - (non_zero_number / total_output_number)

    return output_sparsity


def get_cifar10_snn(TW, batch_size):
    '''
    CIFAR-10
    SNN
    '''
    g = Graph('CIFAR-10-SNN', dataset='Cifar-10', log_level=logging.INFO)
    # batch_size = 16
    # trace_dir = "/home/tangxin/ant_simulator/scale_sim_v2_main/dataset/layer_record_CIFAR10DVS"
    # trace_dir = 'C:\\Users\\Administrator\\Desktop\\ps4-sim\\SystolicArraySNN_dhl-master\\dataset\\layer_record_CIFAR10DVS'
    trace_dir = '/root/datasets/layer_record_CIFAR10DVS'

    with g.as_default():
        with g.name_scope('inputs'):
            i = get_tensor(shape=(batch_size,128,128,2), name='data', dtype=FQDtype.Bin, trainable=False)

        with g.name_scope('conv0'):
            act_fn = "input.L1.conv.conv2d.csv"
            wgt_fn = "weight.L1.conv.conv2d.csv"
            out_fn = "output.L1.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)

            conv0 = conv(i, filters=128, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
        with g.name_scope('pool0'):
            pool1 = maxPool(conv0, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv1'):
            act_fn = "input.L2.conv.conv2d.csv"
            wgt_fn = "weight.L2.conv.conv2d.csv"
            out_fn = "output.L2.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)

            conv1 = conv(conv0, filters=128, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
        with g.name_scope('pool1'):
            pool1 = maxPool(conv1, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv2'):
            act_fn = "input.L3.conv.conv2d.csv"
            wgt_fn = "weight.L3.conv.conv2d.csv"
            out_fn = "output.L3.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)

            conv2 = conv(pool1, filters=128, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
        with g.name_scope('pool2'):
            pool1 = maxPool(conv2, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv3'):
            act_fn = "input.L4.conv.conv2d.csv"
            wgt_fn = "weight.L4.conv.conv2d.csv"
            out_fn = "output.L4.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)

            conv3 = conv(conv2, filters=128, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
        with g.name_scope('pool3'):
            pool3 = maxPool(conv3, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')
        with g.name_scope('flatten3'):
            flatten5 = flatten(pool3)

        with g.name_scope('fc4'):
            act_fn = "input.L5.fc.linear.csv"
            wgt_fn = "weight.L5.fc.linear.csv"
            out_fn = "output.L5.fc.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)

            fc4 = fc(flatten5, output_channels=512, w_dtype=FQDtype.Bin,
                    f_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)

        with g.name_scope('fc5'):
            act_fn = "input.L6.fc.linear.csv"
            wgt_fn = "weight.L6.fc.linear.csv"
            out_fn = "output.L6.fc.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)

            fc5 = fc(fc4, output_channels=10, w_dtype=FQDtype.Bin,
                    f_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)

    return g


def get_dvs128gesture_snn(TW, batch_size):
    '''
    DVS128Gesture
    SNN
    '''
    g = Graph('DVS128Gesture-SNN', dataset='DVS128Gesture', log_level=logging.INFO)
    # batch_size = 100
    trace_dir = "/root/datasets/layer_record_DVS128Gesture"

    with g.as_default():
        with g.name_scope('inputs'):
            i = get_tensor(shape=(batch_size,128,128,2), name='data', dtype=FQDtype.Bin, trainable=False)
        
        with g.name_scope('conv0'):
            act_fn = "input.L1.conv.conv2d.csv"
            wgt_fn = "weight.L1.conv.conv2d.csv"
            out_fn = "output.L1.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)

            conv0 = conv(i, filters=128, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
        with g.name_scope('pool0'):
            pool0 = maxPool(conv0, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')
        
        with g.name_scope('conv1'):
            act_fn = "input.L2.conv.conv2d.csv"
            wgt_fn = "weight.L2.conv.conv2d.csv"
            out_fn = "output.L2.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)
            conv1 = conv(pool0, filters=128, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
        with g.name_scope('pool1'):
            pool1 = maxPool(conv1, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv2'):
            act_fn = "input.L3.conv.conv2d.csv"
            wgt_fn = "weight.L3.conv.conv2d.csv"
            out_fn = "output.L3.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)
            conv2 = conv(pool1, filters=128, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
        with g.name_scope('pool2'):
            pool2 = maxPool(conv2, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv3'):
            act_fn = "input.L4.conv.conv2d.csv"
            wgt_fn = "weight.L4.conv.conv2d.csv"
            out_fn = "output.L4.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)
            conv3 = conv(pool2, filters=128, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
        with g.name_scope('pool3'):
            pool3 = maxPool(conv3, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv4'):
            act_fn = "input.L5.conv.conv2d.csv"
            wgt_fn = "weight.L5.conv.conv2d.csv"
            out_fn = "output.L5.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)
            conv4 = conv(pool3, filters=128, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
        with g.name_scope('pool4'):
            pool4 = maxPool(conv4, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')
        with g.name_scope('flatten4'):
            flatten4 = flatten(pool4)

        with g.name_scope('fc5'):
            act_fn = "input.L6.fc.linear.csv"
            wgt_fn = "weight.L6.fc.linear.csv"
            out_fn = "output.L6.fc.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)
            fc5 = fc(pool4, output_channels=512, w_dtype=FQDtype.Bin,
                    f_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
        
        with g.name_scope('fc6'):
            act_fn = "input.L7.fc.linear.csv"
            wgt_fn = "weight.L7.fc.linear.csv"
            out_fn = "output.L7.fc.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)
            fc6 = fc(fc5, output_channels=11, w_dtype=FQDtype.Bin,
                    f_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
    return g


def get_imagenet_snn(TW, batch_size):  # 新加入
    '''
    ImageNetSNN
    SNN
    '''
    g = Graph('ImageNet-SNN', dataset='ImageNet', log_level=logging.INFO)
    # batch_size = 100
    trace_dir = "/root/datasets/layer_record_IMAGENET"

    # stride = 2 is replaced with maxpooling
    with g.as_default():
        with g.name_scope('inputs'):
            i = get_tensor(shape=(batch_size, 56, 56, 64), name='data', dtype=FQDtype.Bin, trainable=False)
        
        with g.name_scope('conv1'):
            act_fn = "input.L2.conv.conv2d.csv"
            wgt_fn = "weight.L2.conv.conv2d.csv"
            out_fn = "output.L2.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)
            conv1 = conv(i, filters=64, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
            
        with g.name_scope('conv2'):
            act_fn = "input.L3.conv.conv2d.csv"
            wgt_fn = "weight.L3.conv.conv2d.csv"
            out_fn = "output.L3.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)
            conv2 = conv(conv1, filters=64, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
            
        with g.name_scope('conv3'):
            act_fn = "input.L4.conv.conv2d.csv"
            wgt_fn = "weight.L4.conv.conv2d.csv"
            out_fn = "output.L4.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)
            conv3 = conv(conv2, filters=64, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
        
        with g.name_scope('conv4'):
            act_fn = "input.L5.conv.conv2d.csv"
            wgt_fn = "weight.L5.conv.conv2d.csv"
            out_fn = "output.L5.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)
            conv4 = conv(conv3, filters=64, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
        
        with g.name_scope('conv5'):
            act_fn = "input.L6.conv.conv2d.csv"
            wgt_fn = "weight.L6.conv.conv2d.csv"
            out_fn = "output.L6.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)
            conv5 = conv(conv4, filters=128, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
        with g.name_scope('pool5'):
            pool5 = maxPool(conv5, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv6'):
            act_fn = "input.L7.conv.conv2d.csv"
            wgt_fn = "weight.L7.conv.conv2d.csv"
            out_fn = "output.L7.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)
            conv6 = conv(pool5, filters=128, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
        
        with g.name_scope('conv7'):
            act_fn = "input.L8.conv.conv2d.csv"
            wgt_fn = "weight.L8.conv.conv2d.csv"
            out_fn = "output.L8.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)
            conv7 = conv(conv6, filters=128, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
        
        with g.name_scope('conv8'):
            act_fn = "input.L9.conv.conv2d.csv"
            wgt_fn = "weight.L9.conv.conv2d.csv"
            out_fn = "output.L9.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)
            conv8 = conv(conv7, filters=128, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
            
        with g.name_scope('conv9'):
            act_fn = "input.L10.conv.conv2d.csv"
            wgt_fn = "weight.L10.conv.conv2d.csv"
            out_fn = "output.L10.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)
            conv9 = conv(conv8, filters=256, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
        with g.name_scope('pool9'):
            pool9 = maxPool(conv9, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')
        
        with g.name_scope('conv10'):
            act_fn = "input.L11.conv.conv2d.csv"
            wgt_fn = "weight.L11.conv.conv2d.csv"
            out_fn = "output.L11.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)
            conv10 = conv(pool9, filters=256, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
        
        with g.name_scope('conv11'):
            act_fn = "input.L12.conv.conv2d.csv"
            wgt_fn = "weight.L12.conv.conv2d.csv"
            out_fn = "output.L12.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)
            conv11 = conv(conv10, filters=256, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
        
        with g.name_scope('conv12'):
            act_fn = "input.L13.conv.conv2d.csv"
            wgt_fn = "weight.L13.conv.conv2d.csv"
            out_fn = "output.L13.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)
            conv12 = conv(conv11, filters=256, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
        
        with g.name_scope('conv13'):
            act_fn = "input.L14.conv.conv2d.csv"
            wgt_fn = "weight.L14.conv.conv2d.csv"
            out_fn = "output.L14.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)
            conv13 = conv(conv12, filters=512, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
        with g.name_scope('pool13'):
            pool13 = maxPool(conv13, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')
        
        with g.name_scope('conv14'):
            act_fn = "input.L15.conv.conv2d.csv"
            wgt_fn = "weight.L15.conv.conv2d.csv"
            out_fn = "output.L15.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)
            conv14 = conv(pool13, filters=512, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
        
        with g.name_scope('conv15'):
            act_fn = "input.L16.conv.conv2d.csv"
            wgt_fn = "weight.L16.conv.conv2d.csv"
            out_fn = "output.L16.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)
            conv15 = conv(conv14, filters=512, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
        
        with g.name_scope('conv16'):
            act_fn = "input.L17.conv.conv2d.csv"
            wgt_fn = "weight.L17.conv.conv2d.csv"
            out_fn = "output.L17.conv.plif.csv"
            act_fn = os.path.join(trace_dir, act_fn)
            wgt_fn = os.path.join(trace_dir, wgt_fn)
            out_fn = os.path.join(trace_dir, out_fn)
            in_sparsity_util = get_input_sparsity_util(act_fn, tw=TW, totalTimesteps=batch_size)
            out_sparsity_util = cal_output_sparsity(out_fn, TW=TW, totalTimesteps=batch_size)
            conv16 = conv(conv15, filters=512, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.FXP8, act_fname=act_fn, wgt_fname=wgt_fn, out_fname=out_fn, in_sparsity_util=in_sparsity_util, out_sparsity_util=out_sparsity_util)
        
    return g
