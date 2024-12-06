import math
import functools
import time
import logging

from itertools import permutations
from multiprocessing import Pool, cpu_count

from baseline_int.src.utils.utils import ceil_a_by_b, log2
from baseline_int.src.simulator.loop_stack import LoopStack
from baseline_int.src.simulator.stats import Stats

import numpy as np

logger = logging.getLogger('{}.{}'.format(__name__, 'Optimizer'))
logger.setLevel(logging.DEBUG)

tile_deps = {}
tile_deps['B/b']   = {'act': False, 'wgt': False, 'out': False}
tile_deps['OW/ow'] = {'act': True, 'wgt': False, 'out': True}
tile_deps['OH/oh'] = {'act': True, 'wgt': False, 'out': True}
tile_deps['IC/ic'] = {'act': True, 'wgt': True, 'out': False}
tile_deps['OC/oc'] = {'act': False, 'wgt': True, 'out': True}

# inner_loop = {}
# inner_loop['b']  = {'act': True, 'wgt': False, 'out': True}
# inner_loop['ow'] = {'act': True, 'wgt': False, 'out': True}
# inner_loop['oh'] = {'act': True, 'wgt': False, 'out': True}
# inner_loop['ic'] = {'act': True, 'wgt': True, 'out': False}
# inner_loop['oc'] = {'act': False, 'wgt': True, 'out': True}
# inner_loop['kh'] = {'act': True, 'wgt': True, 'out': False}
# inner_loop['kw'] = {'act': True, 'wgt': True, 'out': False}

def get_stats_fast(conv_params, tiling, order_type, verbose=False):
    """
    Returns cycles and memory accesses to DRAM, IBUF, OBUF, and WBUF
        TODOs: Without im2col, the calculation of weight and act size is inexact
    """
    acc_obj, K, O, S, IC, OC, B, iprec, wprec, im2col, TW, Speed, energy_cost = conv_params
    '''
    args:
            K: Kernel Size 卷积核大小
            O: Output Size 输出大小
            S: Input Stride 卷积步长
            IC: Input Channels 输入维度
            OC: Output Channels 输出维度
            B: Batch size for the layer 批处理大小
            iprec: Precision for activations (bits) 激活精度
            wprec: Precision for weights (bits) 权重精度
            im2col = False
            acc_obj.M = PE column number
            acc_obj.N = PE row number
    '''
    # 分片大小
    num_b, b = tiling['B/b']
    num_ow, ow = tiling['OW/ow']
    num_oh, oh = tiling['OH/oh']
    num_ic, ic = tiling['IC/ic']
    num_oc, oc = tiling['OC/oc']
    # kernel长宽
    kw = kh = K

    if Speed:
        iprec = TW
        TW = TW
    else:
        iprec = 1
        TW = 1
    TW_number = ceil_a_by_b(b, TW)
    # print('b: ', tiling['B/b'])
    # print('TW Number check: ', Speed, TW_number)
    oprec = 32
    
    perf_factor = acc_obj.get_perf_factor(iprec, wprec)  # 1

    # dram 到 sram 的读写
    writes = {}  # 写入区
    reads = {}  # 读取区

    if im2col:
        writes['wgt'] = K * K * ic * wprec * ceil_a_by_b(oc, acc_obj.M) * acc_obj.M
    else:
        #TODO: Figure this out
        writes['wgt'] = K * K * ic * wprec * ceil_a_by_b(oc, acc_obj.M) * acc_obj.M
    if im2col:
        writes['act'] = K * K * ic * B * ceil_a_by_b((oh * ow), acc_obj.N) * acc_obj.N
    else:
        #TODO: Figure this out
        iw = K + (ow - 1) * S  # 卷积特征图的大小
        ih = K + (oh - 1) * S
        writes['act'] = iw * ih * ic * B * ceil_a_by_b((oh * ow), acc_obj.N) * acc_obj.N

    writes['out'] = oprec * ceil_a_by_b((oh * ow), acc_obj.N) * ceil_a_by_b(oc, acc_obj.M) * acc_obj.N * acc_obj.M
    # 一个tile的输出对应于在脉动阵列上所占的大小
    reads['out'] = oprec * ceil_a_by_b((oh * ow), acc_obj.N) * ceil_a_by_b(oc, acc_obj.M) * acc_obj.N * acc_obj.M
    # 一个tile的输出对应于在脉动阵列上所占的大小

    # Skip if overutilizing resources
    # 资源超量分配则跳过
    # TODO check bytes/bits
    overflow = False  # 定义资源超量分配标志位
    if writes['wgt'] > acc_obj.sram['wgt']*8/2:  # 判断单个tile的权重矩阵总大小是否超过buffer大小（字节数表示？），超过则标志溢出
        if verbose:
            print('wgt overflow: {}'.format(writes['wgt']))
            print(b, ow, oh, ic, oc)
        overflow = True
    if writes['act'] > acc_obj.sram['act']*8/2:  # 判断单个tile的激活矩阵总大小是否超过buffer大小（字节数表示？），超过则标志溢出
        if verbose:
            print('act overflow')
            print(b, ow, oh, ic, oc)
        overflow = True
    if writes['out'] > acc_obj.sram['out']*8/2:  # 判断单个tile的输出矩阵总大小是否超过buffer大小（字节数表示？），超过则标志溢出
        if verbose:
            print('out overflow')
            print(b, ow, oh, ic, oc)
        overflow = True
    if overflow:  # 如果buffer溢出（不能完整的将一个tile塞入）
        if verbose:
            print('Activation size: {} bytes'.format(writes['act']/8.))
            print('Weights size: {} bytes'.format(writes['wgt']/8.))
            print('Output size: {} bytes'.format(writes['out']/8.))
        return
    # 定义激活、权重和输出的最大写入大小；定义输出的最大读取大小
    max_write_size = {}
    max_read_size = {}
    for namespace in writes:
        max_write_size[namespace] = writes[namespace]
    for namespace in reads:
        max_read_size[namespace] = reads[namespace]

    # First the loop block optimizations
    # 如果buffer能容下多个分片则扩展
    stats = Stats()
    write_promote = {'wgt': True, 'act': True, 'out': True}
    read_promote = {'out': True}
    if verbose:
        logger.debug('Initialize reads/writes')
        logger.debug('\tim2col: {}'.format(im2col))
        logger.debug('\tTiling: {}'.format(tiling))
        logger.debug('\tReads : {}'.format(reads))
        logger.debug('\tWrites: {}'.format(writes))
    for loop in reversed(order_type):  # order type是包含了所有参数的一个集合，如b,w,h,ic,oc等，loop即为其中tile的一个维度
        num_tiles, tile_size = tiling[loop]  # 读取分块的数量和维度
        # promote all writes
        # 此部分功能：① 将每分片读写数 * 分片数得到总读写数；② 计算sram中的最大读写大小
        for namespace in writes:  # 对writes中的每一项(wgt, act, out)
            # promote is true
            if write_promote[namespace]:  # 对writes中的每一项(wgt, act, out)
                # If tile loop depends on the namespace index, make the read size larger
                if tile_deps[loop][namespace]:  # 如果这一项有这个维度
                    writes[namespace] *= num_tiles  # 那么将其原大小 * 分块数
                    # If tile size is larger than the SRAM, set promote to False
                    if writes[namespace] > acc_obj.sram[namespace]*8./2:  # 如果总大小大于buffer大小，将promote置false。也就是作者想如果buffer足够大就快速计算不分片了？
                        write_promote[namespace] = False
                    else:
                        max_write_size[namespace] = writes[namespace]  # 否则把最大写入大小中对应的项更新
            else:
                if tile_deps[loop][namespace]:  # 如果这一项有这个维度
                    writes[namespace] *= num_tiles  # 乘以维度对应分片数

        # promote all reads
        for namespace in reads:
            # promote is true
            if read_promote[namespace]:
                # Tile loop depends on the namespace index
                if tile_deps[loop][namespace]:
                    reads[namespace] *= num_tiles
                    # Tile size is now larger than the SRAM, set promote to False
                    if reads[namespace] > acc_obj.sram[namespace]*8./2:
                        read_promote[namespace] = False
                    else:
                        max_read_size[namespace] = writes[namespace]
            else:
                if tile_deps[loop][namespace]:  # 如果这一项有这个维度
                    reads[namespace] *= num_tiles

        if verbose:
            logger.debug('Loop: {}'.format(loop))
            logger.debug('\tLoop range: {}'.format(tiling[loop]))
            logger.debug('\tMax write size: {}'.format(max_write_size))
            logger.debug('\tMax read size: {}'.format(max_read_size))
            logger.debug('\tLoop Dependencies: {}'.format(tile_deps[loop]))
            logger.debug('\tLoop Promote: {}'.format(write_promote))
            logger.debug('\tReads : {}'.format(reads))
            logger.debug('\tWrites: {}'.format(writes))

    # 将读取区和写入区的大小写到统计数据stats中
    for namespace in writes:
        stats.writes[namespace] = writes[namespace]
        stats.reads['dram'] += writes[namespace]
    for namespace in reads:
        stats.reads[namespace] = reads[namespace]
        stats.writes['dram'] += reads[namespace]


    # Next the inner loop optimizations
    # 做取整运算？
    if im2col:
        # With im2col, loops are:
        # (os_loop: ic x kh x kw): Wgt: True, Out: False, Act: True
        # (ws_loop: b x oh x ow): Wgt: False, Out: True, Act: True
        # (is_loop: oc): Wgt: True, Out: True, Act: False
        is_loop = ceil_a_by_b(oc, acc_obj.M) * acc_obj.M
        os_loop = ceil_a_by_b(ic * kh * kw, acc_obj.N * acc_obj.get_perf_factor(iprec, wprec)) * acc_obj.N * acc_obj.get_perf_factor(iprec, wprec)
        ws_loop = b * oh * ow
        # Output Stationary energy
        # oc * oh * ow * b -> kw * kh * ic
        os_energy = (is_loop * ws_loop) * (oprec + os_loop * (iprec + wprec))
    else:
        is_loop = ceil_a_by_b(oc, acc_obj.M) * acc_obj.M
        os_loop = ceil_a_by_b(ic, acc_obj.N * acc_obj.get_perf_factor(iprec, wprec)) * acc_obj.N * acc_obj.get_perf_factor(iprec, wprec) * kh * kw
        ws_loop = b * oh * ow
        # Output Stationary energy
        # oc * oh * ow * b -> kw * kh * ic
        os_energy = (is_loop * ws_loop) * (oprec + os_loop * (iprec + wprec))

    min_energy = os_energy  # 取三种stationary中能量最小的
    num_tiles = num_b * num_ow * num_oh * num_ic * num_oc  # 分片数量

    # SRAM 到 脉动阵列的读写
    if os_energy == min_energy:
        if verbose:
            logger.debug('SRAM access order: Output Stationary')
        # stats.reads['act'] += num_tiles * (oc * oh * ow * b) * (kw * kh * ic) * iprec
        # stats.reads['out'] += num_tiles * (oc * oh * ow * b) * oprec
        # stats.writes['out'] += num_tiles * (oc * oh * ow * b) * oprec
        # stats.reads['wgt'] += num_tiles * (oc * oh * ow * b) * (kw * kh * ic) * wprec
        # stats.reads['out'] += num_tiles * (oc * oh * ow * TW_number) * oprec
        stats.reads['wgt'] += num_tiles * oc * (kw*kh*ic * TW_number) * wprec
        stats.reads['act'] += num_tiles * oh * ow * kw*kh*ic * B
        stats.writes['out'] += num_tiles * ceil_a_by_b(oc, acc_obj.M) * acc_obj.M * ceil_a_by_b(oh*ow, acc_obj.N) * acc_obj.N * oprec

    # TODO: update
    # 初始化dram的读取和写入次数为0
    initial_dram_reads = 0
    final_dram_writes = 0
    for namespace in max_write_size:
        initial_dram_reads += max_write_size[namespace]
        # 获取不同部分参数的dram读取次数
    for namespace in max_read_size:
        final_dram_writes += max_read_size[namespace]
        # 获取不同部分参数的dram写入次数
    latency = acc_obj.get_mem_read_cycles('dram', initial_dram_reads) + acc_obj.get_mem_write_cycles('dram', final_dram_writes)
    # 根据dram的读写次数计算出latency
    # latency = 计算前对dram的访问cycles + 计算完后对dram的写入cycles

    total_dram_accesses = stats.reads['dram'] + stats.writes['dram']  # 从stats中获得总的dram访问次数
    middle_dram_accesses = total_dram_accesses - initial_dram_reads - final_dram_writes  # 计算过程中的dram访问量
    # 总dram访问次数减去对当前一层神经网络的dram访问次数


    compute_cycles = num_tiles * ceil_a_by_b((oh * ow), acc_obj.N) * ceil_a_by_b(oc, acc_obj.M) * acc_obj.get_compute_cycles_output_stationary(ic, kw, kh, TW_number, im2col)  # 计算所需的时钟周期数
    # 计算时钟周期数 = 分片数量 * 脉动阵列次数 * 单次脉动阵列的时钟周期数
    memory_cycles_required = ceil_a_by_b(middle_dram_accesses, acc_obj.mem_if_width)  # 读取所需的时钟周期数：计算中的内存访问/访存带宽 取整
    # 需要的内存周期数

    memory_stalls = max(0, memory_cycles_required - compute_cycles) + latency  # 内存停顿周期数，表示由于内存访问延迟导致的芯片运行停顿周期数。
    # memory_stalls = 0 or (计算中访存周期数 - 计算周期) latency_hiding? + 计算始末访存带来的latency
    stats.total_cycles = compute_cycles + memory_stalls  # 总周期数 = 计算周期数 + 内存停顿周期数；写入stats中
    stats.mem_stall_cycles = memory_stalls  # 存储停顿周期数；写入stats中

    if verbose:
        logger.debug('Compute cycles : {:>20,}'.format(compute_cycles))
        logger.debug('Memory cycles  : {:>20,}'.format(memory_cycles_required + latency))
        logger.debug('Memory stalls  : {:>20,}'.format(memory_stalls))

    stats.best_tiling = str(tiling)
    stats.best_order = str(order_type)

    return stats


def optimize_for_order(conv_params):
    # Generate permutations for the order
    loops = ['B/b', 'OW/ow', 'OH/oh', 'IC/ic', 'OC/oc']
    order = set(permutations(loops))  # 包括loops中元素以所有顺序排列的可能

    # 过滤掉不符合要求的排列，即排除 'B/b' 出现在 'IC/ic' 后面的排列
    order = [perm for perm in order if perm.index('IC/ic') < perm.index('B/b')]

    return_dict = {}
    acc_obj, K, O, S, IC, OC, B, iprec, wprec, im2col, TW, Speed, energy_cost = conv_params

    _bound_optimizer_method = functools.partial(_optimize_for_order, conv_params)

    try:
        core_number = cpu_count()
        # print(core_number)
        if core_number > 100:
            core_number = 1
            # print(core_number)
        pool = Pool(core_number)
        results = pool.map_async(_bound_optimizer_method, order).get(10000)
        pool.close()
        pool.join()

        # for o in order:
        #     _bound_optimizer_method(o)
        # exit()

        best_cycles = None
        best_energy = None
        # min_cycles = min([x[-4] for x in results])
        # min_energy = min([x[-3] for x in results])
        # cycles_list = [x[-2] for x in results]
        # energy_list = [x[-1] for x in results]
        for r in results:
            tiling, order_type, cycles, energy = r
            if best_cycles is None or best_cycles > cycles or (best_cycles == cycles and best_energy > energy):
                best_cycles = cycles
                best_energy = energy
                best_tiling = tiling
                best_order = order_type
        # print('best tiling:', best_tiling)
        return get_loop_instructions(conv_params, best_tiling, best_order), best_tiling, best_order

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        return


def get_loop_instructions(conv_params, tiling, order_type):
    acc_obj, K, O, S, IC, OC, B, iprec, wprec, im2col, TW, Speed, energy_cost = conv_params
    I = (O - 1) * S + K

    num_b, b = tiling['B/b']
    num_ow, ow = tiling['OW/ow']
    num_oh, oh = tiling['OH/oh']
    num_ic, ic = tiling['IC/ic']
    num_oc, oc = tiling['OC/oc']

    instructions = {}
    instructions['B/b'] = [num_b, I * I * IC * b, 0, O * O * OC * b]
    instructions['OW/ow'] = [num_ow, ow * S, 0, ow]
    instructions['OH/oh'] = [num_oh, I * S, 0, O]
    instructions['IC/ic'] = [num_ic, I * I * ic, K * K * ic, 0]
    instructions['OC/oc'] = [num_oc, 0, K * K * IC * oc, O * O * oc]

    instruction_ordered = LoopStack()
    wgt_stride = []
    act_stride = []
    out_stride = []
    count = 0
    for o in order_type:
        ins = instructions[o]
        if ins[0] > 1:
            stride = {'wgt': ins[2], 'act': ins[1], 'out': ins[3]}
            instruction_ordered.insert_loop(ins[0], stride=stride, level=count, name=o)
            wgt_stride.append(stride['wgt'])
            act_stride.append(stride['act'])
            out_stride.append(stride['out'])
            count += 1
    if count == 0:
        ins = instructions[o]
        stride = {'wgt': ins[2], 'act': ins[1], 'out': ins[3]}
        instruction_ordered.insert_loop(ins[0], stride=stride, level=count, name=o)
        wgt_stride.append(stride['wgt'])
        act_stride.append(stride['act'])
        out_stride.append(stride['out'])
        count += 1

    iw = K + (ow - 1) * S
    ih = K + (oh - 1) * S

    I = K + (O - 1) * S

    if im2col:
        wgt_read_size = \
                ceil_a_by_b(K * K * ic, acc_obj.N) * acc_obj.N * oc * wprec
        max_wgt_size = \
                ceil_a_by_b(K * K * IC, acc_obj.N) * acc_obj.N * OC * wprec
    else:
        wgt_read_size = \
                ceil_a_by_b(K * K * ic, acc_obj.N) * acc_obj.N * \
                ceil_a_by_b(oc, acc_obj.M) * acc_obj.M * \
                wprec
        max_wgt_size = \
                ceil_a_by_b(K * K * IC, acc_obj.N) * acc_obj.N * \
                ceil_a_by_b(OC, acc_obj.M) * acc_obj.M * wprec


    if im2col:
        act_read_size = ow * oh * \
                ceil_a_by_b(K * K, acc_obj.N) * \
                b * iprec * acc_obj.N
        max_act_size = B * O * O * \
                ceil_a_by_b(K * K, acc_obj.N) * acc_obj.N * \
                iprec
    else:
        act_read_size = iw * ih * ic * b * iprec
        max_act_size = B * I * I * IC * iprec


    oprec = 32
    out_read_size = ow * oh * oc * b * oprec
    max_out_size = O * O * OC * B * oprec


    # Skip if overutilizing resources (consider double buffering)
    if wgt_read_size > acc_obj.sram['wgt'] * 8 / 2.0:
        print('error')
        return
    if act_read_size > acc_obj.sram['act'] * 8 / 2.0:
        return
    if out_read_size > acc_obj.sram['out'] * 8 / 2.0:
        return

    # Skip tiling if underutilizing resources
    # underutilization_count = 0
    # if act_read_size < 0.5 * acc_obj.sram['act'] and max_act_size >= 0.5 * acc_obj.sram['act']:
    #     underutilization_count += 1
    # if out_read_size < 0.5 * acc_obj.sram['out'] and max_out_size >= 0.5 * acc_obj.sram['out']:
    #     underutilization_count += 1
    # if wgt_read_size < 0.5 * acc_obj.sram['wgt'] and max_wgt_size >= 0.5 * acc_obj.sram['wgt']:
    #     underutilization_count += 1
    # if underutilization_count > 1:
    #     return

    # Memory Instructions
    instruction_ordered.insert_mem_read(name='Wgt RD', namespace='wgt', addr=0,
                                        size=wgt_read_size, stride=wgt_stride, level=count - 0)
    instruction_ordered.insert_mem_read(name='Act RD', namespace='act', addr=0,
                                        size=act_read_size, stride=act_stride, level=count - 0)
    instruction_ordered.insert_mem_read(name='Out RD', namespace='out', addr=0,
                                        size=out_read_size, stride=out_stride, level=count - 0)
    instruction_ordered.insert_mem_write(name='Out WR', namespace='out', addr=0,
                                         size=out_read_size, stride=out_stride, level=count - 0)
    ni = K * K * ic
    no = oh * ow * oc
    b = b

    instruction_ordered.insert_compute(acc_obj.get_compute_stats, ic, oc, ow, oh, b, K, K, iprec, wprec, im2col)

    # stats = acc_obj.loop_estimate_stats(instruction_ordered)
    instruction_ordered.promote_mem_ops(acc_obj.sram)

    return instruction_ordered


def _optimize_for_order(conv_params, order_type, verbose=False):
    """
    For a given ordering, optimizes tiling
    Args:
        conv_params: A tuple with convolution params
        order_type: ordering loop
    """
    acc_obj, K, O, S, IC, OC, B, iprec, wprec, im2col, TW, Speed, energy_cost = conv_params
    I = (O - 1) * S + K

    # We do not tile the "K" dimension and compute an entire 2-D conv at a
    # time
    num_O_tiles = int(math.ceil(log2(O))) + 1
    num_IC_tiles = int(math.ceil(log2(IC))) + 1

    # TODO: Fix?
    if im2col:
        num_OC_tiles = int(math.ceil(log2(OC))) + 1
    else:
        num_OC_tiles = int(math.ceil(log2(math.ceil(float(OC)/acc_obj.M)))) + 1

    num_B_tiles = int(math.ceil(log2(B))) + 1

    best_cycles = None
    best_energy = None
    best_tiling = None

    for _b in range(int(math.log(TW, 2)), num_B_tiles):
        b = min(1 << _b, B)
        num_b = ceil_a_by_b(B, b)

        for _o in range(num_O_tiles):
            ow = min(1 << _o, O)
            oh = ow
            num_ow = ceil_a_by_b(O, ow)
            num_oh = ceil_a_by_b(O, oh)

            for _ic in range(num_IC_tiles):
                ic = min(1 << _ic, IC)
                num_ic = ceil_a_by_b(IC, ic)

            # for _ic in range(2):
            #     ic = max(0*_ic, IC)
            #     num_ic = ceil_a_by_b(IC, ic)

                for _oc in range(num_OC_tiles):

                    if im2col:
                        oc = min((1 << _oc), OC)
                    else:
                        oc = min((1 << _oc) * acc_obj.M, OC)

                    num_oc = ceil_a_by_b(OC, oc)

                    iw = K + (ow - 1) * S
                    ih = K + (oh - 1) * S

                    tiling = {}
                    tiling['B/b'] = (num_b, b)
                    tiling['OW/ow'] = (num_ow, ow)
                    tiling['OH/oh'] = (num_oh, oh)
                    tiling['IC/ic'] = (num_ic, ic)
                    tiling['OC/oc'] = (num_oc, oc)

                    stats = get_stats_fast(conv_params, tiling, order_type, verbose=False)

                    if stats is None:
                        continue
                    # print('Stats: ', stats)

                    cycles = stats.total_cycles
                    energy = stats.get_energy(energy_cost)
                    mem_cycles = stats.mem_stall_cycles

                    if best_cycles is None or best_cycles > cycles or (best_cycles == cycles and best_energy > energy):
                    # if best_energy is None or best_energy > energy or (best_energy == energy and best_cycles > cycles):
                        best_energy = energy
                        best_cycles = cycles
                        best_mem_cycles = mem_cycles
                        best_order = order_type
                        best_tiling = tiling


    if best_cycles is None:
        print('Not found')
        print(conv_params)
        # stats = get_stats_fast(conv_params, tiling, order_type, verbose=True)

    return (best_tiling, order_type, best_cycles, best_energy)
