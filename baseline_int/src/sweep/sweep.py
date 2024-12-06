import pandas
import os
import logging

from baseline_int.src.simulator.simulator import Simulator
from baseline_int.src.utils.utils import lookup_pandas_dataframe
import baseline_int.src.benchmarks.benchmarks as benchmarks

class SimulatorSweep(object):
    def __init__(self, csv_filename, config_file, verbose=False):
        self.sim_obj = Simulator(config_file, verbose=False)
        self.csv_filename = csv_filename
        if verbose:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO
        self.logger = logging.getLogger('{}.{}'.format(__name__, 'Simulator'))
        self.logger.setLevel(log_level)

        self.columns = ['N', 'M', 'Max Precision (bits)', 'Min Precision (bits)',
                'Network', 'Layer',
                'Cycles', 'Memory wait cycles',
                'WBUF Read', 'WBUF Write',
                'OBUF Read', 'OBUF Write',
                'IBUF Read', 'IBUF Write',
                'DRAM Read', 'DRAM Write',
                'Bandwidth (bits/cycle)',
                'WBUF Size (bits)', 'OBUF Size (bits)', 'IBUF Size (bits)',
                'Timesteps', 'in_sparsity_util', 'out_sparsity_util', 'best_tiling', 'TW', 'Speed', 'best_order']  # 有修改

        if os.path.exists(csv_filename):
            self.sweep_df = pandas.read_csv(csv_filename)
        else:
            self.sweep_df = pandas.DataFrame(columns=self.columns)

    def sweep(self, sim_obj, TW=None, list_n=None, list_m=None, list_pmax=None, list_pmin=None, list_bw=None, list_bench=None, list_wbuf=None, list_ibuf=None, list_obuf=None, list_batch=None, Speed=None):
        """
        Sweep the parameters of the accelerator
        """
        # 参数读取
        if list_n is None:
            list_n = [sim_obj.accelerator.N]
        if list_m is None:
            list_m = [sim_obj.accelerator.M]

        if list_pmax is None:
            list_pmax = [sim_obj.accelerator.pmax]
        if list_pmin is None:
            list_pmin = [sim_obj.accelerator.pmin]

        if list_bw is None:
            list_bw = [sim_obj.accelerator.mem_if_width]

        if list_bench is None:
            list_bench = benchmarks.benchlist

        if list_wbuf is None:
            list_wbuf = [sim_obj.accelerator.sram['wgt']]
        if list_ibuf is None:
            list_ibuf = [sim_obj.accelerator.sram['act']]
        if list_obuf is None:
            list_obuf = [sim_obj.accelerator.sram['out']]

        if list_batch is None:
            list_batch = [1]

        data_line = []
        for batch_size in list_batch:  # 对每个batch
            for n in list_n:
                for m in list_m:
                    # self.logger.info('N x M = {} x {}'.format(n, m))
                    for pmax in list_pmax:
                        for pmin in list_pmin:
                            if pmin > pmax:
                                continue
                            for wbuf in list_wbuf:
                                for ibuf in list_ibuf:
                                    for obuf in list_obuf:
                                        for bw in list_bw:
                                            sim_obj.accelerator.N = n
                                            sim_obj.accelerator.M = m
                                            sim_obj.accelerator.pmax = pmax
                                            sim_obj.accelerator.pmin = pmin
                                            sim_obj.accelerator.mem_if_width = bw
                                            sim_obj.accelerator.sram['wgt'] = wbuf
                                            sim_obj.accelerator.sram['out'] = obuf
                                            sim_obj.accelerator.sram['act'] = ibuf
                                            for b in list_bench:
                                                lookup_dict = {}
                                                lookup_dict['N'] = n
                                                lookup_dict['M'] = m
                                                lookup_dict['Max Precision (bits)'] = pmax
                                                lookup_dict['Min Precision (bits)'] = pmin
                                                lookup_dict['Network'] = b
                                                lookup_dict['Bandwidth (bits/cycle)'] = sim_obj.accelerator.mem_if_width
                                                lookup_dict['WBUF Size (bits)'] = wbuf
                                                lookup_dict['OBUF Size (bits)'] = obuf
                                                lookup_dict['IBUF Size (bits)'] = ibuf
                                                lookup_dict['Timesteps'] = batch_size
                                                results = lookup_pandas_dataframe(self.sweep_df, lookup_dict)
                                                nn = benchmarks.get_bench_nn(b, TW=TW, batch_size=batch_size, WRPN=True)
                                                if len(results) == 0:
                                                    self.logger.info('Simulating Benchmark: {}'.format(b))
                                                    self.logger.info('N x M = {} x {}'.format(n, m))  # m代表脉动矩阵PE列数
                                                    self.logger.info('Max Precision (bits): {}'.format(pmax))
                                                    self.logger.info('Min Precision (bits): {}'.format(pmin))
                                                    self.logger.info('Batch size: {}'.format(batch_size))
                                                    self.logger.info('Bandwidth (bits/cycle): {}'.format(bw))
                                                    stats = benchmarks.get_bench_numbers(nn, sim_obj, TW, Speed)  # 获取性能基准
                                                    for layer in stats:
                                                        cycles = stats[layer].total_cycles
                                                        reads = stats[layer].reads
                                                        writes = stats[layer].writes
                                                        stalls = stats[layer].mem_stall_cycles
                                                        data_line.append((n,m,pmax,pmin,b,layer,
                                                            cycles,stalls,
                                                            reads['wgt'],writes['wgt'],
                                                            reads['out'],writes['out'],
                                                            reads['act'],writes['act'],
                                                            reads['dram'],writes['dram'],
                                                            sim_obj.accelerator.mem_if_width,
                                                            wbuf, obuf, ibuf, batch_size,  # 有修改
                                                            stats[layer].in_sparsity_util,  # 有修改
                                                            stats[layer].out_sparsity_util,
                                                            stats[layer].best_tiling,
                                                            TW, Speed, stats[layer].best_order))  # 有修改
                                            if len(data_line) > 0:
                                                if os.path.exists(self.csv_filename):
                                                    self.sweep_df = pandas.read_csv(self.csv_filename)
                                                else:
                                                    self.sweep_df = pandas.DataFrame(columns=self.columns)
                                                # self.sweep_df = self.sweep_df.append(pandas.DataFrame(data_line, columns=self.columns))
                                                new_df = pandas.DataFrame(data_line, columns=self.columns)  # 有修改
                                                self.sweep_df = pandas.concat([self.sweep_df, new_df], ignore_index=True)  # 有修改
                                                # self.sweep_df.to_csv(self.csv_filename, index=False)
                                                data_line = []
        return self.sweep_df


def check_pandas_or_run(sim, dataframe, sim_sweep_csv, batch_size, config_file, list_bench, TW, Speed):  # 有修改
    ld = {}
    ld['N'] = sim.accelerator.N
    ld['M'] = sim.accelerator.M
    ld['Max Precision (bits)'] = sim.accelerator.pmax
    ld['Min Precision (bits)'] = sim.accelerator.pmin
    ld['Bandwidth (bits/cycle)'] = sim.accelerator.mem_if_width
    ld['WBUF Size (bits)'] = sim.accelerator.sram['wgt']
    ld['OBUF Size (bits)'] = sim.accelerator.sram['out']
    ld['IBUF Size (bits)'] = sim.accelerator.sram['act']
    ld['TW'] = TW

    results = lookup_pandas_dataframe(dataframe, ld)  # 尝试从['sim_sweep_columns']列中读取结果

    sweep_obj = SimulatorSweep(sim_sweep_csv, config_file)  # 根据conf.ini中的参数和csv存储文件创建sweep对象
    dataframe = sweep_obj.sweep(sim, list_batch=[batch_size], list_bench=list_bench, TW=TW, Speed=Speed)
    # dataframe.to_csv(sim_sweep_csv, index=False)  # 将扫描结果写回sim_sweep_csv
    return lookup_pandas_dataframe(dataframe, ld)  # 将ld中的列对应的数据返回
