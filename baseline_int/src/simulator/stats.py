class Stats(object):
    """
    Stores the stats from the simulator
    """

    def __init__(self):
        self.total_cycles = 0
        self.ptb_cycles = 0  # 有修改
        self.ps4_cycles = 0  # 有修改
        self.mem_stall_cycles = 0
        self.namespaces = ['act', 'wgt', 'out', 'dram']
        self.reads = {}
        self.writes = {}
        for n in self.namespaces:
            self.reads[n] = 0
            self.writes[n] = 0
        self.in_sparsity_util = 0  # 有修改
        self.out_sparsity_util = 0  # 有修改
        self.best_tiling = ''

    def __iter__(self):
        return iter([\
                     self.total_cycles,
                     self.mem_stall_cycles,
                     self.reads['act'],
                     self.reads['wgt'],
                     self.reads['out'],
                     self.reads['dram'],
                     self.writes['out'],
                     self.writes['dram']
                    ])

    def __add__(self, other):
        ret = Stats()
        ret.total_cycles = self.total_cycles + other.total_cycles
        ret.mem_stall_cycles = self.mem_stall_cycles + other.mem_stall_cycles
        for n in self.namespaces:
            ret.reads[n] = self.reads[n] + other.reads[n]
            ret.writes[n] = self.writes[n] + other.writes[n]
        return ret

    def __mul__(self, other):
        ret = Stats()
        ret.total_cycles = self.total_cycles * other
        ret.mem_stall_cycles = self.mem_stall_cycles * other
        for n in self.namespaces:
            ret.reads[n] = self.reads[n] * other
            ret.writes[n] = self.writes[n] * other
        return ret

    def __str__(self):
        ret = '\tStats'
        ret+= '\n\t{0:>20}   : {1:>20,}, '.format('Total', self.total_cycles)
        ret+= '\n\t{0:>20}   : {1:>20,}, '.format('Memory Stalls', self.mem_stall_cycles)
        ret+= '\n\tReads: '
        for n in self.namespaces:
            ret+= '\n\t{0:>20} rd: {1:>20,} bits, '.format(n, self.reads[n])
        ret+= '\n\tWrites: '
        for n in self.namespaces:
            ret+= '\n\t{0:>20} wr: {1:>20,} bits, '.format(n, self.writes[n])
        return ret

    def get_energy(self, energy_cost, dram_cost=15.e-3):
        dyn_energy = (self.total_cycles - self.mem_stall_cycles) * energy_cost.core_dynamic_energy

        dyn_energy += self.reads['wgt'] * energy_cost.wbuf_read_energy
        dyn_energy += self.writes['wgt'] * energy_cost.wbuf_write_energy

        dyn_energy += self.reads['act'] * energy_cost.ibuf_read_energy
        dyn_energy += self.writes['act'] * energy_cost.ibuf_write_energy

        dyn_energy += self.reads['out'] * energy_cost.obuf_read_energy
        dyn_energy += self.writes['out'] * energy_cost.obuf_write_energy

        # Assuming that the DRAM requires 6 pJ/bit
        dram_cost = 6e-3  # 有修改
        dyn_energy += self.reads['dram'] * dram_cost
        dyn_energy += self.writes['dram'] * dram_cost

        return dyn_energy

    def get_energy_breakdown(self, energy_cost, mode="Baselin", dram_cost=15.e-3):  # 获得各个部分的能耗
        # core energy = （总周期数 - 内存阻塞周期数） * PE阵列核心能耗
        if mode == 'Baseline':
            core_energy = (self.total_cycles - self.mem_stall_cycles) * energy_cost.core_dynamic_energy
        elif mode == 'PTB':  # 有修改
            core_energy = (self.ptb_cycles - self.mem_stall_cycles) * energy_cost.core_dynamic_energy
        elif mode == 'PS4':  # 有修改
            core_energy = (self.ps4_cycles - self.mem_stall_cycles) * energy_cost.core_dynamic_energy

        breakdown = [core_energy]

        # sram能耗 = 权重读写次数/大小？ * PE阵列weight buffer读写能耗 + 激活读写次数/大小 * PE阵列activation buffer读写能耗
        #           + 激活读次数 * 每bit popcnt能耗 + 输出读写次数/大小？ * PE阵列output buffer读写能耗
        sram_energy = self.reads['wgt'] * energy_cost.wbuf_read_energy
        sram_energy += self.writes['wgt'] * energy_cost.wbuf_write_energy

        sram_energy += self.reads['act'] * energy_cost.ibuf_read_energy
        sram_energy += self.writes['act'] * energy_cost.ibuf_write_energy

        if mode == 'PS4':  # 有修改
            popcnt_cost = 4.9882e-3 * 0.15 / 8      # pJ/bit
            popcnt_energy = self.reads['act'] * popcnt_cost
        else:  # 有修改
            popcnt_energy = 0

        sram_energy += self.reads['out'] * energy_cost.obuf_read_energy
        sram_energy += self.writes['out'] * energy_cost.obuf_write_energy

        breakdown.append(sram_energy)
        # breakdown.append(0)

        # dram能耗 = dram读写次数 * dram能耗
        dram_energy = self.reads['dram'] * dram_cost
        dram_energy += self.writes['dram'] * dram_cost
        breakdown.append(dram_energy)

        # popcnt能耗
        breakdown.append(popcnt_energy)  # 有修改
        return breakdown

def get_energy_from_results(results, acc_obj):
    stats = Stats()
    stats.total_cycles = int(results['Cycles'])
    stats.mem_stall_cycles = int(results['Memory wait cycles'])
    stats.reads['act'] = int(results['IBUF Read'])
    stats.reads['out'] = int(results['OBUF Read'])
    stats.reads['wgt'] = int(results['WBUF Read'])
    stats.reads['dram'] = int(results['DRAM Read'])
    stats.writes['act'] = int(results['IBUF Write'])
    stats.writes['out'] = int(results['OBUF Write'])
    stats.writes['wgt'] = int(results['WBUF Write'])
    stats.writes['dram'] = int(results['DRAM Write'])
    energy = stats.get_energy(acc_obj)
    return energy

