import pyopencl as cl
import numpy as np

class Kernels:
    def __init__(self, ctx, queue):
        self.ctx = ctx
        self.queue = queue
        self.mf = cl.mem_flags
        self.program = cl.Program(ctx, """
            __kernel void fused_kernel(
                __global float *banks,
                __global float *consumers,
                __global float *goods,
                __global float *interest_rate,
                __global float *buy_amounts,
                __global float *sell_amounts,
                __global float *new_prices,
                __global float *inflation,
                __global float *gdp_growth,
                __global float *unemployment_rate,
                __global float *interest_rate_adjustment,
                __global int *recession_status,
                __global float *bond_price,
                int num_banks,
                int num_consumers,
                int num_goods
            ) {
                int gid = get_global_id(0);
                if (gid < num_consumers) {
                    for (int i = 0; i < num_goods; i++) {
                        float purchase_prob = goods[i] * (1 + *interest_rate);
                        int buy = (purchase_prob > 0.5) ? 1 : 0;
                        int sell = (purchase_prob < 0.1) ? 1 : 0;
                        int bid_ask_spread = buy - sell;
                        float price_adjustment = 1 + 0.01 * bid_ask_spread / num_consumers;
                        new_prices[i] = goods[i] * price_adjustment;
                    }
                    consumers[gid * 4 + 3] -= buy_amounts[gid];
                    consumers[gid * 4 + 3] += sell_amounts[gid];
                } else if (gid < num_consumers + num_banks) {
                    int bank_id = gid - num_consumers;
                    float margin_requirement = max(0.05 - (*interest_rate * 0.1), 0.01);
                    if (consumers[gid - num_banks] >= goods[0] * margin_requirement) {
                        consumers[gid - num_banks] += goods[0] * (1 - margin_requirement);
                        banks[bank_id * 4 + 3] -= goods[0] * (1 - margin_requirement);
                    }
                }

                barrier(CLK_GLOBAL_MEM_FENCE);

                if (gid == 0) {
                    float total_weight = 0;
                    float weighted_prices = 0;
                    for (int i = 0; i < num_goods; i++) {
                        weighted_prices += goods[i] * weights[i];
                        total_weight += weights[i];
                    }
                    *inflation = weighted_prices / total_weight;

                    if (*inflation <= 0.03 && *interest_rate > 0.025) {
                        *interest_rate -= 0.01;
                    }
                    *interest_rate = max(*interest_rate, 0.025);

                    for (int i = 0; i < num_banks; i++) {
                        if (*recession_status) {
                            if (banks[i * 4 + 2] > 0) {
                                banks[i * 4 + 2] -= 1;
                                banks[i * 4 + 3] += *bond_price;
                            }
                        } else {
                            banks[i * 4 + 2] += 1;
                            banks[i * 4 + 3] -= *bond_price;
                        }
                    }
                }
            }

            __kernel void fused_update(
                __global float *banks,
                __global float *consumers,
                __global float *goods,
                __global float *interest_rate,
                __global float *buy_amounts,
                __global float *sell_amounts,
                __global float *new_prices,
                __global float *inflation,
                __global float *gdp_growth,
                __global float *unemployment_rate,
                __global float *interest_rate_adjustment,
                __global int *recession_status,
                __global float *bond_price,
                int num_banks,
                int num_consumers,
                int num_goods
            ) {
                int gid = get_global_id(0);
                if (gid < num_consumers) {
                    for (int i = 0; i < num_goods; i++) {
                        float purchase_prob = goods[i] * (1 + *interest_rate);
                        int buy = (purchase_prob > 0.5) ? 1 : 0;
                        int sell = (purchase_prob < 0.1) ? 1 : 0;
                        int bid_ask_spread = buy - sell;
                        float price_adjustment = 1 + 0.01 * bid_ask_spread / num_consumers;
                        new_prices[i] = goods[i] * price_adjustment;
                    }
                    consumers[gid * 4 + 3] -= buy_amounts[gid];
                    consumers[gid * 4 + 3] += sell_amounts[gid];
                } else if (gid < num_consumers + num_banks) {
                    int bank_id = gid - num_consumers;
                    float margin_requirement = max(0.05 - (*interest_rate * 0.1), 0.01);
                    if (consumers[gid - num_banks] >= goods[0] * margin_requirement) {
                        consumers[gid - num_banks] += goods[0] * (1 - margin_requirement);
                        banks[bank_id * 4 + 3] -= goods[0] * (1 - margin_requirement);
                    }
                }

                barrier(CLK_GLOBAL_MEM_FENCE);

                if (gid == 0) {
                    float total_weight = 0;
                    float weighted_prices = 0;
                    for (int i = 0; i < num_goods; i++) {
                        weighted_prices += goods[i] * weights[i];
                        total_weight += weights[i];
                    }
                    *inflation = weighted_prices / total_weight;

                    if (*inflation <= 0.03 && *interest_rate > 0.025) {
                        *interest_rate -= 0.01;
                    }
                    *interest_rate = max(*interest_rate, 0.025);

                    for (int i = 0; i < num_banks; i++) {
                        if (*recession_status) {
                            if (banks[i * 4 + 2] > 0) {
                                banks[i * 4 + 2] -= 1;
                                banks[i * 4 + 3] += *bond_price;
                            }
                        } else {
                            banks[i * 4 + 2] += 1;
                            banks[i * 4 + 3] -= *bond_price;
                        }
                    }
                }
            }
        """).build()

    def create_buffers(self, *args):
        buffers = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=arg)
            else:
                raise ValueError("Unsupported type")
            buffers.append(buf)
        return buffers

    def fused_kernel(self, banks, consumers, goods, interest_rate, buy_amounts, sell_amounts, gdp_growth, unemployment_rate, interest_rate_adjustment, recession_status, bond_price):
        num_banks = len(banks)
        num_consumers = len(consumers)
        num_goods = len(goods)

        new_prices = np.empty_like(goods)
        inflation = np.empty(1, dtype=np.float32)

        buffers = self.create_buffers(
            banks, consumers, goods, interest_rate, buy_amounts, sell_amounts, new_prices, inflation, gdp_growth, unemployment_rate, interest_rate_adjustment, recession_status, bond_price
        )

        self.program.fused_kernel(self.queue, (num_consumers,), None, *buffers, np.int32(num_banks), np.int32(num_consumers), np.int32(num_goods))

        cl.enqueue_copy(self.queue, new_prices, buffers[6])
        cl.enqueue_copy(self.queue, inflation, buffers[7])

        return new_prices, inflation[0]

    def fused_update(self, banks, consumers, goods, interest_rate, buy_amounts, sell_amounts, gdp_growth, unemployment_rate, interest_rate_adjustment, recession_status, bond_price):
        num_banks = len(banks)
        num_consumers = len(consumers)
        num_goods = len(goods)

        new_prices = np.empty_like(goods)
        inflation = np.empty(1, dtype=np.float32)

        buffers = self.create_buffers(
            banks, consumers, goods, interest_rate, buy_amounts, sell_amounts, new_prices, inflation, gdp_growth, unemployment_rate, interest_rate_adjustment, recession_status, bond_price
        )

        self.program.fused_update(self.queue, (num_consumers,), None, *buffers, np.int32(num_banks), np.int32(num_consumers), np.int32(num_goods))

        cl.enqueue_copy(self.queue, banks, buffers[0])
        cl.enqueue_copy(self.queue, consumers, buffers[1])
        cl.enqueue_copy(self.queue, goods, buffers[2])
        cl.enqueue_copy(self.queue, np.array([interest_rate], dtype=np.float32), buffers[3])
        cl.enqueue_copy(self.queue, new_prices, buffers[6])
        cl.enqueue_copy(self.queue, inflation, buffers[7])

        return banks, consumers, goods, interest_rate, new_prices, inflation[0]
