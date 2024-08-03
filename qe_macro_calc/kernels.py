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
                __global float *bank_bond_buying,
                __global float *bank_bond_selling,
                __global float *weights,
                __global float *money_supply,
                __global float *salary,
                __global float *affordability_ratio,
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
                    float margin_requirement = (0.05 - (*interest_rate * 0.1) > 0.01) ? 0.05 - (*interest_rate * 0.1) : 0.01;
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

                    // Increase inflation proportional to money supply
                    *inflation *= (1 + *money_supply);

                    if (*inflation <= 0.03 && *interest_rate > 0.025) {
                        *interest_rate -= 0.01;
                    }
                    *interest_rate = (*interest_rate > 0.025) ? *interest_rate : 0.025;

                    for (int i = 0; i < num_banks; i++) {
                        if (*recession_status) {
                            if (banks[i * 4 + 2] > 0) {
                                banks[i * 4 + 2] -= 1;
                                banks[i * 4 + 3] += *bond_price;
                                bank_bond_buying[i] += 1;
                            }
                        } else {
                            banks[i * 4 + 2] += 1;
                            banks[i * 4 + 3] -= *bond_price;
                            bank_bond_selling[i] += 1;
                        }
                    }

                    // QE during low growth periods
                    if (*gdp_growth < 0.01) {
                        *money_supply += 0.01;
                    }

                    // Check price-to-income ratio and trigger recession if necessary
                    float average_income = 0;
                    for (int i = 0; i < num_consumers; i++) {
                        average_income += consumers[i * 4 + 3];
                    }
                    average_income /= num_consumers;

                    float average_price = 0;
                    for (int i = 0; i < num_goods; i++) {
                        average_price += new_prices[i];
                    }
                    average_price /= num_goods;

                    *affordability_ratio = average_income / average_price;

                    if (average_price / average_income > 1.5) {
                        *recession_status = 1;
                    } else {
                        *recession_status = 0;
                    }

                    // Update goods prices
                    for (int i = 0; i < num_goods; i++) {
                        goods[i] = new_prices[i];
                    }

                    // Update bond price
                    *bond_price *= (1 + *interest_rate);
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

    def fused_kernel(self, banks, consumers, goods, interest_rate, buy_amounts, sell_amounts, gdp_growth, unemployment_rate, interest_rate_adjustment, recession_status, bond_price, weights, money_supply, salary):
        num_banks = len(banks)
        num_consumers = len(consumers)
        num_goods = len(goods)

        new_prices = np.empty(num_goods, dtype=np.float32)
        inflation = np.empty(1, dtype=np.float32)
        bank_bond_buying = np.zeros(num_banks, dtype=np.float32)
        bank_bond_selling = np.zeros(num_banks, dtype=np.float32)
        affordability_ratio = np.empty(1, dtype=np.float32)

        buffers = self.create_buffers(
            banks, consumers, goods, interest_rate, buy_amounts, sell_amounts, new_prices, inflation, gdp_growth, unemployment_rate, interest_rate_adjustment, recession_status, bond_price, bank_bond_buying, bank_bond_selling, weights, money_supply, salary, affordability_ratio
        )

        self.program.fused_kernel(self.queue, (num_consumers,), None, *buffers, np.int32(num_banks), np.int32(num_consumers), np.int32(num_goods))

        cl.enqueue_copy(self.queue, new_prices, buffers[6])
        cl.enqueue_copy(self.queue, inflation, buffers[7])
        cl.enqueue_copy(self.queue, bank_bond_buying, buffers[14])
        cl.enqueue_copy(self.queue, bank_bond_selling, buffers[15])
        cl.enqueue_copy(self.queue, money_supply, buffers[16])
        cl.enqueue_copy(self.queue, salary, buffers[17])
        cl.enqueue_copy(self.queue, affordability_ratio, buffers[18])

        return new_prices, inflation[0], bank_bond_buying, bank_bond_selling, money_supply[0], salary[0], affordability_ratio[0]