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
                __global float *money_supply,
                __global float *salary,
                __global float *bank_bond_buying,
                __global float *bank_bond_selling,
                __global float *updated_money_supply,
                __global float *updated_salary,
                __global float *affordability_ratio,
                int num_banks,
                int num_consumers,
                int num_goods,
                float money_supply_increment
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
                    float margin_requirement = 0.05 - (*interest_rate * 0.1);
                    margin_requirement = (margin_requirement > 0.01) ? margin_requirement : 0.01;
                    if (consumers[gid - num_banks] >= goods[0] * margin_requirement) {
                        consumers[gid - num_banks] += goods[0] * (1 - margin_requirement);
                        banks[bank_id * 4 + 3] -= goods[0] * (1 - margin_requirement);
                    }
                }

                barrier(CLK_GLOBAL_MEM_FENCE);

                if (gid == 0) {
                    float total_prices = 0;
                    for (int i = 0; i < num_goods; i++) {
                        total_prices += new_prices[i];
                    }
                    float average_price = total_prices / num_goods;
                    *affordability_ratio = *money_supply / (*salary * num_consumers);
                    *inflation = average_price * *affordability_ratio;

                    if (*inflation <= 0.03 && *interest_rate > 0.025) {
                        *interest_rate -= 0.01;
                    }
                    *interest_rate = (*interest_rate > 0.025) ? *interest_rate : 0.025;

                    float total_bond_buying = 0;
                    float total_bond_selling = 0;
                    for (int i = 0; i < num_banks; i++) {
                        if (*recession_status) {
                            if (banks[i * 4 + 2] > 0) {
                                banks[i * 4 + 2] -= 1;
                                banks[i * 4 + 3] += *bond_price;
                                total_bond_buying += 1;
                            }
                        } else {
                            banks[i * 4 + 2] += 1;
                            banks[i * 4 + 3] -= *bond_price;
                            total_bond_selling += 1;
                        }
                    }
                    *bank_bond_buying = total_bond_buying;
                    *bank_bond_selling = total_bond_selling;

                    *updated_money_supply = *money_supply + total_bond_buying * *bond_price - total_bond_selling * *bond_price + money_supply_increment;
                    *updated_salary = *salary * (1 + *inflation);
                }
            }
        """).build()

    def fused_kernel(self, banks, consumers, goods, interest_rate, buy_amounts, sell_amounts, gdp_growth, unemployment_rate, interest_rate_adjustment, recession_status, bond_price, money_supply, salary, money_supply_increment):
        num_banks = len(banks)
        num_consumers = len(consumers)
        num_goods = len(goods)

        new_prices = np.empty_like(goods)
        inflation = np.empty(1, dtype=np.float32)
        bank_bond_buying = np.empty(1, dtype=np.float32)
        bank_bond_selling = np.empty(1, dtype=np.float32)
        updated_money_supply = np.empty(1, dtype=np.float32)
        updated_salary = np.empty(1, dtype=np.float32)
        affordability_ratio = np.empty(1, dtype=np.float32)

        # Manually create buffers
        banks_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=banks)
        consumers_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=consumers)
        goods_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=goods)
        interest_rate_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=4)
        buy_amounts_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=buy_amounts)
        sell_amounts_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=sell_amounts)
        new_prices_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=new_prices)
        inflation_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=4)
        gdp_growth_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=gdp_growth)
        unemployment_rate_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=unemployment_rate)
        interest_rate_adjustment_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=4)
        recession_status_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=4)
        bond_price_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=4)
        money_supply_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=4)
        salary_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=4)
        bank_bond_buying_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=4)
        bank_bond_selling_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=4)
        updated_money_supply_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=4)
        updated_salary_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=4)
        affordability_ratio_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=4)

        # Copy data to buffers
        cl.enqueue_copy(self.queue, interest_rate_buf, np.array([interest_rate], dtype=np.float32))
        cl.enqueue_copy(self.queue, interest_rate_adjustment_buf, np.array([interest_rate_adjustment], dtype=np.float32))
        cl.enqueue_copy(self.queue, recession_status_buf, np.array([recession_status], dtype=np.int32))
        cl.enqueue_copy(self.queue, bond_price_buf, np.array([bond_price], dtype=np.float32))
        cl.enqueue_copy(self.queue, money_supply_buf, np.array([money_supply], dtype=np.float32))
        cl.enqueue_copy(self.queue, salary_buf, np.array([salary], dtype=np.float32))

        self.program.fused_kernel(self.queue, (num_consumers + num_banks,), None,
                                  banks_buf, consumers_buf, goods_buf, interest_rate_buf, buy_amounts_buf, sell_amounts_buf,
                                  new_prices_buf, inflation_buf, gdp_growth_buf, unemployment_rate_buf, interest_rate_adjustment_buf,
                                  recession_status_buf, bond_price_buf, money_supply_buf, salary_buf, bank_bond_buying_buf,
                                  bank_bond_selling_buf, updated_money_supply_buf, updated_salary_buf, affordability_ratio_buf,
                                  np.int32(num_banks), np.int32(num_consumers), np.int32(num_goods), np.float32(money_supply_increment))

        # Copy data back from buffers
        cl.enqueue_copy(self.queue, banks, banks_buf)
        cl.enqueue_copy(self.queue, consumers, consumers_buf)
        cl.enqueue_copy(self.queue, goods, goods_buf)
        cl.enqueue_copy(self.queue, np.array([interest_rate], dtype=np.float32), interest_rate_buf)
        cl.enqueue_copy(self.queue, new_prices, new_prices_buf)
        cl.enqueue_copy(self.queue, inflation, inflation_buf)
        cl.enqueue_copy(self.queue, bank_bond_buying, bank_bond_buying_buf)
        cl.enqueue_copy(self.queue, bank_bond_selling, bank_bond_selling_buf)
        cl.enqueue_copy(self.queue, updated_money_supply, updated_money_supply_buf)
        cl.enqueue_copy(self.queue, updated_salary, updated_salary_buf)
        cl.enqueue_copy(self.queue, affordability_ratio, affordability_ratio_buf)

        return banks, consumers, goods, interest_rate, inflation[0], bank_bond_buying[0], bank_bond_selling[0], updated_money_supply[0], updated_salary[0], affordability_ratio[0]