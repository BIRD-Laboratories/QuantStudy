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
                __global float *historical_inflation,
                __global float *previous_prices,
                int num_banks,
                int num_consumers,
                int num_goods,
                int current_round,
                float money_supply_increment
            ) {
                int gid = get_global_id(0);
                if (gid < num_consumers) {
                    float age = consumers[gid * 11 + 3];
                    float salary = consumers[gid * 11 + 4];
                    float family_status = consumers[gid * 11 + 7];
                    float spend_need = consumers[gid * 11 + 8];
                    float credit_status = consumers[gid * 11 + 9];
                    float job_id = consumers[gid * 11 + 10];

                    for (int i = 0; i < num_goods; i++) {
                        float base_purchase_prob = goods[i] * (1 + *interest_rate);
                        float age_factor = (age > 30) ? 1.2 : 0.8;
                        float salary_factor = (salary > 50000) ? 1.5 : 0.5;
                        float family_status_factor = (family_status == 1) ? 1.3 : 0.7;
                        float spend_need_factor = (spend_need == 2) ? 1.4 : (spend_need == 1) ? 1.1 : 0.9;
                        float credit_status_factor = (credit_status == 0) ? 1.2 : 0.8;
                        float job_id_factor = (job_id == 0) ? 1.0 : 1.0; // Assuming job_id 0 for now

                        float purchase_prob = base_purchase_prob * age_factor * salary_factor * family_status_factor * spend_need_factor * credit_status_factor * job_id_factor;
                        int buy = (purchase_prob > 0.5) ? 1 : 0;
                        int sell = (purchase_prob < 0.1) ? 1 : 0;
                        int bid_ask_spread = buy - sell;
                        float price_adjustment = 1 + 0.01 * bid_ask_spread / num_consumers;
                        new_prices[i] = goods[i] * price_adjustment;
                    }
                    consumers[gid * 11 + 3] -= buy_amounts[gid];
                    consumers[gid * 11 + 3] += sell_amounts[gid];
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

                    // Calculate inflation as the percentage change in average price from the previous round
                    if (current_round > 0) {
                        float previous_average_price = 0;
                        for (int i = 0; i < num_goods; i++) {
                            previous_average_price += previous_prices[i];
                        }
                        previous_average_price /= num_goods;
                        *inflation = (average_price - previous_average_price) / previous_average_price;
                    } else {
                        *inflation = 0; // No inflation in the first round
                    }

                    historical_inflation[current_round] = *inflation;

                    if (*inflation > 0.03) {
                        *interest_rate_adjustment += 0.01;
                        *money_supply -= money_supply_increment;
                    } else if (*inflation < -0.01) {
                        *interest_rate_adjustment -= 0.01;
                        *money_supply += money_supply_increment;
                    } else if (*inflation < 0.03 && *inflation > -0.01) {
                        *interest_rate_adjustment -= 0.005;
                        *money_supply += money_supply_increment * 0.5;
                    }

                    // Ensure money supply is over 10000
                    if (*money_supply < 10000) {
                        *money_supply = 10000;
                    }

                    // Check for hyperinflation in the last 15 rounds
                    bool hyperinflation_in_last_15_rounds = false;
                    for (int i = max(0, current_round - 15); i < current_round; i++) {
                        if (historical_inflation[i] > 0.1) {
                            hyperinflation_in_last_15_rounds = true;
                            break;
                        }
                    }

                    if (hyperinflation_in_last_15_rounds) {
                        *interest_rate_adjustment += 0.05; // Stronger increase in interest rate
                        *money_supply -= money_supply_increment * 2; // Stronger decrease in money supply
                    }

                    // Check for extremely low inflation after hyperinflation
                    if (hyperinflation_in_last_15_rounds && *inflation < 0.01) {
                        *interest_rate_adjustment *= 0.95; // Decrease interest rate by 5%
                        *money_supply += money_supply_increment * 2; // Increase money supply at an accelerated rate
                    }

                    float total_bond_buying = 0;
                    float total_bond_selling = 0;
                    for (int i = 0; i < num_banks; i++) {
                        if (banks[i * 4 + 2] > 0) {
                            banks[i * 4 + 2] -= 1;
                            banks[i * 4 + 3] += *bond_price;
                            total_bond_buying += 1;
                        } else {
                            banks[i * 4 + 2] += 1;
                            banks[i * 4 + 3] -= *bond_price;
                            total_bond_selling += 1;
                        }
                    }
                    *bank_bond_buying = total_bond_buying;
                    *bank_bond_selling = total_bond_selling;

                    *updated_money_supply = *money_supply + total_bond_buying * *bond_price - total_bond_selling * *bond_price + money_supply_increment;
                    *updated_salary = *salary * (1 + *inflation); // Adjust salary based on inflation

                    // Store current prices as previous prices for the next round
                    for (int i = 0; i < num_goods; i++) {
                        previous_prices[i] = new_prices[i];
                    }
                }
            }
        """).build()
        print("Kernel compiled successfully")

    def fused_kernel(self, banks, consumers, goods, interest_rate, buy_amounts, sell_amounts, gdp_growth, unemployment_rate, interest_rate_adjustment, recession_status, bond_price, money_supply, salary, money_supply_increment, historical_inflation, previous_prices, current_round):
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
        historical_inflation_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=historical_inflation)
        previous_prices_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=previous_prices)

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
                                  historical_inflation_buf, previous_prices_buf, np.int32(num_banks), np.int32(num_consumers), np.int32(num_goods), np.int32(current_round), np.float32(money_supply_increment))

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
        cl.enqueue_copy(self.queue, historical_inflation, historical_inflation_buf)
        cl.enqueue_copy(self.queue, previous_prices, previous_prices_buf)

        return banks, consumers, goods, interest_rate + interest_rate_adjustment, inflation[0], bank_bond_buying[0], bank_bond_selling[0], updated_money_supply[0], updated_salary[0], affordability_ratio[0], historical_inflation, previous_prices