void compute();#include <cuda.h>

void populate_acceleration(vector3* values, values3** accel, int local_start, int local_end);

void compute_pairwise_acceleration(vector3* values, values3** accel, int local_start, int local_end);

void sum_rows_from_accel_sum(vector3* accel_sum, values3** accel, int loop_index, int local_start, int local_end);