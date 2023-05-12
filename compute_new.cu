#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "compute.h"
#include "vector.h"
#include "config.h"

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL

__global__ void populate_acceleration(vector3* device_values, vector3** accel, int local_start, int local_end);

__global__ void compute_pairwise_acceleration(vector3** accel, int local_start, int local_end);

__global__ void sum_rows_from_accel_sum(vector3** accel, int local_start, int local_end);
 
void compute(){
    //int index = blockIdx.x * blockDim.x + threadIdx.x;
    //int stride = blockDim.x * gridDim.x;
	//make an acceleration matrix which is NUMENTITIES squared in size;

	dim3 blocksize(16,16);
	dim3 grid_size(NUMENTITIES / blocksize.x, NUMENTITIES / blocksize.y);

    /* Probably need cudamalloc*/
	vector3* h_values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	vector3** h_accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);

	cudaMalloc(&d_hVel, NUMENTITIES * sizeof(vector3));
	cudaMalloc(&d_hPos, NUMENTITIES * sizeof(vector3));

	cudaMemcpy(d_hVel, hVel, NUMENTITIES * sizeof(vector3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_hPos, hPos, NUMENTITIES * sizeof(vector3), cudaMemcpyHostToDevice);
	// copy mass to device
	cudaMemcpy(d_mass, mass, sizeof(double), cudaMemcpyHostToDevice);
	// malloc and copy accels to device
	cudaMalloc(&d_accels, NUMENTITIES * sizeof(vector3*));
	cudaMemcpy(&d_accels, h_accels, NUMENTITIES * sizeof(vector3*), cudaMemcpyHostToDevice);

	cudaMalloc(&d_values, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
	// copy all host values into device 
	cudaMemcpy(&d_values, h_values, NUMENTITIES * NUMENTITIES * sizeof(vector3), cudaMemcpyHostToDevice);

	// need to copy hval and hpos onto the GPU
	// copy values to values and accel 

    /* kernel function */
	populate_acceleration<<<grid_size,blocksize>>>(d_values,d_accels,0,NUMENTITIES);
	//copy stuff back to host
	cudaMemcpy(&h_values, d_values, NUMENTITIES * NUMENTITIES * sizeof(vector3), cudaMemcpyDeviceToHost);
	
    compute_pairwise_acceleration<<<grid_size,blocksize>>>(d_accels,0, NUMENTITIES);

    sum_rows_from_accel_sum<<<grid_size,blocksize>>>(d_accels, 0, NUMENTITIES); 
	// copy d_accels back to host
	cudaMemcpy(&h_accels, d_accels, NUMENTITIES * sizeof(vector3*), cudaMemcpyHostToDevice);


	free(h_accels);
	free(h_values);
}


__global__ void populate_acceleration(vector3* device_values, vector3** device_accel, int local_start, int local_end){
	for (int i=local_start;i<local_end;i++)
	device_accel[i]=&device_values[i*local_end];
	__syncthreads();
	//Copy values from device back to host values??
}

__global__ void compute_pairwise_acceleration(vector3** accel, int local_start, int local_end){
	int stride = blockDim.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i=index;0<local_end;i++){
		for (int j=local_start;j<local_end;j++){
			if (i==j) {
				FILL_VECTOR(accel[i][j],0,0,0);
			}
			else{
				vector3 distance;
				for (int k=0;k<3;k++) distance[k]=d_hPos[i][k]-d_hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*d_mass[j]/magnitude_sq;
				FILL_VECTOR(accel[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			}
		}
	}
	__syncthreads();
}

__global__ void sum_rows_from_accel_sum(vector3** accel, int local_start, int local_end){
		for (int i=local_start;i<local_end;i++){

			int stride = blockDim.x;
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			vector3 accel_sum={0,0,0};
			for (int j=0;j<local_end;j++){
				for (int k=0;k<3;k++)
					accel_sum[k]+=(accel[i][j][k]);
			}
			//compute the new velocity based on the acceleration and time interval
			//compute the new position based on the velocity and time interval
			for (int k=0;k<3;k++){
				d_hVel[i][k]+=accel_sum[k]*INTERVAL;
				d_hPos[i][k]=d_hVel[i][k]*INTERVAL;
			}
		}
		__syncthreads();
}