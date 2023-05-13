#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "compute.h"
#include "vector.h"
#include "config.h"

vector3 *d_hVel;
vector3 *d_hPos;
double *d_mass;

vector3** device_accels;
vector3* device_values; 
//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL

__global__ void populate_acceleration(vector3* device_values, vector3** accel);

__global__ void compute_pairwise_acceleration(vector3** accel, vector3 *device_hPos, double* mass);

__global__ void sum_rows_from_accel_sum(vector3** accel, vector3 *device_hVel, vector3 *device_hPos);
 
void initCudaMemory(){
	    /* Probably need cudamalloc*/


	//extern vector3 *d_hVel;
	//extern vector3 *d_hPos;

	cudaMalloc(&d_hVel, NUMENTITIES * sizeof(vector3));
	cudaMalloc(&d_hPos, NUMENTITIES * sizeof(vector3));
	cudaMalloc(&d_mass, NUMENTITIES * sizeof(double));
	cudaMemcpy(d_hVel, hVel, NUMENTITIES * sizeof(vector3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_hPos, hPos, NUMENTITIES * sizeof(vector3), cudaMemcpyHostToDevice);
	// copy mass to device
	cudaMemcpy(d_mass, mass, sizeof(double), cudaMemcpyHostToDevice);
	// malloc and copy accels to device
	cudaMalloc(&device_accels, NUMENTITIES * sizeof(vector3*));
	//cudaMemcpy(&d_accels, h_accels, NUMENTITIES * sizeof(vector3*), cudaMemcpyHostToDevice);
	
	cudaMalloc(&device_values, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
	// copy all host values into device 
	//cudaMemcpy(&d_values, h_values, NUMENTITIES * NUMENTITIES * sizeof(vector3), cudaMemcpyHostToDevice);

	// need to copy hval and hpos onto the GPU
	// copy values to values and accel 

}
void compute(){
    //int index = blockIdx.x * blockDim.x + threadIdx.x;
    //int stride = blockDim.x * gridDim.x;
	//make an acceleration matrix which is NUMENTITIES squared in size;

	dim3 blocksize(16,16);
	dim3 grid_size((NUMENTITIES+15) / blocksize.x, (NUMENTITIES+15) / blocksize.y);

    /* kernel function */
	populate_acceleration<<<NUMENTITIES,1>>>(device_values,device_accels);
	//copy stuff back to host
	// ** SEGFAULT cudaMemcpy(&h_values, d_values, NUMENTITIES * NUMENTITIES * sizeof(vector3), cudaMemcpyDeviceToHost);
	
	//printf("%d:  %s\n", __LINE__,cudaGetErrorString(cudaGetLastError()));
	//printf("############################################\n");
    compute_pairwise_acceleration<<<grid_size,blocksize>>>(device_accels, d_hPos, mass);
	//printf("%d:  %s\n", __LINE__,cudaGetErrorString(cudaGetLastError()));
	//printf("############################################\n");
    sum_rows_from_accel_sum<<<NUMENTITIES,3>>>(device_accels, d_hPos, d_hVel); 
	//copy hpos and hvel back.
	// copy d_accels back to host
	//printf("%d:  %s\n", __LINE__,cudaGetErrorString(cudaGetLastError()));
	//printf("############################################\n");

	//cudaMemcpy(&h_accels, d_accels, NUMENTITIES * sizeof(vector3*), cudaMemcpyDeviceToHost);

	//free(h_accels);
	//free(h_values);
	cudaMemcpy(hVel, d_hVel, NUMENTITIES * sizeof(vector3), cudaMemcpyDeviceToHost);
	cudaMemcpy(hPos, d_hPos, NUMENTITIES * sizeof(vector3), cudaMemcpyDeviceToHost);
}


__global__ void populate_acceleration(vector3* device_values, vector3** device_accel){
	int i=blockIdx.x;
	device_accel[i]=&device_values[i*NUMENTITIES];
	//Copy values from device back to host values??
}

__global__ void compute_pairwise_acceleration(vector3** accel, vector3* device_hPos, double* mass){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;

	if (i==j){
		FILL_VECTOR(accel[i][j],0,0,0);
	}
	else if (i<NUMENTITIES && j<NUMENTITIES){
				vector3 distance;
				for (int k=0;k<3;k++) distance[k]=device_hPos[i][k]-device_hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
				FILL_VECTOR(accel[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
	}
	// int stride = blockDim.x;
	// int index = blockIdx.x * blockDim.x + threadIdx.x;
	// for (int i=index;0<local_end;i++){
	// 	for (int j=local_start;j<local_end;j++){
	// 		if (i==j) {
	// 			FILL_VECTOR(accel[i][j],0,0,0);
	// 		}
	// 		else{
	// 			vector3 distance;
	// 			for (int k=0;k<3;k++) distance[k]=device_hPos[i][k]-device_hPos[j][k];
	// 			double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
	// 			double magnitude=sqrt(magnitude_sq);
	// 			double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
	// 			FILL_VECTOR(accel[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
	// 		}
	// 	}
	// }
	// __syncthreads();
}

__global__ void sum_rows_from_accel_sum(vector3** accel, vector3 *device_hVel, vector3 *device_hPos){
	int i=blockIdx.x;
	int k=threadIdx.x;
	double accelSum=0;
	for (int j=0;j<NUMENTITIES;j++){
		accelSum+=accel[i][j][k];
	}
	device_hVel[i][k]+=accelSum*INTERVAL;
	device_hPos[i][k]+=device_hVel[i][k]*INTERVAL;
		// for (int i=local_start;i<local_end;i++){

		// 	int stride = blockDim.x;
		// 	int index = blockIdx.x * blockDim.x + threadIdx.x;
		// 	vector3 accel_sum={0,0,0};
		// 	for (int j=0;j<local_end;j++){
		// 		for (int k=0;k<3;k++)
		// 			accel_sum[k]+=(accel[i][j][k]);
		// 	}
		// 	//compute the new velocity based on the acceleration and time interval
		// 	//compute the new position based on the velocity and time interval
		// 	for (int k=0;k<3;k++){
		// 		device_hVel[i][k]+=accel_sum[k]*INTERVAL;
		// 		device_hPos[i][k]+=device_hVel[i][k]*INTERVAL;
		// 	}
		// }
		// __syncthreads();
}