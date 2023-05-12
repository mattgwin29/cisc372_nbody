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

__global__ void populate_acceleration(vector3* values, vector3** accel, int local_start, int local_end);

__global__ void compute_pairwise_acceleration(vector3* values, vector3** accel, int local_start, int local_end);

__global__ void sum_rows_from_accel_sum(vector3** accel, int local_start, int local_end);
 
void compute(){
    //int index = blockIdx.x * blockDim.x + threadIdx.x;
    //int stride = blockDim.x * gridDim.x;
	//make an acceleration matrix which is NUMENTITIES squared in size;

	dim3 blocksize(16,16);
	dim3 grid_size(NUMENTITIES / blocksize.x, NUMENTITIES / blocksize.y);

    /* Probably need cudamalloc*/
	vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);

	cudaMalloc(&d_hVel, NUMENTITIES * sizeof(vector3));
	cudaMalloc(&d_hPos, NUMENTITIES * sizeof(vector3));

	cudaMemcpy(d_hVel, hVel, NUMENTITIES * sizeof(vector3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_hPos, hPos, NUMENTITIES * sizeof(vector3), cudaMemcpyHostToDevice);

	cudaMemcpy(d_mass, mass, sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc(&d_accels, NUMENTITIES * sizeof(vector3*));
	cudaMemcpy(&d_accels, accels, NUMENTITIES * sizeof(vector3*), cudaMemcpyHostToDevice);



	//need to copy hval and hpos onto the GPU
	//copy values to values and accel 

    /* kernel function */
	populate_acceleration<<<grid_size,blocksize>>>(values,d_accels,0,NUMENTITIES);

    //populate_acceleration<<<grid_size,blocksize>>>(values,accels,0,NUMENTITIES);

	/*for (i=0;i<NUMENTITIES;i++)
		accels[i]=&values[i*NUMENTITIES];*/



	//first compute the pairwise accelerations.  Effect is on the first argument.
	
    compute_pairwise_acceleration<<<grid_size,blocksize>>>(values, d_accels,0, NUMENTITIES);

    /*for (i=0;i<NUMENTITIES;i++){
		for (j=0;j<NUMENTITIES;j++){
			if (i==j) {
				FILL_VECTOR(accels[i][j],0,0,0);
			}
			else{
				vector3 distance;
				for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
				FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			}
		}
	}*/

	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
		//printf("STOPPING HERE LALALA\n");
        sum_rows_from_accel_sum<<<grid_size,blocksize>>>(d_accels, 0, NUMENTITIES); //new kernel function
		/*for (j=0;j<NUMENTITIES;j++){
			for (k=0;k<3;k++)
				accel_sum[k]+=accels[i][j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			hVel[i][k]+=accel_sum[k]*INTERVAL;
			hPos[i][k]=hVel[i][k]*INTERVAL;
		}*/
	
	free(accels);
	free(values);
}


__global__ void populate_acceleration(vector3* values, vector3** accel, int local_start, int local_end){
	for (int i=local_start;i<local_end;i++)
	accel[i]=&values[i*local_end];
}

__global__ void compute_pairwise_acceleration(vector3* values, vector3** accel, int local_start, int local_end){
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