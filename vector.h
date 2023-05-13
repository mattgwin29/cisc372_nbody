#ifndef __TYPES_H__
#define __TYPES_H__

typedef double vector3[3];
#define FILL_VECTOR(vector,a,b,c) {vector[0]=a;vector[1]=b;vector[2]=c;}
extern vector3 *hVel;
extern vector3 *hPos;

extern vector3 *d_hVel;
extern vector3 *d_hPos;
extern double * d_mass;

extern double *mass;
//__shared__ extern double *d_mass;
extern vector3** device_accels;
extern vector3* device_values; 


#endif