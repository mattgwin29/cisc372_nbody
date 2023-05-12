#ifndef __TYPES_H__
#define __TYPES_H__

typedef double vector3[3];
#define FILL_VECTOR(vector,a,b,c) {vector[0]=a;vector[1]=b;vector[2]=c;}
extern vector3 *hVel;
extern vector3 *hPos;

__shared__ extern vector3 *d_hVel;
__shared__ extern vector3 *d_hPos;

extern double *mass;
__shared__ extern double *d_mass;
__shared__ vector3** d_accels;

#endif