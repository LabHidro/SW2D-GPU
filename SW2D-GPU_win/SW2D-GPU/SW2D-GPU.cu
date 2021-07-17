//************************************************************************************************
// Two-dimensional shallow water model accelerated by GPGPU (SW2D-GPU)
//
// Sequential code developer(Original version) : Seungsoo Lee   | Code written in FORTRAN 90
//
// Developer of parallel code in GPGPU: Tomas Carlotto          | Code written in CUDA C/C++

//************************************************************************************************
// Prerequisites for using parallel code:
//         Computer equipped with NVIDIA GPU (compatible with CUDA technology).
//         Software required: CUDA™ Toolkit 8.0 or later 
//                  
//         System: Windows
//         To view and edit the code you must have:
//                  Visual Studio community 2013 or later version
//************************************************************************************************


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <stdio.h>
#include <math.h>
#include <ctime>
#include <chrono>
#include <sstream>

__global__ void init_inf(int rows, int cols, double *d_ho, double *d_h, int *d_inf, double *d_baseo, int N, double NaN){

	int id_inf = blockDim.x*blockIdx.x + threadIdx.x;

	while (id_inf < N){
		//int inj = id_inf % cols;
		//int ini = id_inf / cols;

		d_ho[id_inf] = d_h[id_inf];

		if (d_baseo[id_inf] != NaN){
			d_inf[id_inf] = 1;
		}
		else{
			d_inf[id_inf] = 0;
		}

		// ***************************************************************************
		id_inf += gridDim.x * blockDim.x;
	}
}


__global__ void initiald(int rows, int cols, double *d_h, int *d_infx, int *d_infy, int *d_inf, double *d_hm, double *d_hn, double *d_baseo, int N, double NaN){

	int id_init = blockDim.x*blockIdx.x + threadIdx.x;
	//double hmn;

	while (id_init < N){
		int inj = id_init % cols;
		int ini = id_init / cols;

		// ***************************************************************************
		if (ini == 0){
			d_hm[id_init] = d_h[id_init];
			d_infx[id_init] = 1;
		}
		else if (ini == rows){
			d_hm[id_init] = d_h[id_init - cols];
			d_infx[id_init] = 1;
		}
		else{
			d_hm[id_init] = 0.50*(d_h[id_init] + d_h[id_init - cols]);
			d_infx[id_init] = abs(d_inf[id_init] - d_inf[id_init - cols]);
		}
		//d_hm[id_init] = hmn;
		// ****************************************************************************
		if (inj == 0){
			d_hn[id_init] = d_h[id_init];
			d_infy[id_init] = 1;
		}
		else if (inj == cols){
			d_hn[id_init] = d_h[id_init - 1];
			d_infy[id_init] = 1;
		}
		else{
			d_hn[id_init] = 0.50*(d_h[id_init] + d_h[id_init - 1]);
			d_infy[id_init] = abs(d_inf[id_init] - d_inf[id_init - 1]);
		}
		//d_hn[id_init] = hmn;
		// ***************************************************************************
		id_init += gridDim.x * blockDim.x;
	}
}


// ************************************************************
//          2D SHALLOW WATER CALCULATION
// ************************************************************

__global__ void flux(double *d_th, double gg, double rn, int* d_inf, double* d_h, int* d_infx, int* d_infy, \
	double* d_baseo, double* d_um, double* d_hm, double* d_uu1, double* d_umo, double* d_vv1, \
	double* d_vva, double* d_vn, double*d_hn, double* d_vno, double* d_uua, \
	double* ho, int N, int cols, int rows, double dx, double dy, double dt2)
{
	double hhn, hhs, hhnp, hhsp, hhe, hhw, hhep, hhwp, hhan, sgnm, hh3, u13, u11uur, u11uul, umr, uml, \
		u11, u12vvu, u12vvd, umu, umd, u12, sqx, ram, v13, v11, v11uur, v11uul, vnr, vnl, v12vvu, v12vvd, vnu, vnd, v12, sqy;

	int f_id = blockDim.x*blockIdx.x + threadIdx.x;

	while (f_id < N){
		int inj = f_id % cols;
		int ini = f_id / cols;

		//      ----------------------
		//      X - DIRECTION
		//      ----------------------

		if (ini > 0 && ini < rows) {
			if (d_inf[f_id] != 0) {
				if (d_inf[f_id - cols] != 0){
					if ((d_h[f_id - cols] > d_th[0]) || (d_h[f_id] > d_th[0])){
						if (d_infx[f_id] != 1){
							hhe = d_h[f_id] + d_baseo[f_id];
							hhw = d_h[f_id - cols] + d_baseo[f_id - cols];
							hhep = d_h[f_id] - d_th[0];
							hhwp = d_h[f_id - cols] - d_th[0];

							//      ----------------------
							//      DRY BED TREATMENT (1)
							//      ----------------------

							if (hhe<d_baseo[f_id - cols]){
								if (d_h[f_id - cols]>d_th[0]){
									d_um[f_id] = 0.5440*d_h[f_id - cols] * sqrt(gg*d_h[f_id - cols]);
								}
								else{
									d_um[f_id] = 0;
								}
							}
							else if (hhw < d_baseo[f_id]){
								if (d_h[f_id]>d_th[0]){
									d_um[f_id] = -0.544*d_h[f_id] * sqrt(gg*d_h[f_id]);
								}
								else{
									d_um[f_id] = 0;
								}
							}
							//      ----------------------
							//      DRY BED TREATMENT (2)
							//      ----------------------
							else if (hhep*hhwp < 0){
								if ((d_h[f_id]>0) || (d_h[f_id - cols]>0)){
									hhan = hhep - hhwp;
									sgnm = hhan / abs(hhan);
									hh3 = fmax((d_h[f_id] + d_baseo[f_id]), (d_h[f_id - cols] + d_baseo[f_id - cols])) - fmax(d_baseo[f_id], d_baseo[f_id - cols]);
									d_um[f_id] = -sgnm*0.350*hh3*sqrt(2.00*gg*hh3);
								}
								else{
									d_um[f_id] = 0;
								}
							}

							else{

								//      ----------------------
								//      GRAVITY TERM
								//      ----------------------
								u13 = gg*d_hm[f_id] * (dt2 / dx)*(d_h[f_id] + d_baseo[f_id] - d_h[f_id - cols] - d_baseo[f_id - cols]);

								//      ----------------------
								//      CONVECTION TERM
								//      ----------------------

								u11uur = 0.50*(d_uu1[f_id + cols] + d_uu1[f_id]);
								u11uul = 0.50*(d_uu1[f_id] + d_uu1[f_id - cols]);
								umr = u11uur*(d_umo[f_id + cols] + d_umo[f_id])*0.50 + abs(u11uur)*(d_umo[f_id] - d_umo[f_id + cols])*0.50;
								uml = u11uul*(d_umo[f_id] + d_umo[f_id - cols])*0.50 + abs(u11uul)*(d_umo[f_id - cols] - d_umo[f_id])*0.50;
								u11 = (dt2 / dx)*(umr - uml);

								if (inj == 0){
									u12 = 0;
								}
								else if (inj == cols){
									u12 = 0;
								}
								else{
									u12vvu = 0.50*(d_vv1[f_id + 1] + d_vv1[f_id - cols + 1]);
									u12vvd = 0.50*(d_vv1[f_id] + d_vv1[f_id - cols]);
									umu = u12vvu*(d_umo[f_id + 1] + d_umo[f_id])*0.50 + abs(u12vvu)*(d_umo[f_id] - d_umo[f_id + 1])*0.50;
									umd = u12vvd*(d_umo[f_id - 1] + d_umo[f_id])*0.50 + abs(u12vvd)*(d_umo[f_id - 1] - d_umo[f_id])*0.50;
									u12 = (dt2 / dy)*(umu - umd);
								}
								//      ----------------------
								//      FRICTION TERM
								//      ----------------------
								sqx = sqrt(d_uu1[f_id] * d_uu1[f_id] + d_vva[f_id] * d_vva[f_id]);
								double expx = pow(double(d_hm[f_id]), double(1.3333330));
								ram = gg*rn*rn*sqx / expx;
								if (d_hm[f_id] <= d_th[0]){
									ram = 0.00;
								}
								//      ----------------------
								//      UM
								//      ----------------------
								d_um[f_id] = ((1.00 - dt2*ram*0.50)*d_umo[f_id] + (-u11 - u12 - u13)) / (1.00 + dt2*ram*0.50);
							}

						}
						else{
							d_um[f_id] = 0;
						}
					}
					else{
						d_um[f_id] = 0;
					}
				}
				else{
					d_um[f_id] = 0;
				}
			}
			else{
				d_um[f_id] = 0;
			}
		}
		else{

			d_um[f_id] = 0;

		}

		//      ----------------------
		//      Y - DIRECTION
		//      ----------------------

		if (inj > 0 && inj < cols) {
			if (d_inf[f_id] != 0) {
				if (d_inf[f_id - 1] != 0){
					if ((d_h[f_id - 1] > d_th[0]) || (d_h[f_id] > d_th[0])){
						if (d_infy[f_id] != 1){
							hhn = d_h[f_id] + d_baseo[f_id];
							hhs = d_h[f_id - 1] + d_baseo[f_id - 1];
							hhnp = d_h[f_id] - d_th[0];
							hhsp = d_h[f_id - 1] - d_th[0];


							//      ----------------------
							//      DRY BED TREATMENT (1)
							//      ----------------------

							if (hhn<d_baseo[f_id - 1]){
								if (d_h[f_id - 1]>d_th[0]){
									d_vn[f_id] = 0.5440*d_h[f_id - 1] * sqrt(gg*d_h[f_id - 1]);
								}
								else{
									d_vn[f_id] = 0;
								}
							}
							else if (hhs < d_baseo[f_id]){
								if (d_h[f_id]>d_th[0]){
									d_vn[f_id] = -0.544*d_h[f_id] * sqrt(gg*d_h[f_id]);
								}
								else{
									d_vn[f_id] = 0;
								}
							}
							//      ----------------------
							//      DRY BED TREATMENT (2)
							//      ----------------------
							else if (hhnp*hhsp < 0){
								if ((d_h[f_id]>0) || (d_h[f_id - 1]>0)){
									hhan = hhnp - hhsp;
									sgnm = hhan / abs(hhan);
									hh3 = fmax((d_h[f_id] + d_baseo[f_id]), (d_h[f_id - 1] + d_baseo[f_id - 1])) - fmax(d_baseo[f_id], d_baseo[f_id - 1]);
									d_vn[f_id] = -sgnm*0.350*hh3*sqrt(2.00*gg*hh3);
								}
								else{
									d_vn[f_id] = 0;
								}
							}

							else{

								//      ----------------------
								//      GRAVITY TERM
								//      ----------------------
								v13 = gg*d_hn[f_id] * (dt2 / dy)*(d_h[f_id] + d_baseo[f_id] - d_h[f_id - 1] - d_baseo[f_id - 1]);

								//      ----------------------
								//      CONVECTION TERM
								//      ----------------------			

								if (ini == 0){
									v11 = 0;
								}
								else if (ini == rows){                                                                                                             //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< rows
									v11 = 0;
								}
								else{
									v11uur = 0.50*(d_uu1[f_id + cols] + d_uu1[f_id + cols - 1]);
									v11uul = 0.50*(d_uu1[f_id] + d_uu1[f_id - 1]);
									vnr = v11uur*(d_vno[f_id + cols] + d_vno[f_id])*0.50 + abs(v11uur)*(d_vno[f_id] - d_vno[f_id + cols])*0.50;
									vnl = v11uul*(d_vno[f_id] + d_vno[f_id - cols])*0.50 + abs(v11uul)*(d_vno[f_id - cols] - d_vno[f_id])*0.50;
									v11 = (dt2 / dx)*(vnr - vnl);
								}

								v12vvu = 0.50*(d_vv1[f_id + 1] + d_vv1[f_id]);
								v12vvd = 0.50*(d_vv1[f_id] + d_vv1[f_id - 1]);
								vnu = v12vvu*(d_vno[f_id + 1] + d_vno[f_id])*0.50 + abs(v12vvu)*(d_vno[f_id] - d_vno[f_id + 1])*0.50;
								vnd = v12vvd*(d_vno[f_id - 1] + d_vno[f_id])*0.50 + abs(v12vvd)*(d_vno[f_id - 1] - d_vno[f_id])*0.50;
								v12 = (dt2 / dy)*(vnu - vnd);
								//      ----------------------
								//      FRICTION TERM
								//      ----------------------
								sqy = sqrt(d_uua[f_id] * d_uua[f_id] + d_vv1[f_id] * d_vv1[f_id]);
								double expy = pow(double(d_hn[f_id]), double(1.3333330));
								ram = gg*rn*rn*sqy / expy;
								if (d_hn[f_id] <= d_th[0]){
									ram = 0.00;
								}
								//      ----------------------
								//      VN
								//      ----------------------
								d_vn[f_id] = ((1.00 - dt2*ram*0.50)*d_vno[f_id] + (-v11 - v12 - v13)) / (1.00 + dt2*ram*0.50);
							}

						}
						else{
							d_vn[f_id] = 0;
						}
					}
					else{
						d_vn[f_id] = 0;
					}
				}
				else{
					d_vn[f_id] = 0;
				}
			}
			else{
				d_vn[f_id] = 0;
			}
		}
		else{
			d_vn[f_id] = 0;
		}

		//      +++++++++++++++++++++++++++++++++++++++++++
		//      CONTINUITY EQUATION
		//      +++++++++++++++++++++++++++++++++++++++++++

		//      +++++++++++++++++++++++++++++++++++++++++++


		f_id += gridDim.x * blockDim.x;
	}

}

//   +++++++++++++++++++++++++++++
//   PREPARING NEXT CALCULATION
//   +++++++++++++++++++++++++++++
//   ------------------------------
//   hm, hn, calculation
//   ------------------------------

__global__ void hm_hn(double* d_hm, double* d_hn, double* d_h, int N, int cols, int rows){


	int hmhn_id = blockDim.x*blockIdx.x + threadIdx.x;

	while (hmhn_id < N){
		int inj = hmhn_id % cols;
		int ini = hmhn_id / cols;
		//   ------------------------------
		//   hm, hn, calculation
		//   ------------------------------
		if ((inj>0)){
			d_hn[hmhn_id] = 0.50*(d_h[hmhn_id] + d_h[hmhn_id - 1]);
		}

		if ((ini>0)){
			d_hm[hmhn_id] = 0.50*(d_h[hmhn_id] + d_h[hmhn_id - cols]);
		}

		if (ini == 0){
			d_hm[hmhn_id] = d_h[hmhn_id];
		}
		if (ini == rows){
			d_hm[hmhn_id] = d_h[hmhn_id - cols];
		}

		if (inj == 0){
			d_hn[hmhn_id] = d_h[hmhn_id];
		}
		if (inj == cols){
			d_hn[hmhn_id] = d_h[hmhn_id - 1];
		}

		hmhn_id += gridDim.x * blockDim.x;
	}

}

//   ------------------------------
//   uu1, vv1, calculation         
//   ------------------------------

__global__ void uu1_vv1(double *d_th, double* d_hm, double* d_hn, double* d_uu1, double* d_um, double* d_vv1, double* d_vn, int N, int cols, int rows){


	int u1v1_id = blockDim.x*blockIdx.x + threadIdx.x;

	while (u1v1_id < N){
		int inj = u1v1_id % cols;
		int ini = u1v1_id / cols;

		//   ------------------------------
		//   uu1, vv1, calculation         
		//   ------------------------------

		if (d_hm[u1v1_id] >= d_th[0]){
			d_uu1[u1v1_id] = d_um[u1v1_id] / d_hm[u1v1_id];
		}
		else{
			d_uu1[u1v1_id] = 0.00;
		}

		if (d_hn[u1v1_id] >= d_th[0]){
			d_vv1[u1v1_id] = d_vn[u1v1_id] / d_hn[u1v1_id];
		}
		else{
			d_vv1[u1v1_id] = 0.0;
		}

		if (ini == rows){
			if (d_hm[u1v1_id] >= d_th[0]){
				d_uu1[u1v1_id] = d_um[u1v1_id] / d_hm[u1v1_id];
			}
			else{
				d_uu1[u1v1_id] = 0.00;
			}
		}

		if (inj == cols){
			if (d_hn[u1v1_id] >= d_th[0]){
				d_vv1[u1v1_id] = d_vn[u1v1_id] / d_hn[u1v1_id];
			}
			else{
				d_vv1[u1v1_id] = 0.00;
			}
		}

		u1v1_id += gridDim.x * blockDim.x;
	}

}

//   ------------------------------
//   uu, vv calculation
//   ------------------------------

__global__ void uu_vv(double *d_th, double* d_h, double* d_uu1, double* d_vv1, double*d_uu, double*d_vv, int N, int cols){


	int uuvv_id = blockDim.x*blockIdx.x + threadIdx.x;

	while (uuvv_id < N){
		//int inj = uuvv_id % cols;
		//int ini = uuvv_id / cols;

		//   ------------------------------
		//   uu, vv calculation
		//   ------------------------------

		if (d_h[uuvv_id] >= d_th[0]){
			d_uu[uuvv_id] = (d_uu1[uuvv_id + cols] + d_uu1[uuvv_id]) / 2.00;
			d_vv[uuvv_id] = (d_vv1[uuvv_id + 1] + d_vv1[uuvv_id]) / 2.00;
		}
		else{
			d_uu[uuvv_id] = 0.00;
			d_vv[uuvv_id] = 0.00;
		}

		uuvv_id += gridDim.x * blockDim.x;
	}

}

//   ------------------------------
//   uua, vva calculation
//   ------------------------------

__global__ void uua_vva(double* d_uu1, double* d_vv1, double*d_uua, double*d_vva, int N, int cols, int rows){


	int ua_va_id = blockDim.x*blockIdx.x + threadIdx.x;

	while (ua_va_id < N){
		int inj = ua_va_id % cols;
		int ini = ua_va_id / cols;

		//   ------------------------------
		//   uua, vva calculation
		//   ------------------------------

		//if ((inj>0) && (inj<cols) && (ini>0) && (ini < rows-1)){
		if (inj>0){
		d_uua[ua_va_id] = 0.250*(d_uu1[ua_va_id] + d_uu1[ua_va_id + cols] + d_uu1[ua_va_id - 1] + d_uu1[ua_va_id + cols - 1]);
		}
		
		//if ((ini>0) && (ini < rows-1) && (inj>0) && (inj<cols)){
		if (ini>0){
			d_vva[ua_va_id] = 0.250*(d_vv1[ua_va_id] + d_vv1[ua_va_id + 1] + d_vv1[ua_va_id - cols] + d_vv1[ua_va_id - cols + 1]);
		}		

		if (inj == 0){
			d_uua[ua_va_id] = 0.50*(d_uu1[ua_va_id] + d_uu1[ua_va_id + cols]);
		}
		if (inj == (cols)){
			d_uua[ua_va_id] = 0.50*(d_uu1[ua_va_id] + d_uu1[ua_va_id + cols]);
		}
		if (ini == 0){
			d_vva[ua_va_id] = 0.50*(d_vv1[ua_va_id] + d_vv1[ua_va_id + 1]);
		}
		if (ini == (rows)){
			d_vva[ua_va_id] = 0.50*(d_vv1[ua_va_id] + d_vv1[ua_va_id + 1]);
		}
		
		ua_va_id += gridDim.x * blockDim.x;
	}

}

//**************************************************************************************************************
__host__ void stream_flow(int cols, int rows, double xcoor, double ycoor, double time, double dtrain, double *h_rain, double **h_qq, double *h_ql, double dtoq, double *h_brx, double *h_bry, double dx, double dy, int nst, double* h_rr){

	// nst = Input Numbers

	double ql;
	int it, qiny, qinx;
	for (int i = 0; i < nst; i++){
		if (time <= 1.00){
			ql = h_qq[0][i] * time;
		}
		else{
			it = int(time / dtoq);
			ql = h_qq[it][i] + (h_qq[it + 1][i] - h_qq[it][i]) / (dtoq * (time - dtoq * (it)));
		}
		ql = ql / (dx*dy);  //m3 / s->m / s

		qinx = round(abs(xcoor - h_brx[i]));
		qiny = rows - round(abs(ycoor - h_bry[i]));

		h_ql[(qiny)*cols - (cols - (qinx))] = ql;

	}
	if (time <= 1.00){
		h_rr[0] = h_rain[0] * time;
	}
	else{
		it = int(time / dtrain);
		h_rr[0] = h_rain[it] + (h_rain[it + 1] - h_rain[it]) / (dtrain * (time - dtrain * (it)));// [mm]
	}
	h_rr[0] = h_rr[0] / (dtrain*1000.0); //  mm->m / s
}

__global__ void gpu_evaporation_calc(double albedo, double* d_T, double*d_Rg, double* d_Rs, double* d_pw, double* d_lv, double*d_Evapo, double dtime, int N){

	int ev_id = blockDim.x*blockIdx.x + threadIdx.x;
	while (ev_id < N){
		d_Rs[ev_id] = (1 - albedo)*d_Rg[ev_id]; //net radiation[w/m^2/h];

		//Water density as a function of temperature[kg/m ^ 3] 5°C < Temperature < 40°C
		//ITS-90 Density of Water Formulation for Volumetric Standards Calibration (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4909168/)

		d_pw[ev_id] = 999.85308 + 6.32693*(0.01)*d_T[ev_id] - 8.523829*(0.001)*d_T[ev_id] * d_T[ev_id] + 6.943248*(0.00001)*d_T[ev_id] * d_T[ev_id] * d_T[ev_id] - 3.821216*(0.0000001)*d_T[ev_id] * d_T[ev_id] * d_T[ev_id] * d_T[ev_id];

		//Latent heat of vaporization [j/kg]; 273k < Temperature < 308k
		//A new formula for latent heat of vaporization of water as a function of temperature
		//By B.HENDERSON - SELLERS(1984)
		//Department of Mathematics, University of Salford		

		d_lv[ev_id] = (((d_T[ev_id] + 273.15) / ((d_T[ev_id] + 273.15) - 33.91))*((d_T[ev_id] + 273.15) / ((d_T[ev_id] + 273.15) - 33.91)))*1.91846 * 1000000;

		//Applied hydrology Cap 3.5(Chow etal 1988)
		//Evaporation by the energy balance method
		d_Evapo[ev_id] = (d_Rs[ev_id] / (d_lv[ev_id] * d_pw[ev_id]))*dtime*1000.0; //[mm/dtime];


		if (d_Evapo[ev_id] < 0){
			d_Evapo[ev_id] = 0;
		}

		ev_id += gridDim.x * blockDim.x;

	}
}

__host__ void evaporation_load(double time, double dtrain, double* h_Evapo, double* h_Ev){
	int it;
	if (time <= 1.00){
		h_Ev[0] = h_Evapo[0] * time;
	}
	else{
		it = int(time / dtrain);
		h_Ev[0] = h_Evapo[it] + (h_Evapo[it + 1] - h_Evapo[it]) / (dtrain * (time - dtrain * (it))); // [mm]
	}
	h_Ev[0] = h_Ev[0] / (dtrain*1000.0); //  mm->m / s
}

__global__ void continuity(double dt2, int cols, int rows, double dx, double dy, double *d_rr, double *d_Ev, double *d_ql, double *d_h, double *d_ho, double *d_um, double *d_vn, double INT, double INF, double LWL, double EV_WL_min, int *d_inf, int N){

	// Peri Lake vertedouro
	//int qiny = 397;
	//int qinx = 142;

	// UFSC 
	//int qinx1 = round(abs(744107.8622132 - 745209.0));
	//int qiny1 = 1588 - round(abs(6943856.7347817 - 6944966.0));
	//int qinx2 = round(abs(744107.8622132 - 745209.0));
	//int qiny2 = 1588 - round(abs(6943856.7347817 - 6944980.0));
	//int qinx3 = round(abs(744107.8622132 - 744912.0));
	//int qiny3 = 1588 - round(abs(6943856.7347817 - 6944300.0));
	// ======================================

	double percent_P2flow, evapo;
	int ct_id = blockDim.x*blockIdx.x + threadIdx.x;
	while (ct_id < N){
		//int inj = ct_id % cols;  
		//int ini = ct_id / cols;  


		//************ percentage of precipitation that becomes runoff **************

		if (d_ho[ct_id] > EV_WL_min){
			percent_P2flow = 1.00 - LWL;
			evapo = d_Ev[0];
		}
		else{
			percent_P2flow = 1.00 - (INT + INF);
			evapo = 0.000;
		}

		//if ((ini < rows-1) && (inj<cols)){
			d_h[ct_id] = d_ho[ct_id] - dt2*((d_um[ct_id + cols] - d_um[ct_id]) / dx + (d_vn[ct_id + 1] - d_vn[ct_id]) / dy - d_ql[ct_id] - d_rr[0] * percent_P2flow + evapo);
		//}


		d_h[ct_id] = fmax(d_h[ct_id], 0.00);
		if (d_inf[ct_id] == 0){
			d_h[ct_id] = 0.00;
		}
		ct_id += gridDim.x * blockDim.x;

	}

}

//**************************************************************************************************************


__global__ void forward(int cols, int rows, double *d_umo, double *d_um, double *d_vno, double *d_vn, double *d_ho, double *d_h, int N){

	int fw_id = blockDim.x*blockIdx.x + threadIdx.x;

	while (fw_id<N){

		int inj = fw_id % cols;
		int ini = fw_id / cols;

		d_umo[fw_id] = d_um[fw_id];
		d_vno[fw_id] = d_vn[fw_id];
		d_ho[fw_id] = d_h[fw_id];

		if (ini == rows){
			d_umo[fw_id] = d_um[fw_id];
		}
		if (inj == cols){
			d_vno[fw_id] = d_vn[fw_id];
		}

		fw_id += gridDim.x * blockDim.x;
	}

}

__global__ void treat_error(int cols, int rows, double *d_th, int *d_inf, double *d_um, double *d_vn, double *d_h, int N){

	int er_id = blockDim.x*blockIdx.x + threadIdx.x;

	while (er_id<N){

		int inj = er_id % cols;
		int ini = er_id / cols;

		if ((d_inf[er_id] == 1) || (d_inf[er_id] == 2)){
			if (d_h[er_id]<d_th[0]){

				if (ini<rows){
					if (d_um[er_id + cols] > 0){ d_um[er_id + cols] = 0; }
				}

				if (d_um[er_id] < 0.0){ d_um[er_id] = 0.00; }

				if (inj<cols){
					if (d_vn[er_id + 1] > 0){ d_vn[er_id + 1] = 0; }
				}

				if (d_vn[er_id] < 0){ d_vn[er_id] = 0; }

			}
		}

		er_id += gridDim.x * blockDim.x;
	}

}


int main()
{

	// Definition of integer type variables - scalar
	int dir_number, out_velocity_x, out_velocity_y, out_elevation, out_depth, out_outlet_on, dir_it, N, cols, rows, i, lpout, lkout, nst, n_out, outx, outy;

	// Definition of variables of type double - scalar
	double resolution, tday0, thour0, tmin0, tsec0, tday, thour, tmin, tsec, dkout, dpout, time0, \
		timmax, dt, dtoq, dt2, gg, manning_coef, dtrain, NaN;// , hmn, rr;// north, south, east, west;

	// Definition of string variables
	std::string dir_parameters, dirfile_setup, dir_DEM, dir_overQ, dir_rain, initi_cond, outlet_file, dir_temperature, dir_solar_radiation;
	std::string tempo;

	std::stringstream out;

	// Definition of processing variables in CPU - vector / array
	double *h_baseo, *h_h, *h_um, *h_hm, *h_uu1, *h_umo, \
		*h_vv1, *h_vva, *h_vn, *h_hn, *h_vno, *h_uua, **h_qq, \
		*h_rain, *h_ho, *h_uu, *h_vv, *h_ql, *h_rr, *h_th, *h_initial_condition;

	double *h_brx, *h_bry, *h_outx, *h_outy, xcoor, ycoor;
	int *h_infx, *h_infy;

	// Evaporation variables definition
	double albedo, INT, INF, LWL, EV_WL_min, *h_T, *h_Rg, *h_Evapo, *h_Ev;
	int evaporation_on;

	// Definition of processing variables in GPU - vector / array
	double *d_baseo, *d_h, *d_um, *d_hm, *d_uu1, *d_umo, \
		*d_vv1, *d_vva, *d_vn, *d_hn, *d_vno, *d_uua, *d_ho, *d_uu, *d_vv, *d_ql, *d_rr, *d_th;
	int *d_inf, *d_infx, *d_infy;// , *d_outlet;

	// Evaporation variables definition
	double *d_T, *d_Rg, *d_Rs, *d_pw, *d_lv, *d_Evapo, *d_Ev;

	double duration;

	int numBlocks;                     //Number of blocks
	int threadsPerBlock;               //Number of threads
	int maxThreadsPerBlock;

	char dirfile[4000];
	//*************************************************
	FILE *dir = fopen("db\\dir.txt", "r");
	fscanf(dir, " dir_number: %d\n", &dir_number);

	for (dir_it = 0; dir_it < dir_number; dir_it++){
		fscanf(dir, " %s\n", dirfile);
		dir_parameters = dirfile;

		FILE *file_setup;
		dirfile_setup = "db\\" + dir_parameters + "\\input\\setup.dat";
		file_setup = fopen(dirfile_setup.c_str(), "r");
		if (file_setup == NULL) {
			printf("unknown file - setup.dat\n");
			system("pause");
			return 0;
		}

		fscanf(file_setup, " tday0_thour0_tmin0_tsec0: %lf %lf %lf %lf\n ", &tday0, &thour0, &tmin0, &tsec0);
		fscanf(file_setup, " tday_thour_tmin_tsec: %lf %lf %lf %lf\n ", &tday, &thour, &tmin, &tsec);
		fscanf(file_setup, " dt: %lf\n ", &dt);
		fscanf(file_setup, " dpout: %lf\n ", &dpout);
		fscanf(file_setup, " dkout: %lf\n ", &dkout);
		fscanf(file_setup, " dtoq: %lf\n ", &dtoq);
		fscanf(file_setup, " evaporation_on: %d\n", &evaporation_on);
		fscanf(file_setup, " EV_WL_min: %lf\n ", &EV_WL_min);
		fscanf(file_setup, " INT: %lf\n ", &INT);
		fscanf(file_setup, " INF: %lf\n ", &INF);
		fscanf(file_setup, " LWL: %lf\n ", &LWL);
		fscanf(file_setup, " manning_coef: %lf\n ", &manning_coef);
		fscanf(file_setup, " out_velocity_x: %d\n ", &out_velocity_x);
		fscanf(file_setup, " out_velocity_y: %d\n ", &out_velocity_y);
		fscanf(file_setup, " out_elevation: %d\n ", &out_elevation);
		fscanf(file_setup, " out_depth: %d\n ", &out_depth);
		fscanf(file_setup, " out_outlet_on: %d\n ", &out_outlet_on);
		fclose(file_setup);

		// ******************************************************************
		dt2 = 2.00*dt;
		gg = 9.80;
		// ******************************************************************
		// ******************************************************************
		//                            Input MDT
		// ******************************************************************

		dir_DEM = "db\\" + dir_parameters + "\\input\\dem.txt";
		FILE *V_DEM = fopen(dir_DEM.c_str(), "r");
		if (V_DEM == NULL) {
			printf("unknown file - dem.txt\n");
			system("pause");
			return 0;
		}
		fscanf(V_DEM, " ncols %d\n", &cols);
		fscanf(V_DEM, " nrows %d\n", &rows);
		fscanf(V_DEM, " xllcorner %lf\n", &xcoor);
		fscanf(V_DEM, " yllcorner %lf\n", &ycoor);
		fscanf(V_DEM, " cellsize %lf\n", &resolution);
		fscanf(V_DEM, " NODATA_value %lf\n", &NaN);

		N = (rows)*(cols)+cols/2;

		h_baseo = (double*)malloc(N*sizeof(double));
		for (i = 0; i < N; i++){
			fscanf(V_DEM, "%lf\n", &h_baseo[i]);
		}
		fclose(V_DEM);
		// ****************************************************************
		//              Input coord source or sink
		// ****************************************************************

		FILE *file_coord_source;
		std::string dirfile_coord_source = "db\\" + dir_parameters + "\\input\\coord_source_sink.dat";
		file_coord_source = fopen(dirfile_coord_source.c_str(), "r");
		if (file_coord_source == NULL) {
			printf("unknown file - coord_source_sink.dat\n");
			system("pause");
			return 0;
		}
		fscanf(file_coord_source, "nst  %d/n ", &nst);
		h_brx = (double*)malloc(nst*sizeof(double));
		h_bry = (double*)malloc(nst*sizeof(double));
		for (int inst = 0; inst < nst; inst++){
			fscanf(file_coord_source, " %lf %lf\n ", &h_brx[inst], &h_bry[inst]);
			h_brx[inst] = h_brx[inst] / resolution;
			h_bry[inst] = h_bry[inst] / resolution;
		}
		fclose(file_coord_source);
		
		// ******************************************************************
		//                            Input outlet
		// ******************************************************************	
		/*
		if (out_outlet_on == 1){
		outlet_file = "db\\" + dir_parameters + "\\input\\outlet.dat";
		FILE *V_outlet = fopen(outlet_file.c_str(), "r");
		if (V_outlet == NULL) {
		printf("unknown file - outlet.dat\n");
		system("pause");
		return 0;
		}
		h_outlet = (int*)malloc(N*sizeof(int));
		for (i = 0; i < N; i++){
		fscanf(V_outlet, "%d\n", &h_outlet[i]);
		}
		fclose(V_outlet);
		}
		*/

		// ******************************************************************
		//                    Input source or sink
		// ******************************************************************
		int cont_qq = 0;
		int ch = 0;
		dir_overQ = "db\\" + dir_parameters + "\\input\\Q_source_sink.dat";
		FILE *V_dir_overQ = fopen(dir_overQ.c_str(), "r");
		if (V_dir_overQ == NULL) {
			printf("unknown file - Q_source_sink.dat\n");
			system("pause");
			return 0;
		}
		//Count the number of elements in the Q_source_sink.dat file
		while (!feof(V_dir_overQ))
		{
			ch = fgetc(V_dir_overQ);
			if (ch == '\n')
			{
				cont_qq++;
			}
		}
		fclose(V_dir_overQ);
		h_qq = (double**)malloc((cont_qq)*sizeof(double*));
		for (int iu = 0; iu < cont_qq; iu++){
			h_qq[iu] = (double*)malloc(nst*sizeof(double));
		}
		FILE *V1_dir_overQ = fopen(dir_overQ.c_str(), "r");
		char skip[10];
		fscanf(V1_dir_overQ, "flow %s\n", skip);
		for (int jq = 0; jq < cont_qq; jq++){
			for (int iq = 0; iq < nst; iq++){
				fscanf(V1_dir_overQ, " %lf\n", &h_qq[jq][iq]);
				/*
				if (iq==(nst-1)){
				printf(" %lf\n", h_qq[jq][iq]);
				}
				else{
				printf(" %lf", h_qq[jq][iq]);
				}
				*/
			}
		}
		fclose(V1_dir_overQ);

		//system("pause");
		// ****************************************************************
		//                  Input meteorological data
		// ****************************************************************

		//>>>>>>>>>>>>>>>>>>> Input Mensurement Rain <<<<<<<<<<<<<<<<<<<<<<
		int cont_rain = 0;
		int chr = 0;
		dir_rain = "db\\" + dir_parameters + "\\input\\rain.dat";
		FILE *V_dir_rain = fopen(dir_rain.c_str(), "r");
		if (V_dir_rain == NULL) {
			printf("unknown file - rain.dat\n");
			system("pause");
			return 0;
		}
		// Count the number of elements in the rain.dat file
		while (!feof(V_dir_rain))
		{
			chr = fgetc(V_dir_rain);
			if (chr == '\n')
			{
				cont_rain++;
			}
		}
		fclose(V_dir_rain);
		FILE *V1_dir_rain = fopen(dir_rain.c_str(), "r");
		char head1_rain[10], head2_rain[10], head3_rain[10], head4_rain[10];//, head5_rain[10], head6_rain[10], head7_rain[10];

		fscanf(V1_dir_rain, " %lf\n", &dtrain);
		fscanf(V1_dir_rain, " %s %s %s %s\n", head1_rain, head2_rain, head3_rain, head4_rain);// , head5_rain, head6_rain, head7_rain);
		h_rain = (double*)malloc((cont_rain - 1)*sizeof(double));
		std::string rtime;
		for (i = 0; i < (cont_rain - 1); i++){
			fscanf(V1_dir_rain, " %s %lf\n", &rtime, &h_rain[i]);
		}
		fclose(V1_dir_rain);

		//>>>>>>>>>> Input Temperatures and solar radiation data <<<<<<<<<<<<<<<<<
		h_Ev = (double*)malloc(sizeof(double));
		cudaMalloc((void**)&d_Ev, sizeof(double));
		h_Ev[0] = 0.00;
		cudaMemcpy(d_Ev, h_Ev, sizeof(double), cudaMemcpyHostToDevice);

		if (evaporation_on == 1){

			char head_solar_rad[30], head_temperature[20];

			h_Evapo = (double*)malloc((cont_rain - 1)*sizeof(double));
			h_T = (double*)malloc((cont_rain - 1)*sizeof(double));
			h_Rg = (double*)malloc((cont_rain - 1)*sizeof(double));

			cudaMalloc((void**)&d_T, (cont_rain - 1)*sizeof(double));
			cudaMalloc((void**)&d_Rg, (cont_rain - 1)*sizeof(double));
			cudaMalloc((void**)&d_Rs, (cont_rain - 1)*sizeof(double));
			cudaMalloc((void**)&d_pw, (cont_rain - 1)*sizeof(double));
			cudaMalloc((void**)&d_lv, (cont_rain - 1)*sizeof(double));
			cudaMalloc((void**)&d_Evapo, (cont_rain - 1)*sizeof(double));

			dir_temperature = "db\\" + dir_parameters + "\\input\\temperatures.txt";
			FILE *V_dir_temperature = fopen(dir_temperature.c_str(), "r");
			if (V_dir_temperature == NULL) {
				printf("unknown file - evaporation.dat\n");
				system("pause");
				return 0;
			}
			dir_solar_radiation = "db\\" + dir_parameters + "\\input\\solar_radiation.txt";
			FILE *V_dir_solar_radiation = fopen(dir_solar_radiation.c_str(), "r");
			if (V_dir_solar_radiation == NULL) {
				printf("unknown file - evaporation.dat\n");
				system("pause");
				return 0;
			}

			fscanf(V_dir_solar_radiation, " albedo: %lf\n", &albedo);
			fscanf(V_dir_solar_radiation, " %s\n", head_solar_rad);
			fscanf(V_dir_temperature, " %s\n", head_temperature);

			for (i = 0; i < (cont_rain - 1); i++){
				//>>>>>>>>>>>>>>>>>>>> Input Temperatures data <<<<<<<<<<<<<<<<<
				fscanf(V_dir_temperature, " %lf\n", &h_T[i]);
				//>>>>>>>>>>>>>>>>>>>> Input Solar Radiation data <<<<<<<<<<<<<<
				fscanf(V_dir_solar_radiation, " %lf\n", &h_Rg[i]);
			}
			cudaMemcpy(d_T, h_T, (cont_rain - 1)*sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Rg, h_Rg, (cont_rain - 1)*sizeof(double), cudaMemcpyHostToDevice);

			fclose(V_dir_temperature);
			fclose(V_dir_solar_radiation);

			//system("pause");

		}

		// *************************************************************
		//                    Input initial condition
		// *************************************************************

		initi_cond = "db\\" + dir_parameters + "\\input\\initial_condition.txt";
		FILE *V_initial_condition = fopen(initi_cond.c_str(), "r");
		if (V_initial_condition == NULL) {
			printf("unknown file - initial_condition.txt\n");
			system("pause");
			return 0;
		}
		h_initial_condition = (double*)malloc(N*sizeof(double));
		for (i = 0; i < N; i++){
			fscanf(V_initial_condition, "%lf\n", &h_initial_condition[i]);
		}
		fclose(V_initial_condition);


		// *************************************************************
		//                   output settings
		// *************************************************************
		lpout = round(dpout / dt);
		lkout = round(dkout / dt);

		// Save Results
		std::string dirRes_Output;
		FILE *WL_Output;
		std::string dirWaterLevel = "db\\" + dir_parameters + "\\output\\WaterLevel.txt";
		WL_Output = fopen(dirWaterLevel.c_str(), "w");

		if (out_outlet_on == 1){

			FILE *file_coord_out;
			std::string dirfile_coord_out = "db\\" + dir_parameters + "\\input\\coord_out.dat";
			file_coord_out = fopen(dirfile_coord_out.c_str(), "r");
			if (file_coord_out == NULL) {
				printf("unknown file - coord_out.dat\n");
				system("pause");
				return 0;
			}
			fscanf(file_coord_out, "n_out  %d\n ", &n_out);
			h_outx = (double*)malloc(n_out*sizeof(double));
			h_outy = (double*)malloc(n_out*sizeof(double));
			for (int in_out = 0; in_out < n_out; in_out++){
				fscanf(file_coord_out, " %lf %lf\n ", &h_outx[in_out], &h_outy[in_out]);
				h_outx[in_out] = h_outx[in_out] / resolution;
				h_outy[in_out] = h_outy[in_out] / resolution;
			}
			fclose(file_coord_out);
		}
		else{
			fprintf(WL_Output,"%s\n", "Outlet off");
		}

		//h_out = (double**)malloc((cont_rain - 1)*sizeof(double*));
		//for (int iu = 0; iu < cont_qq; iu++){
		//	h_out[iu] = (double*)malloc(n_out*sizeof(double));
		//}


		// *************************************************************
		//                  Time settings
		// *************************************************************
		time0 = 3600.0*24.0*tday0 + 3600.0*thour0 + 60.0*tmin0 + tsec0; // Initial time
		timmax = 3600.0*24.0*tday + 3600.0*thour + 60.0*tmin + tsec;    // Final time

		if ((tday == 0) & (thour == 0) & (tmin == 0) & (tsec == 0)){
			timmax = (cont_rain - 2)*dtrain;
		}

		// ***************************************************************
		// ***************************************************************

		printf(" %s\n", " ******************************************************************** ");
		printf(" %s\n", " Two-dimensional shallow water model accelerated by GPGPU (SW2D-GPU)  ");
		printf(" %s\n", " ******************************************************************** ");
		printf(" %s\n", " Month/Year - 11/2020  ");
		printf(" %s\n", " Developer of parallel code in GPGPU: ");
		printf(" %s\n", "     Tomas Carlotto         |   Code written in CUDA C/C++ ");
		printf(" %s\n", " ******************************************************************** ");
		printf(" %s\n", " ******************************************************************** ");
		printf(" %s\n", dirfile);


		// *************************************************************
		//              Memory Allocation - CPU
		// *************************************************************

		//h_inf = (int*)malloc(N*sizeof(int));
		h_infx = (int*)malloc(N*sizeof(int));
		h_infy = (int*)malloc(N*sizeof(int));
		//h_infsw = (int*)malloc(N*sizeof(int));

		h_h = (double*)malloc(N*sizeof(double));
		h_ho = (double*)malloc(N*sizeof(double));
		h_hm = (double*)malloc(N*sizeof(double));
		h_hn = (double*)malloc(N*sizeof(double));

		h_um = (double*)malloc(N*sizeof(double));
		h_umo = (double*)malloc(N*sizeof(double));
		h_uu = (double*)malloc(N*sizeof(double));
		h_uua = (double*)malloc(N*sizeof(double));
		h_uu1 = (double*)malloc(N*sizeof(double));

		h_vn = (double*)malloc(N*sizeof(double));
		h_vno = (double*)malloc(N*sizeof(double));
		h_vv = (double*)malloc(N*sizeof(double));
		h_vva = (double*)malloc(N*sizeof(double));
		h_vv1 = (double*)malloc(N*sizeof(double));

		h_ql = (double*)malloc(N*sizeof(double));
		h_rr = (double*)malloc(sizeof(double));
		h_th = (double*)malloc(sizeof(double));

		h_th[0] = 1.0e-4;

		int km = -1;
		for (int im = 0; im < rows; im++) {
			for (int jm = 0; jm < cols; jm++) {
				km = km + 1;

				h_um[km] = 0.00;
				h_umo[km] = 0.00;
				h_uu[km] = 0.00;
				h_uua[km] = 0.00;
				h_uu1[km] = 0.00;

				h_vn[km] = 0.00;
				h_vno[km] = 0.00;
				h_vv[km] = 0.00;
				h_vva[km] = 0.00;
				h_vv1[km] = 0.00;

				h_ql[km] = 0.000;

				h_h[km] = h_initial_condition[km];
				h_ho[km] = 0.00;
				h_hm[km] = 0.00;
				h_hn[km] = 0.00;

			}
		}

		// *************************************************************
		//                   Memory Allocation - GPU
		// *************************************************************
		cudaMalloc((void**)&d_inf, N*sizeof(int));
		cudaMalloc((void**)&d_infx, N*sizeof(int));
		cudaMalloc((void**)&d_infy, N*sizeof(int));

		cudaMalloc((void**)&d_h, N*sizeof(double));
		cudaMalloc((void**)&d_ho, N*sizeof(double));
		cudaMalloc((void**)&d_hm, N*sizeof(double));
		cudaMalloc((void**)&d_hn, N*sizeof(double));

		cudaMalloc((void**)&d_um, N*sizeof(double));
		cudaMalloc((void**)&d_umo, N*sizeof(double));
		cudaMalloc((void**)&d_uu, N*sizeof(double));
		cudaMalloc((void**)&d_uua, N*sizeof(double));
		cudaMalloc((void**)&d_uu1, N*sizeof(double));

		cudaMalloc((void**)&d_vn, N*sizeof(double));
		cudaMalloc((void**)&d_vno, N*sizeof(double));
		cudaMalloc((void**)&d_vv, N*sizeof(double));
		cudaMalloc((void**)&d_vva, N*sizeof(double));
		cudaMalloc((void**)&d_vv1, N*sizeof(double));

		cudaMalloc((void**)&d_baseo, N*sizeof(double));
		cudaMalloc((void**)&d_ql, N*sizeof(double));
		cudaMalloc((void**)&d_rr, sizeof(double));
		cudaMalloc((void**)&d_th, sizeof(double));

		cudaMemcpy(d_um, h_um, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_umo, h_umo, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_uu, h_uu, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_uua, h_uua, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_uu1, h_uu1, N*sizeof(double), cudaMemcpyHostToDevice);

		cudaMemcpy(d_vn, h_vn, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_vno, h_vno, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_vv, h_vv, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_vva, h_vva, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_vv1, h_vv1, N*sizeof(double), cudaMemcpyHostToDevice);

		cudaMemcpy(d_th, h_th, sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_h, h_h, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_ho, h_ho, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_hn, h_hn, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_hm, h_hm, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_infx, h_infx, N*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_infy, h_infy, N*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_baseo, h_baseo, N*sizeof(double), cudaMemcpyHostToDevice);


		// *******************************************************************
		// Definition of the number of blocks and threads for mesh (N cells)
		// *******************************************************************
		/*
		int iz;
		int jz;
		for (int sw = 0; sw < N;sw++){
			jz = sw % (cols);
			iz = sw / (cols);
			
		}
		printf("%d    %d   %d   %d\n", rows, cols, iz, jz);
		system("pause");
		*/

		cudaDeviceProp prop;
		int count;	cudaGetDeviceCount(&count);
		for (int i = 0; i < count; i++){
			cudaGetDeviceProperties(&prop, i);
			maxThreadsPerBlock = prop.maxThreadsPerBlock;
		}
		if (N < maxThreadsPerBlock){
			threadsPerBlock = N;
			numBlocks = (N + N - 1) / N;
		}
		else{
			threadsPerBlock = maxThreadsPerBlock;
			numBlocks = (N + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
		}

		double dx = resolution;
		double dy = resolution;
		init_inf << < numBlocks, threadsPerBlock >> >(rows, cols, d_ho, d_h, d_inf, d_baseo, N, NaN);
		cudaDeviceSynchronize();

		initiald << < numBlocks, threadsPerBlock >> >(rows, cols, d_h, d_infx, d_infy, d_inf, d_hm, d_hn, d_baseo, N, NaN);
		cudaDeviceSynchronize();
		if (evaporation_on == 1){
			gpu_evaporation_calc << <numBlocks, threadsPerBlock >> >(albedo, d_T, d_Rg, d_Rs, d_pw, d_lv, d_Evapo, dtrain, (cont_rain - 1));
			cudaMemcpy(h_Evapo, d_Evapo, (cont_rain - 1)*sizeof(double), cudaMemcpyDeviceToHost);
		}
		
		double time = time0;
		int mstep = 0;

		//Start time
		std::clock_t start;
		start = std::clock();

		int tq = -1;
		int out0 = -1;
		while (time + dt <= timmax){

			out0 = out0 + 1;

			if (mstep % lpout == 0){
				printf(" %d\n", int(time));
			}

			// ************************************************
			//           2D SHALLOW WATER CALCULATION
			// ************************************************
			
			flux <<< numBlocks, threadsPerBlock >>>(d_th, gg, manning_coef, d_inf, d_h, d_infx, d_infy, d_baseo, d_um, d_hm, d_uu1, \
				d_umo, d_vv1, d_vva, d_vn, d_hn, d_vno, d_uua, d_ho, N, cols, rows, dx, dy, dt2);			
			cudaDeviceSynchronize();

		    /*
	        cudaError_t cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "addKernel launch failed 1: %s\n", cudaGetErrorString(cudaStatus));
			}
			system("pause");
			*/

			// ************************************************
			//              CONTINUITY EQUATION
			// ************************************************

			stream_flow(cols, rows, (xcoor/resolution), (ycoor/resolution), time, dtrain, h_rain, h_qq, h_ql, dtoq, h_brx, h_bry, dx, dy, nst, h_rr);
			cudaMemcpy(d_ql, h_ql, N*sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(d_rr, h_rr, sizeof(double), cudaMemcpyHostToDevice);
			

			if (evaporation_on == 1){
				evaporation_load(time, dtrain, h_Evapo, h_Ev);
				cudaMemcpy(d_Ev, h_Ev, sizeof(double), cudaMemcpyHostToDevice);
			}
			

			continuity << < numBlocks, threadsPerBlock >> >(dt2, cols,rows, dx, dy, d_rr, d_Ev, d_ql, d_h, d_ho, d_um, d_vn, INT, INF, LWL, EV_WL_min, d_inf, N);
			cudaDeviceSynchronize();

	
			// ************************************************
			//                 ERROR TREATMENT
			// ************************************************
			treat_error << < numBlocks, threadsPerBlock >> >(cols, rows, d_th, d_inf, d_um, d_vn, d_h, N);
			cudaDeviceSynchronize();


			// time step **************************************
			time = time + dt;
			mstep = mstep + 1;
			//*************************************************
			//              CONTINUITY EQUATION
			//*************************************************
			
			stream_flow(cols, rows, (xcoor/resolution), (ycoor/resolution), time, dtrain, h_rain, h_qq, h_ql, dtoq, h_brx, h_bry, dx, dy, nst, h_rr);
			cudaMemcpy(d_ql, h_ql, N*sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(d_rr, h_rr, sizeof(double), cudaMemcpyHostToDevice);			

			if (evaporation_on == 1){
				evaporation_load(time, dtrain, h_Evapo, h_Ev);
				cudaMemcpy(d_Ev, h_Ev, sizeof(double), cudaMemcpyHostToDevice);
			}

			//system("pause");
						
			continuity << < numBlocks, threadsPerBlock >> >(dt2, cols,rows, dx, dy, d_rr, d_Ev, d_ql, d_h, d_ho, d_um, d_vn, INT, INF, LWL, EV_WL_min, d_inf, N);
			cudaDeviceSynchronize();
			
			//*************************************************
			//           PREPARING NEXT CALCULATION
			//*************************************************			
			hm_hn << < numBlocks, threadsPerBlock >> >(d_hm, d_hn, d_h, N, cols, rows);
			cudaDeviceSynchronize();	

			uu1_vv1 << < numBlocks, threadsPerBlock >> >(d_th, d_hm, d_hn, d_uu1, d_um, d_vv1, d_vn, N, cols, rows);
			cudaDeviceSynchronize();	
           
			uu_vv << < numBlocks, threadsPerBlock >> >(d_th, d_h, d_uu1, d_vv1, d_uu, d_vv, N, cols);
			cudaDeviceSynchronize();
			
			uua_vva << < numBlocks, threadsPerBlock >> >(d_uu1, d_vv1, d_uua, d_vva, N, cols, rows);
			cudaDeviceSynchronize();			
			
			//**************************************************
			//                   FORWARD
			//**************************************************
			forward << < numBlocks, threadsPerBlock >> >(cols, rows, d_umo, d_um, d_vno, d_vn, d_ho, d_h, N);
			cudaDeviceSynchronize();
			//**************************************************
			
			time = time + dt;
			mstep = mstep + 1;
			/*
			if (mstep % (lkout) == 0){
				//Time
				duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
				std::cout << "printf: " << duration << '\n';

				// Save Times
				FILE *TimeOutput;
				//std::string dirTimeOutput = "db\\" + dir_parameters + "\\output\\TimeSimu_" + tempo + ".txt";
				std::string dirTimeOutput = "db\\" + dir_parameters + "\\output\\TimeSimu_" + "t" + ".txt";
				TimeOutput = fopen(dirTimeOutput.c_str(), "w");
				fprintf(TimeOutput, " %lf\n", duration);
				fclose(TimeOutput);
			}
			*/
			//output
			if ((out_depth == 1) || (out_velocity_x == 1) || (out_velocity_y == 1) || (out_elevation == 1)){
				if ((mstep % (lkout) == 0) || (out0 == 0)){
					tq = tq + 1;

					if (out0 > 0){
						out << round(mstep*dt / dkout);
						tempo = out.str();
						out.str("");

					}
					else{
						out << 0;
						tempo = out.str();
						out.str("");
					}


					FILE *Res_Output;
					dirRes_Output = "db\\" + dir_parameters + "\\output\\Results_" + tempo + ".vtk";
					Res_Output = fopen(dirRes_Output.c_str(), "w");

					// output .vtk format

					fprintf(Res_Output, "%s\n", "# vtk DataFile Version 2.0");
					fprintf(Res_Output, "%s\n", "Brazil");
					fprintf(Res_Output, "%s\n", "ASCII");
					fprintf(Res_Output, "%s\n", "DATASET STRUCTURED_POINTS");
					fprintf(Res_Output, "DIMENSIONS %d %d %d\n", cols, rows, 1);
					fprintf(Res_Output, "ASPECT_RATIO %lf %lf %lf\n", dx, dy, 1.0000);
					fprintf(Res_Output, "ORIGIN %lf %lf %lf\n", xcoor, ycoor, 0.000);
					fprintf(Res_Output, "POINT_DATA %d\n", cols*rows);
					int posxy;
					int npout = 0;
					//                 Water depth
					if (out_depth == 1){
						cudaMemcpy(h_h, d_h, N*sizeof(double), cudaMemcpyDeviceToHost);
						fprintf(Res_Output, "%s\n", "SCALARS Depth float 1");
						fprintf(Res_Output, "%s\n", "LOOKUP_TABLE default");
						int km = -1;
						for (int im = 0; im < rows; im++) {
							posxy = rows*cols - (im + 1)*cols;
							posxy = posxy - 1;
							for (int jm = 0; jm < cols; jm++) {
								km = km + 1;
								posxy = posxy + 1;
								if (out_outlet_on == 1){
									//***********************************************								
									for (int i = 0; i < n_out; i++){
										outx = round(abs((xcoor / resolution) - h_outx[i]));
										outy = rows - round(abs((ycoor / resolution) - h_outy[i]));

										if (km == ((outy)*cols - (cols - (outx + 1)))){
											npout = npout + 1;
											if (out0 == 0){
												if (npout == n_out){
													fprintf(WL_Output, " %lf\n", h_initial_condition[km]);
												}
												else{
													fprintf(WL_Output, " %lf", h_initial_condition[km]);
												}
											}
											else{
												if (npout == n_out){
													fprintf(WL_Output, " %lf\n", h_h[km]);
												}
												else{
													fprintf(WL_Output, " %lf", h_h[km]);
												}
											}
										}
									}
									//***********************************************
								}

								if (out0 == 0){
									fprintf(Res_Output, "%f\n", h_initial_condition[posxy]);
								}
								else{
									fprintf(Res_Output, "%f\n", h_h[posxy]);
								}
							}
						}
					}
					//                Velocity x direction
					if (out_velocity_x == 1){
						cudaMemcpy(h_vv, d_vv, N*sizeof(double), cudaMemcpyDeviceToHost);
						fprintf(Res_Output, "%s\n", "SCALARS x_velocity float 1");
						fprintf(Res_Output, "%s\n", "LOOKUP_TABLE default");
						km = -1;
						for (int im = 0; im < rows; im++) {
							posxy = rows*cols - (im + 1)*cols;
							posxy = posxy - 1;
							for (int jm = 0; jm < cols; jm++) {
								km = km + 1;
								posxy = posxy + 1;
								fprintf(Res_Output, "%f\n", h_vv[posxy]);
							}
						}
					}
					//                Velocity y direction
					if (out_velocity_y == 1){
						cudaMemcpy(h_uu, d_uu, N*sizeof(double), cudaMemcpyDeviceToHost);
						fprintf(Res_Output, "%s\n", "SCALARS y_velocity float 1");
						fprintf(Res_Output, "%s\n", "LOOKUP_TABLE default");
						km = -1;
						for (int im = 0; im < rows; im++) {
							posxy = rows*cols - (im + 1)*cols;
							posxy = posxy - 1;
							for (int jm = 0; jm < cols; jm++) {
								km = km + 1;
								posxy = posxy + 1;
								fprintf(Res_Output, "%f\n", -h_uu[posxy]);
							}
						}
					}
					//                 Elevations
					if (out_elevation == 1){
						cudaMemcpy(h_baseo, d_baseo, N*sizeof(double), cudaMemcpyDeviceToHost);
						fprintf(Res_Output, "%s\n", "SCALARS Elevations float 1");
						fprintf(Res_Output, "%s\n", "LOOKUP_TABLE default");
						km = -1;
						for (int im = 0; im < rows; im++) {
							posxy = rows*cols - (im + 1)*cols;
							posxy = posxy - 1;
							for (int jm = 0; jm < cols; jm++) {
								km = km + 1;
								posxy = posxy + 1;
								fprintf(Res_Output, "%f\n", h_baseo[posxy]);
							}
						}
					}
					fclose(Res_Output);
				}
			}
		}

		//Time
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "printf: " << duration << '\n';

		// Save Times
		FILE *TimeOutput;
		std::string dirTimeOutput = "db\\" + dir_parameters + "\\output\\TimeSimu_" + tempo + ".txt";
		TimeOutput = fopen(dirTimeOutput.c_str(), "w");
		fprintf(TimeOutput, " %lf\n", duration);
		fclose(TimeOutput);
		fclose(WL_Output);

		// Cleaning Up (GPU memory)
		if (evaporation_on == 1){
			cudaFree(d_Ev);
			cudaFree(d_T);
			cudaFree(d_Rg);
			cudaFree(d_Rs);
			cudaFree(d_pw);
			cudaFree(d_lv);
			cudaFree(d_Evapo);
		}
		cudaFree(d_inf);
		cudaFree(d_infx);
		cudaFree(d_infy);
		cudaFree(d_h);
		cudaFree(d_ho);
		cudaFree(d_hm);
		cudaFree(d_hn);
		cudaFree(d_um);
		cudaFree(d_umo);
		cudaFree(d_uu);
		cudaFree(d_uua);
		cudaFree(d_uu1);
		cudaFree(d_vn);
		cudaFree(d_vno);
		cudaFree(d_vv);
		cudaFree(d_vva);
		cudaFree(d_vv1);
		cudaFree(d_baseo);
		cudaFree(d_ql);
		//cudaFree(d_rr);
		//cudaFree(d_th);

	}

	return 0;

}