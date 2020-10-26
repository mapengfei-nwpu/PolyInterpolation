#include <iostream>
void linear_transformation(double* p, double** F, double* b) {
	
	///**************************************************
	///        point1    point2    point3    point4
	///
	/// x  =    x[0]      x[1]      x[2]      x[3]
	/// y  =    y[0]      y[1]      y[2]      y[3]
	/// z  =    z[0]      z[1]      z[2]      z[3]
	///
	///**************************************************
	double* x = p;
	double* y = &(p[4]);
	double* z = &(p[8]);

	double det = 
		  x[0] * y[1] * z[2] - x[0] * y[2] * z[1] - x[1] * y[0] * z[2] 
		+ x[1] * y[2] * z[0] + x[2] * y[0] * z[1] - x[2] * y[1] * z[0] 
		- x[0] * y[1] * z[3] + x[0] * y[3] * z[1] + x[1] * y[0] * z[3] 
		- x[1] * y[3] * z[0] - x[3] * y[0] * z[1] + x[3] * y[1] * z[0] 
		+ x[0] * y[2] * z[3] - x[0] * y[3] * z[2] - x[2] * y[0] * z[3] 
		+ x[2] * y[3] * z[0] + x[3] * y[0] * z[2] - x[3] * y[2] * z[0] 
		- x[1] * y[2] * z[3] + x[1] * y[3] * z[2] + x[2] * y[1] * z[3] 
		- x[2] * y[3] * z[1] - x[3] * y[1] * z[2] + x[3] * y[2] * z[1];

	F[0][0] =   (y[0] * z[2] - y[2] * z[0] - y[0] * z[3] + y[3] * z[0] + y[2] * z[3] - y[3] * z[2]) / det;
	F[0][1] =  -(x[0] * z[2] - x[2] * z[0] - x[0] * z[3] + x[3] * z[0] + x[2] * z[3] - x[3] * z[2]) / det;
	F[0][2] =   (x[0] * y[2] - x[2] * y[0] - x[0] * y[3] + x[3] * y[0] + x[2] * y[3] - x[3] * y[2]) / det;

	F[1][0] =  -(y[0] * z[1] - y[1] * z[0] - y[0] * z[3] + y[3] * z[0] + y[1] * z[3] - y[3] * z[1]) / det;
	F[1][1] =   (x[0] * z[1] - x[1] * z[0] - x[0] * z[3] + x[3] * z[0] + x[1] * z[3] - x[3] * z[1]) / det;
	F[1][2] =  -(x[0] * y[1] - x[1] * y[0] - x[0] * y[3] + x[3] * y[0] + x[1] * y[3] - x[3] * y[1]) / det;

	F[2][0] =   (y[0] * z[1] - y[1] * z[0] - y[0] * z[2] + y[2] * z[0] + y[1] * z[2] - y[2] * z[1]) / det;
	F[2][1] =  -(x[0] * z[1] - x[1] * z[0] - x[0] * z[2] + x[2] * z[0] + x[1] * z[2] - x[2] * z[1]) / det;
	F[2][2] =   (x[0] * y[1] - x[1] * y[0] - x[0] * y[2] + x[2] * y[0] + x[1] * y[2] - x[2] * y[1]) / det;

	b[0] = - F[0][0]*x[0] - F[0][1]*y[0] - F[0][2]*z[0];
	b[1] = - F[1][0]*x[0] - F[1][1]*y[0] - F[1][2]*z[0];
	b[2] = - F[2][0]*x[0] - F[2][1]*y[0] - F[2][2]*z[0];
}

void evaluate_basis_at_point(double* point_out, double* point_in, double** F, double* b) {

	point_out[0] = - F[0][0]*point_in[0] - F[0][1]*point_in[1] - F[0][2]*point_in[2];
	point_out[1] = - F[1][0]*point_in[0] - F[1][1]*point_in[1] - F[1][2]*point_in[2];
	point_out[2] = - F[2][0]*point_in[0] - F[2][1]*point_in[1] - F[2][2]*point_in[2];
	
	point_out[0] += b[0];
	point_out[1] += b[1];
	point_out[2] += b[2];
}

int main(){
    std::cout<<"Hello world!"<<std::endl;
    return 0;
}