#include <iostream>
#include <eigen3/Eigen/Eigen>
#include <assert.h>
void linear_transformation(double* p, double F[3][3], double* b) {
	
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

void point_local_to_ref(double* point_out, double* point_in, double F[3][3], double* b) {

	point_out[0] = - F[0][0]*point_in[0] - F[0][1]*point_in[1] - F[0][2]*point_in[2];
	point_out[1] = - F[1][0]*point_in[0] - F[1][1]*point_in[1] - F[1][2]*point_in[2];
	point_out[2] = - F[2][0]*point_in[0] - F[2][1]*point_in[1] - F[2][2]*point_in[2];
	
	point_out[0] += b[0];
	point_out[1] += b[1];
	point_out[2] += b[2];
}

void evaluate_basis_at_point(double* point, double* basis){
    basis[0] = 1.0;
    basis[1] = point[0];
    basis[2] = point[1];
    basis[3] = point[2];
    basis[4] = point[0]*point[1];
    basis[5] = point[0]*point[2];
    basis[6] = point[1]*point[2];
    basis[7] = point[0]*point[0];
    basis[8] = point[1]*point[1];
    basis[9] = point[2]*point[2];
}

void ref_basis_matrix(double second_tetrahedron_dof_points[10][3],double G_inv[10][10]){
    double G[10][10]={{0}};
    for (size_t i = 0; i < 10; i++){
        evaluate_basis_at_point(second_tetrahedron_dof_points[i],G[i]);
    }
    // G_inv = G^{-1}
    Eigen::MatrixXd G_eigen(10,10);
    for (size_t i = 0; i < 10; i++)
        for (size_t j = 0; j < 10; j++)
            G_eigen(i,j)=G[i][j];
    
    Eigen::MatrixXd G_inv_eigen = G_eigen.inverse();
    
    for (size_t i = 0; i < 10; i++)
        for (size_t j = 0; j < 10; j++)
            G_inv[i][j]=G_inv_eigen(i,j);
}


void evaluate_function_at_points(double G_inv[10][10], double* dof, size_t num_dof, double* points, size_t num_points, size_t value_size, double* results){
    assert(num_dof%value_size == 0);
    assert(num_dof%10 == 0);
    double** parameters = (double**)malloc(sizeof(double*)*value_size);
    for (size_t i = 0; i < value_size; i++){
        parameters[i] = (double*)malloc(sizeof(double)*10);
        for (size_t j = 0; j < 10; j++)
        {
            parameters[i][j] = 0.0;
            for (size_t k = 0; k < 10; k++)
            {
                parameters[i][j] += G_inv[j][k]*dof[value_size*k+i];
            }
        }
    }

    for(size_t i = 0; i < num_points; i++){
        for (size_t j = 0; j < value_size; j++){
            double result = 0.0;
            double basis[10] = {0};
            evaluate_basis_at_point(&(points[3*i]), basis);
            for (size_t k = 0; k < 10; k++){
                result += parameters[j][k] * basis[k];
            }
            results[value_size*i+j] = result;
        }
    }
    for (size_t i = 0; i < value_size; i++)
        free(parameters[i]);
    free(parameters);
}


int main(){
    std::cout<<"Hello world!"<<std::endl;
    double p[] = {
        0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    };
    double F[3][3]={{0}};
    double b[3]={0};
    double point_in[3]={0.4,0.3,0.1};
    double point_out[3]={0};

    linear_transformation(p,F,b);
    point_local_to_ref(point_out, point_in, F, b);

    std::cout<<point_out[0]<<std::endl;
    std::cout<<point_out[1]<<std::endl;
    std::cout<<point_out[2]<<std::endl;


    return 0;
}