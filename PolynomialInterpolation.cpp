#include "PolynomialInterpolation.h"



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

    PolynomialInterpolation pnit;
    pnit.linear_transformation(p,F,b);
    pnit.point_local_to_ref(point_out, point_in, F, b);

    std::cout<<point_out[0]<<std::endl;
    std::cout<<point_out[1]<<std::endl;
    std::cout<<point_out[2]<<std::endl;


    return 0;
}