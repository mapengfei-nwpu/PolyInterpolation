#include "PolynomialInterpolation.h"

int main(){
    double coordinates_dof[] = {
        -0.92763589, -0.8593514, -0.68097846, -0.81305826,
        -0.16811177, -0.15810367, -0.14980267, -0.3576369,
        -0.31854147, -0.47776737, -0.21945887, -0.44912451
    };

    double point_out[12]={0};
    double point_in[12]={
        -0.8106048443681848, -0.1991376661554919, -0.3078406152100412,
        -0.8030754195199337, -0.20805803576948978, -0.34339014701082765,
        -0.7789952615535772, -0.2040375962048887, -0.30148065258529005,
        -0.750647609202964, -0.17414982786948008, -0.2793393260689801};
    
    double dofs[10] = {-1.41428913, -1.49522244, -1.05024, -1.61981967, -1.33502984, -1.55752106, -1.27273122, -1.5170544, -1.23226457, -1.45475579};
    
    size_t num_points = 4;
    size_t value_size = 1;
    double results[1] = {0.0};
    
    PolynomialInterpolation pit;
    pit.transform_points(point_out,point_in,coordinates_dof,num_points);
    pit.evaluate_function_at_points(dofs,10,point_out,num_points,value_size,results);
    for (size_t i = 0; i < num_points; i++){
        for (size_t j = 0; j < 3; j++){      
           std::cout<<point_in[i*3+j]<<"  ";
        }
        std::cout<<std::endl;
    }
    for (size_t i = 0; i < num_points; i++){
        for (size_t j = 0; j < value_size; j++){      
           std::cout<<results[i]<<"  ";
        }
        std::cout<<std::endl;
    }
    return 0;
}