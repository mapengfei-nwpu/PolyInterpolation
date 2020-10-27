#include "PolynomialInterpolation.h"

int main(){
    double coordinates_dof[] = {
        -0.29818503,0.32396691,-0.30418461,
        -0.27370655,0.41245756,-0.09687491,
        -0.54348923,0.44764928,-0.17334703,
        -0.45002428,0.19606839,-0.09479055
    };

    double point_out[12]={0};
    double point_in[12]={
        -0.31155828132356245,0.36252222425410413,-0.1215027206568566,
        -0.3874348560848294,0.39428857502565634,-0.16709429556821262,
        -0.2997841286509151,0.3803884230947467,-0.09744683032775989,
        -0.36582137367226497,0.37177520002864645,-0.16656906336246266
        };
    
    double dofs[10] = {0.28639715, 0.25442126, 0.52581962, 0.24994991, 0.36833483, 0.23270645, 0.37015316, 0.24735877, 0.3829609,  0.25755743};
    
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