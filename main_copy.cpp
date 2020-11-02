#include"PolynomialInterpolation.h"
#include "Myio.h"

int main(){

    size_t num_gauss = 5;           // the number of gauss points in every cell.
    size_t num_cells;           // the number of cells totally.
    size_t value_size;          // the value of every function.
    size_t num_dofs;            // the dofs of a function on every cell.
	std::vector<double> coordinates;
    std::vector<double> function_dofs;
    std::vector<double> gauss_points;
    std::vector<double> gauss_weights;
    std::vector<double> results;

    input("coordinates.txt", coordinates);
    input("function_dofs.txt", function_dofs);
    input("gauss_points.txt", gauss_points);
    input("gauss_weights.txt", gauss_weights);

    num_cells = gauss_points.size()/num_gauss/3;
    value_size = 3;
    num_dofs = function_dofs.size()/num_cells;

    std::cout<<num_dofs<<std::endl;

    
    results.resize(gauss_points.size()/3*value_size);
    PolynomialInterpolation pli;
    pli.evaluate_function(num_cells,num_gauss,value_size,num_dofs,coordinates.data(),
                          function_dofs.data(),gauss_points.data(),results.data());
    
    for(size_t i = 0; i < 100; i++){
        std::cout << gauss_points[i] << " : " << results[i] << std::endl;
    }    

    return 0;
}