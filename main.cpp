#include<dolfin.h>
#include <dolfin/geometry/SimplexQuadrature.h>
#include"PolynomialInterpolation.h"
#include"TetPoisson.h"

using namespace dolfin;

// Function for inflow boundary condition for velocity
class Inflow : public Expression
{
public:
    Inflow() : Expression(3) {}

    void eval(Array<double> &values, const Array<double> &x) const
    {
        values[0] = x[0];
        values[1] = x[1];
        values[2] = x[2];
    }
};

void get_gauss_rule(
    std::shared_ptr<const Function> function,
    std::vector<double> &gauss_points,
    std::vector<double> &gauss_weights,
    std::vector<double> &function_dofs,
    std::vector<double> &coordinates,
    size_t &num_gauss)
{
    auto order = 3;
    auto mesh = function->function_space()->mesh();
    auto dim = mesh->topology().dim();
    auto element = function->function_space()->element();
    auto dofmap  = function->function_space()->dofmap();

    // Construct Gauss quadrature rule
    SimplexQuadrature gauss_quadrature(dim, order);

    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
        // Create ufc_cell associated with dolfin cell.
        ufc::cell ufc_cell;
        cell->get_cell_data(ufc_cell);

        // Compute quadrature rule for the cell.
        // push back gauss points and gauss weights.
        auto quadrature_rule = gauss_quadrature.compute_quadrature_rule(*cell);
        assert(quadrature_rule.second.size() == quadrature_rule.first.size() / 3);
        num_gauss = quadrature_rule.second.size();

        gauss_points.insert(gauss_points.end(),quadrature_rule.first.begin(),quadrature_rule.first.end());
        gauss_weights.insert(gauss_weights.end(),quadrature_rule.second.begin(),quadrature_rule.second.end());
        
        // push back function dofs on the cell.
        auto dofs = dofmap->cell_dofs(cell->index());
        std::vector<double> cell_function_dofs(dofs.size());
        function->vector()->get(cell_function_dofs.data(), dofs.size(), dofs.data());
        function_dofs.insert(function_dofs.end(),cell_function_dofs.begin(),cell_function_dofs.end());

        // push back cell coordinates.
        std::vector<double> cell_coordinates;
        cell->get_vertex_coordinates(cell_coordinates);
        coordinates.insert(coordinates.end(),cell_coordinates.begin(),cell_coordinates.end());
    }
}

int main(){
    auto mesh = std::make_shared<Mesh>(UnitCubeMesh::create({{32, 32, 32}}, CellType::Type::tetrahedron));
    auto V = std::make_shared<TetPoisson::FunctionSpace>(mesh);
    auto in_flow = std::make_shared<Inflow>();
    auto f = std::make_shared<Function>(V);
    f->interpolate(*in_flow);

    size_t num_gauss;           // the number of gauss points in every cell.
    size_t num_cells;           // the number of cells totally.
    size_t value_size;          // the value of every function.
    size_t num_dofs;            // the dofs of a function on every cell.
	std::vector<double> coordinates;
    std::vector<double> function_dofs;
    std::vector<double> gauss_points;
    std::vector<double> gauss_weights;
    std::vector<double> results;
    
    get_gauss_rule(f,gauss_points,gauss_weights,function_dofs,coordinates,num_gauss);

    num_cells = gauss_points.size()/num_gauss/3;
    value_size = f->value_size();
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