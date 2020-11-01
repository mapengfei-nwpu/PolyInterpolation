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
        values[1] = 0.0;
        values[2] = 0.0;
    }
};

void get_gauss_rule(
    std::shared_ptr<const Function> function,
    std::vector<double> &gauss_points,
    std::vector<double> &function_dofs,
    std::vector<double> &coordinate_dofs,
    size_t &num_gauss)
{
    auto order = 3;
    auto mesh = function->function_space()->mesh();
    auto dim = mesh->topology().dim();

    // Construct Gauss quadrature rule
    SimplexQuadrature gauss_quadrature(dim, order);

    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
        // Create ufc_cell associated with dolfin cell.
        ufc::cell ufc_cell;
        cell->get_cell_data(ufc_cell);

        // Compute quadrature rule for the cell.
        auto quadrature_rule = gauss_quadrature.compute_quadrature_rule(*cell);
        assert(quadrature_rule.second.size() == quadrature_rule.first.size() / 3);
        num_gauss = quadrature_rule.second.size();
    }
}

int main(){
    auto mesh = std::make_shared<Mesh>(UnitCubeMesh::create({{32, 32, 32}}, CellType::Type::tetrahedron));
    auto V = std::make_shared<TetPoisson::FunctionSpace>(mesh);
    auto in_flow = std::make_shared<Inflow>();
    auto f = std::make_shared<Function>(V);

    size_t num_cells;
    size_t num_gauss;
    size_t value_size;
    size_t num_dofs;
	std::vector<double> coordinate_dofs;
    std::vector<double> function_dofs;
    std::vector<double> gauss_points;
    std::vector<double> results;
    //get_gauss_rule()
    get_gauss_rule(f,gauss_points,function_dofs,coordinate_dofs,num_gauss);

    std::cout<<num_gauss<<std::endl;


    return 0;
}