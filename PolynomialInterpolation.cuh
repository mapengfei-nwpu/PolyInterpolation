extern "C" 
{
    void setCudaG_inv(double *G_inv);
    void outputG_inv();
    
    void transform_points_all(
        size_t num_cells,
        size_t num_gauss,
        double *coordinates_host,
        double *gauss_points_host,
        double *gauss_points_ref_host
    );
}