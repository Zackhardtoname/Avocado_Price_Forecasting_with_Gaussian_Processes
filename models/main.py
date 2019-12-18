import sklearn.gaussian_process.kernels as Kernels
from avocado.models.gp import run_gp


def main():
    # Specify type_ and region of data
    type_ = 'organic' 
    region = 'WestTexNewMexico'

    # Specify the kernel functions; please see the paper for the rationale behind the choices
    kernel = Kernels.ExpSineSquared(20., periodicity=358., periodicity_bounds=(1e-2, 1e8)) \
        + 0.8 * Kernels.RationalQuadratic(alpha=20., length_scale=80.) \
        + Kernels.WhiteKernel(1e2)


    # Fit gp model and plot
    run_gp(kernel, n_restarts_optimizer=10, type_=type_, region=region)


    # + Kernels.ExpSineSquared(20., periodicity=158., periodicity_bounds=(1e-2, 1e8)) \
    # + Kernels.ExpSineSquared(20., periodicity=79., periodicity_bounds=(1e-2, 1e8)) \
    # + Kernels.ExpSineSquared(20., periodicity=30., periodicity_bounds=(1e-2, 1e8)) \



if __name__ == '__main__':
    main()
