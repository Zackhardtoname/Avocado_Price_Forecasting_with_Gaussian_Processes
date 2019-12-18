import sklearn.gaussian_process.kernels as Kernels
from avocado.models.gp import run_gp


def main():
    # Specify type_ and region of data
    type_ = 'organic' 
    region = 'WestTexNewMexico'

    # Specify the kernel functions; please see the paper for the rationale behind the choices
    kernel = Kernels.ExpSineSquared(length_scale=20., periodicity=365.) \
        + 0.8 * Kernels.RationalQuadratic(alpha=20., length_scale=80.) \
        + Kernels.WhiteKernel(.2)

    # Fit gp model and plot
    run_gp(kernel, n_restarts_optimizer=10, type_=type_, region=region)

if __name__ == '__main__':
    main()
