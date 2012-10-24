
import numpy as np
import pyfluids
from pyfish import boundary, driving, gravity


class TestProblem(object):
    tfinal = 1.0
    lower_bounds = [-0.5, -0.5, -0.5]
    upper_bounds = [+0.5, +0.5, +0.5]
    resolution = [128]
    plot_fields = ['rho', 'pre', 'vx']

    def __init__(self, **kwargs):
        for k,v in kwargs.iteritems():
            setattr(self, k, v)
        self._descr = pyfluids.FluidDescriptor(fluid=self.fluid,
                                               gamma=self.gamma)

    @property
    def fluid_descriptor(self):
        return self._descr

    def build_boundary(self, mara):
        return boundary.Outflow()

    def ginit(self, x, y, z):
        return 0.0


class OneDimensionalUpsidedownGaussian(TestProblem):
    '''
    A good test for test-fluid gravity. Should only work when gravity is
    implemented as a momentum and energy source term, i.e. when $del^2 phi =
    rho$ does not have to be satisfied. Sets up an upside-down Gaussian
    potential well with a pressure profile in hydrostatic equillibrium.
    '''
    fluid = 'gravs'
    gamma = 1.4
    sig = 0.05
    sie = 2.00
    ph0 = 1.0

    def pinit(self, x, y, z):
        phi = self.ginit(x, y, z)[0]
        e0 = self.sie
        D0 = 1.0
        rho = D0 * np.exp(-phi / (e0 * (self.gamma - 1.0)))
        pre = rho * e0 * (self.gamma - 1.0)
        return [rho, pre, 0.0, 0.0, 0.0]

    def ginit(self, x, y, z):
        phi = -self.ph0 * np.exp(-0.5 * x**2 / self.sig**2)
        gph = -x/self.sig**2 * phi
        return [phi, gph, 0.0, 0.0]

    def build_boundary(self, mara):
        ng = mara.number_guard_zones()
        return boundary.Inflow(mara.fluid[0:ng], mara.fluid[-ng:])


class OneDimensionalPolytrope(TestProblem):
    '''
    Provides the initial conditions of a 1d polytrope for Gamma=2. Pressure
    density satisfy Poisson equation and hydrostatic equillibrium.
    '''
    fluid = 'grave' # or gravp, grave
    gamma = 2.0
    Dc = 1.0 # central density
    Da = 1e-2 # atmosphere density
    Pa = 1e-3 # atmosphere pressure
    R = 0.25
    four_pi_G = 1.0
    plot_fields = ['rho', 'pre', 'phi', 'gph']
    pauls_fix = True

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        if self.selfgrav:
            self.poisson_solver = gravity.PoissonSolver1d()

    def pinit(self, x, y, z):
        R = self.R
        K = 0.5 * R**2 / np.pi**2
        if abs(x) < R/2:
            rho = self.Dc * np.cos(np.pi * x / R)
            pre = K * rho**2.0
        else:
            if self.pauls_fix:
                rho = 0.0
                pre = 0.0
            else:
                rho = self.Da
                pre = K * rho**2.0
        if self.pauls_fix:
            if rho < self.Da: rho = self.Da
            if pre < self.Pa: pre = self.Pa
        return [rho, pre, 0.0, 0.0, 0.0]

    def ginit(self, x, y, z):
        return [0.0, 0.0, 0.0, 0.0]

    def build_boundary(self, mara):
        return boundary.Periodic()


class PeriodicDensityWave(TestProblem):
    '''
    Gives a stationary density wave without pressure variation.
    '''
    fluid = 'nrhyd'
    gamma = 1.4
    p0 = 1.00 # background pressure
    D0 = 10.00 # background density
    D1 = 1.00 # density fluctuation
    v0 = 0.00 # velocity of the wave
    n0 = 4 # integer valued wave-number
    plot_fields = ['rho', 'vx']

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        if self.fluid in ['gravs', 'gravp', 'grave']:
            self.plot_fields.append('phi')
            self.poisson_solver = gravity.PoissonSolver1d()
            self.fluid_descriptor.rhobar = self.D0

    def pinit(self, x, y, z):
        L = self.upper_bounds[0] - self.lower_bounds[0]
        rho = self.D0 + self.D1 * np.cos(2 * self.n0 * np.pi * x / L)
        return [rho, self.p0, self.v0, 0.0, 0.0]

    def ginit(self, x, y, z):
        return [0.0, 0.0, 0.0, 0.0]

    def build_boundary(self, mara):
        return boundary.Periodic()


class BrioWuShocktube(TestProblem):
    fluid = 'nrhyd'
    gamma = 1.4
    tfinal = 0.2
    geometry = 'planar'
    direction = 'x'

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        if len(self.resolution) == 2:
            self.plot_fields.append('vy')

    def pinit(self, x, y, z):
        if self.geometry == 'planar':
            r = {'x': x, 'y': y, 'z': z}[self.direction]
            R = 0.0
        elif self.geometry == 'cylindrical':
            r = (x**2 + y**2)**0.5
            R = 0.125
        elif self.geometry == 'spherical':
            r = (x**2 + y**2 + z**2)**0.5
            R = 0.125
        else:
            raise ValueError("invalid problem geometry: %s" % self.geometry)

        if r > R:
            return [0.125, 0.100, 0.0, 0.0, 0.0]
        else:
            return [1.000, 1.000, 0.0, 0.0, 0.0]


class DrivenTurbulence2d(TestProblem):
    fluid = 'nrhyd'
    gamma = 1.4
    tfinal = 1.0
    resolution = [128, 128]
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.driving = driving.DrivingModule2d(self.resolution)

    def pinit(self, x, y, z):
        return [1.0, 1.0, 0.0, 0.0, 0.0]

    def build_boundary(self, mara):
        return boundary.Periodic()


def polytrope3d(x, y, z):
    rho_c = 1.0    # central density
    rho_f = 1.0e-3 # floor (atmospheric) density
    G = 1.0        # gravitational constant
    b = 0.3        # beta, stellar radius
    a = b / np.pi  # alpha
    n = 1.0        # polytropic index
    K = 4*np.pi*G * a**2 / ((n + 1) * rho_c**(1.0/n - 1.0))
    r = (x**2 + y**2 + z**2)**0.5 / a
    if r < 1e-6:
        rho = rho_c
    elif r >= np.pi:
        rho = rho_f
    else:
        rho = rho_c * np.sin(r) / r
    pre = K * rho**2
    return [rho, pre, 0.0, 0.0, 0.0]


def central_mass3d(x, y, z):
    rho_c = 1.0    # central density
    rho_f = 1.0e-2 # floor (atmospheric) density
    a = 0.3        # alpha, stellar radius
    r = (x**2 + y**2 + z**2)**0.5 / a
    if r < 0.5:
        rho = rho_c
    else:
        rho = rho_f
    pre = 1.0
    return [rho, pre, 0.0, 0.0, 0.0]


def ginit(self, x, y, z):
    """
    An attempt at a static gravity solution to the 1d polytrop. In practive we
    just use the FFT.
    """
    R = self.R
    s = (2 * self.Dc * R) / np.pi # enclosed mass per unit area
    if abs(x) < R/2:
        phi = -(self.Dc / (np.pi/R)**2) * np.cos(np.pi*x/R) * self.four_pi_G
        gph = +(self.Dc / (np.pi/R)**1) * np.sin(np.pi*x/R) * self.four_pi_G
    else:
        phi = abs(x) * self.four_pi_G * s
        gph = self.four_pi_G * s * np.sign(x)
    return [phi, gph, 0.0, 0.0]
