import sys
import os
import time
import pstats
import cProfile
import cPickle
import numpy as np
import pyfluids
import pyfish

class MaraEvolutionOperator(object):
    def __init__(self, problem, scheme):
        descr = problem.fluid_descriptor
        X0 = problem.lower_bounds
        X1 = problem.upper_bounds
        ng = self.number_guard_zones()

        self.shape = tuple([n + 2*ng for n in problem.resolution])
        self.fluid = pyfluids.FluidStateVector(self.shape, descr)
        self.scheme = scheme
        self.driving = getattr(problem, 'driving', None)
        self.poisson_solver = getattr(problem, 'poisson_solver', None)
        self.pressure_floor = 1e-6

        if len(self.shape) == 1:
            Nx, Ny, Nz = self.fluid.shape + (1, 1)
        if len(self.shape) == 2:
            Nx, Ny, Nz = self.fluid.shape + (1,)
        if len(self.shape) == 3:
            Nx, Ny, Nz = self.fluid.shape

        dx = (X1[0] - X0[0])/(Nx - (2*ng if Nx > 1 else 0))
        dy = (X1[1] - X0[1])/(Ny - (2*ng if Ny > 1 else 0))
        dz = (X1[2] - X0[2])/(Nz - (2*ng if Nz > 1 else 0))

        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.X0, self.X1 = X0, X1

    @property
    def fields(self):
        P = self.fluid.primitive
        G = self.fluid.gravity
        return {'rho': P[...,0],
                'pre': P[...,1],
                'vx' : P[...,2],
                'vy' : P[...,3],
                'vz' : P[...,4],
                'phi': G[...,0] if self.fluid.descriptor.ngravity else None,
                'gph': G[...,1] if self.fluid.descriptor.ngravity else None}

    def write_checkpoint(self, status, dir=".", update_status=True, **extras):
        if update_status:
            status.chkpt_last = status.time_current
            status.chkpt_number += 1
        try:
            os.makedirs(dir)
            print "creating data directory", dir
        except OSError: # Directory exists
            pass
        chkpt = { "prim": self.fluid.primitive, "status": status.__dict__ }
        chkpt.update(extras)
        chkpt_name = "%s/chkpt.%04d.pk" % (dir, status.chkpt_number)
        chkpt_file = open(chkpt_name, 'w')
        print "Writing checkpoint", chkpt_name
        cPickle.dump(chkpt, chkpt_file)

    def measure(self):
        meas = { }
        P = self.fluid.primitive
        U = self.fluid.conserved()
        rho = P[...,0]
        vx = P[...,2]
        vy = P[...,3]
        vz = P[...,4]
        meas["kinetic"] = (rho * (vx*vx + vy*vy + vz*vz)).mean()
        meas["density_max"] = rho.max()
        meas["density_min"] = rho.min()
        meas["conserved_avg"] = [U[...,i].mean() for i in range(5)]
        meas["primitive_avg"] = [P[...,i].mean() for i in range(5)]
        return meas

    def min_grid_spacing(self):
        return min([self.dx, self.dy, self.dz][:len(self.shape)])

    def number_guard_zones(self):
        return 3

    def coordinate_grid(self):
        ng = self.number_guard_zones()
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        dx, dy, dz = self.dx, self.dy, self.dz
        x0, y0, z0 = self.X0
        x1, y1, z1 = self.X1
        if self.Nx > 1: x0 -= ng * dx
        if self.Ny > 1: y0 -= ng * dy
        if self.Nz > 1: z0 -= ng * dz
        if self.Nx > 1: x1 += ng * dx
        if self.Ny > 1: y1 += ng * dy
        if self.Nz > 1: z1 += ng * dz
        return np.mgrid[x0+dx/2 : x1+dx/2 : dx,
                        y0+dy/2 : y1+dy/2 : dy,
                        z0+dz/2 : z1+dz/2 : dz]

    def initial_model(self, pinit, ginit=None):
        npr = self.fluid.descriptor.nprimitive
        ngr = self.fluid.descriptor.ngravity
        if ginit is None: ginit = lambda x,y,z: np.zeros(ngr)
        shape = self.shape
        X, Y, Z = self.coordinate_grid()
        P = np.ndarray(
            shape=shape + (npr,), buffer=np.array(
                [pinit(x, y, z) for x, y, z in zip(X.flat, Y.flat, Z.flat)]))
        G = np.ndarray(
            shape=shape + (ngr,), buffer=np.array(
                [ginit(x, y, z) for x, y, z in zip(X.flat, Y.flat, Z.flat)]))
        self.fluid.primitive = P
        self.fluid.gravity = G

    def set_boundary(self):
        """
        This function does not call BC's on gravitational field right now.
        """
        ng = self.number_guard_zones()
        self.boundary.set_boundary(self.fluid.primitive, ng, field='prim')

    def advance(self, dt, rk=3):
        start = time.clock()
        U0 = self.fluid.conserved()
        # RungeKuttaSingleStep
        if rk == 1:
            U1 = U0 + dt * self.dUdt(U0)
        # RungeKuttaRk2Tvd
        if rk == 2:
            U1 =      U0 +      dt*self.dUdt(U0)
            U1 = 0.5*(U0 + U1 + dt*self.dUdt(U1))
        # RungeKuttaShuOsherRk3
        if rk == 3:
            U1 =      U0 +                  dt * self.dUdt(U0)
            U1 = 3./4*U0 + 1./4*U1 + 1./4 * dt * self.dUdt(U1)
            U1 = 1./3*U0 + 2./3*U1 + 2./3 * dt * self.dUdt(U1)
        # RungeKuttaClassicRk4
        if rk == 4:
            L1 = self.dUdt(U0)
            L2 = self.dUdt(U0 + (0.5*dt) * L1)
            L3 = self.dUdt(U0 + (0.5*dt) * L2)
            L4 = self.dUdt(U0 + (1.0*dt) * L3)
            U1 = U0 + dt * (L1 + 2.0*L2 + 2.0*L3 + L4) / 6.0

        try:
            ng = self.number_guard_zones()
            S = self.driving.source_terms(self.fluid.primitive[ng:-ng,ng:-ng])
            U1[ng:-ng,ng:-ng] += S * dt
            self.driving.advance(dt)
        except AttributeError:
            # no driving module
            pass

        ng = self.number_guard_zones()
        self.boundary.set_boundary(U1, ng)
        self.fluid.from_conserved(U1)
        return time.clock() - start

    def update_gravity(self):
        """
        Notes:
        ------

        (1) Only works in 1d for now.

        (2) To see the bug introduced by not accounting for the background
        density, do

        self.fluid.descriptor.rhobar = 0.0 #rhobar

        """
        if self.poisson_solver is None:
            return
        if len(self.shape) > 1:
            raise NotImplementedError
        try:
            ng = self.number_guard_zones()
            G, rhobar = self.poisson_solver.solve(self.fields['rho'][ng:-ng],
                                                  retrhobar=True)
            self.fluid.descriptor.rhobar = rhobar
            self.fluid.gravity[ng:-ng] = G
            self.boundary.set_boundary(self.fluid.gravity, ng, field='grav')

        except AttributeError: # no poisson_solver
            pass
        except ValueError: # no gravity array
            pass

    def validate_gravity(self):
        if len(self.shape) > 1:
            raise NotImplementedError
        import matplotlib.pyplot as plt
        ng = self.number_guard_zones()
        phi0 = self.fields['phi']
        gph0 = self.fields['gph']
        gph1 = np.gradient(phi0, self.dx)
        plt.semilogy(abs(((gph1 - gph0))[ng:-ng]))
        plt.show()

    def dUdt(self, U):
        ng = self.number_guard_zones()
        dx = [self.dx, self.dy, self.dz]
        self.boundary.set_boundary(U, ng)
        self.fluid.from_conserved(U)

        if (U[...,0] < 0.0).any():
            raise RuntimeError("negative density (conserved)")
        if (U[...,1] < 0.0).any():
            raise RuntimeError("negative energy")
        self.fluid.from_conserved(U)
        P = self.fluid.primitive
        if (P[...,0] < 0.0).any():
            raise RuntimeError("negative density (primitive)")
        if (P[...,1] < 0.0).any():
            #P[P[...,1] < self.pressure_floor, 1] = self.pressure_floor
            #print "set pressure floor on some zones"
            raise RuntimeError("negative pressure")

        self.update_gravity()
        L = self.scheme.time_derivative(self.fluid, dx)
        if self.fluid.descriptor.fluid in ['gravp', 'gravs', 'grave']:
            S = self.fluid.source_terms()
            return L + S
        else:
            return L


class SimulationStatus:
    pass


def main():
    # Problem options
    problem_cfg = dict(resolution=[128],
                       tfinal=10.)
    problem = pyfish.problems.OneDimensionalPolytrope(selfgrav=True, **problem_cfg)
    #problem = pyfish.problems.PeriodicDensityWave(**problem_cfg)
    #problem = pyfish.problems.DrivenTurbulence2d(tfinal=0.01)

    # Status setup
    status = SimulationStatus()
    status.CFL = 0.3
    status.iteration = 0
    status.time_step = 0.0
    status.time_current = 0.0
    status.chkpt_number = 0
    status.chkpt_last = 0.0
    status.chkpt_interval = 1.0
    measlog = { }

    # Plotting options
    plot_fields = problem.plot_fields
    plot_interactive = False
    plot_initial = True
    plot_final = True
    plot = [plot1d, plot2d, plot3d][len(problem.resolution) - 1]

    scheme = pyfish.FishSolver()
    scheme.solver_type = ["godunov", "spectral"][0]
    scheme.reconstruction = "plm"
    scheme.riemann_solver = "hllc"
    scheme.shenzha10_param = 100.0
    scheme.smoothness_indicator = ["jiangshu96", "borges08", "shenzha10"][2]

    mara = MaraEvolutionOperator(problem, scheme)
    mara.initial_model(problem.pinit, problem.ginit)
    mara.boundary = problem.build_boundary(mara)

    if plot_interactive:
        import matplotlib.pyplot as plt
        plt.ion()
        lines = plot(mara, plot_fields, show=False)

    if plot_initial:
        plot(mara, plot_fields, show=False, label='start')

    while status.time_current < problem.tfinal:
        if plot_interactive:
            for f in plot_fields:
                lines[f].set_ydata(mara.fields[f])
            plt.draw()

        ml = abs(mara.fluid.eigenvalues()).max()
        dt = status.CFL * mara.min_grid_spacing() / ml
        try:
            wall_step = 1e-10 +  mara.advance(dt, rk=3)
        except RuntimeError as e:
            print e
            break

        status.time_step = dt
        status.time_current += status.time_step
        status.iteration += 1

        status.message = "%05d(%d): t=%5.4f dt=%5.4e %3.1fkz/s %3.2fus/(z*Nq)" % (
            status.iteration, 0, status.time_current, dt,
            (mara.fluid.size / wall_step) * 1e-3,
            (wall_step / (mara.fluid.size*5)) * 1e6)

        if status.time_current - status.chkpt_last > status.chkpt_interval:
            mara.write_checkpoint(status, dir="data/test", update_status=True,
                                  measlog=measlog)

        measlog[status.iteration] = mara.measure()
        measlog[status.iteration]["time"] = status.time_current
        measlog[status.iteration]["message"] = status.message
        print status.message

    mara.set_boundary()
    if plot_final:
        plot(mara, plot_fields, show=True, label='end')


def plot1d(mara, fields, show=True, **kwargs):
    import matplotlib.pyplot as plt
    lines = { }
    x, y, z = mara.coordinate_grid()
    try:
        axes = plot1d.axes
    except:
        plot1d.axes = [plt.subplot(len(fields),1,n+1) for n,f in enumerate(fields)]
        axes = plot1d.axes

    for ax, f in zip(axes, fields):
        lines[f], = ax.plot(x.flat, mara.fields[f], '-o', mfc='none', label=(
                f + ' ' + kwargs.get('label', '')))
    if show:
        for ax in axes:
            ax.legend(loc='best')
            yl = ax.get_ylim()
            ax.set_ylim(yl[0]+1e-8, yl[1]-1e-8)
        for ax in axes[:-1]:
            ax.set_xticks([])
        axes[-1].set_xlabel('position')
        plt.subplots_adjust(hspace=0.0, wspace=0)
        plt.show()
    return lines


def plot2d(mara, fields, show=True, **kwargs):
    import matplotlib.pyplot as plt
    lines = { }
    x, y, z = mara.coordinate_grid()
    ng = mara.number_guard_zones()
    try:
        axes = plot2d.axes
    except:
        nr = 2
        nc = np.ceil(len(fields) / float(nr))
        plot2d.axes = [plt.subplot(nc,nr,n+1) for n,f in enumerate(fields)]
        axes = plot2d.axes
    for ax, f in zip(axes, fields):
        lines[f] = ax.imshow(mara.fields[f][ng:-ng,ng:-ng].T, interpolation='nearest')
    if show:
        for ax, f in zip(axes, fields):
            ax.set_title(f)
        plt.show()
    return lines


def plot3d(mara, fields, show=True, **kwargs):
    import matplotlib.pyplot as plt
    lines = { }
    x, y, z = mara.coordinate_grid()
    ng = mara.number_guard_zones()
    try:
        axes = plot2d.axes
    except:
        nr = 2
        nc = np.ceil(len(fields) / float(nr))
        plot2d.axes = [plt.subplot(nc,nr,n+1) for n,f in enumerate(fields)]
        axes = plot2d.axes
    for ax, f in zip(axes, fields):
        i0 = mara.fields[f].shape[0] / 2
        lines[f] = ax.imshow(mara.fields[f][i0,ng:-ng,ng:-ng].T, interpolation='nearest')
    if show:
        for ax, f in zip(axes, fields):
            ax.set_title(f)
        plt.show()
    return lines


if __name__ == "__main__":
    main()
