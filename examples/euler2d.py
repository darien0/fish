
import time
import numpy as np
import pyfluids
import pyfish

N = 128
dx = 1.0 / N
ng = 3


def set_boundary1d(U):
    U[:+ng] = U[+(ng+0)]
    U[-ng:] = U[-(ng+1)]


def set_boundary2d(U):
    U[:,:+ng] = U[:,+(ng+0)][:,np.newaxis,:]
    U[:,-ng:] = U[:,-(ng+1)][:,np.newaxis,:]
    U[:+ng,:] = U[+(ng+0),:][np.newaxis,:,:]
    U[-ng:,:] = U[-(ng+1),:][np.newaxis,:,:]


def set_boundary3d(U):
    U[:,:,:+ng] = U[:,:,+(ng+0)][:,:,np.newaxis,:]
    U[:,:,-ng:] = U[:,:,-(ng+1)][:,:,np.newaxis,:]
    U[:,:+ng,:] = U[:,+(ng+0),:][:,np.newaxis,:,:]
    U[:,-ng:,:] = U[:,-(ng+1),:][:,np.newaxis,:,:]
    U[:+ng,:,:] = U[+(ng+0),:,:][np.newaxis,:,:,:]
    U[-ng:,:,:] = U[-(ng+1),:,:][np.newaxis,:,:,:]


def dUdt1d(fluid, solver):
    L = np.zeros([N,5])
    for j in range(N):
        Fiph = solver.intercellflux(fluid._states[:,j], dim=0)
        L[1:] += -(Fiph[1:] - Fiph[:-1]) / dx
    return L


def dUdt2d(fluid, solver):
    L = np.zeros([N,N,5])
    for j in range(N):
        Fiph = solver.intercellflux(fluid._states[:,j], dim=0)
        L[1:,j] += -(Fiph[1:] - Fiph[:-1]) / dx
    for i in range(N):
        Giph = solver.intercellflux(fluid._states[i,:], dim=1)
        L[i,1:] += -(Giph[1:] - Giph[:-1]) / dx
    return L


def dUdt3d(fluid, solver):
    L = np.zeros([N,N,N,5])
    for j in range(N):
        for k in range(N):
            Fiph = solver.intercellflux(fluid._states[:,j,k], dim=0)
            L[1:,j,k] += -(Fiph[1:] - Fiph[:-1]) / dx
    for k in range(N):
        for i in range(N):
            Giph = solver.intercellflux(fluid._states[i,:,k], dim=1)
            L[i,1:,k] += -(Giph[1:] - Giph[:-1]) / dx
    for i in range(N):
        for j in range(N):
            Hiph = solver.intercellflux(fluid._states[i,j,:], dim=2)
            L[i,j,1:] += -(Hiph[1:] - Hiph[:-1]) / dx
    return L


def advance_midpoint(fluid, solver, dt):
    U0 = fluid.get_conserved()

    U1 = U0 + 0.5 * dt * dUdt(fluid, solver)
    set_boundary(U1)
    fluid.set_conserved(U1)

    U1 = U0 + 1.0 * dt * dUdt(fluid, solver)
    set_boundary(U1)
    fluid.set_conserved(U1)


def advance_shuosher_rk3(fluid, solver, dt):
    U0 = fluid.get_conserved()

    U1 =      U0 +                  dt * dUdt(fluid, solver)
    set_boundary(U1)
    fluid.set_conserved(U1)

    U1 = 3./4*U0 + 1./4*U1 + 1./4 * dt * dUdt(fluid, solver)
    set_boundary(U1)
    fluid.set_conserved(U1)

    U1 = 1./3*U0 + 2./3*U1 + 2./3 * dt * dUdt(fluid, solver)
    set_boundary(U1)
    fluid.set_conserved(U1)


P = np.zeros([N,N,5])
X, Y = np.mgrid[-0.5:0.5:dx,-0.5:0.5:dx]
P[np.where(X**2 + Y**2 >= 0.05)] = [0.1, 0.125, 0.0, 0.0, 0.0]
P[np.where(X**2 + Y**2 <  0.05)] = [1.0, 1.000, 0.0, 0.0, 0.0]

solver = pyfish.FishSolver()
solver.reconstruction = "plm"
solver.riemannsolver = "hllc"

fluid = pyfluids.FluidStateVector([N,N])
fluid.set_primitive(P)

set_boundary = set_boundary2d
dUdt = dUdt2d
advance = advance_shuosher_rk3

def main():
    iter = 0
    tcur = 0.0
    CFL = 0.5
    while tcur < 0.025:
        ml = abs(np.array([f.eigenvalues() for f in fluid._states.flat])).max()
        dt = CFL * dx / ml
        start = time.clock()
        advance(fluid, solver, dt)
        tcur += dt
        iter += 1
        wall_step = time.clock() - start
        print "%05d(%d): t=%5.4f dt=%5.4e %3.1fkz/s %3.2fus/(z*Nq)" % (
            iter, 0, tcur, dt,
            (fluid._states.size / wall_step) * 1e-3,
            (wall_step / P.size) * 1e6)

def plot():
    import matplotlib.pyplot as plt
    plt.imshow(fluid.get_primitive()[:,:,0], interpolation='nearest')
    plt.show()


if __name__ == "__main__":
    import cProfile
    #cProfile.run('main()')
    main()
    plot()