import numpy as np
from numpy.fft import *

class PoissonSolver1d(object):
    L = 1.0
    four_pi_G = 1.0
    gradient_method = ['spectral', 'difference'][0]

    def __init__(self):
        pass

    def solve(self, rho, retrhobar=False):
        Nx, = rho.shape
        k = fftfreq(Nx) * 2*np.pi*Nx
        k[0] = 1.0
        rhohat = fft(rho)
        rhobar = rhohat[0] / rho.size
        phihat = self.four_pi_G * rhohat / -k**2
        phihat[0] = 0.0
        gphhat = 1.j * k * phihat
        phi = ifft(phihat).real

        if self.gradient_method == 'spectral':
            gph = ifft(1.j * k * phihat).real
        elif self.gradient_method == 'difference':
            gph = np.gradient(phi, self.L / Nx)

        soln = np.zeros(rho.shape + (4,))
        soln[:,0] = phi
        soln[:,1] = gph
        if retrhobar:
            return soln, rhobar.real
        else:
            return soln

    def self_test(self):
        pass


class SelfGravitySourceTerms(object):
    def __init__(self, G=1.0):
        self.G = G

    def source_terms(self, mara, retphi=False):
        from numpy.fft import fftfreq, fftn, ifftn
        ng = mara.number_guard_zones()
        G = self.G
        L = 1.0
        Nx, Ny, Nz = mara.fluid.shape
        Nx -= 2*ng
        Ny -= 2*ng
        Nz -= 2*ng
        P = mara.fluid.primitive[ng:-ng,ng:-ng,ng:-ng]
        rho = P[...,0]
        vx = P[...,2]
        vy = P[...,3]
        vz = P[...,4]

        K = [fftfreq(Nx)[:,np.newaxis,np.newaxis] * (2*np.pi*Nx/L),
             fftfreq(Ny)[np.newaxis,:,np.newaxis] * (2*np.pi*Ny/L),
             fftfreq(Nz)[np.newaxis,np.newaxis,:] * (2*np.pi*Nz/L)]
        delsq = -(K[0]**2 + K[1]**2 + K[2]**2)
        delsq[0,0,0] = 1.0 # prevent division by 0

        rhohat = fftn(rho)
        phihat = (4*np.pi*G) * rhohat / delsq
        fx = -ifftn(1.j * K[0] * phihat).real
        fy = -ifftn(1.j * K[1] * phihat).real
        fz = -ifftn(1.j * K[2] * phihat).real

        S = np.zeros(mara.fluid.shape + (5,))
        S[ng:-ng,ng:-ng,ng:-ng,0] = 0.0
        S[ng:-ng,ng:-ng,ng:-ng,1] = rho * (fx*vx + fy*vy + fz*vz)
        S[ng:-ng,ng:-ng,ng:-ng,2] = rho * fx
        S[ng:-ng,ng:-ng,ng:-ng,3] = rho * fy
        S[ng:-ng,ng:-ng,ng:-ng,4] = rho * fz
        return (S, ifftn(phihat).real) if retphi else S


class EnclosedMassMonopoleGravity(object):
    def __init__(self, G=1.0):
        self.G = G

    def source_terms(self, mara, retphi=False):
        from scipy.interpolate import interp1d

        G = self.G
        X, Y, Z = mara.coordinate_grid()
        r2 = X**2 + Y**2 + Z**2
        r = r2**0.5

        P = mara.fluid.primitive
        rho = P[...,0]
        vx = P[...,2]
        vy = P[...,3]
        vz = P[...,4]

        dV = mara.dx * mara.dy * mara.dz
        renc = np.linspace(0.0, 1.0, 10)
        Menc = [ ]

        for r0 in renc:
            Menc.append(rho[r < r0].sum() * dV)

        Menc_f = interp1d(renc, Menc, kind='cubic')
        Menc_i = Menc_f(r)

        rhatx, rhaty, rhatz = X/r, Y/r, Z/r
        fx = -G * Menc_i * rhatx / r2
        fy = -G * Menc_i * rhaty / r2
        fz = -G * Menc_i * rhatz / r2

        S = np.zeros(mara.fluid.shape + (5,))
        S[...,0] = 0.0
        S[...,1] = rho * (fx*vx + fy*vy + fz*vz)
        S[...,2] = rho * fx
        S[...,3] = rho * fy
        S[...,4] = rho * fz
        return S


class StaticCentralGravity(object):
    def __init__(self, G=1.0, M=1.0):
        self.G = G
        self.M = M

    def source_terms(self, mara, retphi=False):
        G = self.G
        M = self.M

        X, Y, Z = mara.coordinate_grid()
        r2 = X**2 + Y**2 + Z**2
        r = r2**0.5

        rhatx, rhaty, rhatz = X/r, Y/r, Z/r
        fx = -G * M * rhatx / r2
        fy = -G * M * rhaty / r2
        fz = -G * M * rhatz / r2

        phi = -G * M / r

        P = mara.fluid.primitive
        rho = P[...,0]
        vx = P[...,2]
        vy = P[...,3]
        vz = P[...,4]

        S = np.zeros(mara.fluid.shape + (5,))
        S[...,0] = 0.0
        S[...,1] = rho * (fx*vx + fy*vy + fz*vz)
        S[...,2] = rho * fx
        S[...,3] = rho * fy
        S[...,4] = rho * fz

        return (S, phi) if retphi else S

class PoissonSolver1dA(object):
    '''
    Solves the equation del^2 phi = rho for one-dimensional periodic arrays rho
    using FFT's. The return value of the 'solve' function a 4-component array
    containing phi in soln[:,0] and its gradient in soln[:,1:4].

    Use of FFT's results in a solution to the equation del^2 phi = rho - <rho>,
    so it's formally wrong whenever there's a net charge (which of course there
    is for gravity). The lack of a solution follows from the periodicity imposed
    by the FFT. In 1d, we can obtain an exact solution by adding the quadratic
    <rho> x^2 / 2 to phi.

    In many cases, it's perfectly fine to use a phi satisfying del^2 phi = rho -
    <rho>. However, the 'gravp' and 'grave' schemes operate on the assumption of
    a real solution and thus may require the addition of the quadratic term.
    '''
    L = 1.0

    def __init__(self):
        pass

    def solve(self, x, rho):
        Nx, = rho.shape
        k = fftfreq(Nx) * 2*np.pi*Nx
        k[0] = 1.0
        rhohat = fft(rho)
        rhobar = rhohat[0] / rho.size
        phihat = rhohat / -k**2 + fft(0.5 * rhobar * x**2)
        gphhat = 1.j * k * phihat
        phi = ifft(phihat).real
        gph = ifft(1.j * k * phihat).real
        soln = np.zeros(rho.shape + (4,))
        soln[:,0] = phi
        soln[:,1] = gph
        return soln

    def self_test(self):
        pass
