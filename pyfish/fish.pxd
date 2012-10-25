
from pyfluids.fluids cimport *
cimport fish
cimport numpy as np
import numpy as np

cdef extern from "fish.h":
    enum:
        FISH_PCM, # piecewise constant reconstruction
        FISH_PLM, # piecewise linear reconstruction
        FISH_AVG, # piecewise linear reconstruction
        FISH_WENO5, # weno-5 reconstruction
        FISH_GODUNOV, # conservative finite volume Riemann-solver intercell fluxes
        FISH_SPECTRAL, # conservative finite differencing of characteristic fields

        # ---------------------------------------------------------------------------
        # smoothness indicators for WENO reconstruction
        # ---------------------------------------------------------------------------
        FISH_ISK_JIANGSHU96, # original smoothness indicator of Jiang & Shu (1996)
        FISH_ISK_BORGES08, # improved by Borges (2008) NOTE: might be 4th order
        FISH_ISK_SHENZHA10, # improved by Shen & Zha (2010)

        # ---------------------------------------------------------------------------
        # names of parameters for solver description
        # ---------------------------------------------------------------------------

        # ------------------
        # integer parameters
        # ------------------
        FISH_SOLVER_TYPE,
        FISH_RIEMANN_SOLVER,
        FISH_RECONSTRUCTION,
        FISH_GRAVRECNSTRUCT,
        FISH_SMOOTHNESS_INDICATOR,

        # -----------------
        # double parameters
        # -----------------
        FISH_PLM_THETA, # [1 -> 2 (most aggressive)]
        FISH_SHENZHA10_PARAM, # [0 -> ~100 (most aggressive)]

        FISH_ERROR_BADARG,

    struct fish_state

    fish_state *fish_new()
    int fish_del(fish_state *S)
    int fish_evolve(fish_state *S, double dt)
    int fish_intercellflux(fish_state *S, fluids_state **fluid, double *F, int N,
                           int dim)
    int fish_timederivative(fish_state *S, fluids_state **fluid,
                            int ndim, int *shape, double *dx, double *L)
    int fish_getparami(fish_state *S, int *param, long flag)
    int fish_setparami(fish_state *S, int param, long flag)
    int fish_getparamd(fish_state *S, double *param, long flag)
    int fish_setparamd(fish_state *S, double param, long flag)

cdef class FishSolver(object):
    cdef fish_state *_c
