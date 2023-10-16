import traceback
import numpy as np
from mpi4py import MPI
import h5py
import time
import glob
import os
from dedalus import public as de
from dedalus.extras import flow_tools
import logging
logger = logging.getLogger(__name__)

# Parameters
Lx, Lz = (2.*np.pi, 2.*np.pi)
nx, nz = (128, 128)
dealias = 3/2
stop_sim_time = 20 # change to 10
timestepper = de.timesteppers.RK443

# vortex eqn
n=1
k=10
Uo = 1.0
nu = 1e-1 # only this varies
Omega = (n/k)*Uo/nu
Re = Omega

# active scalar
Ao = 1
Rm = 1.0e3 # fix, but can vary
Bo = 0.03 # only thing that varies.
M = Bo / Uo

# passive scalar
co = 1
D = 1.0e3

# Create bases and domain
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
z_basis = de.Fourier('z', nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['c','q','psi','u', 'v','bx','bz'])
# substitutions
problem.parameters['omega'] = Omega
problem.parameters['Rm'] = Rm
problem.parameters['D'] = D
problem.parameters['Msq'] = M**2
problem.parameters['n'] = n
problem.substitutions['Fw'] = '-(n**3)*(cos(n*x)+cos(n*z))/omega'
problem.substitutions['L(A)'] = 'dx(dx(A))+dz(dz(A))'
problem.add_equation("dt(c) - D*L(c) = -( dx(psi)*dz(c) - dz(psi)*dx(c) )", condition="(nx!=0) or (nz!=0)")
problem.add_equation("dt(q) - (1/Rm)*L(q) = -( dx(psi)*dz(q) - dz(psi)*dx(q) )", condition="(nx!=0) or (nz!=0)")
problem.add_equation("dt(L(psi)) - (1/omega)*(L(L(psi))) = Fw - ( dx(psi)*dz(L(psi))- dz(psi)*dx(L(psi)) ) + Msq*( dx(q)*dz(L(q)) - dz(q)*dx(L(q)) )",condition="(nx!=0) or (nz!=0)")
problem.add_equation("u + dz(psi) = 0")
problem.add_equation("v - dx(psi) = 0")
problem.add_equation("bx + dz(q) = 0")
problem.add_equation("bz - dx(q) = 0")
problem.add_equation("psi=0", condition="(nx == 0) and (nz == 0)")
problem.add_equation("q = 0", condition="(nx == 0) and (nz == 0)")
problem.add_equation("c = 0", condition="(nx == 0) and (nz == 0)")

# Build solver
solver = problem.build_solver(timestepper)
logger.info('Solver built')
solver.step_sim_time = stop_sim_time

# initial conditions
x = domain.grid(0)
z = domain.grid(1)
u = solver.state['u']
v = solver.state['v']
c = solver.state['c']
q = solver.state['q']

# Initializing Gaussian monopoles on omega
f = sorted(glob.glob('./snapshots_Re_*'))
f_size = len(f)
if f_size == 0:
    # noise
    # wnoise = []
    # for v in range(1,3):
        # for b in range(1,3):
            # wn_tmp = (np.sin(v*x+b*z) + np.cos(v*x+b*z))*(b**2/np.sqrt(v**2 + b**2))
            # wnoise.append(wn_tmp)
    u['g'] = Uo*np.sin(n*z) # + 0.0001*sum(wnoise)
    v['g'] = -Uo*np.sin(n*x) # + 0.0001*sum(wnoise)
    q['g'] = Ao*np.cos(x)
    c['g'] = co*np.cos(x)
else:
    latest = sorted(glob.glob(f[-1]+'/snapshots_*/*'), key=os.path.getmtime)
    print(latest[-1])
    tasks = ['c','q','psi','u', 'v', 'bx', 'bz']
    with h5py.File(latest[-1],mode='r') as file:
        u['g'] = file['tasks']['u'][-1,:,:]
        v['g'] = file['tasks']['v'][-1,:,:]
        q['g'] = file['tasks']['q'][-1,:,:]
        c['g'] = file['tasks']['c'][-1,:,:]


# Initial timestep
dt = 0.01
# Integration parameters
solver.stop_sim_time = 200
solver.stop_wall_time = 30 * 60.
solver.stop_iteration = np.inf

# Analysis
if Re < 10:
    snapshots = solver.evaluator.add_file_handler('snapshots_Re_0'+str(int(Omega)), sim_dt=0.05, max_writes=10)
else:
    snapshots = solver.evaluator.add_file_handler('snapshots_Re_'+str(int(Omega)), sim_dt=0.05, max_writes=10)
snapshots.add_system(solver.state)


# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=0.6,
                     max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.05)
CFL.add_velocities(('u', 'v'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(v*v + u*u) ", name='|u|')
flow.add_property("v", name='v')
flow.add_property("u", name='u')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max |u| = %f' %flow.max('|u|'))
            logger.info('v = %f' %flow.max('v'))
            logger.info('u = %f' %flow.max('u'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))