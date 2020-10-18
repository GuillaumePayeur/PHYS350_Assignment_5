# Guillaume Payeur
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre

R = 1
V0 = 2
ntheta = 10001
theta = np.linspace(0,np.pi,ntheta)
ncoeffs = 500

def V_and_rho(theta_0):
    # First filling in the lhs matrix (mat)
    # mat1 is correct between 0 and theta_0, mat2 is correct between theta_0 and pi
    # mat is the combination of the correct parts of mat1 and mat2
    mat = np.zeros((ntheta,ncoeffs))
    mat1 = np.zeros((ntheta,ncoeffs))
    mat2 = np.zeros((ntheta,ncoeffs))

    for l in range(ncoeffs):
        mat1[:,l] = R**l*legendre(l)(np.cos(theta))
    for l in range(ncoeffs):
        mat2[:,l] = (2*l+1)*R**(l-1)*legendre(l)(np.cos(theta))
    mat[0:int(ntheta*(theta_0/np.pi))] = mat1[0:int(ntheta*(theta_0/np.pi))]
    mat[int(ntheta*(theta_0/np.pi)):] = mat2[int(ntheta*(theta_0/np.pi)):]

    # Now filling in the rhs matrix
    # First setting it to V0 and then filling in the hole
    rhs = np.zeros((ntheta)) + V0
    rhs[int(ntheta*(theta_0/np.pi)):] = 0

    # Solving the matrix system
    fitp=np.linalg.inv(mat.T@mat)@(mat.T@rhs)

    # Finding the potential on the shell
    V=mat1@fitp

    # Finding the charge density on the shell
    rho =mat2@fitp

    return V,rho

# Displaying the potential on the shell for three values of theta_0
theta_0_array = [np.pi/8, np.pi/2, np.pi-0.01]
plt.style.use('seaborn-whitegrid')
fig, axes = plt.subplots(nrows=2, ncols=3)
axes[0,0].set_title('Theta_0 = pi/8')
axes[0,1].set_title('Theta_0 = pi/2')
axes[0,2].set_title('Theta_0 = pi-delta')
for i,theta_0 in enumerate(theta_0_array):
    axes[0,i].plot([0,np.pi],[V0,V0],label='V_0',color='gray',alpha=0.5)
    axes[0,i].plot(theta,V_and_rho(theta_0)[0])
    axes[1,i].plot(theta,V_and_rho(theta_0)[1],color='green')
plt.setp(axes[1,0], xlabel='theta')
plt.setp(axes[1,1], xlabel='theta')
plt.setp(axes[1,2], xlabel='theta')
plt.setp(axes[0,0], ylabel='V')
plt.setp(axes[1,0], ylabel='rho')
axes[0,2].legend(loc=2,frameon=True)
plt.show()
