import numpy as np
from tqdm import tqdm 

import jax
import jax.numpy as jnp
from jax import grad
jax.config.update("jax_enable_x64", True)


#create the left density matrix
def density_L(Bl_jax,ML_jax,NR_jax):
    rho_L=jnp.einsum('abc,cde,egh,igh,kji,lbk->adlj',
                     ML_jax,Bl_jax,NR_jax,NR_jax.conj(),Bl_jax.conj(),ML_jax.conj())
    #density_L: virtual_left_normal,physical_normal,physical_conj,virtual_left_conj
    return rho_L


#create the right density matrix
def density_R(Bl_jax,ML_jax,NR_jax):
    rho_R=jnp.einsum('cba,edc,gfe,gfh,hzj,jbl->adlz',
                     NR_jax,Bl_jax,ML_jax,ML_jax.conj(),Bl_jax.conj(),NR_jax.conj())
    return rho_R


# Function to calculate original_density_L and original_density_R from C_MPS
def calculate_original_densities(C_MPS):
    original_density_L = jnp.einsum('abc,dec->abde', C_MPS, jnp.conj(C_MPS))
    original_density_R = jnp.einsum('abc,ade->cbed', C_MPS, jnp.conj(C_MPS))
    return original_density_L, original_density_R

#Cost function
def cost_function(Bl_jax,ML_jax,NR_jax,original_density_L,original_density_R):
    #create the density matrices
    rho_L=density_L(Bl_jax,ML_jax,NR_jax)
    rho_R=density_R(Bl_jax,ML_jax,NR_jax)

    dif_L=rho_L-original_density_L
    dif_R=rho_R-original_density_R

    #compute the cost function
    cost1=jnp.einsum('ijkl,ijkl',dif_L,dif_L.conj())
    cost2=jnp.einsum('ijkl,ijkl',dif_R,dif_R.conj())
    total_cost=jnp.real(cost1)+jnp.real(cost2)
    return total_cost

# Function to calculate the norm
def calculate_norm(ML, Bl, NR):
    norm = jnp.einsum('abc,cde,efg,abh,hde,efg',
                      ML, Bl, NR, jnp.conj(ML), jnp.conj(Bl), jnp.conj(NR))
    return jnp.real(norm)  # Norm should be a real value

# Normalize the tensors to ensure norm = 1
def normalize_tensors(ML, Bl, NR):
    norm = calculate_norm(ML, Bl, NR)
    norm_factor = norm ** (1/6)
    ML = ML / norm_factor
    Bl = Bl / norm_factor
    NR = NR / norm_factor
    return ML, Bl, NR

#Start the loop for the gradient descent
def gradient_descent_opt(Bl,ML,NR,C_MPS,learning_rate=1.,num_steps=100):
    original_density_L, original_density_R=calculate_original_densities(C_MPS)

    Bl_jax=jnp.array(Bl)
    #Bl:v_left(slow),physical(up\down),v_right(slow)
    NR_jax=jnp.array(NR)
    #NR:slow,fast,virtual_right
    ML_jax=jnp.array(ML)
    #ML:virtual_left,slow,fast

    ML_jax,Bl_jax,NR_jax=normalize_tensors(ML_jax,Bl_jax,NR_jax)

    #grad_cost=grad(cost_function,argnums=(0,1,2),holomorphic=True)
    grad_cost=grad(cost_function,argnums=(0,1,2))
    convergence=True

    for i in range(num_steps):
        old_opt=cost_function(Bl_jax,ML_jax,NR_jax,original_density_L,original_density_R)
        print('old iteration'+str(i)+':',old_opt)
        #We compute the gradient of each tensor. But as an argument we have to put the cost function argument.
        #Compute the gradient of the cost function, without taking original_density_L,original_density_R as variables
        gradient_Bl,gradient_ML,gradient_NR=grad_cost(Bl_jax,ML_jax,NR_jax,original_density_L,original_density_R)
        #pass the initial arrays in jax form
        #in jax :https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html it says we need to use the conjugate of the 
        #gradient
        #Bl_jax-=learning_rate*gradient_Bl.conj()
        #NR_jax-=learning_rate*gradient_NR.conj()
        #ML_jax-=learning_rate*gradient_ML.conj()
        Bl_old=jnp.copy(Bl_jax)
        NR_old=jnp.copy(NR_jax)
        ML_old=jnp.copy(ML_jax)
        Bl_jax-=learning_rate*gradient_Bl.conj()
        NR_jax-=learning_rate*gradient_NR.conj()
        ML_jax-=learning_rate*gradient_ML.conj()
        ML_jax,Bl_jax,NR_jax=normalize_tensors(ML_jax,Bl_jax,NR_jax)
        opt=cost_function(Bl_jax,ML_jax,NR_jax,original_density_L,original_density_R)
        print('optimized iteration'+str(i)+':',opt)
        if old_opt<opt:
            convergence=False
            Bl_jax=jnp.copy(Bl_old)
            NR_jax=jnp.copy(NR_old)
            ML_jax=jnp.copy(ML_old)
            break
        

    opt_cost=cost_function(Bl_jax,ML_jax,NR_jax,original_density_L,original_density_R)
    print(opt_cost)  
    Bl_new=np.array(Bl_jax)
    NR_new=np.array(NR_jax)
    ML_new=np.array(ML_jax)
    return Bl_new,NR_new,ML_new,opt_cost,convergence

