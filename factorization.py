import numpy as np 
from ncon import ncon
import scipy.linalg as LA
import utils

#UhL:slow_L,fast_L,virtual_L--> slow,fast-UhL-C-
#VhR:virtual_R,slow_R,fast_R --> -C-VhR--slow,fast
def Optimize_slow(slow,fast,C,UhL,VhR):
    #Analytical optimum slow:
    Tensors=[UhL,C,VhR,fast.conj()]
    labels=[(-1,1,2),(2,-2,3),(3,-3,4),(1,4)]
    slow_optimum=ncon(Tensors,labels)
    #Normalize
    norm=ncon([slow_optimum,slow_optimum.conj()],[(1,2,3),(1,2,3)])
    slow_optimum=slow_optimum/np.sqrt(norm)
    return slow_optimum


#UhL:slow_L,fast_L,virtual_L--> slow,fast-UhL-C-
#VhR:virtual_R,slow_R,fast_R --> -C-VhR--slow,fast
def Optimize_fast(slow,fast,C,UhL,VhR):
    #Analytical optimum fast:
    Tensors=[UhL,C,VhR,slow.conj()]
    labels=[(1,-1,2),(2,3,4),(4,5,-2),(1,3,5)]
    fast_optimum=ncon(Tensors,labels)
    #Normalize
    norm=ncon([fast_optimum,fast_optimum.conj()],[(1,2),(1,2)])
    fast_optimum=fast_optimum/np.sqrt(norm)
    return fast_optimum


def Optimize_UhL(slow,fast,C,UhL,VhR):
    #Analytical optimum unitary
    #Tensors=[C,VhR,slow.conj(),fast.conj()]
    Tensors=[VhR,C,slow.conj(),fast.conj()]
    #labels=[(-1,1,2),(2,3,4),(-2,1,3),(-3,4)]
    labels=[(1,2,3),(-1,4,1),(-2,4,2),(-3,3)]
    #New tensor to optimize index:C_v_left,slow_L,fast_L
    enviroment=ncon(Tensors,labels)
    enviroment=enviroment.reshape(C.shape[0],slow.shape[0]*fast.shape[0])
    #apply SVD:
    U,S,Vh=LA.svd(enviroment, full_matrices=False)
    UhL_optimum=np.transpose(Vh.conj())@np.transpose(U.conj())
    UhL_optimum=UhL_optimum.reshape(slow.shape[0],fast.shape[0],C.shape[0])
    return UhL_optimum


def Optimize_VhR(slow,fast,C,UhL,VhR):
    #Analytical optimum unitary
    Tensors=[UhL,C,slow.conj(),fast.conj()]
    labels=[(1,2,3),(3,4,-3),(1,4,-1),(2,-2)]
    #New tensor to optimize index:slow_R,fast_R,C_v_left
    enviroment=ncon(Tensors,labels)
    enviroment=enviroment.reshape(slow.shape[2]*fast.shape[1],C.shape[2])
    #apply SVD:
    U,S,Vh=LA.svd(enviroment, full_matrices=False)
    VhR_optimum=np.transpose(Vh.conj())@np.transpose(U.conj())
    VhR_optimum=VhR_optimum.reshape(C.shape[2],slow.shape[2],fast.shape[1])
    return VhR_optimum




#To see how close we are from a product state using this method. If a product state, S(rho_fast)=0
def rho_fast(UhL,C,VhR):
    #UhL:slow_L,fast_L,virtual_L--> slow,fast-UhL-C-
    #VhR:virtual_R,slow_R,fast_R --> -C-VhR--slow,fast
    #fast:v_left,v_right
    #slow: v_left,physical(up\down),v_right
    
    #reduced density matrix of rho fast
    Tensors=[UhL,C,VhR,UhL.conj(),C.conj(),VhR.conj()]
    labels=[(1,-1,2),(2,3,4),(4,5,-2),(1,-3,6),(6,3,7),(7,5,-4)]
    rho_fast=ncon(Tensors,labels)
    matrix=rho_fast.reshape(rho_fast.shape[0]*rho_fast.shape[1],rho_fast.shape[2]*rho_fast.shape[3])
    eig_rho,_=LA.eigh(matrix)
    #compute entropy:
    cut_off =1.e-12
    mask_sl=np.greater_equal(np.abs(eig_rho),cut_off)
    eig_rho =eig_rho[mask_sl]
    vn_entropy =sum(-eig_rho*np.log(eig_rho))
    return vn_entropy



#Function that does the disentangling and computes the entropy
def disentangling_state(C_MPS,dfast1,max_iter=100,eps=1e-10):
    #Save the dimensions of C tensor
    dim_L=C_MPS.shape[0]
    dim_s=C_MPS.shape[1]
    dim_R=C_MPS.shape[2]

    UhL,VhR, fast,slow=utils.random_tensors(dim_L,dim_s,dim_R,dfast1)
    entropy=10000.
    fast=np.diag((1./np.sqrt(2))*np.array([1.+0j]*dfast1))
    fin_entropy=1000
    

    for j in range(0,6):

        fast=np.diag((1./np.sqrt(2))*np.array([1.+0j]*dfast1))
        for k in range(0,max_iter):
            #______________________________________optimization__________________________________________________-
            
            slow=Optimize_slow(slow,fast,C_MPS,UhL,VhR)
            if k>=int(max_iter*(0.1)):
                fast=Optimize_fast(slow,fast,C_MPS,UhL,VhR)
            UhL=Optimize_UhL(slow,fast,C_MPS,UhL,VhR)
            VhR=Optimize_VhR(slow,fast,C_MPS,UhL,VhR)
        
            #____________________________________study convergence_______________________________________________
        
            #The equivalent of C is applying U and V to the fast and slow degrees of freedom
            #UL:virtual_left,slow,fast
            #VR:slow,fast,virtual_right
            UhL_inv=(np.transpose(np.conjugate(UhL.reshape(int(dim_L/dfast1)*dfast1,dim_L)))).reshape(dim_L,int(dim_L/dfast1),dfast1)
            VhR_inv=(np.transpose(np.conjugate(VhR.reshape(dim_L,int(dim_L/dfast1)*dfast1)))).reshape(int(dim_L/dfast1),dfast1,dim_L)
            C_equiv=ncon([UhL_inv,slow,fast,VhR_inv],[(-1,1,2),(1,-2,3),(2,4),(3,4,-3)])
            overlap=ncon([C_MPS,C_equiv.conj()],[(1,2,3),(1,2,3)])
            #compute entropy of rho_fast
            entropy_new=rho_fast(UhL,C_MPS,VhR)
            entropy=entropy_new
        if entropy<fin_entropy:
            fin_UhL=np.copy(UhL)
            fin_VhR=np.copy(VhR)
            fin_fast=np.copy(fast)
            fin_fast=np.copy(fast)
            fin_slow=np.copy(slow)
            fin_entropy=entropy
            fin_overlap=np.abs(overlap)


        
    
    return fin_UhL,fin_VhR, fin_fast,fin_slow,fin_entropy,fin_overlap

    
    
#Funtion that computes the fast dimension we will use
def fast_dim_MPS(bond_dim):
    factors=[2,3,5]
    factor=False
    for i in factors:
        if bond_dim%i==0:
            factor=True
            final_dim=i
            break
    
    if not factor:
        final_dim=1
    return final_dim
