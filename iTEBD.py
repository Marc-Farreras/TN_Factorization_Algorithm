import numpy as np 
from ncon import ncon
import scipy.linalg as LA

#This is the simple infinite TEBD, where we apply a unitary operation to an MPS which is already in canonical form
def iTEBD(A,sAB,B,sBA,gateAB,chi=120,min_sv=1e-12):
    #Ensure that if sAB only has one element it has the matrix form (if not when apply diag an error arises)
    if sAB.shape==(np.ones(1)).shape:
        sAB_matrix=sAB.reshape(1,1)
    else:
        sAB_matrix=np.diag(sAB)
    # ensure singular values are above tolerance threshold to avoid numerical inestabilities
    stol=1e-9
    sBA_trim = sBA * (sBA > stol) + stol * (sBA < stol)
    
    if sBA_trim.shape==(np.ones(1)).shape:
        sBA_matrix=sBA_trim.reshape(1,1)
        sBA_inv_matrix=(1./sBA_trim).reshape(1,1)
    else:
        sBA_matrix=np.diag(sBA_trim)
        sBA_inv_matrix=np.diag(1./sBA_trim)
    
    #AB_tensor index:virtual_l,physical_A,physical_B,virtual_R
    AB_tensor=ncon([sBA_matrix,A,sAB_matrix,B,sBA_matrix],[(-1,1),(1,-2,2),(2,3),(3,-3,4),(4,-4)])
    
    #Apply the time evolution operator:
        #gate AB: s1out,s2out,s1in,s2in
        #AB_new:virtual_l,physical_A,physical_B,virtual_R
    AB_new=ncon([AB_tensor,gateAB],[(-1,2,3,-4),(-2,-3,2,3)])
    
    #Reshape to a matrix form and apply SVD
    U,sAB,Vh=LA.svd(AB_new.reshape(AB_new.shape[0]*AB_new.shape[1],AB_new.shape[2]*AB_new.shape[3]), full_matrices=False)
    #First truncation:
    d_tol=sum(sAB > min_sv)
    chitemp=min(chi,d_tol)
    U=U[:,:chitemp]
    sAB=sAB[:chitemp]
    Vh=Vh[:chitemp,:]
    
    #Now add the sBA inverse to return in the canonical form
    A=sBA_inv_matrix@(U.reshape(AB_new.shape[0],AB_new.shape[1]*chitemp))
    B=Vh.reshape(chitemp*AB_new.shape[2],AB_new.shape[3])@sBA_inv_matrix
    
    #Reshape the matrix A and B:
    A=A.reshape(AB_new.shape[0],AB_new.shape[1],chitemp)
    B=B.reshape(chitemp,AB_new.shape[2],AB_new.shape[3])
    
    #normalize the metrics
    sAB = sAB / LA.norm(sAB)
    
    
    return A,sAB,B

