import numpy as np 
from ncon import ncon
import scipy.linalg as LA

#Define an iTEBD but for this state which is an MPO
def iTEBD_truncation(Bl,ML,NR,sBA,gate,chi=120,min_sv=1e-9):
    #Reminder:
    #Bl:v_left(slow),physical(up\down),v_right(slow)
    #NR:slow,fast,virtual_right
    #ML:virtual_left,fast,slow

    #Firts apply the local gate directly to Bl:
    
    d=2
    #dim_BL=Bl.shape[0]
    #dim_BR=Bl.shape[2]
    Bl=Bl.reshape(Bl.shape[0],d,d,Bl.shape[2])
    Bl=ncon([Bl,gate],[(-1,1,2,-4),(-2,-3,1,2)])
    
    #Separate Bl
    Bl=Bl.reshape(Bl.shape[0]*Bl.shape[1],Bl.shape[2]*Bl.shape[3])
    U_Bl,sBl,Vh_Bl=LA.svd(Bl,full_matrices=False)
    #Truncate:
    #First truncation:
    d_tol=sum(sBl > min_sv)
    chitemp=min(chi,d_tol)
    U_Bl=U_Bl[:,:chitemp]
    sBl=sBl[:chitemp]
    sBl=sBl/LA.norm(sBl)
    Vh_Bl=Vh_Bl[:chitemp,:]
    U_Bl=U_Bl@np.diag(sBl)
    Vh_Bl=np.diag(sBl)@Vh_Bl
    #reshape it
    U_Bl=U_Bl.reshape(int(Bl.shape[0]/d),d,chitemp)
    Vh_Bl=Vh_Bl.reshape(chitemp,d,int(Bl.shape[1]/d))

    #DMRG part:
    #Contract the time evolution operator with the whole initial tensor. 
    #The new tensor gets the name Full_tensor and the order of the legs is: left,physical_left,purification_left,purification_right,physical_right,right
    Full_tensor=ncon([Vh_Bl,NR,np.diag((1./sBA)),ML,U_Bl],[(-1,-2,1),(1,-3,2),(2,3),(3,-4,4),(4,-5,-6)])
    Full_tensor=ncon([Full_tensor,gate],[(-1,1,-3,-4,2,-6),(-2,-5,1,2)])

    #Obtain B tensor truncated
    #print(f"ft shape: {Full_tensor.shape[0]*Full_tensor.shape[1]*Full_tensor.shape[2]},{Full_tensor.shape[3]*Full_tensor.shape[4]*Full_tensor.shape[5]}")
    U,sBA,Vh=LA.svd(Full_tensor.reshape(Full_tensor.shape[0]*Full_tensor.shape[1]*Full_tensor.shape[2],Full_tensor.shape[3]*Full_tensor.shape[4]*Full_tensor.shape[5]),
                    full_matrices=False)
    #First truncation:
    d_tol=sum(sBA > min_sv)
    chitemp=min(chi,d_tol)
    sBA=sBA[:chitemp]
    #normalize
    sBA=sBA/LA.norm(sBA)
    U=U[:,:chitemp]@np.diag(sBA)
    U=U.reshape(Full_tensor.shape[0]*Full_tensor.shape[1],Full_tensor.shape[2]*len(sBA))
    Vh=Vh[:chitemp,:]
    #add the s1 norm in the Vh
    Vh=(np.diag(sBA)@Vh).reshape(len(sBA)*Full_tensor.shape[3],Full_tensor.shape[4]*Full_tensor.shape[5])

    #Obtain B and NR
    B,sR,NR=LA.svd(U,full_matrices=False)
    min_sv=1e-16
    d_tol=sum(sR > min_sv)
    chitemp=min(chi,d_tol)
    sR=sR[:chitemp]
    sR=sR/LA.norm(sR)
    B=B[:,:chitemp]
    B=B@np.diag(sR)
    B=B.reshape(Full_tensor.shape[0],Full_tensor.shape[1],len(sR))
    NR=NR[:chitemp,:]
    NR=NR.reshape(len(sR),Full_tensor.shape[2],len(sBA))

    #Obtain ML and A
    ML,sL,A=LA.svd(Vh,full_matrices=False)
    min_sv=1e-16
    d_tol=sum(sL > min_sv)
    chitemp=min(chi,d_tol)
    sL=sL[:chitemp]
    sL=sL/LA.norm(sL)
    ML=ML[:,:chitemp]
    ML=ML.reshape(len(sBA),Full_tensor.shape[3],len(sL))
    A=A[:chitemp,:]
    A=np.diag(sL)@A
    A=A.reshape(len(sL),Full_tensor.shape[4],Full_tensor.shape[5])

    #Create Bl
    Bl=ncon([A,np.diag((1./sBl)),B],[(-1,-2,1),(1,2),(2,-3,-4)])
    Bl=Bl.reshape(Bl.shape[0],2*2,Bl.shape[3])
    
    



    return Bl,ML,NR,sBA

