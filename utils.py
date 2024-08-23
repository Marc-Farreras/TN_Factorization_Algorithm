import numpy as np 
from ncon import ncon
from scipy.stats import unitary_group

def expect_val(A,B,sAB,sBA):
    sigma_y = np.array([[0.,-1j],[1j,0.]])
    sigma_x = np.array([[0.,1.],[1.,0.]])
    sigma_z = np.array([[1.,0.],[0.,-1.]])
    if sAB.shape==(np.ones(1)).shape:
        sAB=sAB.reshape(1,1)
    else:
        sAB=np.diag(sAB)
    if sBA.shape==(np.ones(1)).shape:
        sBA=sBA.reshape(1,1)
    else:
        sBA=np.diag(sBA)
    
    Full_A=ncon([sBA,A,sAB],[(-1,1),(1,-2,2),(2,-3)])
    Full_B=ncon([sAB,B,sBA],[(-1,1),(1,-2,2),(2,-3)])
    
    norm_A=ncon([Full_A,Full_A.conj()],[(1,2,3),(1,2,3)])
    norm_B=ncon([Full_B,Full_B.conj()],[(1,2,3),(1,2,3)])
    
    X_A=ncon([Full_A,sigma_x,Full_A.conj()],[(1,2,3),(4,2),(1,4,3)])/norm_A
    X_B=ncon([Full_B,sigma_x,Full_B.conj()],[(1,2,3),(4,2),(1,4,3)])/norm_B
    
    Y_A=ncon([Full_A,sigma_y,Full_A.conj()],[(1,2,3),(4,2),(1,4,3)])/norm_A
    Y_B=ncon([Full_B,sigma_y,Full_B.conj()],[(1,2,3),(4,2),(1,4,3)])/norm_B
    
    Z_A=ncon([Full_A,sigma_z,Full_A.conj()],[(1,2,3),(4,2),(1,4,3)])/norm_A
    Z_B=ncon([Full_B,sigma_z,Full_B.conj()],[(1,2,3),(4,2),(1,4,3)])/norm_B
    
    return norm_A,norm_B,X_A,X_B,Y_A,Y_B,Z_A,Z_B





#First let's create the Unitary transformations from we will start to optimization process.
def random_tensors(dim_L,dim_s,dim_R,dfast1):
    #___________________________________________Random Unitaries________________________________________________________________
    #Random unitary matrix of dim=dim_L
    UhL = unitary_group.rvs(dim_L)
    #Random unitary matrix of dim=dim_L
    VhR = unitary_group.rvs(dim_R)

    #From here reshape the unitaries
    UhL=UhL.reshape(int(dim_L/dfast1),dfast1,dim_L)
    VhR=VhR.reshape(dim_R,int(dim_R/dfast1),dfast1)
    #order of index: First for order of position from left to right. If we have both in the left, the slow comes before the fast.
    #UhL:slow_L,fast_L,virtual_L--> slow,fast-UhL-C-
    #VhR:virtual_R,slow_R,fast_R --> -C-VhR--slow,fast

    #____________________________________________Random Slow and Fast MPS_________________________________________________________
    #fast:v_left,v_right
    fast = np.random.rand(dfast1,dfast1) +1j*np.random.rand(dfast1,dfast1)
    #normalize:
    norm=ncon([fast,fast.conj()],[(1,2),(1,2)])
    fast=fast/np.sqrt(norm)

    #slow: v_left,physical(up\down),v_right
    slow = np.random.rand(int(dim_L/dfast1),dim_s,int(dim_L/dfast1)) +1j*np.random.rand(int(dim_L/dfast1),dim_s,int(dim_L/dfast1))
    norm=ncon([slow,slow.conj()],[(1,2,3),(1,2,3)])
    slow=slow/np.sqrt(norm)

    return UhL,VhR, fast,slow


