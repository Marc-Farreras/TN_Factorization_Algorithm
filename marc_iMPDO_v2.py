import numpy as np
import scipy.linalg as LA
from scipy.sparse.linalg import LinearOperator, eigs
from functools import partial
from ncon import ncon
import json
import jax
import jax.numpy as jnp
from jax import grad
from tqdm import tqdm 
import utils 
import factorization
import iTEBD
import iTEBD_factorized
import gradient_descent

jax.config.update("jax_enable_x64", True)



#parameters needed:
sigma_y = np.array([[0.,-1j],[1j,0.]])
sigma_x = np.array([[0.,1.],[1.,0.]])
sigma_z = np.array([[1.,0.],[0.,-1.]])
identity=np.eye(2)
d=2
#the Hamiltonian has two parameters for each type of interaction (J,g). Hence we will have two different epsilon
delta_t=0.05
J=1.
g=2.
Ham_2body=-J*np.kron(sigma_z,sigma_z)-(g/2.)*(np.kron(identity,sigma_x)+np.kron(sigma_x,identity))
#time_op index:spin1_out,spin2_out,Spin1_in,Spin2_in
gateBA=LA.expm(-1j*delta_t*Ham_2body).reshape(2,2,2,2)
gateAB=LA.expm(-1j*delta_t*Ham_2body).reshape(2,2,2,2)




def expect_Bl(Bl,ML,NR):
    sigma_x = np.array([[0.,1.],[1.,0.]])
    Bl=Bl.reshape(Bl.shape[0],2,2,Bl.shape[2])
    norm=ncon([ML,Bl,NR,ML.conj(),Bl.conj(),NR.conj()],[(1,2,3),(3,4,5,7),(7,8,10),(1,2,6),(6,4,5,9),(9,8,10)])
    expect_X=ncon([ML,Bl,NR,sigma_x,ML.conj(),Bl.conj(),NR.conj()],[(1,2,3),(3,4,5,8),(8,9,11),(6,5),(1,2,7),(7,4,6,10),(10,9,11)])/norm
    return expect_X


#ENSAMBLE THE MIXED STATE FORM BY THE TENSORS: Bl,NR,ML

def ensamble_mix_state(UhL,VhR, fast,slow):
    #Reminder:
        #UL:virtual_left,slow,fast
        #VR:slow,fast,virtual_right
        #fast:v_left,v_right
        #slow: v_left,physical(up\down),v_right

    #Now we have the decopled states fast and slow. We can finally obtain the purification state. First,the real state psi
    #consist in the application of U-slow_fast-V, but from above we have the Uh and Vh
    dim_L=UhL.shape[2]
    dfast1=fast.shape[0]

    UL=(np.transpose(np.conjugate(UhL.reshape(int(dim_L/dfast1)*dfast1,dim_L)))).reshape(dim_L,int(dim_L/dfast1),dfast1)
    VR=(np.transpose(np.conjugate(VhR.reshape(dim_L,int(dim_L/dfast1)*dfast1)))).reshape(int(dim_L/dfast1),dfast1,dim_L)

    #The Bl tensor corresponds to slow:
    Bl=np.copy(slow)
    #Bl:v_left(slow),physical(up\down),v_right(slow)

    #NR:contract fast and VR
    NR=ncon([fast,VR],[(-2,1),(-1,1,-3)])
    #NR:slow,fast\purification,virtual_right

    #ML: contract UL with fast
    ML=ncon([UL,fast],[(-1,-3,1),(1,-2)])
    #ML:virtual_left,fast\purification,slow

    return Bl,NR,ML





def canonical_form_trunc(Bl,ML,NR,sBA):
    chi_L=ML.shape[0]
    sig_L= np.eye(chi_L) / chi_L
        #put in vector form
    v0 = sig_L.reshape(np.prod(sig_L.shape))
            
        #Define tensor network
        # define network for transfer operator contract
    tensors=[ML,Bl,NR,ML.conj(),Bl.conj(),NR.conj(),np.diag((1./sBA)),np.diag((1./sBA))]
    labels=[[1,2],[1,3,4],[4,6,7],[7,9,10],[2,3,5],[5,6,8],[8,9,11],[10,-1],[11,-2]]
            
        # define function for boundary contraction and pass to eigs
    def left_iter(sig_L):
        return ncon([sig_L.reshape([chi_L, chi_L]), *tensors],labels).reshape([chi_L**2, 1])
    Dtemp, sig_L = eigs(LinearOperator((chi_L**2, chi_L**2), matvec=left_iter), k=1, which='LM', v0=v0, tol=1e-10)
            
        # normalize the environment density matrix sigBA
    sig_L = sig_L.reshape(chi_L, chi_L)
    herm=0.5 * (sig_L + np.conj(sig_L.T))
    sig_L = 0.5 * (sig_L + np.conj(sig_L.T))
    sig_L = sig_L / np.trace(sig_L)
        #remember now sig_L (goes to the left of the transfer matrix): up,down--> up-sig_L-down

        #__________________Find leading right vector__________________________:

        #initialize a random starting left vector
    chi_R=NR.shape[2]
    mu_R= np.eye(chi_R) / chi_R
        #put in vector form
    v0 = mu_R.reshape(np.prod(mu_R.shape))
            
        #Define tensor network
        # define network for transfer operator contract
    tensors=[np.diag((1./sBA)),np.diag((1./sBA)),ML,Bl,NR,ML.conj(),Bl.conj(),NR.conj()]
    labels=[[1,2],[-1,10],[-2,11],[10,9,7],[7,6,4],[4,3,1],[11,9,8],[8,6,5],[5,3,2]]
            
        # define function for boundary contraction and pass to eigs
    def right_iter(mu_R):
        return ncon([mu_R.reshape([chi_R, chi_R]), *tensors],labels).reshape([chi_R**2, 1])
    Dtemp, mu_R = eigs(LinearOperator((chi_R**2, chi_R**2), matvec=right_iter), k=1, which='LM', v0=v0, tol=1e-10)
            
        # normalize the environment density matrix muBA
    mu_R = mu_R.reshape(chi_R, chi_R)
    mu_R = 0.5 * (mu_R + np.conj(mu_R.T))
    mu_R = mu_R / np.trace(mu_R)
            
        #remember now mu_R (goes to the right of the transfer matrix): up,down--> up-muBA-down

        #_______________________Orthogonalize____________________________
        #Add the inverse sBA in the ML and NR tensor
    ML=ncon([np.diag((1./sBA)),ML],[(-1,1),(1,-2,-3)])
    NR=ncon([NR,np.diag((1./sBA))],[(-1,-2,1),(1,-3)])

        # diagonalize left environment matrix
        #M_original=U_M @ np.diag(eig_M) @U_M.T.conj()
    dtemp, utemp = LA.eigh(sig_L)
    dtol=1e-12
    chitemp = sum(dtemp > dtol)
    #chitemp=(len(dtemp))
    DL = dtemp[range(-1, -chitemp - 1, -1)]
    UL = utemp[:, range(-1, -chitemp - 1, -1)]
        #Decompose the left vector: up-sigBA-down=up-Y-Y.adj()-down where sigBA=UL@DL@UL.T.adj(), Y=UL@sqrt(DL)
        #but in fact we need for the calculations the vector:down-sigBA.T-up=down-Y.T.adj()-Y.T-up  where Y.T=sqrt(DL)@UL.T
    Y_trans=np.diag(np.sqrt(DL))@np.transpose(UL)
    Y_trans_inv=UL.conj()@np.diag(1 / np.sqrt(DL))
            
        # diagonalize right environment matrix
        #M_original=U_M @ np.diag(eig_M) @U_M.T.conj()
    dtemp, utemp = LA.eigh(mu_R)
    chitemp = sum(dtemp > dtol)
    #chitemp=(len(dtemp))
    DR = dtemp[range(-1, -chitemp - 1, -1)]
    UR = utemp[:, range(-1, -chitemp - 1, -1)]
        #Decompose the right vector: up-muBA-down=up-X-X.adj()-down where X=UR@sqrt(DR)
    X=UR@np.diag(np.sqrt(DR))
    X_inv=np.diag(1 / np.sqrt(DR))@np.transpose(UR.conj())
            
        # compute new weights for B-A link
    weighted_mat = Y_trans @ np.diag(sBA) @ X
    UBA, stemp, VhBA = LA.svd(weighted_mat, full_matrices=False)
    sBA=stemp/LA.norm(stemp)

        #New ML and NR tensors:
    ML=ncon([np.diag(sBA),VhBA,X_inv,ML],[(-1,1),(1,2),(2,3),(3,-2,-3)])
    NR=ncon([NR,Y_trans_inv,UBA,np.diag(sBA)],[(-1,-2,1),(1,2),(2,3),(3,-3)])
        #Test the left and right transfer matrix:
        #Transfer R and L (in Roman notation):
        
    #Transfer_M_R=ncon([np.diag((1./sBA)),np.diag((1./sBA)),ML,Bl,NR,ML.conj(),Bl.conj(),NR.conj()],[(-1,8),(-2,9),(8,1,2),(2,3,5),(5,6,10),(9,1,4),(4,3,7),(7,6,10)])
    #Transfer_M_L=ncon([ML,Bl,NR,ML.conj(),Bl.conj(),NR.conj(),np.diag((1./sBA)),np.diag((1./sBA))],[(10,1,2),(2,3,5),(5,6,8),(10,1,4),(4,3,7),(7,6,9),(8,-1),(9,-2)])
    #print('Transfer L',Transfer_M_L/np.abs(Transfer_M_L[0,0]))
    #print('Transfer L',Transfer_M_L)
    #print('Transfer R',Transfer_M_R/np.abs(Transfer_M_R[0,0]))
    #print('Transfer R',Transfer_M_R)
    norm=ncon([ML,Bl,NR,ML.conj(),Bl.conj(),NR.conj()],[(1,2,3),(3,4,5),(5,6,7),(1,2,8),(8,4,9),(9,6,7)])
    ML=ML/np.sqrt(np.sqrt(norm))
    NR=NR/np.sqrt(np.sqrt(norm))
    #UL,sL,VL=LA.svd(ML.reshape(ML.shape[0]*ML.shape[1],ML.shape[2]),full_matrices=False)
    #VL=np.diag(sL)@VL
    #ML=UL.reshape(ML.shape[0],ML.shape[1],len(sL))
    #UR,sR,VR=LA.svd(NR.reshape(NR.shape[0],NR.shape[1]*NR.shape[2]),full_matrices=False)
    #UR=UR@np.diag(sR)
    #NR=VR.reshape(len(sR),NR.shape[1],NR.shape[2])
    norm=ncon([ML,Bl,NR,ML.conj(),Bl.conj(),NR.conj()],[(1,2,3),(3,4,5),(5,6,7),(1,2,8),(8,4,9),(9,6,7)])
    return Bl,ML,NR,sBA





def correct_state(ML,Bl,NR):
    #Obtain ML 
    UL,sL,VL=LA.svd(ML.reshape(ML.shape[0]*ML.shape[1],ML.shape[2]),full_matrices=False)
    min_sv=1e-16
    d_tol=sum(sL > min_sv)
    chitemp=min(len(sL),d_tol)
    sL=sL[:chitemp]
    UL=UL[:,:chitemp]
    UL=UL.reshape(ML.shape[0],ML.shape[1],len(sL))
    VL=VL[:chitemp,:]
    VL=np.diag(sL)@VL

    #obtain NR
    UR,sR,VR=LA.svd(NR.reshape(NR.shape[0],NR.shape[1]*NR.shape[2]),full_matrices=False)
    min_sv=1e-16
    d_tol=sum(sR > min_sv)
    chitemp=min(len(sR),d_tol)
    sR=sR[:chitemp]
    UR=UR[:,:chitemp]
    UR=UR@np.diag(sR)
    VR=VR[:chitemp,:]
    VR=VR.reshape(len(sR),NR.shape[1],NR.shape[2])

    #put all into Bl
    Bl=ncon([VL,Bl,UR],[(-1,1),(1,-2,2),(2,-3)])
    ML=np.copy(UL)
    NR=np.copy(VR)
    
    return ML,Bl,NR




def reduce_purification(Bl,ML,NR,dpur=30):
    Bl=Bl.reshape(Bl.shape[0]*2,2*Bl.shape[2])
    U_Bl,sBl,Vh_Bl=LA.svd(Bl,full_matrices=False)
    #First truncation:
    d_tol=sum(sBl > 1e-9)
    chitemp=d_tol
    U_Bl=U_Bl[:,:chitemp]
    sBl=sBl[:chitemp]
    sBl=sBl/LA.norm(sBl)
    Vh_Bl=Vh_Bl[:chitemp,:]
    U_Bl=U_Bl@np.diag(sBl)
    Vh_Bl=np.diag(sBl)@Vh_Bl
    #reshape it
    U_Bl=U_Bl.reshape(int(Bl.shape[0]/d),d,chitemp)
    Vh_Bl=Vh_Bl.reshape(chitemp,d,int(Bl.shape[1]/d))

    #Add ML and NR to form an MPO
    A=ncon([ML,U_Bl],[(-2,-1,1),(1,-3,-4)])
    B=ncon([Vh_Bl,NR],[(-1,-2,1),(1,-4,-3)])
    
    #truncate purification
    _,spur_A,U_A=LA.svd(A.reshape(A.shape[0],A.shape[1]*A.shape[2]*A.shape[3]),full_matrices=False)
    spur_A=spur_A[:dpur]
    spur_A=spur_A/LA.norm(spur_A)
    U_A=U_A[:dpur,:]
    U_A=np.diag(spur_A)@U_A
    A=U_A.reshape(len(spur_A),A.shape[1],A.shape[2],A.shape[3])
    A=ncon([A],[(-2,-1,-3,-4)])

    U_B,spur_B,_=LA.svd(B.reshape(B.shape[0]*B.shape[1]*B.shape[2],B.shape[3]),full_matrices=False)
    spur_B=spur_B[:dpur]
    spur_B=spur_B/LA.norm(spur_B)
    U_B=U_B[:,:dpur]
    U_B=U_B@np.diag(spur_B)
    B=U_B.reshape(B.shape[0],B.shape[1],B.shape[2],len(spur_B))
    B=ncon([B],[(-1,-2,-4,-3)])

    #Recover the original form
    ML,sA,U_A=LA.svd(A.reshape(A.shape[0]*A.shape[1],A.shape[2]*A.shape[3]),full_matrices=False)
    d_tol=sum(sA > 1e-16)
    chitemp=d_tol
    ML=ML[:,:chitemp]
    sA=sA[:chitemp]
    sA=sA/LA.norm(sA)
    U_A=U_A[:chitemp,:]
    U_A=np.diag(sA)@U_A
    ML=ML.reshape(A.shape[0],A.shape[1],len(sA))
    A=U_A.reshape(len(sA),A.shape[2],A.shape[3])

    U_B,sB,NR=LA.svd(B.reshape(B.shape[0]*B.shape[1],B.shape[2]*B.shape[3]),full_matrices=False)
    d_tol=sum(sB > 1e-16)
    chitemp=d_tol
    U_B=U_B[:,:chitemp]
    sB=sB[:chitemp]
    sB=sB/LA.norm(sB)
    NR=NR[:chitemp,:]
    U_B=U_B@np.diag(sB)
    NR=NR.reshape(len(sB),B.shape[2],B.shape[3])
    B=U_B.reshape(B.shape[0],B.shape[1],len(sB))

    #add together A and B to form Bl
    Bl=ncon([A,np.diag((1./sBl)),B],[(-1,-2,1),(1,2),(2,-3,-4)])
    Bl=Bl.reshape(Bl.shape[0],2*2,Bl.shape[3])
    return Bl,ML,NR





#First iTEB(normal)

#Initial state:
sAB = np.ones(1)
sBA = np.ones(1)
A = ((1/np.sqrt(2))*np.array([1.,1.])).reshape(1,d,1)
B = ((1/np.sqrt(2))*np.array([1.,1.])).reshape(1,d,1)
time=[]
list_XA=[]
list_XB=[]
list_YA=[]
list_YB=[]
list_ZA=[]
list_ZB=[]
time_truncation=[]
bound_small_list=[]
bound_big_list=[]
cost_func_list=[]
print("begin TEBD normal")
for k in range(0,70):
    time.append(delta_t*k)
    print(delta_t*k)
    #compute expected values:
    norm_A,norm_B,X_A,X_B,Y_A,Y_B,Z_A,Z_B=utils.expect_val(A,B,sAB,sBA)
    list_XA.append(np.real(X_A))
    list_XB.append(X_B)
    list_YA.append(Y_A)
    list_YB.append(Y_B)
    list_ZA.append(Z_A)
    list_ZB.append(Z_B)
    bound_small_list.append(A.shape[2])
    bound_big_list.append(A.shape[0])


    #Apply iTEBD:
    A,sAB,B=iTEBD.iTEBD(A,sAB,B,sBA,gateAB,chi=200,min_sv=1e-5)
    B,sBA,A=iTEBD.iTEBD(B,sBA,A,sAB,gateBA,chi=200,min_sv=1e-5)

    if delta_t*k>2.1:

        
        #C=sBA-A-sAB-B-sBA
        C_MPS=ncon([np.diag(sBA),A,np.diag(sAB),B,np.diag(sBA)],[(-1,1),(1,-2,2),(2,3),(3,-3,4),(4,-4)])
        #Add the physical dimensions together. In this case the index order are: virtual_L,phisical_s,virtual_R
        C_MPS=C_MPS.reshape(C_MPS.shape[0],C_MPS.shape[1]*C_MPS.shape[2],C_MPS.shape[3])
        
        #compute the lowest factor and put it as the dfast dimension
        bond_dim=C_MPS.shape[0]
        #compute the slowest purification dmension
        dfast=2
        if bond_dim%2!=0:
            continue
        UhL,VhR, fast,slow,entropy,overlap=factorization.disentangling_state(C_MPS,dfast,max_iter=150,eps=1e-5)
        
        if entropy<0.003:
            Bl,NR,ML=ensamble_mix_state(UhL,VhR, fast,slow)
            num_loop=30
            rate_list=np.linspace(0.5,0.05,80)
            print('rate list',rate_list)
            for j in range(num_loop):
                print('num loop:',k)
                for rate in rate_list:
                    print('rate',rate)
                    Bl,NR,ML,opt_cost,convergence=gradient_descent.gradient_descent_opt(Bl,ML,NR,C_MPS,learning_rate=rate,num_steps=150)
                    print('Convergence:',convergence)
                    if convergence==True:
                        break
            if opt_cost>9e-6:
                    print('skip this')
                    continue
            cost_func_list.append(float(opt_cost))
            
            #compute expected values:
            norm_A,norm_B,X_A,X_B,Y_A,Y_B,Z_A,Z_B=utils.expect_val(A,B,sAB,sBA)
            list_XA.append(np.real(X_A))
            list_XB.append(X_B)
            list_YA.append(Y_A)
            list_YB.append(Y_B)
            list_ZA.append(Z_A)
            list_ZB.append(Z_B)
            bound_small_list.append(A.shape[0])
            bound_big_list.append(A.shape[0])
            time.append(delta_t*(k+1))
            t0=delta_t*(k+1)
            time_truncation.append(delta_t*(k+1))
            break


#put canonical form
Bl,ML,NR,sBA=canonical_form_trunc(Bl,ML,NR,sBA)
ML,Bl,NR=correct_state(ML,Bl,NR)
list_XA.append(np.real(expect_Bl(Bl,ML,NR)))
bound_small_list.append(Bl.shape[0])
bound_big_list.append(ML.shape[0])
time.append(t0)

#Real Algorithm to perform time evolution factorizing
for k in range(0,12):
    print("I have done", k+1, "truncations") 
    #initial time:
    for j in (pbar := tqdm(range(1,100))):
        pbar.set_description(f"T{t0+delta_t}, shapes= {Bl.shape}, {ML.shape}, {NR.shape}") 
    #print("time", t0+delta_t, "shapes=", Bl.shape, ML.shape, NR.shape) 
    #for j in range(1,100):
        t0=time[len(time)-1]

        #Perform iTEB 
        Bl,ML,NR,sBA=iTEBD_factorized.iTEBD_truncation(Bl,ML,NR,sBA,gateAB,chi=256,min_sv=1e-5,)
        time.append(t0+delta_t)

        #compute expected values:
        list_XA.append(np.real(expect_Bl(Bl,ML,NR)))
        bound_small_list.append(Bl.shape[0])
        bound_big_list.append(ML.shape[0])
        
        

        if j>=12:
            bond_dim=Bl.shape[0]
            #compute the slowest purification dmension
            dfast=2
            if bond_dim%2!=0 or (Bl.shape[0]!=Bl.shape[2]):
                continue
            UhL,VhR, fast,slow,entropy,overlap=factorization.disentangling_state(Bl,dfast,max_iter=150,eps=1e-7)
            print("disentangling entropy", entropy)
            if entropy<0.003:
                print("Good entropy, ", entropy, "factorizing")
                #Create the new mixed state
                Bl_new,NR_new,ML_new=ensamble_mix_state(UhL,VhR, fast,slow)
                #optimization process
                num_loop=30
                rate_list=np.linspace(0.5,0.05,80)
                for k in range(num_loop):
                    print('num loop:',k)
                    for rate in rate_list:
                        print('rate',rate)
                        Bl_new,NR_new,ML_new,opt_cost,convergence=gradient_descent.gradient_descent_opt(Bl_new,ML_new,NR_new,Bl,learning_rate=rate,num_steps=125)
                        print('Convergence:',convergence)
                        if convergence==True:
                            break
                if opt_cost>9e-6:
                    continue
                cost_func_list.append(float(opt_cost))
                #ended the optimization
                Bl=np.copy(Bl_new)
                #Add the impurity tensors together:
                ML=ncon([ML,ML_new],[(-1,-2,1),(1,-3,-4)])
                NR=ncon([NR_new,NR],[(-1,-2,1),(1,-3,-4)])
                #add together the impurity dimensions(fast)
                ML=ML.reshape(ML.shape[0],ML.shape[1]*ML.shape[2],ML.shape[3])
                NR=NR.reshape(NR.shape[0],NR.shape[1]*NR.shape[2],NR.shape[3])
                #ML:virtual_left,slow,fast
                #NR:slow,fast,virtual_right
                #put canonical form
                Bl,ML,NR,sBA=canonical_form_trunc(Bl,ML,NR,sBA)
                #Distribute well the sibgular values
                ML,Bl,NR=correct_state(ML,Bl,NR)
                dpur=32
                if ML.shape[1]>dpur:
                    Bl,ML,NR=reduce_purification(Bl,ML,NR,dpur)
                time_truncation.append(t0+delta_t)
                
                break


#save the data using json
# Combine the lists into a dictionary
data = {
    'list_time': time,
    'list_XA': list_XA,
    'list_truncation': time_truncation,
    'list_small_bound': bound_small_list,
    'list_big_bound': bound_big_list,
    'ML_real': np.real(ML).tolist(),
    'ML_imaginary': np.imag(ML).tolist(),
    'NR_real': np.real(NR).tolist(),
    'NR_imaginary': np.imag(NR).tolist(),
    'Bl_real': np.real(Bl).tolist(),
    'Bl_imaginary': np.imag(Bl).tolist(),
    'sBA_real': np.real(sBA).tolist(),
    'sBA_imaginary': np.imag(sBA).tolist(),
    'cost_func_list': cost_func_list
}

# Save the dictionary to a text file
with open('results_MPS.txt', 'w') as f:
    json.dump(data, f)








