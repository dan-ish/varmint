import numpy as np
np.seterr(all='raise')
import scipy
import scipy.linalg
import time as time
from functools import wraps
import inspect
import sys
import traceback
import scipy.optimize as sciop

deriv_tol = 1e-6
max_iter = None
eq_tol = 1e-10

class TraceException(Exception):
    def __init__(self,e,location):
        super(TraceException,self).__init__(e,location)
        self.err = e
        self.loc = location

    def __repr__(self):
        return self.loc+"->"+repr(self.err)

    def __str__(self):
        return self.loc+"->"+str(self.err)

def tracing(f):
    @wraps(f)
    def traced(*args,**kwargs):
        try:
            return f(*args,**kwargs)
        except Exception as e:
            lineno = traceback.extract_tb(sys.exc_info()[2],2)[1][1]
            raise TraceException(e,f.__module__+":"+f.__name__+"@"+str(lineno))
    return traced


class Run_Result(object):
    @tracing
    def __init__(self,N,u,Tstep,numT):
        self.elapsed = -time.time()
        self.N = N
        self.u = u

        self.numT =  numT
        self.fe = np.zeros(self.numT)
        self.s = np.zeros(self.numT)
        self.cv = np.zeros(self.numT)
        self.c2 = np.zeros(self.numT)
        self.g2 = np.zeros(self.numT)
        self.x = np.zeros(self.numT)

    @tracing
    def record_const(self,Bnorm):
        self.fe_const = -self.u*(Bnorm**2)/2

    @tracing
    def record_ev(self,ev,x,tranInd,Tstep):
        self.ev = ev
        self.tranInd = tranInd
        if tranInd < self.numT:
            self.fe[tranInd:] = (-Tstep*(np.arange(tranInd,self.numT)+1)
                                    *self.N*np.log(np.sqrt(2))
                                    +self.fe_const)
            self.s[tranInd:] = self.N*np.log(np.sqrt(2))
            self.x[tranInd:] = x


    @tracing
    def record_T_res(self,T_ind,f_val):
        if T_ind >= self.numT:
            return
        self.fe[T_ind] = f_val.f - f_val.T*f_val.N*np.log(np.sqrt(2))+self.fe_const
        self.s[T_ind]  = (f_val.N*np.log(np.sqrt(2))-nent(f_val.g/f_val.T))
        self.cv[T_ind] = (f_val.G
                            .dot(f_val.pert)
                            .dot(f_val.G)/f_val.T)

        self.c2[T_ind] = 2*np.linalg.norm(1-OM_tanh(f_val.g/f_val.T))**2
        self.g2[T_ind] = 2*np.linalg.norm(f_val.g)**2
        self.x[T_ind] = -np.trace(f_val.pert)

    @tracing
    def finish(self,status,message=None):
        self.elapsed = self.elapsed+time.time()
        self.stat = status
        self.mess = message

class F_val(object):
    @tracing
    def __init__(self,L,T,N,G):
        self.L = L
        self.T = T
        self.N = N
        self.G = G

    @tracing
    def F(self,G):
        if not np.all(np.abs(G-self.G)<eq_tol):
            self.G = G
            self.flush()
        return self.f

    @tracing
    def Grad(self,G):
        if not np.all(np.abs(G-self.G)<eq_tol):
            self.G = G
            self.flush()
        return self.grad

    @tracing
    def flush(self):
        try:
            del self.f
        except AttributeError:
            pass

        try:
            del self.g
        except AttributeError:
            pass

        try:
            del self.phi
        except AttributeError:
            pass

        try:
            del self.C
        except AttributeError:
            pass

        try:
            del self.grad_like
        except AttributeError:
            pass

        try:
            del self.eng_grad
        except AttributeError:
            pass

        try:
            del self.free_hess
        except AttributeError:
            pass

        try:
            del self.grad
        except AttributeError:
            pass

        try:
            del self.pert
        except AttributeError:
            pass

    @tracing
    def __getattr__(self,name):
        if name == 'g' or name == 'phi':
            (self.g,self.phi) = diag(self.G,self.N)
            return self.__getattribute__(name)

        if name == 'C':
            self.C = make_mat(1-OM_tanh(self.g/self.T),self.phi)
            return self.C

        if name == 'f':
            self.f = self.eng_grad.dot(self.C)/2+self.T*nent(self.g/self.T)
            return self.f

        elif name == 'eng_grad':
            self.eng_grad = self.L.dot(self.C)
            return self.eng_grad

        elif name == 'grad_like':
            self.grad_like = self.eng_grad + self.G
            return self.grad_like

        elif name == 'free_hess':
            self.free_hess = free_hess(self.g,self.phi,self.T)
            return self.free_hess

        elif name == 'grad':
            self.grad = -self.free_hess.dot(self.grad_like)
            return self.grad

        elif name == 'pert':
            self.pert = self.free_hess.dot(
                        np.linalg.inv(
                            self.L.dot(self.free_hess) - np.eye(self.N*(self.N-1)/2)
                                     )
                                           )
            return self.pert

    @tracing
    def changeG(self,G):
        self.G = G
        self.flush()

    @tracing
    def change_T(self,new_T):
        self.T = new_T
        self.flush()


@tracing
def flatind(i,j,N):
    if np.any(i<0) or np.any(i>= j) or np.any(j>=N):
        raise ValueError('Index out of bounds')
    return i*N - i*(i+1)/2 + (j - i-1)

@tracing
def make_J(N):
    np.random.seed()
    dim = N*(N-1)/2
    J = np.zeros((dim,dim))
    for i in range(N):
        for j in range(i+1,N):
            for k in range(j+1,N):
                for l in range(k+1,N):
                    rand = np.random.normal(0.0,1.0/N**1.5)
                    J[flatind(i,j,N),flatind(k,l,N)] = rand
                    J[flatind(k,l,N),flatind(i,j,N)] = rand

                    J[flatind(i,k,N),flatind(j,l,N)] = -rand
                    J[flatind(j,l,N),flatind(i,k,N)] = -rand

                    J[flatind(i,l,N),flatind(j,k,N)] = rand
                    J[flatind(j,k,N),flatind(i,l,N)] = rand
    return J

@tracing
def make_U(N,result):
    np.random.seed()

    dim = N*(N-1)/2

    B = np.array([np.random.normal(0.0,1.0/N) for i in range(dim)])
    result.record_const(np.linalg.norm(B))

    U = -np.outer(B,B) + out_prod(B,N)
    return U

@tracing
def out_prod(M,N):
    proj = np.zeros((N**2,N*(N-1)/2))
    for i in range(N):
        for j in range(i+1,N):
            proj[i+N*j,flatind(i,j,N)] = 1/np.sqrt(2)
            proj[j+N*i,flatind(i,j,N)] = -1/np.sqrt(2)

    full_mat = np.zeros((N,N))
    full_mat[np.triu_indices(N,1)] = M
    full_mat = full_mat-np.transpose(full_mat)

    rv = np.kron(full_mat,full_mat)
    return np.transpose(proj).dot(rv).dot(proj)

@tracing
def make_mat(eig,phi):
    N = 2*eig.shape[0]
    M = np.zeros((N,N))
    inds = 2*np.arange(N/2)
    M[inds,inds+1] = eig
    M[inds+1,inds] = -eig
    M = phi.dot(M).dot(np.transpose(phi))
    return M[np.triu_indices(N,1)]

@tracing
def diag(G,N):
    if np.all(G==0):
        return (np.zeros(N/2),np.identity(N))
    mat = np.zeros((N,N))+0j
    mat[np.triu_indices(N,1)] = 1j*G
    eigs = scipy.linalg.eigh(mat,lower=False,eigvals=(N/2,N-1))
    phi = np.zeros((N,N))
    phi[:,2*np.arange(N/2)] = np.sqrt(2)*np.real(eigs[1])
    phi[:,2*np.arange(N/2)+1] = -np.sqrt(2)*np.imag(eigs[1])
    return (eigs[0],phi)


#actually computes the negative of the entropy
#Note the constant factor in the entropy has been dropped
#so the trial free energy at G = 0 is 0
@tracing
def nent(e):  
    return np.sum(OM_tanh(e)*np.log(OM_tanh(e)) + (2-OM_tanh(e))*np.log(2-OM_tanh(e)))/2

#computes 1- tanh
@tracing
def OM_tanh(e):
    return 2.0/(1+np.exp(2*e))

#computes the hessian of the parametrization
@tracing
def free_hess(g,phi,T):
    N = 2*np.size(g)
    if np.all(g==0):
        return -np.identity(N*(N-1)/2)/T
    R = np.kron(np.transpose(phi),np.transpose(phi))
    proj = np.zeros((N**2,N*(N-1)/2))
    for i in range(N):
        for j in range(i+1,N):
            proj[i+N*j,flatind(i,j,N)] = 1/np.sqrt(2)
            proj[j+N*i,flatind(i,j,N)] = -1/np.sqrt(2)
    R = np.transpose(proj).dot(R.dot(proj))

    hess = np.zeros((N*(N-1)/2,N*(N-1)/2))
    for mu in range(N/2):
        hess[flatind(2*mu,2*mu+1,N),flatind(2*mu,2*mu+1,N)] = 1.0/(np.cosh(g[mu]/T)**2)/T
        for nu in range(mu+1,N/2):

            diff = (OM_tanh(g[mu]/T) - OM_tanh(g[nu]/T))/(g[nu] - g[mu])/2
            suma = (2-OM_tanh(g[mu]/T) - OM_tanh(g[nu]/T))/(g[mu] + g[nu])/2

            hess[flatind(2*mu,2*nu,N),flatind(2*mu,2*nu,N)] = diff + suma
            hess[flatind(2*mu+1,2*nu+1,N),flatind(2*mu+1,2*nu+1,N)] = diff + suma
            hess[flatind(2*mu,2*nu,N),flatind(2*mu+1,2*nu+1,N)]= diff - suma 
            hess[flatind(2*mu+1,2*nu+1,N),flatind(2*mu,2*nu,N)] = diff - suma

            hess[flatind(2*mu+1,2*nu,N),flatind(2*mu+1,2*nu,N)] = diff + suma
            hess[flatind(2*mu,2*nu+1,N),flatind(2*mu,2*nu+1,N)] = diff + suma
            hess[flatind(2*mu+1,2*nu,N),flatind(2*mu,2*nu+1,N)]= suma - diff 
            hess[flatind(2*mu,2*nu+1,N),flatind(2*mu+1,2*nu,N)] = suma - diff

    return -np.transpose(R).dot(hess.dot(R))

@tracing
def diagnostic(N,u,Tstep,numT):
    result = Run_Result(N,u,Tstep,numT)
    Gs = np.zeros((numT,N*(N-1)/2))
    gs = np.zeros((numT,N/2))
    phis = np.zeros((numT,N,N))
    L = make_J(N)+ u*make_U(N,result)
    if u<=0:
        L = 3*np.sqrt(N)*L/4
    else:
        L = 2*L
    eigs = scipy.linalg.eigh(L,eigvals =(0,0))
    tranInd = int(np.floor(np.abs(eigs[0][0])/Tstep))
    dim = N*(N-1)/2
    Ts = Tstep*(np.arange(tranInd,numT)+1)
    Ts.shape = (Ts.shape[0],1,1)
    x = np.reshape(L,(1,dim,dim))/Ts + np.reshape(np.identity(dim),(1,dim,dim))
    Ts.shape = (Ts.shape[0],)
    x = -np.trace(np.linalg.inv(x),axis1=1,axis2=2)/Ts
    result.record_ev(eigs[0][0],x,tranInd,Tstep)

    T_inds = np.arange(tranInd,0,-1)

    f_val = F_val(L,tranInd*Tstep,N,np.zeros(N*(N-1)/2)+0.1*eigs[1][1])

    for i in T_inds:
        print "T = {0}".format(i*Tstep)
        if i<tranInd:
                f_val.change_T(i*Tstep)
        try:
            res= sciop.minimize(f_val.F,f_val.G,
                    method='BFGS',jac=f_val.Grad,
                    options={'gtol':deriv_tol,'norm':2,'maxiter':max_iter})
        except Exception as e:
            result.finish('f {0} {1}'.format(tranInd - i+1,tranInd),repr(e))
            return result
        if res.success:
            f_val.changeG(res.x)
            if i-1<numT:
                Gs[i-1] = res.x
                gs[i-1] = f_val.g
                phis[i-1] = f_val.phi
        else:
            result.finish('f {0} {1}'.format(tranInd - i+1,tranInd),
                            "minimizer failed. Message:"+res.message)

        result.record_T_res(i-1,f_val)

    result.finish('s')
    return (result,L,Gs,gs,phis)

@tracing
def do_instance(N,u,Tstep,numT):
    result = Run_Result(N,u,Tstep,numT)
    L = make_J(N)+ u*make_U(N,result)
    if u<=0:
        L = np.sqrt(N)*L/2
    else:
        L = 2*L
    eigs = scipy.linalg.eigh(L,eigvals =(0,0))
    tranInd = int(np.floor(np.abs(eigs[0][0])/Tstep))
    dim = N*(N-1)/2
    Ts = Tstep*(np.arange(tranInd,numT)+1)
    Ts.shape = (Ts.shape[0],1,1)
    x = np.reshape(L,(1,dim,dim))/Ts + np.reshape(np.identity(dim),(1,dim,dim))
    Ts.shape = (Ts.shape[0],)
    x = -np.trace(np.linalg.inv(x),axis1=1,axis2=2)/Ts
    result.record_ev(eigs[0][0],x,tranInd,Tstep)

    T_inds = np.arange(tranInd,0,-1)

    f_val = F_val(L,tranInd*Tstep,N,np.zeros(N*(N-1)/2)+0.1*eigs[1][1])

    for i in T_inds:
        print "T = {0}".format(i*Tstep)
        if i<tranInd:
                f_val.change_T(i*Tstep)
        try:
            res= sciop.minimize(f_val.F,f_val.G,
                    method='BFGS',jac=f_val.Grad,
                    options={'gtol':deriv_tol,'norm':2,'maxiter':max_iter})
        except Exception as e:
            result.finish('f {0} {1}'.format(tranInd - i+1,tranInd),repr(e))
            return result
        if res.success:
            f_val.changeG(res.x)
        else:
            result.finish('f {0} {1}'.format(tranInd - i+1,tranInd),
                            "minimizer failed. Message:"+res.message)

        result.record_T_res(i-1,f_val)

    result.finish('s')
    return result
    

