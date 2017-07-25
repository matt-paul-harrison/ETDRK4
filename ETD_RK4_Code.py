
import numpy as np
import pylab as plb
import matplotlib.pyplot as plt
from scipy import asarray as ar,exp
import random as rand
import time
from scipy.signal import argrelmax
import warnings
import matplotlib as mpl
import os
import scipy
class etd:
    def __init__(self,N,L,h,T_Final,T_snapshots,uxx_coeff,uxxxx_coeff,tdiv,u_initial):
        self.h=h
        self.N=N
        self.L=L
        self.h=h
        self.T_Final=T_Final
        self.T_snapshots=T_snapshots
        self.uxx_coeff=uxx_coeff
        self.uxxxx_coeff=uxxxx_coeff
        self.n=int(N/2+1)
        self.tdiv=T_snapshots
        self.u_initial=u_initial
        self.dx=(1.*L)/(1.*N)
        self.Nv = Nv_glob
        self.FF=[]
        #print("init run")
    def anaAnal(self,ana,thresh=10**-2):
        i=0
        while ana[i]>thresh:
            i+=1
            if i==len(ana)-1:
                return len(ana)
        return i
    def _ux(self,u,dx):
        um1 = np.roll(u,1,axis=0)
        up1 = np.roll(u,-1,axis=0)
        out = (-um1/2+up1/2)/dx
        return out
    def spacetime(self,FF):
        QQ=np.array(FF[:])
        phi_m = np.arange(0, len(FF[1]),dtype="int")*self.dx
        phi_p = np.arange(0, len(FF),dtype="int")*self.tdiv
        fig, ax = plt.subplots(figsize=[20,10])
        plt.figure(figsize=[20,10])
        p = ax.pcolor(phi_m, phi_p, QQ, cmap=plt.cm.gist_heat, vmin=np.min(QQ), vmax=np.max(QQ))
        cb = fig.colorbar(p, ax=ax)
        ax.set_ylabel('Time',fontsize=40)
        ax.set_xlabel('$x$',fontsize=40)

    def linmax(self,L,A=1,D=1):
        return np.sqrt((1.*A)/(1.*D))*self.L/(2*np.pi*np.sqrt(2))
    def yrejNEW(self,yshort):
        n=self.n
        N=self.N
        GG=np.zeros(N,dtype=complex)
        GG[n:]=np.conjugate(yshort[::-1][0:-1])
        GG[0:n]=yshort
        return GG

    def A(self,n):
        L=self.L
        n=self.n
        QR=np.zeros([n-1,n-1])
        for i in range(n-1):
            QR[i][i]=(2*np.pi/L)**2*(i+1)**2-(2*np.pi/L)**4*(i+1)**4
        return QR
        ###A and GALNEW take the half length vector
    def sparexpm(self,A):
        for i in range(len(A)):
            A[i][i]=np.exp(A[i][i])
        return A
    def EAF(self,z):
        h=self.h
        return self.sparexpm(h*z)
    def histplot(self,FF):
        QQ=np.fft.ifft(FF[:])
        phi_m = np.arange(0, N,dtype="int")
        phi_p = np.arange(0, len(FF),dtype="int")
        fig, ax = plt.subplots(figsize=[20,10])
        p = ax.pcolor(phi_m, phi_p, QQ, cmap=plt.cm.RdBu, vmin=np.min(QQ), vmax=np.max(QQ))
        cb = fig.colorbar(p, ax=ax)
    def matrix_to_list(self,M):
        spy=[]
        for i in range(len(M)):
            spy.append(M[i][i])
        return spy
    def AN(self,N):
        uxx_coeff=self.uxx_coeff
        uxxxx_coeff=self.uxxxx_coeff
        L=self.L

        n=int(N/2+1)
        QR=np.zeros([N,N])
        for i in range(n):
            QR[i][i]=uxx_coeff*(2*np.pi/L)**2*(i)**2-uxxxx_coeff*(2*np.pi/L)**4*(i)**4
        for i in range(n):
            QR[-i][-i]=uxx_coeff*(2*np.pi/L)**2*(i)**2-uxxxx_coeff*(2*np.pi/L)**4*(i)**4
        return QR
    
    def a(self,v,t):
        return self.E2*v+self.Q*self.Nv(self.gv,v,t)
    def Na(self,v,t):
        return self.Nv(self.gv,self.a(v,t),t)
    def b(self,v,t):
        return self.E2*v+(self.Q*self.Na(v,t))
    def Nb(self,v,t):
        return self.Nv(self.gv,self.b(v,t),t)
    def c(self,v,t):
        return (self.E2*self.a(v,t))+(self.Q*(2*self.Nb(v,t)-self.Nv(self.gv,v,t)))
    def Nc(self,v,t):
        return self.Nv(self.gv,self.c(v,t),t)
    def vnew(self,v,t):
        E=self.E
        f1=self.f1
        f2=self.f2
        f3=self.f3
        return E*v+self.Nv(self.gv,v,t)*f1+2*(self.Na(v,t)+self.Nb(v,t))*f2+(self.Nc(v,t)*f3)
    def FAN(self,z):
        N=self.N
        h=self.h
        return np.dot(np.linalg.inv(self.AN(N)+z*np.identity(N)),(scipy.linalg.expm((self.AN(N)+z*np.identity(N))*h/2)-np.identity(N)))
    def alpha(self,z):
        h=self.h
        N=self.N
        return h*np.dot(np.identity(N)*(-4)-z*h+np.dot(self.EAF(z),(np.identity(N)*4-3*z*h+h**2*np.dot(z,z))),np.linalg.matrix_power(h*z,-3))
    def beta(self,z):
        h=self.h
        N=self.N
        return h*np.dot(2*np.identity(N)+z*h+np.dot(self.EAF(z),(-2*np.identity(N)+h*z)),np.linalg.matrix_power(h*z,-3))
        
    def gamma(self,z):
        h=self.h
        N=self.N
        return np.dot((1/h**2)*np.linalg.matrix_power(z,-3),-4*np.identity(N)-3*h*z-h**2*np.dot(z,z)+np.dot(self.EAF(z),(np.identity(N)*4-h*z)))
        
    def _uxx(self,u,dx):
        um1 = np.roll(u,1)
        um2 = np.roll(u,2)
        um3 = np.roll(u,3)
        um4 = np.roll(u,4)
        up1 = np.roll(u,-1)
        up2 = np.roll(u,-2)
        up3 = np.roll(u,-3)
        up4 = np.roll(u,-4)
        out = (-um4/560+8*um3/315-um2/5+8*um1/5-205*u/72+8*up1/5-up2/5+8*up3/315-up4/560)/(dx**2)
        return out
    def vtrans(self,v):
        N=len(v)
        ksub=np.zeros(1,dtype=complex)
        ksub[0]=v[0]
        for i in range(1,int(N/2+1)):
            ksub=np.append(ksub,v[i]*(1.*L)/(2.*np.pi*i*1.0j))
        ksub=self.yrejNEW(ksub)
        return ksub

    def calculate_matrices(self):
        N=self.N
        L=self.L
        uxx_coeff=self.uxx_coeff
        uxxxx_coeff=self.uxxxx_coeff
        h=self.h
        
        self.start_time = time.time()
        RESET=1
        RECALC=1
        self.n=int(N/2+1)
        n=self.n
        
        self.dx=(1.*L)/(1.*N)
        dx=self.dx

        ss=16
        if RESET==1:
            FF=[]
            FH=[]
            t=0
            GG=np.arange(N,dtype=complex)
            jj=np.arange(2*n-1)
            #################################

        print("Calculating B")
        LL=self.AN(N)
        EAL=self.sparexpm(h*LL/2.)
        EALF=self.sparexpm(h*LL)
        SP=np.zeros([N,N])
        for i in range(n):
            SP[i][i]=(i*2*np.pi/L)
        for i in range(n):
            SP[-i][-i]=-(i*2*np.pi/L)
        g=-.5*1.0j*SP
        SP2=np.zeros([N,N])
        for i in range(n):
            SP2[i][i]=(i*2*np.pi/L)**3
        for i in range(n):
            SP2[-i][-i]=-(i*2*np.pi/L)**3
        g2=-.5*1.0j*SP2
        ML=[]
        for i in range(ss):
            ML.append(np.exp(2*np.pi*1.0j*(i+.5)/(1.0*ss)))
        Qr=np.zeros([N,N])
        for i in range(ss):
            Qr=Qr+np.real(self.FAN(ML[i]))
        APROP=Qr/ss #####FIXED APROP
        Qg=np.zeros(N)
        for i in range(ss):
            Qg=Qg+np.real(self.alpha(LL+np.identity(N)*ML[i]))
        ALPHA=Qg/ss ########FIXED ALPHA
        Qg=np.zeros(N)
        for i in range(ss):
            Qg=Qg+np.real(self.beta(LL+np.identity(N)*ML[i]))
        BETA=Qg/ss
        Qg=np.zeros(N)
        for i in range(ss):
            Qg=Qg+np.real(self.gamma(LL+np.identity(N)*ML[i]))
        GAMMA=Qg/ss
        ###############FULL LENGTH DEFINITIONS GIVEN ABOVE##################
        self.Q=self.matrix_to_list(APROP)
        self.E=self.matrix_to_list(EALF)
        self.E2=self.matrix_to_list(EAL)
        self.f1=self.matrix_to_list(ALPHA)
        self.f2=self.matrix_to_list(BETA)
        self.f3=self.matrix_to_list(GAMMA)
        self.gv=np.array(self.matrix_to_list(g))
        self.gv2=np.array(self.matrix_to_list(g2))
        ##############################

        ##############################
        print("startup time %s" %(time.time() - self.start_time))
        #################
        #return Q,E,E2,f1,f2,f3,gv,gv2

    
    def integrate(self,t=0):
        st=time.time()
        N=self.N
        n=self.n
        dx=self.dx
        Q=self.Q
        E=self.E
        TIME=self.T_Final
        use_stored=1
        FF=[]
        FH=[]
        t=0
        tmax=TIME
        tstart=0
        GG=np.arange(N,dtype=complex)
        jj=np.arange(2*n-1)
        y0=list(np.arange(N))
        u=self.u_initial
        counter=1
        v=np.fft.fft(u)
        t=0
        while t<tmax:
            v=self.vnew(v,t)
            u=np.real(np.fft.ifft(v))
            v=np.fft.fft(u)
            t+=h
            v[0]=0
            if t>tstart:
                FF.append(np.real(np.fft.ifft(self.vtrans(v)))) ##Keeps track of the real space surface
                FH.append(np.real(np.fft.ifft(v)))         #Keeps track of the slope
                tstart+=self.tdiv
                counter+=1
                if counter%100==0:
                    counter=1
                    print(str(round(100*t/float(tmax),2))+"% done")
        print("Integration time was "+str(time.time()-st)+" seconds")
        return FF
    def similarity1(self,u1,u2):
        su1=np.roll(u1,-np.argmax(u1))
        su2=np.roll(u2,-np.argmax(u2))
        return np.sqrt(np.sum((su1-su2)**2))/(len(su1))
    def similarity2(self,u1,u2):

        a=(len(u1))
        sk=[]
        for i in range(len(u1)):
            sk.append(np.sqrt(np.sum((u1-np.roll(u2,i))**2)))
        return min(sk)/a
    def Anal(self,FF,refu,m=1):
        sk=[]
        if m==1:
            for i in range(len(FF)):
                sk.append(self.similarity1(np.fft.ifft(FF[i]),refu))
        if m==2:
            for i in range(len(FF)):
                sk.append(self.similarity2(np.fft.ifft(FF[i]),refu))
        return sk
    def snapshot(self,t):
        i=int(t/self.tdiv)
        plt.figure(figsize=[7.5,5.5])
        plt.title("T="+str(t),fontsize=28)
        plt.plot(np.arange(N)*self.dx,FF[i])
        plt.xlabel('x',fontsize=28)
        plt.ylabel('u',fontsize=28)

        
#### By default, ETD method integrates the derivative of the dependent variable, Nv should be defined accordingly
#### The output FF is the real space surface. FF[i] is the surface at time t=i*T_snapshots
#### Function etd.histplotR(FF) gives a spacetime plot of the simulation
#### 
### Options for etd class:
N=451                    # N  -   Number of discretized points (Must be odd)
L=200                    # L  -   Full length of the simulation region
h=0.05                   # h  -   Length of one timestep
T_Final=300              # T_Final  -   How long to run the simulation for
T_snapshots=1            # T_snapshots - How long in between saved surface snapshots
uxx_coeff=1              # uxx_coeff - Coefficient of the uxx term, positive is destabilizing
uxxxx_coeff=1            # uxxxx_coeff - Coefficient of the uxxxx term, positive is stabilizing
lmbda=1                  # lmbda  - coefficient of the ux^2 term
r2=.35                   # r2     - coefficient of the ux^3 term
#########################
def Nv_glob(gv,v,t):
    # Nv_glob -   Nonlinear operator which takes inputs and outputs in Fourier Space.
    # Whatever function this is set to when the ETD class is instantiated will be used for that object forever
    return lmbda*gv*np.fft.fft(np.real(np.fft.ifft(v))**2)+r2*gv*np.fft.fft(np.real(np.fft.ifft(v))**3)
#########################
def Initial_Condition(N,amp=10**-4):
    ##Defines initial condition, by default it's low amplitude white noise
    initial_u=np.random.randn(N)*amp
    return initial_u
##########################
u_initial=Initial_Condition(N)
ETD=etd(N,L,h,T_Final,T_snapshots,uxx_coeff,uxxxx_coeff,T_snapshots,u_initial) ##instantiates the ETD RK4 Class, named ETD here
ETD.calculate_matrices() #Necessary if and only if simulation parameters have changed
ETD.FF=ETD.integrate()
plt.plot(np.arange(ETD.N)*ETD.dx,ETD.FF[-1])
plt.show()
ETD.spacetime(ETD.FF)
plt.show()


