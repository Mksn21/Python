import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import numpy as np
import matplotlib.pyplot as plt

time_values = []
theta1_values = []
theta2 = []
theta3 = [] 

#variabel 
thdd = np.zeros(3,dtype=float)
thdf = np.zeros(3,dtype=float)
M = np.zeros((3,3),dtype=float)
C = np.zeros((3,3),dtype=float)
G = np.zeros(3,dtype=float)

th = np.zeros(3,dtype=float)
th1 = 90* np.pi /180
th2 = -40 * np.pi/180
th3 = 0 *np.pi/180

thd1 = 0
thd2 = 0
thd3 = 0
thdd = np.zeros(3)

i1 = 0.133
l1 = 0.383 *1.25
a1 = 0.383 * 0.42
m1 = 6.86

i2 =0.048
l2 = 0.407 *1.25
a2 = 0.407 * 0.41
m2 = 2.76

i3 = 0.004
l3 = 0.149*1.25
a3 = 0.149 * 0.4
m3 = 0.89

g=9.8

tau = np.zeros(3)
tau[0] = 0
tau[1] = 0
tau[2] = 0
t = 0
dt = 0.01

fric = np.zeros(3)
# Muscle Model Iliopsoas
Fmilopas = 1100 
loptilopas = 0.35 
Cpdilopas = 275 
kpeilopas = 5.85
MailopasH = 0.132
ai = 0
#Muscle Model BFLH
FmBFLH = 2750
lopyBFLH = 0.46
CpdBflh = 275 
kpeBFLH = 4.1 
MaBFLHH = 0.054
MaBFLHK = 0.049
aBL = 0
#Muscle Model BFSH 
FmBFSH = 100 
loptBFSH = 0.29 
CpdBFSH = 200 
kpeBFSH = 1.6 
MaBFSHK = 0.049
aBH = 0
#Muscle Model Rectus Femoris 
FMRF = 1800
loptRF = 0.48 
CpdRF = 300 
kpeRF = 5.4 
MaRfH = 0.049 
MaRfK = 0.025
aRF =0 
#Muscle Model Gastroc med 
FmGm = 1150 
loptGm = 0.56 
CpdGm = 275 
kpeGm = 8.25 
MaGmK = 0.05
MaGmA = 0.04 
aGm = 0 
#Muscle Model Soleus 
FmS = 2150 
lopts = 0.35 
CpdS = 200 
kpeS = 6.5 
MasA = 0.036 
aS = 0
#Muscle Model Tibias Anterior 
FmTA = 1650 
loptTA = 0.3 
CpdTA = 200 
kpeTA = 1.3 
MaTA = 0.023 
aTa = 0

tau = np.zeros(3)
tau[0] = 0
tau[1] = 0
tau[2] = 0
thdd1 = 0
thdd2 = 0
thdd3 = 0 

def ttd1 (th1,thd1):
    M[0,0] = i1 + m1*(a1**2) + i2 + m2*((l1**2)+(a2**2)+(2*l1*a2*np.cos(th2-th1))) + i3 + m3*((l1**2)+(l2**2)+(a3**2)+(2*l1*l2*np.cos(th2))+(2*l2*a3*np.cos(th3))+(2*l2*a3*np.cos(th3-th2)))
    M[0,1] = -i2 - m2*((a2**2)+(l1*a2*np.cos(th2))) - i3 - m3*((l2**2)-(a3**2)-(l1*l2*np.cos(th2))-(2*l2*a3*np.cos(th3))-(l1*a3*np.cos(th3-th2)))
    M[0,2] = i3 + m3*((a3**2)-(l2*a3*np.cos(th3))+(l1*a3*np.cos(th3-th2)))

    M[1,0] = -i2 - m2*((a2**2)+(l1*a2*np.cos(th2-th1))) - i3 - m3*((l2**2)+(a3**2)+(l1*l2*np.cos(th2))+(2*l2*a3*np.cos(th3))+(l1*a3*np.cos(th3-th2)))
    M[1,1] = i2 + m2*(a2**2) +i3 + m3*((l2**2)+(a3**2)+(2*l2*a3*np.cos(th3)))
    M[1,2] = -i3 -m3*((a3**2)+(l2*a3*np.cos(th3)))

    M[2,0] = i3 + m3*((a3**2)+(l2*a3*np.cos(th3))+(l1*a3*np.cos(th3-th2)))
    M[2,1] = -i3 - m3*((a3)+(l2*a3*np.cos(th3)))
    M[2,2] = i3 + m3*(a3**2)

    C[1,0] = -(m2*l1*a2*thd1*np.sin(th2)) - (m3*l1*l2*thd1*np.sin(th2)) + (m3*l1*a3*thd1*np.sin(th3-th2))
    C[1,1] = (m2*l1*a2*thd1*np.sin(th2)) + (m3*l1*a3*thd1*np.sin(th3-th2)) + (m3*l1*a3*thd1*np.sin(th2))
    C[1,2] = m3*l1*a3*thd1*np.sin(th3-th1)
    C[2,0] = (m2*l2*a3*((2*thd2)-thd1)*np.sin(th3)) - (m3*l1*a3*np.sin(th3-th2))
    C[2,1] = -(m2*l2*a3*thd2*np.sin(th3)) + (m3*l1*a3*thd1*np.sin(th3-th2))
    C[2,2] = (m2*l2*a3*(thd2-thd1)*np.sin(th3)) - (m3*l1*a3*thd1*np.sin(th3-th2))

    G[0] = (m1*g*np.sin(th1)) + m2*g*((l1*np.sin(th1))-(a2*np.sin(th2-th1))) +m3*g*((l1*np.sin(th1))-(l2*np.sin(th2-th1))+(a3*np.sin(th1-th2+th3)))
    G[1] = (m2*g*a2*np.sin(th2-th1)) + m3*g*((l2*np.sin(th2-th1))-(a3*np.sin(th1-th2+th3)))
    G[2] = m3*g*a3*np.sin(th1-th2+th3)
    thdd1 = ((tau[0]-(C[0,0]*thd1+C[0,1]*thd2+C[0,2]*thd3)-G[0])-M[0,1]*thdd2-M[1,2]*thdd3)/M[0,0]
    return thdd1


def ttd2 (th2,thd2):
    M[0,0] = i1 + m1*(a1**2) + i2 + m2*((l1**2)+(a2**2)+(2*l1*a2*np.cos(th2-th1))) + i3 + m3*((l1**2)+(l2**2)+(a3**2)+(2*l1*l2*np.cos(th2))+(2*l2*a3*np.cos(th3))+(2*l2*a3*np.cos(th3-th2)))
    M[0,1] = -i2 - m2*((a2**2)+(l1*a2*np.cos(th2))) - i3 - m3*((l2**2)-(a3**2)-(l1*l2*np.cos(th2))-(2*l2*a3*np.cos(th3))-(l1*a3*np.cos(th3-th2)))
    M[0,2] = i3 + m3*((a3**2)-(l2*a3*np.cos(th3))+(l1*a3*np.cos(th3-th2)))

    M[1,0] = -i2 - m2*((a2**2)+(l1*a2*np.cos(th2-th1))) - i3 - m3*((l2**2)+(a3**2)+(l1*l2*np.cos(th2))+(2*l2*a3*np.cos(th3))+(l1*a3*np.cos(th3-th2)))
    M[1,1] = i2 + m2*(a2**2) +i3 + m3*((l2**2)+(a3**2)+(2*l2*a3*np.cos(th3)))
    M[1,2] = -i3 -m3*((a3**2)+(l2*a3*np.cos(th3)))

    M[2,0] = i3 + m3*((a3**2)+(l2*a3*np.cos(th3))+(l1*a3*np.cos(th3-th2)))
    M[2,1] = -i3 - m3*((a3)+(l2*a3*np.cos(th3)))
    M[2,2] = i3 + m3*(a3**2)

    C[1,0] = -(m2*l1*a2*thd1*np.sin(th2)) - (m3*l1*l2*thd1*np.sin(th2)) + (m3*l1*a3*thd1*np.sin(th3-th2))
    C[1,1] = (m2*l1*a2*thd1*np.sin(th2)) + (m3*l1*a3*thd1*np.sin(th3-th2)) + (m3*l1*a3*thd1*np.sin(th2))
    C[1,2] = m3*l1*a3*thd1*np.sin(th3-th1)
    C[2,0] = (m2*l2*a3*((2*thd2)-thd1)*np.sin(th3)) - (m3*l1*a3*np.sin(th3-th2))
    C[2,1] = -(m2*l2*a3*thd2*np.sin(th3)) + (m3*l1*a3*thd1*np.sin(th3-th2))
    C[2,2] = (m2*l2*a3*(thd2-thd1)*np.sin(th3)) - (m3*l1*a3*thd1*np.sin(th3-th2))

    G[0] = (m1*g*np.sin(th1)) + m2*g*((l1*np.sin(th1))-(a2*np.sin(th2-th1))) +m3*g*((l1*np.sin(th1))-(l2*np.sin(th2-th1))+(a3*np.sin(th1-th2+th3)))
    G[1] = (m2*g*a2*np.sin(th2-th1)) + m3*g*((l2*np.sin(th2-th1))-(a3*np.sin(th1-th2+th3)))
    G[2] = m3*g*a3*np.sin(th1-th2+th3)
    thdd2 = ((tau[1]-(C[1,0]*thd1+C[1,1]*thd2+C[1,2]*thd3)-G[1])-M[1,0]*thdd1-M[1,2]*thdd3) / M[1,1]
    return thdd2


def ttd3 (th3,thd3):
    M[0,0] = i1 + m1*(a1**2) + i2 + m2*((l1**2)+(a2**2)+(2*l1*a2*np.cos(th2-th1))) + i3 + m3*((l1**2)+(l2**2)+(a3**2)+(2*l1*l2*np.cos(th2))+(2*l2*a3*np.cos(th3))+(2*l2*a3*np.cos(th3-th2)))
    M[0,1] = -i2 - m2*((a2**2)+(l1*a2*np.cos(th2))) - i3 - m3*((l2**2)-(a3**2)-(l1*l2*np.cos(th2))-(2*l2*a3*np.cos(th3))-(l1*a3*np.cos(th3-th2)))
    M[0,2] = i3 + m3*((a3**2)-(l2*a3*np.cos(th3))+(l1*a3*np.cos(th3-th2)))

    M[1,0] = -i2 - m2*((a2**2)+(l1*a2*np.cos(th2-th1))) - i3 - m3*((l2**2)+(a3**2)+(l1*l2*np.cos(th2))+(2*l2*a3*np.cos(th3))+(l1*a3*np.cos(th3-th2)))
    M[1,1] = i2 + m2*(a2**2) +i3 + m3*((l2**2)+(a3**2)+(2*l2*a3*np.cos(th3)))
    M[1,2] = -i3 -m3*((a3**2)+(l2*a3*np.cos(th3)))

    M[2,0] = i3 + m3*((a3**2)+(l2*a3*np.cos(th3))+(l1*a3*np.cos(th3-th2)))
    M[2,1] = -i3 - m3*((a3)+(l2*a3*np.cos(th3)))
    M[2,2] = i3 + m3*(a3**2)

    C[1,0] = -(m2*l1*a2*thd1*np.sin(th2)) - (m3*l1*l2*thd1*np.sin(th2)) + (m3*l1*a3*thd1*np.sin(th3-th2))
    C[1,1] = (m2*l1*a2*thd1*np.sin(th2)) + (m3*l1*a3*thd1*np.sin(th3-th2)) + (m3*l1*a3*thd1*np.sin(th2))
    C[1,2] = m3*l1*a3*thd1*np.sin(th3-th1)
    C[2,0] = (m2*l2*a3*((2*thd2)-thd1)*np.sin(th3)) - (m3*l1*a3*np.sin(th3-th2))
    C[2,1] = -(m2*l2*a3*thd2*np.sin(th3)) + (m3*l1*a3*thd1*np.sin(th3-th2))
    C[2,2] = (m2*l2*a3*(thd2-thd1)*np.sin(th3)) - (m3*l1*a3*thd1*np.sin(th3-th2))

    G[0] = (m1*g*np.sin(th1)) + m2*g*((l1*np.sin(th1))-(a2*np.sin(th2-th1))) +m3*g*((l1*np.sin(th1))-(l2*np.sin(th2-th1))+(a3*np.sin(th1-th2+th3)))
    G[1] = (m2*g*a2*np.sin(th2-th1)) + m3*g*((l2*np.sin(th2-th1))-(a3*np.sin(th1-th2+th3)))
    G[2] = m3*g*a3*np.sin(th1-th2+th3)
    thdd3 = ((tau[2]-(C[2,0]*thd1+C[2,1]*thd2+C[2,2]*thd2)-G[2])-M[2,0]*thdd1-M[2,1]*thdd2) / M[2,2]
    return thdd3

#nilai u
def mr(s):
    u = 0.5*np.tanh(15*(s-0.5)) + 0.5
    return u 
#nilai adot, nanti menggunakan RK orde 4 
def madt(u,a):
    adt = (1/0.02)*(u-a) + (1/0.2)*(u-a-(u-a)*u)
    return adt
#fungsi f(l)
def fl(lt,lopt):
    fl = 1-(np.square((lt-lopt)/(0.5*lopt)))
    return fl 
# fungsi f(V)
def fv(l,lopt,v):
    if l <= lopt:
        fv = (3-v) / (3+(2.5*v))
    else  : 
        fv = 1.3 - 0.3*((3-(2.5*3)/(1+(2.5**2*v))))
    return fv
#passive element elastis
def fpe(kpe,l,lopt):
    fpe = kpe*np.exp(15*(l-lopt)-1)
    return fpe
#passive element dumping 
def fpd(cpd,V):
    fpd = -cpd*V
    return fpd


# Set up the display
pygame.init()
width, height = 800, 600

pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
glViewport(0, 0, width, height)
glMatrixMode(GL_PROJECTION)
gluPerspective(45, (width / height), 1, 100.0)
glMatrixMode(GL_MODELVIEW)
glTranslatef(0.0, 0.0, -5)
glEnable(GL_DEPTH_TEST)
glLoadIdentity()

glClearColor(0.0, 0.0, 0.0, 1.0)	   
glShadeModel(GL_SMOOTH)          
glClearDepth(1.0)                  
glEnable(GL_DEPTH_TEST)              
glDepthFunc(GL_LESS)
glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
glEnable(GL_TEXTURE_2D)
mat_specular = [8.0, 8.0, 1.0, 0]
mat_shininess = [40.0]
light_position = [120.6, 14.0, 41.0, 10.7]

Sphere = gluNewQuadric()
cylinder= gluNewQuadric()
disk=gluNewQuadric()

gluQuadricNormals(Sphere, GLU_SMOOTH)
gluQuadricNormals(cylinder, GLU_SMOOTH)
gluQuadricNormals(disk, GLU_SMOOTH)

glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)

glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)
glEnable(GL_LIGHT1)
glEnable(GL_LIGHT2)
glEnable(GL_LIGHT3)
glDepthFunc(GL_LEQUAL)

xposk=-1.4
yposk=0.7
zposk=-4.5
x=90

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
    rate = 1 / dt
    ss = 0
    #Iliopsas 
    K1aI = madt(mr(ss),ai)
    K2aI = madt(mr(ss)+(0.5*dt),ai+(0.5*K1aI*dt)) 
    k3aI = madt(mr(ss)+(0.5*dt),ai+(0.5*K2aI*dt)) 
    k4aI = madt(mr(ss)+(dt),ai+(k3aI*dt))
    ai += dt*(K1aI+2*K2aI+2*k3aI+k4aI) / 6 

    lHI = MailopasH*th1
    if lHI < 0: 
        lHI = 0 
    flI = fl(lHI,loptilopas)
    if flI <0 : 
        flI = 0
    VHI = MailopasH *thd1
    if VHI < 0: 
        VHI = 0   
    fVI = fv(lHI,loptilopas,VHI)
    if fVI <0: 
        fVI = 0
    FactI = ai*flI*fVI*Fmilopas
    if FactI > Fmilopas :
        FactI = Fmilopas 
    if FactI <0:
        FactI = 0 
    FpeI = fpe(kpeilopas,lHI,loptilopas)
    FpdI = fpd(Cpdilopas,VHI)
    FmIH = FactI + FpeI + FpdI
    taumIH = FmIH * MailopasH
    #BFLH 
    ss =1
    K1aBL = madt(mr(ss),aBL)
    K2aBL = madt(mr(ss)+(0.5*dt),aBL+(0.5*K1aBL*dt)) 
    k3aBL = madt(mr(ss)+(0.5*dt),aBL+(0.5*K2aBL*dt)) 
    k4aBL = madt(mr(ss)+(dt),aBL+(k3aBL*dt))
    aBL += dt*(K1aBL+2*K2aBL+2*k3aBL+k4aBL) / 6 
    lHBL =MaBFLHH  * th1
    lKBL = MaBFLHK * th2
    if lHBL < 0: 
        lHBL = 0 
    if lKBL < 0: 
        lKBL = 0 
    flHBL = fl(lHBL,lopyBFLH )
    if flHBL <0 : 
        flHBl = 0
    flKBL = fl(lKBL,lopyBFLH )
    if flKBL < 0: 
        flKBL = 0
    VHBL  = MaBFLHH *thd1
    VKBL = MaBFLHK *thd2
    if VHBL < 0: 
        VHBL = 0 
    if VKBL < 0: 
        VKBL = 0     
    fVHBL = fv(lHBL,lopyBFLH,VHBL)
    if fVHBL < 0: 
        fVHBL = 0
    fVKBL = fv(lKBL,lopyBFLH,VKBL)
    if fVKBL < 0: 
        fVKBL = 0
    FactHBL = aBL*flHBL*fVHBL*FmBFLH 
    FactKBL = aBL*flKBL*fVKBL*FmBFLH
    if FactHBL >FmBFLH:
        FactHBL = FmBFLH
    if FactKBL >FmBFLH:
        FactKBL = FmBFLH
    if FactHBL < 0: 
        FactHBL =0 
    if FactKBL < 0: 
        FactKBL = 0 
    FpeHBL = fpe(kpeBFLH,lHBL,lopyBFLH)
    FpeKBL = fpe(kpeBFLH,lKBL,lopyBFLH)
    fpdHBL = fpd(CpdBflh,VHBL)
    fpdKBL = fpd(CpdBflh,VKBL)
    FmHBL = FactHBL+fpdHBL+FpeHBL 
    FmKBL = FactKBL + fpdKBL + FpeKBL
    taumHBL = FmKBL * MaBFLHH 
    taumKBL = FmKBL * MaBFLHK 

    #BFSH 
    ss= 0
    K1aBH = madt(mr(ss),aBH)
    K2aBH = madt(mr(ss)+(0.5*dt),aBH+(0.5*K1aBH*dt)) 
    k3aBH = madt(mr(ss)+(0.5*dt),aBH+(0.5*K2aBH*dt)) 
    k4aBH = madt(mr(ss)+(dt),aBH+(k3aBH*dt))
    aBH += dt*(K1aBH+2*K2aBH+2*k3aBH+k4aBH) / 6 
    lBHK = MaBFSHK*th2
    if lBHK < 0: 
        lBHK = 0 
    flBHK = fl(lBHK,loptBFSH )
    if flBHK <0: 
        flBHK = 0 
    VBHK = MaBFSHK*thd2
    if VBHK < 0: 
        VBHK = 0 
    fVBHK = fv(lBHK,loptBFSH,VBHK)
    if fVBHK < 0 : 
        fVBHK = 0 
    FactBHK = aBH*flBHK*fVBHK*FmBFSH 
    if FactBHK > FmBFSH: 
        FactBHK = FmBFSH
    if FactBHK < 0: 
        FactBHK = 0
    FpeBHK = fpe(kpeBFSH ,lBHK,loptBFSH )
    FpdBHK = fpd(CpdBFSH ,VBHK)
    FmBHK = FactBHK + FpeBHK + FpdBHK
    taumBHK = FmBHK * MaBFSHK 

    #Rectus femoris 
    ss=1
    K1aRF = madt(mr(ss),aRF)
    K2aRF = madt(mr(ss)+(0.5*dt),aRF+(0.5*K1aRF*dt)) 
    k3aRF = madt(mr(ss)+(0.5*dt),aRF+(0.5*K2aRF*dt)) 
    k4aRF = madt(mr(ss)+(dt),aRF+(k3aRF*dt))
    aRF += dt*(K1aRF+2*K2aRF+2*k3aRF+k4aRF) / 6
    lHRF =MaRfH   * th1
    lKRF = MaRfK * th2
    if lHRF < 0: 
        lHRF = 0 
    if lKRF < 0: 
        lKRF = 0 
    flHRF = fl(lHRF,loptRF)
    if flHRF < 0 : 
        flHRF = 0
    flKRF = fl(lKRF,loptRF)
    if flKRF < 0: 
        flKRF = 0
    VHRF  = MaRfH   *thd1
    VKRF = MaRfK *thd2
    if VHRF < 0: 
        VHRF = 0 
    if VKRF < 0: 
        VKRF = 0     
    fVHRF = fv(lHRF,loptRF ,VHRF)
    if fVHRF < 0: 
        fVHRF = 0 
    fVKRF = fv(lKRF,loptRF ,VKRF)
    if fVKRF < 0 : 
        fVKRF = 0
    FactHRF = aRF*flHRF*fVHRF*FMRF
    if FactHRF >  FMRF: 
        FactHRF = FMRF
    if FactHRF < 0: 
        FactHRF =0
    FactKRF = aRF*flKRF*fVKRF*FMRF 
    if FactKRF > FMRF: 
        FactKRF = FMRF
    if FactKRF < 0: 
        FactKRF =0
    FpeHRF = fpe(kpeRF,lHRF,loptRF )
    FpeKRF = fpe(kpeRF,lKRF,loptRF )
    fpdHRF = fpd(CpdRF,VHRF)
    fpdKRF = fpd(CpdRF,VKRF)
    FmHRF = FactHRF+fpdHRF+FpeHRF 
    FmKRF = FactKRF + fpdKRF + FpeKRF
    taumHRF = FmKBL * MaRfH  
    taumKRF = FmKRF * MaRfK 

    #Gastroc Med 
    ss = 0
    K1aGm = madt(mr(ss),aGm)
    K2aGm = madt(mr(ss)+(0.5*dt),aGm+(0.5*K1aGm*dt)) 
    k3aGm = madt(mr(ss)+(0.5*dt),aGm+(0.5*K2aGm*dt)) 
    k4aGm = madt(mr(ss)+(dt),aGm+(k3aGm*dt))
    aGm += dt*(K1aGm+2*K2aGm+2*k3aGm+k4aGm) / 6
    lKGm = MaGmK   * th2
    lAGm = MaGmA * th3
    if lKGm < 0: 
        lKGm = 0 
    if lAGm < 0: 
        lAGm = 0 
    flKGm = fl(lKGm,loptGm)
    if flKGm < 0: 
        flKGm = 0
    flAGm = fl(lAGm,loptGm)
    if flAGm < 0: 
        flAGm =0
    VKGm  = MaGmK*thd2
    VAGm = MaGmA *thd3
    if VKGm < 0: 
        VKGm = 0 
    if VAGm < 0: 
        VAGm = 0     
    fVKGm = fv(lKGm,loptGm ,VKGm)
    if fVKGm <0: 
        fVKGm = 0 
    fVAGm = fv(lAGm,loptGm ,VAGm)
    if fVAGm <0: 
        fVAGm =0
    FactKGm = aGm*flKGm*fVKGm*FmGm  
    if FactKGm > FmGm: 
        FactKGm = FmGm
    if FactKGm < 0:
        FactKGm = 0
    FactAGm = aGm*flAGm*fVAGm*FmGm
    if FactAGm > FmGm: 
        FactAGm = FmGm
    if FactAGm <0: 
        FactAGm = 0
    FpeKGm = fpe(kpeGm,lKGm,loptGm )
    FpeAGm = fpe(kpeGm,lAGm,loptGm )
    fpdKGm = fpd(CpdGm,VKGm)
    fpdAGm = fpd(CpdGm,VAGm)
    FmKGm = FactKGm+fpdKGm+FpeKGm 
    FmAGm = FactAGm + fpdAGm + FpeAGm
    taumKGm = FmAGm * MaGmK 
    taumAGm = FmAGm * MaGmA 

    #Soleus
    ss= 0 
    K1aS = madt(mr(ss),aS)
    K2aS = madt(mr(ss)+(0.5*dt),aS+(0.5*K1aS*dt)) 
    k3aS = madt(mr(ss)+(0.5*dt),aS+(0.5*K2aS*dt)) 
    k4aS = madt(mr(ss)+(dt),aS+(k3aS*dt))
    aS += dt*(K1aS+2*K2aS+2*k3aS+k4aS) / 6 
    lAS = MasA*th3
    if lAS < 0: 
        lAS = 0 
    flAS = fl(lAS,lopts)
    if flAS < 0: 
        flAS = 0
    VAS = MasA*thd3
    if VAS < 0: 
        VAS = 0 
    fVAS = fv(lAS,lopts,VAS)
    if fVAS < 0:
        fVAS = 0
    FactAS = aS*flAS*fVAS*FmS
    if FactAS > FmS: 
        FactAS = FmS 
    if FactAS < 0: 
        FactAS =0
    FpeAS = fpe(kpeS ,lAS,lopts )
    FpdAS = fpd(CpdS ,VAS)
    FmAS = FactAS + FpeAS + FpdAS
    taumAS = FmAS * MasA 

    #Tibialis Anterior
    ss = 0 
    K1TA = madt(mr(ss),aTa)
    K2TA = madt(mr(ss)+(0.5*dt),aTa+(0.5*K1TA*dt)) 
    k3TA = madt(mr(ss)+(0.5*dt),aTa+(0.5*K2TA*dt)) 
    k4TA = madt(mr(ss)+(dt),aTa+(k3TA*dt))
    aTa += dt*(K1TA+2*K2TA+2*k3TA+k4TA) / 6 
    lTA = MaTA*th3
    if lTA < 0: 
        lTA = 0 
    flTA = fl(lTA,loptTA)
    if flTA <0:
        flTA = 0
    VTA = MaTA*thd3
    if VTA < 0: 
        VTA = 0 
    fVTA = fv(lTA,loptTA,VTA)
    if fVTA <0: 
        fVTA =0
    FactTA = aTa*flTA*fVTA*FmTA
    if FactTA > FmTA: 
        FactTA = FmTA
    if FactTA < 0: 
        FactTA =0
    FpeTA = fpe(kpeTA ,lTA,loptTA )
    FpdTA = fpd(CpdTA ,VTA)
    FmTA = FactTA + FpeTA + FpdTA
    taumTA = FmTA * MaTA   

    tauMHIP = taumIH +  taumHBL + taumHRF
    tauMKnee = taumKBL + taumKGm + taumKRF + taumBHK 
    tauMAnkle = taumTA + taumAS + taumAGm 

    tauphip = -3.09*thd1+2.6*np.exp(-5.8*(th1-(-10*np.pi/180)))-8.7*np.exp(-1.3*((10*np.pi/180)-th1))
    taupknee = -10*thd2+6.1*np.exp(-5.9*(th2-(10*np.pi/180)))-10.5*np.exp(-21.8*((67*np.pi/180)-th2))
    taupAnkle =-0.943*thd3+2*np.exp(-5*(th3-(-15*np.pi/180)))-2*np.exp(-5*((25*np.pi/180)-th3))
    tau[0] =  tauphip +tauMHIP
    tau[1] = taupknee + tauMKnee
    tau[2] = taupAnkle + tauMAnkle

    k11 = (0.5*dt) * ttd1(th1,thd1)
    k12 = (0.5*dt) * ttd2(th2,thd2)
    k13 = (0.5*dt) * ttd3(th3,thd3)

    k21 = (0.5*dt) * ttd1(th1+(0.5*dt*(thd1+(k11*0.5))),thd1+k11)
    k22 = (0.5*dt) * ttd2(th2+(0.5*dt*(thd2+(k12*0.5))),thd2+k12)
    k23 = (0.5*dt) * ttd3(th3+(0.5*dt*(thd3+(k13*0.5))),thd3+k13)

    k31 = (0.5*dt) * ttd1(th1+(0.5*dt*(thd1+(k21*0.5))),thd1+k21)
    k32 = (0.5*dt) * ttd2(th2+(0.5*dt*(thd2+(k22*0.5))),thd2+k22)
    k33 = (0.5*dt) * ttd3(th3+(0.5*dt*(thd3+(k23*0.5))),thd3+k23)

    k41 = (0.5*dt) * ttd1(th1+(dt*(thd1+(k11))),thd1+(2*k31))    
    k42 = (0.5*dt) * ttd2(th2+(dt*(thd2+(k12))),thd2+(2*k32))    
    k43 = (0.5*dt) * ttd3(th3+(dt*(thd3+(k13))),thd3+(2*k33))

    th1 += dt*(thd1+((k11+k21+k31)/3))
    th2 += dt*(thd2+((k12+k22+k32)/3))
    th3 += dt*(thd3+((k13+k23+k33)/3))

    thd1 += (k11+(2*k21)+(2*k31)+k41)/3
    thd2 += (k12+(2*k22)+(2*k32)+k42)/3
    thd3 += (k13+(2*k23)+(2*k33)+k43)/3

    rate = 1 / dt

    teta1 = th1 *180/np.pi
    teta2 = th2 *180/np.pi
    teta3 = th3*180/np.pi
    glClearColor(0.0, 0.0, 0, 0.0)                 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  
    glLoadIdentity()                                   

    glTranslate(xposk, yposk,zposk)                                
    glRotate(90,1,0,0)
    glRotate(0,0,1,0)
    glRotate(0,0,0,1)

    glPushMatrix()
    gluSphere(Sphere, 0.08, 32, 32)
    glPopMatrix()

    glPushMatrix()
    glRotate(teta1,0,1,0)
    gluCylinder(cylinder, 0.08, 0.05, l1, 32, 32)
    glPopMatrix()

    glPushMatrix()
    glRotate(teta1,0,1,0)
    glTranslate(0, 0, l1)
    gluSphere(Sphere, 0.05, 32, 32)
    glPopMatrix()

    glPushMatrix()
    glRotate(teta1,0,1,0)
    glTranslate(0, 0, l1)
    glRotate(-teta2, 0,1,0)
    gluCylinder(cylinder, 0.05, 0.02, l2, 32, 32)
    glPopMatrix()

    glPushMatrix()
    glRotate(teta1, 0,1,0)
    glTranslate(0, 0, l1)
    glRotate(-teta2, 0,1,0)
    glTranslate(0, 0, l2)
    gluSphere(Sphere, 0.04, 32, 32)
    glPopMatrix()

    glPushMatrix()
    glRotate(teta1, 0,1,0)
    glTranslate(0, 0, l1)
    glRotate(180, 0, 0, 1)
    glRotate(teta2, 0,1,0)
    glRotate(270,0,1, 0)
    glRotate(180, 0, 0, 1)
    glTranslate(-l2, 0, 0)
    glRotate(-teta3, 0,1,0)
    gluCylinder(cylinder, 0.04, 0.015, l3, 32, 32)
    glPopMatrix()

    glPushMatrix()
    glRotate(teta1, 0,1,0)
    glTranslate(0, 0, l1)
    glRotate(180,0,0,1)
    glRotate(teta2, 0,1,0)
    glRotate(270,0,1,0)
    glRotate(180,0,0,1)
    glTranslate(-l2, 0, 0)
    glRotate(-teta3, 0,1,0)
    glTranslate(0, 0, l3)
    gluSphere(Sphere, 0.015, 32, 32)
    glPopMatrix()


    time_values.append(t)
    theta1_values.append(teta1)
    theta2.append(teta2)
    theta3.append(teta3)
    plt.plot(time_values, theta1_values, label='Theta1')
    plt.plot(time_values, theta2, label='Theta2')
    plt.plot(time_values, theta3, label='Theta3')
    plt.xlabel('Time')
    plt.ylabel('Theta (radians)')
    plt.legend()
    plt.pause(0.01)  # Adjust the pause duration as needed

    # Clear the previous plot to avoid overlapping
    plt.clf()
    t += dt
    pygame.display.flip()
    pygame.time.wait(int(dt * 1))
    pygame.time.Clock().tick(rate)
