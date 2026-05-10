import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist

coords=np.mgrid[-25:26,-25:26]
xgrid=coords[1,:,:]
ygrid=coords[0,:,:]

COULOMB_CONST=1.0
class Charge:

    def __init__(self,x,y,q):
        self.x = x
        self.y = y
        self.q = q
        self.r=np.sqrt((xgrid-self.x-0.0001)**2+(ygrid-self.y-0.0001)**2)
        return

    def potential(self):
        return COULOMB_CONST*self.q/(self.r)
    
    def E_field(self):
        mag=COULOMB_CONST*self.q/(self.r)**2
        return [mag*(xgrid-self.x)/(self.r),mag*(ygrid-self.y)/(self.r)]
    
    def fieldV(self,x,y):
        return COULOMB_CONST*self.q/np.sqrt((x-self.x-0.0001)**2+(y-self.y-.0001)**2)   
    
    def field(self,x,y):
        r=(np.sqrt((x-self.x-0.0001)**2+(y-self.y-.0001)**2))
        mag=COULOMB_CONST*self.q/r**2
        return [mag*(x-self.x)/r, mag*(y-self.y)/r]


def plot(data):
    plt.gca().set_aspect('equal')
    plt.imshow(data, vmin=np.percentile(data,2), vmax=np.percentile(data,98))
    plt.colorbar()
    plt.contour(data)
    plt.show()

def total_V(Charges):
    full_V=[0]
    for i in Charges:
        full_V+=i.potential()
    return full_V

def total_E(Charges):
    full_Ex=[0]
    full_Ey=[0]
    for i in Charges:
        full_Ex+=i.E_field()[0]
        full_Ey+=i.E_field()[1]
    return full_Ey, full_Ex

Charges=[Charge(-7,0,1), Charge(7,0,-1)]

def finite_dif(full_V):
    dx = 1

    dV_dy = np.zeros_like(full_V)
    dV_dx = np.zeros_like(full_V)

    for i in range(full_V.shape[0]):
        for j in range(full_V.shape[1]):

            if i == 0:
                dV_dy[i, j] = (full_V[i+1, j] - full_V[i, j]) / dx
            elif i == full_V.shape[0] - 1:
                dV_dy[i, j] = (full_V[i, j] - full_V[i-1, j]) / dx
            else:
                dV_dy[i, j] = (full_V[i+1, j] - full_V[i-1, j]) / (2*dx)

            if j == 0:
                dV_dx[i, j] = (full_V[i, j+1] - full_V[i, j]) / dx
            elif j == full_V.shape[1] - 1:
                dV_dx[i, j] = (full_V[i, j] - full_V[i, j-1]) / dx
            else:
                dV_dx[i, j] = (full_V[i, j+1] - full_V[i, j-1]) / (2*dx)

    return dV_dy, dV_dx

def r(x,y):
    return np.sqrt(x**2+y**2)

def genE(Charges,x,y):
    full_Ex=[0]
    full_Ey=[0]
    for i in Charges:
        full_Ex+=i.field(x,y)[0]
        full_Ey+=i.field(x,y)[1]
    return full_Ey, full_Ex

x=Charges[0].x+1
y=Charges[0].y+0
def launch(x,y,ra=1,RK2=False,dt=0.1,color="b"):
    plt.axis([-30,30,-22,22])
    xarr=np.array(x)
    yarr=np.array(y)
    sign=np.sign(x+0.000001)
    steps=0
    while True:
        Ey, Ex=genE(Charges,x,y)
        dEx=(genE(Charges,x+dt,y)[1]-genE(Charges,x,y)[1])/dt
        dEy=(genE(Charges,x,y+dt)[0]-genE(Charges,x,y)[0])/dt
        mag = np.sqrt(Ex**2 + Ey**2)
        if RK2:
            x+=(Ex*dt+dt*dt*dEx/2)/mag
            y+=(Ey*dt+dt*dt*dEy/2)/mag
        else:
            x+=Ex/mag*dt
            y+=Ey/mag*dt   
        xarr=np.append(xarr,x)
        yarr=np.append(yarr,y)
        print(Ex)
        print(Ey)
        steps+=1
        if steps > 10000 or np.any([r(x-i.x,y-i.y)<ra for i in Charges]) or r(x,y)/mag>10**9:
            break
    for i in Charges:
        if i.q>0:
            c=plt.Circle((i.x, i.y), ra*np.sqrt(abs(i.q)),color='red', fill=True, linewidth=1)  
            plt.gca().add_patch(c)      
        if i.q<0:
            c=plt.Circle((i.x, i.y), ra*np.sqrt(abs(i.q)),color='blue', fill=True, linewidth=1)  
            plt.gca().add_patch(c) 
    plt.plot(xarr,yarr,color=color, linewidth=2)

line=launch(x,y,1)
plt.show()

# Problem 1

def full_line(Charges,number,ra=1,RK2=False,dt=0.1,color="b"):
    deltheta=360/number        
    thetadeg=22.5
    while thetadeg<360:
        theta=np.deg2rad(thetadeg)
        x=Charges[0].x+ra*np.cos(theta)
        y=Charges[0].y+ra*np.sin(theta)
        launch(x,y,ra,RK2,dt,color)
        thetadeg+=deltheta
#a)
full_line(Charges, 8, 3, False, 0.5,"black")
full_line(Charges, 8, 3, True, 0.5,"grey")
plt.show()

#b)
full_line(Charges, 8, 3, False, 0.1,"black")
full_line(Charges, 8, 3, True, 0.1,"grey")
plt.show()
#c)
# The plot made with second order Runge Kutta was closer to the actual field lines of a dipole.
# However, as we decreased the step size, the difference in error between second order Runge Kutta
# and the Euler method became less noticable. The Runge Kutta method did also seem to take very slightly longer.
# These results are within our expectations as per the formula error that was derived for high order terms in this Unit. 

#Problem 2
def full_line(Charges,number,ra=1,RK2=False,dt=0.1,color="b"):
    deltheta=360/number
    for i in Charges:
        thetadeg=22.5
        while thetadeg<360:
            theta=np.deg2rad(thetadeg)
            x=i.x+ra*np.cos(theta)
            y=i.y+ra*np.sin(theta)
            launch(x,y,ra,RK2,dt,color)
            thetadeg+=deltheta

Charges=[Charge(0,0,1)]
full_line(Charges, 8, 1, True, 0.5,"black")
plt.show()

Charges=[Charge(-7,0,1), Charge(7,0,1)]
full_line(Charges, 8, 1, True, 0.5,"black")
plt.show()

Charges=[Charge(-7,0,1), Charge(7,0,1),  Charge(0,-7,-1)]
full_line(Charges, 8, 1, True, 0.5,"black")
plt.show()
# Problem 3

def Equipotential(x,y,ra=1,RK2=False,dt=0.1,color="b"):
    X=x
    Y=y
    plt.axis([-30,30,-22,22])
    xarr=np.array(x)
    yarr=np.array(y)
    sign=np.sign(x+0.000001)
    steps=0
    while True:
        Ey, Ex=genE(Charges,x,y)
        dEx=(genE(Charges,x+dt,y)[1]-genE(Charges,x,y)[1])/dt
        dEy=(genE(Charges,x,y+dt)[0]-genE(Charges,x,y)[0])/dt
        mag = np.sqrt(Ex**2 + Ey**2)
        if RK2:
            y+=sign*Ex/mag*dt+dt*dt*dEx/2
            x-=sign*Ey/mag*dt+dt*dt*dEy/2
        else:
            y+=sign*Ex/mag*dt
            x-=sign*Ey/mag*dt   
        xarr=np.append(xarr,x)
        yarr=np.append(yarr,y)
        print(Ex)
        print(Ey)
        tol=0.75*dt
        steps+=1
        if steps > 10000 or r(x-X,y-Y)<tol or np.any([r(x-i.x,y-i.y)<ra for i in Charges]) or r(x,y)/mag>10**9:
            break
    for i in Charges:
        if i.q>0:
            c=plt.Circle((i.x, i.y), ra*np.sqrt(abs(i.q)),color='red', fill=True, linewidth=1)  
            plt.gca().add_patch(c)      
        if i.q<0:
            c=plt.Circle((i.x, i.y), ra*np.sqrt(abs(i.q)),color='blue', fill=True, linewidth=1)  
            plt.gca().add_patch(c) 
    plt.plot(xarr,yarr,color=color)

Charges=[Charge(-7,0,1), Charge(7,0,-1)]
Equipotential(1,0,1,True,0.01,"red")
full_line(Charges, 8, 1, True, 0.1, "black")
plt.show()

# Yes, the equipotential looks like it is perpendicular to the fields lines.

#Problem 4
potentialslice=total_V(Charges)[int(np.shape(total_V(Charges))[1]/2)]
level=3
i = np.argmax(potentialslice>level)
full_line(Charges,8,1,True,0.1,"black")
Equipotential(1,0,1,True, 0.01,"red")
Equipotential(2,0,1,True, 0.01,"red")
Equipotential(3,0,1,True, 0.01,"red")
Equipotential(4,0,1,True, 0.01,"red")
Equipotential(-1,0,1,True, 0.01,"red")
Equipotential(-2,0,1,True, 0.01,"red")
Equipotential(-3,0,1,True, 0.01,"red")
Equipotential(-4,0,1,True, 0.01,"red")
plt.show()
