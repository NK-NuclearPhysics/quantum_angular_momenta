import numpy as np


## To calculate Wignar d-matrix
#### The corresponding function is Rd
def Atvec(A,e):
    ''' Solving asymmetrized matrix with 
        diagonal elements e.       '''
    alen = A.size
    alen1 = alen + 1
    X = np.ones(alen1,dtype=np.complex)
    X[1] = e/A[0]
    for i in range(2,alen):
        X[i] = (A[i-2]*X[i-2]+e*X[i-1])/A[i-1]
    C = np.linalg.norm(X)
    return X/C

def Rd(j,m,n,t,d=0):
    ''' Calculate the reduced rotational matrix <j,m|exp(-itJy)|j,n>
            j --- angular momemtum
            m, n --- the quantum number both |m| |n| no more than j
            t --- the angular theta
            d --- the derivative to t
            Use Atvec
             <PhysRevE92(2015)043307>               '''  
    
    j2 = j*(j+1)
    J = np.arange(-j,j+1,1)  
    ndim = int(2*j)
    if(j==0):
        return 1.
    if(abs(m)>j or abs(n)>j):
        return 0.
    Ay = np.zeros(ndim)
    for i in range(int(ndim/2)):
        s = j-1-i
        Ay[i] = sqrt(j2 -s*(s+1))
        Ay[ndim-1-i] = Ay[i]
    ndim = ndim + 1
    vm = np.zeros(ndim)
    vn = np.zeros(ndim)
    
    km = np.where(J == m)
    kn = np.where(J == n)
    
    vm[km] = 1.0
    vn[kn] = 1.0 
    
    dsum = 0.0 
    for i in range(ndim):
        x = J[i]
        expi = np.complex(np.cos(t*x),-np.sin(t*x))
        vi = Atvec(Ay, np.complex(0,2.*x))
        if(d!=0):
            fac = np.complex(0,-x)**d
            expi*=fac
        ti = np.dot(vm,vi)*np.dot(np.conjugate(vi),vn)
        dsum += ti*expi
    return dsum.real

def SHY_list(l,m,t,phi):
    '''
        The Real spherical harmonics:
            l --- angular momentum 
            m --- magenetic quantume number 
            t --- theta 
        By exhaustivity 
    '''
    if(phi == 0):
        ep = 1.
    else:
        ep = complex(cos(m*phi),sin(m*phi))
    if(abs(m)>l): 
        return 0
    
    if(l==0):
        Y = 0.282094791774
    elif(l==1):
        if(m==0):
            Y = 0.488602511903*cos(t)
        elif(m==1):
            Y = -0.345494149471*sin(t)        
        elif(m==-1):
            Y = 0.345494149471*sin(t)        
    elif(l==2):
        if(m==0):
            ct = cos(t)
            Y = 0.315391565253*(3*ct**2-1)
        elif(m==1):
            Y = -0.772548404046*cos(t)*sin(t)
        elif(m==-1):
            Y = 0.772548404046*cos(t)*sin(t)
        else:
            Y = 0.386274202023*sin(t)**2
    elif(l==3):
        if(m==0):
            ct = cos(t)
            Y = 0.373176332590*(5*ct**3-3*ct)
        elif(m==-1):
            ct = cos(t)
            st = sin(t)
            Y = 0.323180184114*st*(5*ct**2-1)
        elif(m==1):
            ct = cos(t)
            st = sin(t)
            Y = -0.323180184114*st*(5*ct**2-1)
        elif(m==-2 or m==2):
            ct = cos(t)
            st = sin(t)
            Y = 1.0219854764332823*st**2*ct
        elif(m==-3):
            st = sin(t)
            Y = 0.4172238236327841*st**3
        elif(m==3):
            st = sin(t)
            Y = -0.4172238236327841*st**3
    elif(l==4):
        if(m==0):
            ct = cos(t)
            Y = 0.1057855469152043*(35*ct**4-30*ct**2+3)
        elif(m==-1):
            st = sin(t)
            ct = cos(t)
            Y = 0.4730873478787800*st*(7*ct**3-3*ct)
        elif(m==1):
            st = sin(t)
            ct = cos(t)
            Y = -0.4730873478787800*st*(7*ct**3-3*ct)
        elif(m==2 or m==-2):
            st = sin(t)
            ct = cos(t)
            Y = 0.3345232717786446*st**2*(7*ct**2-1)
        elif(m==-3):
            st = sin(t)
            ct = cos(t)
            Y = 1.2516714708983523*st**3*ct 
        elif(m==3):
            st = sin(t)
            ct = cos(t)
            Y = -1.2516714708983523*st**3*ct 
        else:
            st = sin(t)
            Y = 0.4425326924449826*st**4
    elif(l==5):
        if(m==0):
            ct = cos(t)
            Y = 0.1169503224534236*(63*ct**5 - 70*ct**3 + 15*ct)
        elif(m==-1):
            ct, st = cos(t), sin(t)
            Y = 0.3202816485762152*st*(21*ct**4 -14*ct**2+1)
        elif(m==1):
            ct, st = cos(t), sin(t)
            Y = -0.3202816485762152*st*(21*ct**4 -14*ct**2+1)
        elif(m==-2 or m==2):
            ct, st = cos(t), sin(t)
            Y = 1.6947711832608994*st**2*(3*ct**3-ct)
        elif(m==-3):
            ct, st = cos(t), sin(t)
            Y = 0.3459437191468403*st**3*(9*ct**2 -1.)
        elif(m==3):
            ct, st = cos(t), sin(t)
            Y = -0.3459437191468403*st**3*(9*ct**2 -1.)
        elif(m==-4 or m==4):
            ct, st = cos(t), sin(t)
            Y = 1.467714898305751*st**4*ct 
        elif(m==-5):
            st = sin(t)
            Y = 0.46413220344085815*st**5
        elif(m==5):
            st = sin(t)
            Y = -0.46413220344085815*st**5
    elif(l==6):
        if(m==0):
            ct = cos(t)
            Y = 0.06356920226762842*(231*ct**6 -315*ct**4 +105*ct**2-5)
        elif(m==-1):
            ct, st = cos(t), sin(t)
            Y = 0.4119755163011408*st*(33*ct**5 -30*ct**3+5*ct)
        elif(m==1):
            ct, st = cos(t), sin(t)
            Y = -0.4119755163011408*st*(33*ct**5 -30*ct**3+5*ct)
        elif(m==-2 or m==2):
            ct, st = cos(t), sin(t)
            Y = 0.3256952429338579*st**2*(33*ct**4-18*ct**2+1)
        elif(m==-3):
            ct, st = cos(t), sin(t)
            Y = 0.6513904858677158*st**3*(11*ct**3 -3*ct)
        elif(m==3):
            ct, st = cos(t), sin(t)
            Y = -0.6513904858677158*st**3*(11*ct**3 -3*ct)
        elif(m==-4 or m==4):
            ct, st = cos(t), sin(t)
            Y = 0.5045649007287242*st**4*(11*ct**2-1) 
        elif(m==-5):
            ct, st = cos(t), sin(t)
            Y = 1.673452458100098*st**5*ct 
        elif(m==5):
            ct, st = cos(t), sin(t)
            Y =-1.673452458100098*st**5*ct 
        else:
            st = sin(t)
            Y = 0.48308411358006625*st**6
    elif(l==7):
        if(m==0):
            ct = cos(t)
            Y = 0.06828427691200495*(429*ct**7-693*ct**5+315*ct**3-35*ct)
        elif(m==-1):
            ct, st = cos(t), sin(t)
            Y = 0.06387409227708014*st*(429*ct**6-495*ct**4+135*ct**2-5)
        elif(m==1):
            ct, st = cos(t), sin(t)
            Y = -0.06387409227708014*st*(429*ct**6-495*ct**4+135*ct**2-5)
        elif(m==-2 or m==2):
            ct, st = cos(t), sin(t)
            Y = 0.15645893386229404*st**2*(143*ct**5-110*ct**3+15*ct)
        elif(m==-3):
            ct, st = cos(t), sin(t)
            Y = 0.11063317311124565*st**3*(143*ct**4-66*ct**2+3)
        elif(m==3):
            ct, st = cos(t), sin(t)
            Y = -0.11063317311124565*st**3*(143*ct**4-66*ct**2+3)
        elif(m==-4 or m==4):
            ct, st = cos(t), sin(t)
            Y = 0.7338574491528755*st**4*(13*ct**3-3*ct) 
        elif(m==-5):
            ct, st = cos(t), sin(t)
            Y = 0.36692872457643777*st**5*(13*ct**2-1) 
        elif(m==5):
            ct, st = cos(t), sin(t)
            Y = -0.36692872457643777*st**5*(13*ct**2-1) 
        elif(m==-6 or m==6):
            ct, st = cos(t), sin(t)
            Y = 1.8709767267129687*st**6*ct
        elif(m==-7):
            st = sin(t)
            Y = 0.5000395635705507*st**7
        elif(m==7):
            st = sin(t)
            Y = -0.5000395635705507*st**7
    elif(l==8):
        if(m==0):
            ct = cos(t)
            Y = 0.009086770491564996*(6435*ct**8-12012*ct**6+6930*ct**4-1260*ct**2+35)
        elif(m==-1):
            ct, st = cos(t), sin(t)
            Y = 0.07710380440405712*st*(715*ct**7-1001*ct**5+385*ct**3-35*ct)
        elif(m==1):
            ct, st = cos(t), sin(t)
            Y = -0.07710380440405712*st*(715*ct**7-1001*ct**5+385*ct**3-35*ct)
        elif(m==-2 or m==2):
            ct, st = cos(t), sin(t)
            Y = 0.32254835519288305*st**2*(143*ct**6-143*ct**4+33*ct**2-1)
        elif(m==-3):
            ct, st = cos(t), sin(t)
            Y = 0.8734650749797142*st**3*(39*ct**5-26*ct**3+3*ct)
        elif(m==3):
            ct, st = cos(t), sin(t)
            Y = -0.8734650749797142*st**3*(39*ct**5-26*ct**3+3*ct)
        elif(m==-4 or m==4):
            ct, st = cos(t), sin(t)
            Y = 0.33829156888902456*st**4*(65*ct**4-26*ct**2+1) 
        elif(m==-5):
            ct, st = cos(t), sin(t)
            Y = 2.439455195373073*st**5*(5*ct**3-ct) 
        elif(m==5):
            ct, st = cos(t), sin(t)
            Y = -2.439455195373073*st**5*(5*ct**3-ct) 
        elif(m==-6 or m==6):
            ct, st = cos(t), sin(t)
            Y = 0.37641610872849457*st**6*(15*ct**2-1)
        elif(m==-7):
            ct, st = cos(t), sin(t)
            Y = 2.0617159375891374*st**7*ct
        elif(m==7):
            ct, st = cos(t), sin(t)
            Y = -2.0617159375891374*st**7*ct
        else:
            st = sin(t)
            Y = 0.5154289843972844*st**8
    elif(l==9):
        if(m==0):
            ct = cos(t)
            Y = 0.0096064272643866*(12155*ct**9-25740*ct**7+18018*ct**5-4620*ct**3+315*ct)
        elif(m==-1):
            ct, st = cos(t), sin(t)
            Y = 0.04556728549830324*st*(2431*ct**8-4004*ct**6+2002*ct**4-308*ct**2+7)
        elif(m==1):
            ct, st = cos(t), sin(t)
            Y = -0.04556728549830324*st*(2431*ct**8-4004*ct**6+2002*ct**4-308*ct**2+7)
        elif(m==-2 or m==2):
            ct, st = cos(t), sin(t)
            Y = 0.42745902806723024*st**2*(221*ct**7-273*ct**5+91*ct**3-7*ct)
        elif(m==-3):
            ct, st = cos(t), sin(t)
            Y = 0.3264772254350559*st**3*(221*ct**6-195*ct**4+39*ct**2-1)
        elif(m==3):
            ct, st = cos(t), sin(t)
            Y = -0.3264772254350559*st**3*(221*ct**6-195*ct**4+39*ct**2-1)
        elif(m==-4 or m==4):
            ct, st = cos(t), sin(t)
            Y = 2.883368783344621*st**4*(17*ct**5-10*ct**3+ct) 
        elif(m==-5):
            ct, st = cos(t), sin(t)
            Y = 0.344628486111519*st**5*(85*ct**4-30*ct**2+1) 
        elif(m==5):
            ct, st = cos(t), sin(t)
            Y = -0.344628486111519*st**5*(85*ct**4-30*ct**2+1) 
        elif(m==-6 or m==6):
            ct, st = cos(t), sin(t)
            Y = 0.8898269248923925*st**6*(17*ct**3-3*ct)
        elif(m==-7):
            ct, st = cos(t), sin(t)
            Y = 0.3853063609640998*st**7*(17*ct**2-1)
        elif(m==7):
            ct, st = cos(t), sin(t)
            Y = -0.3853063609640998*st**7*(17*ct**2-1)
        elif(m==-8 or m==8):
            ct, st = cos(t), sin(t)
            Y = 2.246702855559565*st**8*ct
        elif(m==-9):
            st = sin(t)
            Y = 0.5295529414924496*st**9    
        elif(m==9):
            st = sin(t)
            Y = -0.5295529414924496*st**9   
    elif(l==10):
        if(m==0):
            ct = cos(t)
            Y = 0.005049690376783604*(46189*ct**10-109395*ct**8+90090*ct**6-30030*ct**4\
                                    +3465*ct**2-63)
        elif(m==-1):
            ct, st = cos(t), sin(t)
            Y = 0.052961599476903105*st*(4199*ct**9-7956*ct**7+4914*ct**5-1092*ct**3\
                                         +63*ct)
        elif(m==1):
            ct, st = cos(t), sin(t)
            Y =-0.052961599476903105*st*(4199*ct**9-7956*ct**7+4914*ct**5-1092*ct**3\
                                         +63*ct)
        elif(m==-2 or m==2):
            ct, st = cos(t), sin(t)
            Y = 0.04586609057205472*st**2*(4199*ct**8-6188*ct**6+2730*ct**4-364*ct**2\
                                          +7)
        elif(m==-3):
            ct, st = cos(t), sin(t)
            Y = 0.46774418167824217*st**3*(323*ct**7-357*ct**5+105*ct**3-7*ct)
        elif(m==3):
            ct, st = cos(t), sin(t)
            Y = -0.46774418167824217*st**3*(323*ct**7-357*ct**5+105*ct**3-7*ct)
        elif(m==-4 or m==4):
            ct, st = cos(t), sin(t)
            Y = 0.3307450827252375*st**4*(323*ct**6-255*ct**4+45*ct**2-1) 
        elif(m==-5):
            ct, st = cos(t), sin(t)
            Y = 0.2091815572625123*st**5*(323*ct**5-170*ct**3+15*ct)
        elif(m==5):
            ct, st = cos(t), sin(t)
            Y = -0.2091815572625123*st**5*(323*ct**5-170*ct**3+15*ct) 
        elif(m==-6 or m==6):
            ct, st = cos(t), sin(t)
            Y = 0.11693604541956054*st**6*(323*ct**4-102*ct**2+3)
        elif(m==-7):
            ct, st = cos(t), sin(t)
            Y = 0.9642793334137447*st**7*(19*ct**3-3*ct)
        elif(m==7):
            ct, st = cos(t), sin(t)
            Y = -0.9642793334137447*st**7*(19*ct**3-3*ct)
        elif(m==-8 or m==8):
            ct, st = cos(t), sin(t)
            Y = 0.3936653893957947*st**8*(19*ct**2-1)
        elif(m==-9):
            ct, st = cos(t), sin(t)
            Y = 2.4267164388756717*st**9*ct
        elif(m==9):
            ct, st = cos(t), sin(t)
            Y = -2.4267164388756717*st**9*ct
        else:
            st = sin(t)
            Y = 0.5426302919442214*st**10               
    else:
        # While l = 10 is efficiently large enough. 
        Y = 0.0
    return Y*ep

Vd = lambda n: np.array([-n+i for i in range(int(2*n)+1)])
Va = lambda n: np.array([sqrt(n*(n+1)-i*(i-1)) for i in  Vd(n)[1:]])
Lup = lambda l, m: sqrt(l*(l+1)-(m+1)*m)

Vd = lambda n: np.array([-n+i for i in range(int(2*n)+1)])
Va = lambda n: np.array([sqrt(n*(n+1)-i*(i-1)) for i in  Vd(n)[1:]])
Lup = lambda l, m: sqrt(l*(l+1)-(m+1)*m)

def CGcoin(l,s,j,mj):
    ''' This function is designed to produce the CG coefficients
        <l ml, s ms| j mj >
            l --- orbital quantum number 
            s --- spin quantum number 
            j, mj --- total angular momentum and its projection
    ''' 
    if(abs(l-s)>j or l+s<j):
        #print('The quantum number do not support nonzero coupling.')
        return np.array([0]), np.array([0]), np.array([0])
    if(l<0 or s<0 or j<0):
        return np.array([0]), np.array([0]), np.array([0])
    if(abs(mj)>j):
        return np.array([0]), np.array([0]), np.array([0])
    if(s==0):
        if(j==l):
            return np.array([mj]), np.array([0]), np.array([1])
        else: 
            return np.array([0]), np.array([0]), np.array([0])
    if(l==0):
        if(s==j):
            return np.array([0]), np.array([mj]), np.array([1])
        else:
            return np.array([0]), np.array([0]), np.array([0])
    eg = 0.5*(j*(j+1)-l*(l+1)-s*(s+1))    
    Sd = Vd(s)
    Sa = Va(s) 
    Ms = mj - Sd 
    Ms_dim = len(Ms)
    
    # Now we select all possible |l m> states
    sjump, sindex = True, 0 
    ejump, eindex = True, Ms_dim
    for i in range(Ms_dim):
        ms = Ms[i]
        if(sjump and l>=ms):
            sindex = i
            sjump = False
        if(ejump and ms<=-l):
            eindex = i
            ejump = False
            break 
        
    Sd = Sd[sindex:eindex+1]
    Sa = Sa[sindex:eindex]
    Ml = Ms[sindex:eindex+1]
 
    LSa = np.ones(Sa.shape)
    for i in range(len(Sa)):
        LSa[i] = 0.5*Sa[i]*Lup(l,Ml[i+1])
    LSd = Sd*Ml-eg
 
    C = Tsvec(LSd,LSa)
    for i in range(len(C)):
        print('<{0} {1}, {2} {3}|{4} {5}>={6}\n'.format(repr(l),\
              repr(Ml[i]),repr(s),repr(Sd[i]),repr(j),repr(mj),C[i]) )
    return Ml, Sd,  C


def Tsvec(A,B):
    ''' A --- vector of principle diagonal line A[0:n]
        B --- vector of secondary diagonal line B[0:n-1]
    '''
    n = A.size
    if(n<=1):
        return np.array([1])
    X = np.ones(n)
    X[1] = -A[0]/B[0]
    for i in range(2,n):
        X[i]=-(B[i-2]*X[i-2]+A[i-1]*X[i-1])/B[i-1]
    c = np.linalg.norm(X)
    return X/c


def Tsvec(A,B):
    ''' A --- vector of principle diagonal line A[0:n]
        B --- vector of secondary diagonal line B[0:n-1]
    '''
    n = A.size
    if(n<=1):
        return np.array([1])
    X = np.ones(n)
    X[1] = -A[0]/B[0]
    for i in range(2,n):
        X[i]=-(B[i-2]*X[i-2]+A[i-1]*X[i-1])/B[i-1]
    c = np.linalg.norm(X)
    return X/c