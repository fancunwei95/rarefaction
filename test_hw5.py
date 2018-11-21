import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

## parameters ###################
gamma = 5.0/3.0
#U_p = 0.1*2.0/(gamma-1.0)*np.sqrt(gamma)
U_p =10.0*np.sqrt(gamma)
vis_a = 1.8 
vis_b = 0.2
alpha = 1
dx  = 0.1
n_p = 1000
ratio_of_critical = 0.125  #0.125
plot = True
## functions #####################
# u, R, V, e, P, C, q,  t, V_0, r, alpha, vis_a, vis_b, gamma

def get_nt(U_p, dx, n_p, gamma, vis_b, ratio_of_critical):
    C = np.sqrt(gamma)
    dt = ratio_of_critical*dx/C*vis_b
    T = dx*n_p/abs(2.0/(gamma-1.0)*np.sqrt(gamma))*2.0
    n_t = int(round(T/dt))
    print n_t
    return n_t

def initial(U_p, dx, n_p, n_t, gamma ):
    u = np.zeros((n_t, n_p))
    R = np.zeros((n_t, n_p))
    V = np.zeros((n_t, n_p))
    e = np.zeros((n_t, n_p))
    P = np.zeros((n_t, n_p))
    C = np.zeros((n_t, n_p))
    t = np.zeros(n_t)
    
    r= np.linspace(0.0, n_p - 1.0, n_p)*dx          # initial r
    V_0 = np.zeros(n_p)+1.0                         # initial V
    P[0] = np.zeros(n_p)+1.0                        # initialize P
    V[0] = np.zeros(n_p)+1.0                        # initialize V
    R[0] = np.linspace(0.0, n_p - 1.0, n_p)*dx      # initialize R
    u[0][0] = U_p                                   # initialize u
    C[0] = np.zeros(n_p)+ np.sqrt(gamma)
    e[0] = 1.0/(gamma - 1.0)*P[0]*V[0]
    return u, R, V, e, P, C, t, V_0, r

def get_q( u, V_1, V__0, vis_a):
    # j is the space coordinate of the interest point
    # u is an array with all info about vel for all points at a specific time
    # V_1 is an array with all volume info for all points at a time step
    # after the interest time 
    # V_0 is the array wiht all volume info far all points at the time of interest
    # return a viscousity array from index 0 to n-1 with length n
    # but the useful length is 0 to n-2, that is q[:-1]
    q = np.zeros(len(u)) 
    for j in range(len(q)-1):
        if u[j+1] - u[j] < 0 :
             q[j] = 2.0*vis_a*vis_a*(u[j+1] - u[j])*(u[j+1]-u[j])/(V_1[j]+V__0[j])
    #print u[1] - u[2], (V_1[1]+V__0[2])/2.0, q[1]
    return q

def get_C( V, P,  gamma):
    # the P and V are arrays of P and V at time of interst (1D array)
    C = np.zeros(len(V))
    C[1:] = np.sqrt(gamma*P[1:]*V[1:])
    C[0] = C[1]
    return C

def get_dt(R, C, vis_b, ratio_of_critical ):
    # R and C are 1-D array
    R_1 = R[1:]
    R_0 = R[:-1]
    #print C[:-1]
    dt_arr = vis_b* (R_1 - R_0)/C[:-1]
    return ratio_of_critical*np.amin(dt_arr)
def update_u(dt, u, R, P,  q, V_0, r, alpha):
    # the good points are from index 1 to index n-1 
    # the index 0 is at the piston and is not good
    # and index n are not important
    V_arr = 0.5*(V_0[1:-1]+V_0[2:])         # array of length n -2 
    P_arr = (P[1:-1] - P[:-2])              # P is array of length n and P_arr of length n-2  1:-1 ,  :-2
    q_arr = (q[1:-1] - q[:-2])              # q is array of length n and q_arr of length n-2  1:-1 ,  :-2
    r_arr = 0.5*(r[2:] - r[:-2])            # r is array of length n and r-arr of length n-2
    u_arr = u[1:-1]
    u_array = np.zeros(len(u))
    u_array[1:-1] = u_arr - dt*V_arr*(R[1:-1]/r[1:-1])**(alpha - 1)*(P_arr+q_arr)/(r_arr)
    u_array[0] = u[0]
    u_array[-1] = u[-1]
    return u_array

def update_R(dt, u , R):
    R_new = dt*u + R
    for i in range(1,len(R_new)):
        R_new[i] =max(R_new[i], R_new[i-1])        # R_new[0] is the piston position
    return  R_new

def get_V(R, V, V_0, r):
    # update 0 - n-1 term since there should be only n-1 half grids
    V_arr = np.zeros(len(V))
    V_arr[:-1] = V_0[:-1]*(R[1:]**(alpha) - R[:-1]**(alpha))/(r[1:]**(alpha)-r[:-1]**(alpha))
    V_arr[-1] = V[-1]
    return V_arr 

def get_e(V_1, V__0, e, P, q_1, gamma):
    e_arr = np.zeros(len(e))
    arr_1 =  (1.0+ (gamma - 1.0)*0.5*(V_1[:-1]-V__0[:-1])/V_1[:-1])
    arr_2 = e[:-1]-(0.5*P[:-1]+q_1[:-1])*(V_1[:-1]- V__0[:-1])
    #print P[1], q_1[1],  arr_2[1] 
    e_arr[:-1] = arr_2/arr_1
    e_arr[-1] = e[-1]
    #print e[0], e[1]
    return e_arr

def get_P(P, e_1,V_1, gamma):
    P_arr = np.zeros(len(P))
    P_arr[:-1] = (gamma-1.0)*e_1[:-1]/V_1[:-1]
    P_arr[-1] = P[-1]
    return P_arr

def get_tot_energy_change(V_1, V__0,  P_1, P_0,  e_1, e_0, u_1, u_0, dt ):
    kinetic = ( np.sum((0.5*u_1*u_1)[1:-1]) - np.sum((0.5*u_0*u_0)[1:-1]) )/dt
    thermal = ( np.sum(e_1[1:-1] - e_0[1:-1]) )/dt
    extwork = ( 0.5*(P_1[1]+ P_0[1])*(np.sum(V_1)-np.sum(V__0)) ) #/dt
    total = kinetic + thermal + extwork
    return kinetic , thermal, extwork, total

def get_tot_energy(n, V, P, e, u, q, q_old, extwork, dt, E_0):
    kinetic = np.sum((0.5*u[n]*u[n])[1:-1])      # 1
    thermal = np.sum(e[:-1])               # 0
    #extwork = np.sum(0.5*(P[:,0][1:n+1]+P[:,0][:n])*(V.sum(axis = 1)[1:n+1] - V.sum(axis = 1)[:n]) )
    total = (kinetic + thermal + extwork - E_0)/E_0
    #total = (total - 1498.5) / (abs(kinetic)+abs(thermal)+abs(extwork))
    extwork = extwork + 1.0* np.sum((0.5*P[n][:-1]+ 0.5*P[n+1][:-1] + 0.0* q[:-1] )*(V[n+1][:-1] - V[n][:-1])) + dt * 10.0* np.sum((P[n][1:-1] - P[n][:-2] +0.0* q_old[1:-1] -0.0* q_old[:-2] ) * (u[n+1][1:-1]+u[n][1:-1]))*0.5
    #extwork = extwork + 0.5*(P[n][0]+P[n+1][0])*(np.sum(V[n+1])- np.sum(V[n])) #+ dt * 10.0* np.sum((P[n][1:-1] - P[n][:-2]) * (u[n+1][1:-1]+u[n][1:-1]))*0.5
    #total = kinetic + thermal + extwork
    return kinetic , thermal, extwork, total

def update_all(n,u, R, V, e, P, C, q,  t, V_0, r, alpha, vis_a, vis_b, gamma,dt ,ratio_of_critical):
    
    t[n+1] = t[n] + dt
    u[n+1] = update_u(dt, u[n], R[n], P[n],  q, V_0, r, alpha)
    R[n+1] = update_R(dt, u[n+1] , R[n])
    V[n+1] = get_V(R[n+1],V[n],V_0,r)
    q_new  = get_q(u[n+1], V[n+1], V[n], vis_a)
    e[n+1] = get_e(V[n+1],V[n],e[n],P[n],q_new,gamma)
    P[n+1] = get_P(P[n],e[n+1],V[n+1],gamma)
    C[n+1] = get_C(V[n+1], P[n+1],gamma)
    dt_new = get_dt(R[n+1], C[n+1], vis_b, ratio_of_critical)
    
    return q_new, dt_new

def get_boundary(u_p,C_0, gamma, t):
    u_p = min(abs(u_p), abs(2.0/(gamma-1.0)*C_0))
    x_1 = (C_0 - 0.5 * (gamma+1.0)*u_p)*t
    x_2 = (C_0)*t
    return x_1, x_2

def get_strong_shock_boundary(P, V_0, gamma, x_shock, dt ):
    speed = np.sqrt(V_0/2.0 *( (gamma+1.0)*P+ (gamma -1.0) ))
    #print speed
    #print x_shock
    return x_shock + speed*dt

def rare_analytic(R,t, U_p,gamma ):
    C_0 = np.sqrt(gamma)
    U_p = -1.0*min( abs(U_p), abs(2.0*C_0/(gamma-1.0)) )
    gamma_p_1 = 2.0/(gamma +1.0)
    gamma_m_1 = 2.0/(gamma -1.0)
    B_1 = C_0*t
    B_2 = (C_0 + 0.5*(gamma+1.0)*U_p)*t 
    u = np.zeros(len(R))
    for i in range(len(R)):
        if R[i] > B_1 :
            u[i] = 0.0
        elif (R[i] > B_2 and R[i] <B_1):
            u[i] = - gamma_p_1*(C_0-R[i]/float(t))/C_0
        else:
            u[i] = U_p/C_0
    C = (1.0+u/gamma_m_1)
    V = (C)**gamma_m_1
    P = (C)**(gamma*gamma_m_1)
    return u,C,V,P
def rk_update_all(n,u, R, V, e, P, C, q, t, V_0, r, alpha, vis_a, vis_b, gamma,dt ,ratio_of_critical):
    t[n+1] = t[n] + dt
    dt = dt/6.0
    results = np.zeros((5, 5, len(u[n])))
    results[0] = np.array([u[n],R[n], V[n], e[n], P[n]]) 
    for i in range(4):
        results[i+1][0] = update_u(dt, results[i][0], results[i][1], results[i][4], q, V_0, r, alpha)
        results[i+1][1] = update_R(dt, results[i][0], results[i][1])
        results[i+1][2] = get_V(results[i+1][1], results[i][2], V_0, r)
        q = get_q(results[i+1][0], results[i+1][2], results[i][2], vis_a)
        results[i+1][3] = get_e(results[i+1][2], results[i][2], results[i][3], results[i][4], q, gamma)
        results[i+1][4] = get_P(results[i][4],results[i+1][3], results[i+1][2], gamma)
    u[n+1] = (1.0* results[1][0] + 2.0*results[2][0] + 2.0*results[3][0] + 1.0*results[4][0])/6.0
    R[n+1] = (1.0* results[1][1] + 2.0*results[2][1] + 2.0*results[3][1] + 1.0*results[4][1])/6.0
    V[n+1] = (1.0* results[1][2] + 2.0*results[2][2] + 2.0*results[3][2] + 1.0*results[4][2])/6.0
    e[n+1] = (1.0* results[1][3] + 2.0*results[2][3] + 2.0*results[3][3] + 1.0*results[4][3])/6.0
    P[n+1] = (1.0* results[1][4] + 2.0*results[2][4] + 2.0*results[3][4] + 1.0*results[4][4])/6.0
    C[n+1] = get_C(V[n+1], P[n+1],gamma)
    q_new  = get_q( u[n+1], V[n+1], V[n], vis_a)
    dt_new = get_dt(R[n+1], C[n+1], vis_b, ratio_of_critical)
    return q_new, dt_new

def plot_rare(n,u,R,P,V,C, U_p, gamma, time, savefile):
    plt.figure(1)
    plt.xlabel("x")
    plt.ylabel("ratio")
    x_1, x_2 = get_boundary(U_p,np.sqrt(gamma), gamma, time)
    plt.axvline(x=x_1, color = "black", ls = "dashed")
    plt.axvline(x=x_2, color = "black", ls = "dashed")
    plt.plot(R[n][1:], u[n][1:]/C[0][1], label = "u/C0" , color = "green")
    #plt.plot(R[n][1:], P[n][1:]/P[0][1], label = "P/P0" , color = "deepskyblue")
    #plt.plot(R[n][1:], V[0][1]/V[n][1:], label = "rho/rho0", color = "salmon" )
    #plt.plot(R[n][1:], C[n][1:]/C[0][1], label = "C/C0", color = "yellowgreen" )
    
    u_anly, C_anly, V_anly, P_anly = rare_analytic(R[n][1:],time, U_p,gamma )
    
    plt.plot(R[n][1:], u_anly, color = "black" , ls = "dashed")
    #plt.plot(R[n][1:], P_anly, color = "blue" , ls = "dashed")
    #plt.plot(R[n][1:], V_anly, color = "red" , ls = "dashed")
    #plt.plot(R[n][1:], C_anly, color = "green" , ls = "dashed")

    plt.ylim([-3.0,0.5])
    #plt.legend()
    plt.title("t = "+ str(time))
    plt.savefig(savefile)
    plt.close()
    return

def plot_shock(n, u, R, P, V, e, C, r,  x_shock, P_max, time,  savefile, lagrangean = False ):
    
    u_max = np.amax(u[n][1:]) 
    x_axis = R[n][1:]
    x_name = "x"
    if lagrangean:
        x_axis = r[1:]
        x_name = "r"
    plt.subplot(221)
    plt.xlabel(x_name)
    plt.plot(x_axis, u[n][1:]/C[0][1:], label = "u/u0" )                         
    plt.axvline(x=x_shock, color = "black", ls = "dashed")
    plt.legend()

    plt.subplot(222)
    plt.plot(x_axis, P[n][1:]/P[0][1:], label = "P/P0" )                         
    plt.axvline(x=x_shock, color = "black", ls = "dashed")
    plt.xlabel(x_name)
    plt.legend() 

    plt.subplot(223)
    plt.xlabel(x_name)
    plt.plot(x_axis, V[0][1:]/V[n][1:], label = "rho/rho0")          
    plt.axvline(x=x_shock, color = "black", ls = "dashed") 
    plt.legend() 

    plt.subplot(224)
    plt.xlabel(x_name)
    plt.plot(x_axis, e[n][1:]/e[0][1:], label = "e/e0")
    plt.axvline(x=x_shock, color = "black", ls = "dashed")
    plt.legend() 

    #plt.plot(x_axis[1:], R[n][1:-1]- R[n][:-2])
    #plt.axvline(x=x_shock, color = "black", ls = "dashed")
    #plt.ylim([0.0,1.2])
    plt.suptitle("t = "+ str(time))
    plt.savefig(savefile)
    plt.close()
    return

def main_evolve(U_p, gamma, vis_a, vis_b, alpha, dx, n_p, ratio_of_critical, Plot = True):
    n_t = get_nt(U_p, dx, n_p, gamma, vis_b, ratio_of_critical)
    u, R, V, e, P, C, t, V_0, r = initial(U_p, dx, n_p, n_t, gamma )
    q = get_q( u[0], V[0], V[0], vis_a)
    dt = get_dt(R[0],C[0], vis_b, ratio_of_critical)
    extwork = 0.0
    x_shock = 0.0
    dt_min = 1.0e-7
    E_0 = 0.5*np.sum(u[0][1:]*u[0][1:])+np.sum(e[0][:-1])
    #fo= open("shock_parameter_test.txt","w")
    
    for n in range(int(0.3*n_t)):
        P_max = np.amax(P[n][1:])
        x_shock = get_strong_shock_boundary(P_max, V_0[1], gamma, x_shock, dt_min )
        q_old = q
        q,dt = update_all(n,u, R, V, e, P, C, q, t, V_0, r, alpha, vis_a, vis_b, gamma,dt_min ,ratio_of_critical)
        kinetic , thermal, extwork, total = get_tot_energy( n, V, P, e[n], u, q, q_old,  extwork, dt_min, E_0 )
        time = t[n]
        #print time, u[n][10]/C[0][10], C[n][10]/C[0][10], P[n][10]/P[0][10], V[0][10]/V[n][10]
        
        print time, kinetic , thermal, extwork, total  #
        #fo.write(str(time)+'\t'+str(u[n][10]/C[0][10])+'\t'+str(C[n][10]/C[0][10])+'\t'+str( P[n][10]/P[0][10]) +'\t'+str( V[0][10]/V[n][10])+'\n')
        dt_min = min(dt,dt_min)
    dt = dt_min
    
    for n in range(int(0.3*n_t), n_t-1):
        #print n
        q_old = q
        dt_old = dt
        q,dt = update_all(n,u, R, V, e, P, C, q, t, V_0, r, alpha, vis_a, vis_b, gamma,dt ,ratio_of_critical)
        kinetic , thermal, extwork, total = get_tot_energy( n, V, P, e[n], u, q, q_old , extwork, dt_old, E_0 )
        time = t[n]
        print time, kinetic , thermal, extwork, total  #, total_1
        #print time, u[n][10]/C[0][10], C[n][10]/C[0][10], P[n][10]/P[0][10], V[0][10]/V[n][10]
        #P_max = np.amax(P[n][1:])
        #fo.write(str(time)+'\t'+str(u[n][10]/C[0][10])+'\t'+str(C[n][10]/C[0][10])+'\t'+str( P[n][10]/P[0][10]) +'\t'+str( V[0][10]/V[n][10])+'\n')
        if (n!=0 and n % 10000 == 0 and Plot):
            savefile = "/Users/Chadwick/Documents/my study/Subjects/physics/physics 598/hw_5/shock_10_test/"+str(n).zfill(8)+".png"
            #plot_rare(n,u,R,P,V,C, U_p, gamma, time, savefile)
            #plot_shock(n, u, R, P, V, e, C,  r,  x_shock, P_max, time,  savefile, lagrangean = False)
        #x_shock = get_strong_shock_boundary(P_max, V_0[1], gamma, x_shock, dt )
    return
###### main ###########################

main_evolve(U_p, gamma, vis_a, vis_b, alpha, dx, n_p, ratio_of_critical, Plot = plot)
