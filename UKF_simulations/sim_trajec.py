import numpy as np
import scipy.integrate
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas
import cvxpy
import pynumdiff
import scipy.io

deriv = pynumdiff.finite_difference.first_order
cos = np.cos
sin = np.sin
tan = np.tan

class Solver(object):
    def __init__(self, time_max=2, n_heading_changes=2, headings=None):
        self.u_para = []
        self.u_perp = []
        self.u_phi = []
        self.t = []
        
        self.k = {'km1': 1, 'km2': 0, 'km3': 1, 'km4': 1,
                  'ks1': 1, 'ks2': 1, 'ks3': 0, 'ks4': 1, 'ks5': 0,
                  'C_para': 5, 'C_perp': 5, 'C_phi': 5}
        
        self.time_max = time_max
        self.n_heading_changes = n_heading_changes
        if headings is None:
            self.headings = np.linspace(-np.pi/2., np.pi/2., self.n_heading_changes)
        else:
            self.headings = headings
        self.time_of_heading_changes = np.linspace(0, time_max, self.n_heading_changes+3)
        
        self.current_heading_idx = 0
        
        self.direction_of_travel = []
        self.heading_des = []
    
    def fly_diffeq_dynamic_wind(self, x, t):
        '''
        Fly inspired control. 
        Change u_perp to initiate turn, 
        phi controller keeps phi oriented with airspeed.
        Constant u_para.
        '''
        v_para, v_perp, phi, phidot, w, zeta = x

        a_para = v_para - w*cos(phi-zeta)
        a_perp = v_perp + w*sin(phi-zeta)

        # sensors
        s0 = phi*self.k['ks1']
        s1 = np.arctan2(a_perp, a_para)*self.k['ks2'] + self.k['ks3']
        s2 = np.arctan2(v_perp, v_para)*self.k['ks4'] + self.k['ks5']
        direction_of_travel = s0 + s2
        self.direction_of_travel.append(direction_of_travel)

        wdot = 0.01*np.sin(0.1*t)
        zetadot = 0.05*np.sin(0.14*t)
            
        f0 = np.array([ v_perp*phidot - self.k['C_para']*(v_para-w*cos(phi-zeta)),
                        -v_para*phidot - self.k['C_perp']*(v_perp+w*sin(phi-zeta)),
                        phidot,
                        -self.k['C_phi']*phidot,
                        wdot,
                        zetadot])

        # control
        # u_perp to control direction of travel
        try:
            if t > self.time_of_heading_changes[self.current_heading_idx+1]:
                self.current_heading_idx += 1
                if self.current_heading_idx >= len(self.headings):
                    self.current_heading_idx = len(self.headings)-1
        except:
            pass
        heading_des = self.headings[self.current_heading_idx]
        self.heading_des.append(heading_des)

        u_perp = 100*(heading_des - direction_of_travel)
        self.u_perp.append(u_perp)

        # need positive airspeed
        # can use constant u (offsets the drag)
        # or control to keep constant airspeed or groundspeed
        u_para = 30
        self.u_para.append(u_para)

        # phi: stay oriented with airspeed
        u_phi = -50*(s0 - s1 ) - 10*phidot
        self.u_phi.append(u_phi)
        
        self.t.append(t)

        f_para = np.array([self.k['km1'], 0, 0, self.k['km2'], 0, 0])*u_para
        f_perp = np.array([0, self.k['km3'], 0, 0, 0, 0])*u_perp
        f_phi = np.array([0,0,0,self.k['km4'],0,0])*u_phi

        xdot = f0 + f_para + f_perp + f_phi

        return xdot
    

def get_positions(df):
    dt = np.median(np.diff(df['t']))

    v_x = df['v_para']*cos(df['phi']) - df['v_perp']*sin(df['phi'])
    v_y = df['v_para']*sin(df['phi']) + df['v_perp']*cos(df['phi'])

    x_pos = np.cumsum(v_x)*dt
    y_pos = np.cumsum(v_y)*dt

    df['v_x'] = v_x
    df['v_y'] = v_y

    df['x_pos'] = x_pos
    df['y_pos'] = y_pos

    return df

def data_to_pandas(sol, solver, t):
    v_para = sol[:,0]
    v_perp = sol[:,1]
    phi = sol[:,2]
    phidot = sol[:,3]
    w = sol[:,4]
    zeta = sol[:,5] 
    u_para = np.array(solver.u_para)
    u_perp = np.array(solver.u_perp)
    u_phi = np.array(solver.u_phi)
    u_t = np.array(solver.t)

    a_para = v_para - w*cos(phi-zeta)
    a_perp = v_perp + w*sin(phi-zeta)

    d = {  'v_para': v_para,
           'v_perp': v_perp,
           'phi': phi,
           'phidot': phidot,
           'w': w,
           'zeta': zeta,
           'u_para': np.interp(t, u_t, u_para),
           'u_perp': np.interp(t, u_t, u_perp),
           'u_phi': np.interp(t, u_t, u_phi),
           't': t,
           'a_para': a_para,
           'a_perp': a_perp}
    d.update(solver.k)
    
    df = pandas.DataFrame(d)

    df = get_positions(df)

    return df

def get_many_turns_dynamic_wind(turns=200, Lsec=None):
    dt = 0.01
    if Lsec is None:
        Lsec = turns # sec
    n_heading_changes = turns+1
    
    zeta = np.pi/2.
    headings = [zeta - 0.9*np.pi/2., zeta + 0.9*np.pi/2.]*(turns)
    
    t = np.arange(0, Lsec, dt)
    Lf = len(t) # frames

    x0 = [0.542061360272459, -1.5107554743297094, -0.7094850253100653, 0.005134773205195395, 0.5, zeta]

    solver = Solver(time_max=Lsec, n_heading_changes=n_heading_changes, headings=headings)
    sol = odeint(solver.fly_diffeq_dynamic_wind, x0, t)
    df = data_to_pandas(sol, solver, t)
    #df.to_hdf('twohundred_turns_dynamic_wind.hdf', 'twohundred_turns_dynamic_wind')
    #scipy.io.savemat('twentyone_turns.mat', {name: col.values for name, col in df.items()})
    
    return df

def get_many_turns_dynamic_wind_three(turns=200, Lsec=None):
    dt = 0.01
    if Lsec is None:
        Lsec = turns # sec
    n_heading_changes = turns+1
    
    zeta = np.pi/2.
    #headings = [zeta - 0.9*np.pi/2., zeta + 0.9*np.pi/2., zeta]*int(np.ceil(turns/3.))
    headings = [-np.pi/2.5, np.pi/2.5, 0]*int(np.ceil(turns/3.))
    
    t = np.arange(0, Lsec, dt)
    Lf = len(t) # frames

    x0 = [0.542061360272459, -1.5107554743297094, -0.7094850253100653, 0.005134773205195395, 0.5, zeta]

    solver = Solver(time_max=Lsec, n_heading_changes=n_heading_changes, headings=headings)
    sol = odeint(solver.fly_diffeq_dynamic_wind, x0, t)
    df = data_to_pandas(sol, solver, t)
    #df.to_hdf('twohundred_turns_dynamic_wind.hdf', 'twohundred_turns_dynamic_wind')
    #scipy.io.savemat('twentyone_turns.mat', {name: col.values for name, col in df.items()})
    
    return df


if __name__ == '__main__':

    df = get_many_turns_dynamic_wind()
