'''
Stanley Bak
Autopilot State-Machine Logic

There is a high-level advance_discrete_state() function, which checks if we should change the current discrete state,
and a get_u_ref(f16_state) function, which gets the reference inputs at the current discrete state.
'''

import abc
from math import pi, sin, sqrt, cos, asin, atan2

import numpy as np
from numpy import deg2rad

from aerobench.code.aerobench.util import StateIndex
from aerobench.code.aerobench.lowlevel.low_level_controller import LowLevelController
from aerobench.code.aerobench.util import Freezable
from utils.util import wrap_to_pi

class Autopilot(Freezable):
    '''A container object for the hybrid automaton logic for a particular autopilot instance'''

    def __init__(self, init_mode, llc=None):

        assert isinstance(init_mode, str), 'init_mode should be a string'

        if llc is None:
            # use default
            llc = LowLevelController()

        self.llc = llc
        self.xequil = llc.xequil
        self.uequil = llc.uequil
        
        self.mode = init_mode # discrete state, this should be overwritten by subclasses

        self.freeze_attrs()

    def advance_discrete_mode(self, t, x_f16):
        '''
        advance the discrete mode based on the current aircraft state. Returns True iff the discrete mode
        has changed. It's also suggested to update self.mode to the current mode name.
        '''

        return False

    def is_finished(self, t, x_f16):
        '''
        returns True if the simulation should stop (for example, after maneuver completes)

        this is called after advance_discrete_state
        '''

        return False

    @abc.abstractmethod
    def get_u_ref(self, t, x_f16):
        '''
        for the current discrete state, get the reference inputs signals. Override this one
        in subclasses.

        returns four values per aircraft: Nz, ps, Ny_r, throttle
        '''

        return

    def get_checked_u_ref(self, t, x_f16):
        '''
        for the current discrete state, get the reference inputs signals and check them against ctrl limits
        '''

        rv = np.array(self.get_u_ref(t, x_f16), dtype=float)

        assert rv.size % 4 == 0, "get_u_ref should return Nz, ps, Ny_r, throttle for each aircraft"

        for i in range(rv.size //4):
            Nz, _ps, _Ny_r, _throttle = rv[4*i:4*(i+1)]

            l, u = self.llc.ctrlLimits.NzMin, self.llc.ctrlLimits.NzMax
            assert l <= Nz <= u, f"autopilot commanded invalid Nz ({Nz}). Not in range [{l}, {u}]"

        return rv

class FixedSpeedAutopilot(Autopilot):
    '''Simple Autopilot that gives a fixed speed command using proportional control'''

    def __init__(self, setpoint, p_gain):
        self.setpoint = setpoint
        self.p_gain = p_gain

        init_mode = 'tracking speed'
        Autopilot.__init__(self, init_mode)

    def get_u_ref(self, t, x_f16):
        '''for the current discrete state, get the reference inputs signals'''

        x_dif = self.setpoint - x_f16[0]

        return 0, 0, 0, self.p_gain * x_dif


class FollowerAutopilot(Autopilot):
    '''Autopilot that follows given heading, height and speed using PD control'''

    def __init__(self, gain_str='old'):
        # default control when not waypoint tracking
        self.cfg_u_ol_default = (0, 0, 0, 0.3)

        # control config
        # Gains for speed control
        self.cfg_k_vt = 0.25
        self.cfg_airspeed = 540

        # Gains for altitude tracking
        self.cfg_kp_alt = 0.00005
        self.cfg_kp_h_dot = 0.0

        # Gains for heading tracking
        self.cfg_k_prop_psi = 0
        self.cfg_k_der_psi = 0.

        # Gains for roll tracking
        self.cfg_k_prop_phi = 0.
        self.cfg_k_der_phi = 0.
        self.cfg_max_bank_deg = 65 # maximum bank angle setpoint

        # Ranges for Nz
        self.cfg_max_nz_cmd = 4
        self.cfg_min_nz_cmd = -1

        llc = LowLevelController(gain_str=gain_str)

        Autopilot.__init__(self, 'Following', llc=llc)


    def get_u_ref(self, x_f16, delta_value):
        '''get outloop command: Nz, ps, Ny_r, throttle'''

        delta_h = delta_value[0]
        delta_heading = delta_value[1]
        delta_v = delta_value[2]

        if self.mode != "Done":
            # Get desired roll angle given desired heading
            phi_cmd = self.get_phi_to_track_heading(x_f16, delta_heading)
            ps_cmd = self.track_roll_angle(x_f16, phi_cmd)
            nz_cmd = self.track_altitude(x_f16, delta_h)
            throttle = self.track_airspeed(x_f16, delta_v)
        else:
            # Waypoint Following complete: fly level.
            phi_cmd = 0
            delta_h = 0
            throttle = self.track_airspeed(x_f16, delta_v)
            ps_cmd = self.track_roll_angle(x_f16, phi_cmd)
            nz_cmd = self.track_altitude_wings_level(x_f16, delta_h)

        # trim to limits
        nz_cmd = max(self.cfg_min_nz_cmd, min(self.cfg_max_nz_cmd, nz_cmd))
        throttle = max(min(throttle, 1), 0)

        # Create reference vector
        rv = [nz_cmd, ps_cmd, 0, throttle]

        return rv
    
    def track_altitude(self, x_f16, delta_h):
        'get nz to track altitude, taking turning into account'

        phi = x_f16[StateIndex.PHI]

        nz_alt = self.track_altitude_wings_level(x_f16, delta_h)
        nz_roll = get_nz_for_level_turn_ol(x_f16)

        if delta_h > 0:
            # Ascend wings level or banked
            nz = nz_alt + nz_roll
        elif abs(phi) < np.deg2rad(15):
            # Descend wings (close enough to) level
            nz = nz_alt + nz_roll
        else:
            # Descend in bank (no negative Gs)
            nz = max(0, nz_alt + nz_roll)

        return nz

    def get_phi_to_track_heading(self, x_f16, delta_psi):
        'get phi from psi_cmd, aerobench version'

        # PD Control on heading angle using phi_cmd as control

        # Pull out important variables for ease of use
        r = x_f16[StateIndex.R]

        phi_cmd = wrap_to_pi(delta_psi) * self.cfg_k_prop_psi - r * self.cfg_k_der_psi

        # Bound to acceptable bank angles:
        max_bank_rad = np.deg2rad(self.cfg_max_bank_deg)

        phi_cmd = min(max(phi_cmd, -max_bank_rad), max_bank_rad)

        # 'get phi_cmd form desired heading, L1 guidance version'

        # Va = x_f16[StateIndex.ALT]
        # HeadingAngle = x_f16[StateIndex.PSI]
        # YawAngle = HeadingAngle

        # HeadingControl = self.cfg_k_prop_psi*wrap_to_pi(psi_cmd - HeadingAngle) - self.cfg_k_der_psi*x_f16[StateIndex.R]
        # phi_cmd = atan2(Va*HeadingControl, 9.81*cos(HeadingAngle - YawAngle))

        # # Bound to acceptable bank angles:
        # max_bank_rad = np.deg2rad(self.cfg_max_bank_deg)
        # phi_cmd = min(max(phi_cmd, -max_bank_rad), max_bank_rad)

        return phi_cmd


    def track_roll_angle(self, x_f16, phi_cmd):
        'get roll angle command (ps_cmd)'

        # PD control on roll angle using stability roll rate

        # Pull out important variables for ease of use
        phi = wrap_to_pi(x_f16[StateIndex.PHI])
        p = x_f16[StateIndex.P]

        # Calculate PD control
        ps = wrap_to_pi(phi_cmd - phi) * self.cfg_k_prop_phi - p * self.cfg_k_der_phi

        return ps

    def track_airspeed(self, x_f16, delta_v):
        'get throttle command'

        # Proportional control on airspeed using throttle
        throttle = self.cfg_k_vt * (self.cfg_airspeed - x_f16[StateIndex.VT])

        return throttle

    def track_altitude_wings_level(self, x_f16, delta_h):
        'get nz to track altitude'

        vt = x_f16[StateIndex.VT]

        # Proportional-Derivative Control
        h_error = delta_h
        gamma = get_path_angle(x_f16)
        h_dot = vt * sin(gamma) # Calculated, not differentiated

        # Calculate Nz command
        nz = self.cfg_kp_alt*h_error - self.cfg_kp_h_dot*h_dot
        
        return nz

    def is_finished(self, t, x_f16):
        'is the maneuver done?'

        rv = self.waypoint_index >= len(self.waypoints) and self.done_time + 5.0 < t

        return rv
    


def get_nz_for_level_turn_ol(x_f16):
    'get nz to do a level turn'

    # Pull g's to maintain altitude during bank based on trig

    # Calculate theta
    phi = x_f16[StateIndex.PHI]

    if abs(phi) != 0: # if cos(phi) ~= 0, basically
        nz = 1 / cos(phi) - 1 # Keeps plane at altitude
    else:
        nz = 0

    return nz

def get_path_angle(x_f16):
    'get the path angle gamma'

    alpha = x_f16[StateIndex.ALPHA]       # AoA           (rad)
    beta = x_f16[StateIndex.BETA]         # Sideslip      (rad)
    phi = x_f16[StateIndex.PHI]           # Roll angle     (rad)
    theta = x_f16[StateIndex.THETA]       # Pitch angle   (rad)

    gamma = asin((cos(alpha)*sin(theta)- \
        sin(alpha)*cos(theta)*cos(phi))*cos(beta) - \
        (cos(theta)*sin(phi))*sin(beta))

    return gamma

def cart2sph(pt3d):
    '''
    Cartesian to spherical coordinates

    returns az, elev, r
    '''

    x, y, z = pt3d

    h = sqrt(x*x + y*y)
    r = sqrt(h*h + z*z)

    elev = atan2(z, h)
    az = atan2(y, x)

    return az, elev, r
