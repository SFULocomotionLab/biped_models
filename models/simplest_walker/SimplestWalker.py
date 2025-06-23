import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from biped.BipedBaseClass import BipedBaseClass


class SimplestWalker(BipedBaseClass):
    """Class to simulate the simplest walker for one step.
    Everything is contained within the class with no external dependencies
    except for standard Python libraries.
    """

    def __init__(self, x0, s_nominal, u_nominal):
        """Initialize the walker.

        Args:
            x0: Initial state vector [stance_angle, stance_vel]
            s_nominal: Nominal states (or step length and step frequency)  
            u_nominal: Nominal control inputs or pushoff and hip spring stiffness
        """
        super().__init__()  # Call parent class constructor
        
        # Initial conditions and parameters
        self.GAMMA = 0  # slope
        self.L = 1  # walker length

        # Event tracking
        self.foot_contact_flag = False
        self.fall_flag = False
        self.foot_contact_count = 0
        
        # Step measures
        self.step_time = None
        self.step_length = None
        self.step_length_total = 0
        self.speed = None
        self.hip_dist = None  # hip distance in X direction

        # Work/energy measures
        self.w_toe = None  # pushoff work
        self.w_swing = None  # swing work
        self.w_k = None  # spring work
        self.w_total = None  # total work

        # Simulation parameters
        self.anim = False  # animation flag

        # Integration solutions
        self.INT_ABS_TOL = 1e-12
        self.INT_REL_TOL = 1e-12

        # Position vectors
        self.stance_x = [0]    # stance foot position

        # State variables
        self.next_state = None
        self.x0 = x0  # actual step length/frequency
        self.s_nominal = s_nominal  # nominal step length/frequency
        self.u_nominal = u_nominal  # nominal control inputs

        # Initialize posture
        self.st_foot, self.sw_foot, self.hip = self.get_trajectory(self.x0)

        # Initialize animation figure
        self.fig = None

    def take_one_step(self, x0, u0):
        """Simulate one step of the walker.

        Args:
            x0: Initial state vector [stance_angle, stance_vel]
            u0: Control inputs [pushoff, k1, k2]

        Returns:
            next_state: Final state at end of step
            traj: State trajectory during step
            time: Time points for trajectory
        """
        # Get full state vector by computing swing angle/velocity
        x0 = np.array(x0)
        x_full = np.zeros(4)
        x_full[0] = x0[0]
        x_full[1] = x0[1]
        x_full[2] = 2 * x0[0]
        x_full[3] = (1 - np.cos(2*x0[0])) * x0[1]

        # Set up integration parameters
        max_time = 20  # Maximum simulation time
        t_span = [0, max_time]

        # Define event functions
        # Foot contact event
        def _get_foot_height(t, x):
            # Only detect foot contact if stance angle is less than -0.1
            _, sw_foot, _ = self.get_trajectory(x)
            if x[2] > -0.1 or np.abs(sw_foot[0][1]) > 0.01:
                return 1.0  # Return positive value to prevent event detection
            return sw_foot[0][1]  # Return height of swing foot
        _get_foot_height.terminal = True  # Stop on valid foot contact
        _get_foot_height.direction = -1   # Only detect when foot is coming down
        
        def _get_hip_height(t, x):
            _, _, hip = self.get_trajectory(x)
            return hip[0][1] - 0.2    # Return hip height above threshold
        _get_hip_height.terminal = True    # Always stop on fall
        _get_hip_height.direction = 0      # Detect any crossing

        # Reset flags at start of each step
        self.foot_contact_flag = False
        self.fall_flag = False
        foot_contact_occurred = False
        fall_occurred = False

        # Integrate equations of motion until event
        sol = solve_ivp(
            lambda t, x: self._get_equations_of_motion(t, x, u0),
            t_span,
            x_full,
            method='RK45',
            events=[_get_foot_height, _get_hip_height],
            rtol=self.INT_REL_TOL,
            atol=self.INT_ABS_TOL
        )
        self.int_sol = sol
        next_state = sol.y[:, -1]

        # Check which event triggered the stop
        foot_contact_occurred = len(sol.t_events[0]) > 0  # First event is foot contact
        fall_occurred = len(sol.t_events[1]) > 0  # Second event is hip height

        # If foot contact (not fall), apply impulse
        if foot_contact_occurred and not fall_occurred:
            # Update stance position to the new swing foot position
            _, sw_foot, _ = self.get_trajectory(next_state)
            self.stance_x.append(sw_foot[0][0])  # Add to stance history
            pushoff = u0[0]
            next_state = self._apply_impulse(next_state, pushoff)
            self.foot_contact_count += 1
            self.foot_contact_flag = True
            # print(f'foot contact occurred at {sol.t_events[0][0]}')
        elif fall_occurred:
            self.fall_flag = True
            self.foot_contact_flag = False
            # print(f'fall occurred at {sol.t_events[1][0]}')
        # If neither event occurred, it means the walker is falling
        elif not foot_contact_occurred and not fall_occurred:
            self.fall_flag = True
            self.foot_contact_flag = False
            # print('neither event occurred')
        next_state = next_state.T

        return next_state, sol.y, sol.t

    def _get_equations_of_motion(self, t, x, u):
        """Calculate equations of motion for the walker.

        Args:
            t: Time (not used but required for ODE solver)
            x: State vector [stance_angle, stance_vel, swing_angle, swing_vel]
            u: Control inputs [pushoff, k1, k2]

        Returns:
            dx: State derivatives [stance_vel, stance_acc, swing_vel, swing_acc]
        """
        # Extract states
        theta = x[0]  # stance angle
        dtheta = x[1]  # stance angular velocity
        phi = x[2]  # swing angle
        dphi = x[3]  # swing angular velocity

        # First derivatives (velocities)
        y1 = dtheta
        y2 = np.sin(theta - self.GAMMA)
        y3 = dphi
        y4 = np.sin(theta - self.GAMMA) + dtheta**2 * np.sin(phi) - np.cos(theta -
                                                                           self.GAMMA) * np.sin(phi) - (phi > 0) * u[1] * phi - (phi <= 0) * u[2] * phi
        
        return np.array([y1, y2, y3, y4])

    def _apply_impulse(self, x, P):
        """Apply impulse at foot contact.

        Args:
            x: Pre-impact state [stance_angle, stance_vel, swing_angle, swing_vel]
            P: Push off impulse

        Returns:
            q_plus: Post-impact state [stance_angle, stance_vel, swing_angle, swing_vel]
        """
        # Extract states
        theta = x[0]  # stance angle
        dtheta = x[1]  # stance angular velocity
        phi = x[2]    # swing angle
        dphi = x[3]   # swing angular velocity

        # Matrix multiplication for state transformation
        q_plus = np.array([
            -1, 0, 0, 0,
            0, np.cos(2*theta), 0, 0,
            -2, 0, 0, 0,
            0, np.cos(2*theta) * (1 - np.cos(2*theta)), 0, 0
        ]).reshape(4, 4) @ np.array([theta, dtheta, phi, dphi]) + \
            np.array([0, np.sin(2*theta), 0, np.sin(2*theta)
                     * (1-np.cos(2*theta))]) * P

        return q_plus

    def get_step_measures(self, next_state):
        """Calculate step measurements for the walker.

        Args:
            next_state: The state vector after a step

        Returns:
            tuple: (step_length, speed, hip_distance, step_time)
        """
        # Calculate postures at this step
        st_foot, sw_foot, hip = self.get_trajectory(next_state)

        # Update object's foot and hip positions
        self.st_foot = np.vstack((self.st_foot, st_foot[-1, :]))
        self.sw_foot = np.vstack((self.sw_foot, sw_foot[-1, :]))
        self.hip = np.vstack((self.hip, hip[-1, :]))

        # Calculate step measurements
        step_time = self.int_sol.t[-1]  # Last time point from integration
        hip_dist = np.diff(self.hip[:, 0])  # Distance hip travelled
        hip_dist = hip_dist[-1]  # Get last value
        speed = hip_dist / step_time  # Hip velocity
        step_length = self.st_foot[-1, 0] - self.st_foot[-2, 0]  # Step length

        # Update object properties
        self.step_time = step_time
        self.hip_dist = hip_dist
        self.speed = speed
        self.step_length = step_length

        return step_length, speed, hip_dist, step_time

    def calculate_energy(self):
        """Calculate energetic costs of walking.

        Returns:
            tuple: (w_toe, w_swing, w_total) containing:
                - w_toe: Push-off work per unit distance (cost of transport)
                - w_swing: Swing leg work per unit distance
                - w_total: Total work per unit distance
        """
        # Get final state values
        k = self.int_sol.y[2, -1]  # Hip spring constant
        v_com = self.int_sol.y[1, -1]  # COM velocity before heel strike
        alpha = self.int_sol.y[0, -1]  # Stance angle at ground contact

        # -------------------- toe off work ---------------------------
        # 1- Conservation of angular momentum method
        # Uses law of conservation of angular momentum twice:
        # Once for before/after pushoff and once for before/after heel strike
        # Total work divided by 2
        w1 = 0.25 * (v_com**2) * ((np.tan(abs(alpha))**2)/(np.cos(abs(alpha))**2) + np.tan(abs(alpha))**2)
        w_toe1 = w1/self.step_length

        # 2- Manoj's dissertation method
        # No small angle approximation
        # Derived from conservation of linear momentum before/after push off
        # w2 = 0.5 * (v_com**2) * np.tan(abs(alpha))**2  # Eqn 2.8 pg 24
        # w_toe2 = w2/self.step_length

        # 3- Ruina's paper (2005) method - uses small angle approximation
        # phi = 2 * abs(alpha)
        # b = 1  # For cost of transport
        # em = b * (phi**2) * (v_com**2)/8
        # w_toe3 = em/self.step_length

        # 4- Kuo (2002) method - uses small angle approximation
        # 4-1 Based on push-off impulse
        # w4 = 0.5 * p**2
        # w_toe4 = w4/self.step_length

        # 4-2 Based on COM velocity and stance angle
        # w_toe4 = 0.25 * v_com**2 * abs(alpha)

        # ---------------------- swing work ---------------------------
        # Kuo's methods - all are cost of transport (divided by 2*alpha)
        # 4 versions from Table 1, Kuo 2001 paper
        e_swing1 = k * alpha  # Based on swing work (eqn 8)
        # e_swing2 = k  # Based on peak force (eqn 9)
        # e_swing3 = k * self.step_time  # Based on impulse (eqn 10)
        # e_swing4 = k / self.step_time  # Based on force/time (eqn 11)

        # Use first method for now
        w_toe = w_toe1
        w_swing = e_swing1
        w_total = w_toe + w_swing

        return w_toe, w_swing, w_total
    
    def apply_feedback_controller(self, x, K_gain):
        """Calculate control inputs using feedback control.

        Args:
            x: State vector [stance_angle, stance_vel]'
            K_gain: Gain matrix for feedback control

        Returns:
            u: Control inputs [pushoff, k1, k2]
        """
        # Get nominal values
        s_nom = self.s_nominal
        u_nom = self.u_nominal

        # Calculate state error
        ds = x - s_nom

        # Calculate control input changes
        du = -K_gain @ ds

        # Add changes to nominal control
        u = u_nom + du

        return u
    
    def do_linear_analysis(self, x0, u0, eps=1e-5):
        """Perform linear analysis around nominal trajectory.
        
        Calculates A, B, C, and D matrices for linearized dynamics around nominal trajectory
        using numerical perturbation method with central difference.

        Args:
            x0: Nominal state vector [stance_angle, stance_vel]
            u0: Nominal control vector [pushoff, k1, k2] 
            eps: Small perturbation size for numerical derivatives

        Returns:
            A: State transition matrix (2x2)
            B: Control input matrix (2x3)
            C: Output-state matrix (2x2)
            D: Output-control matrix (2x3)
        """
        # Initialize matrices
        n_states = 2  # Only using stance angle and velocity
        n_inputs = 3  # Pushoff and two spring constants
        n_outputs = 2  # Step length and step frequency
        
        A = np.zeros((n_states, n_states))
        B = np.zeros((n_states, n_inputs))
        C = np.zeros((n_outputs, n_states))
        D = np.zeros((n_outputs, n_inputs))

        # Get nominal trajectory and outputs
        x_nom, _, _ = self.take_one_step(x0, u0)
        x_nom = x_nom[0:2]  # Only keep stance states

        # Calculate A and C matrices - state derivatives using central difference
        for i in range(n_states):
            # Positive perturbation
            x_perturb_plus = x0.copy()
            x_perturb_plus[i] += eps
            x_plus, _, _ = self.take_one_step(x_perturb_plus, u0)
            x_plus_states = x_plus[0:2]
            step_length_plus, _, _, step_time_plus = self.get_step_measures(x_plus)
            step_freq_plus = 1.0 / step_time_plus
            y_plus = np.array([step_length_plus, step_freq_plus])
            
            # Negative perturbation
            x_perturb_minus = x0.copy()
            x_perturb_minus[i] -= eps
            x_minus, _, _ = self.take_one_step(x_perturb_minus, u0)
            x_minus_states = x_minus[0:2]
            step_length_minus, _, _, step_time_minus = self.get_step_measures(x_minus)
            step_freq_minus = 1.0 / step_time_minus
            y_minus = np.array([step_length_minus, step_freq_minus])
            
            # Calculate columns of A and C matrices using central difference
            A[:, i] = (x_plus_states - x_minus_states) / (2 * eps)
            C[:, i] = (y_plus - y_minus) / (2 * eps)

        # Calculate B and D matrices - control derivatives using central difference
        for i in range(n_inputs):
            # Positive perturbation
            u_perturb_plus = u0.copy()
            u_perturb_plus[i] += eps
            x_plus, _, _ = self.take_one_step(x0, u_perturb_plus)
            x_plus_states = x_plus[0:2]
            step_length_plus, _, _, step_time_plus = self.get_step_measures(x_plus)
            step_freq_plus = 1.0 / step_time_plus
            y_plus = np.array([step_length_plus, step_freq_plus])
            
            # Negative perturbation
            u_perturb_minus = u0.copy()
            u_perturb_minus[i] -= eps
            x_minus, _, _ = self.take_one_step(x0, u_perturb_minus)
            x_minus_states = x_minus[0:2]
            step_length_minus, _, _, step_time_minus = self.get_step_measures(x_minus)
            step_freq_minus = 1.0 / step_time_minus
            y_minus = np.array([step_length_minus, step_freq_minus])
            
            # Calculate columns of B and D matrices using central difference
            B[:, i] = (x_plus_states - x_minus_states) / (2 * eps)
            D[:, i] = (y_plus - y_minus) / (2 * eps)

        return A, B, C, D

    def calculate_linear_stability(self, x0, u0, eps=1e-6):
        """Perform stability analysis of the walker around nominal trajectory.

        Calculates eigenvalues and eigenvectors of the linearized dynamics
        to determine local stability properties.

        Args:
            x0: Nominal state vector [stance_angle, stance_vel]
            u0: Nominal control vector [pushoff, k1, k2]
            eps: Small perturbation size for numerical derivatives

        Returns:
            eig_vals: Eigenvalues of linearized dynamics
            eig_vecs: Eigenvectors of linearized dynamics
        """
        # Get linearized dynamics matrices
        A, _, _, _ = self.do_linear_analysis(x0, u0, eps)

        # Calculate eigenvalues and eigenvectors
        eig_vals, eig_vecs = np.linalg.eig(A)

        return eig_vals, eig_vecs
    
    def get_trajectory(self, x):
        """Calculate foot and hip positions.

        Args:
            x: State vector [stance_angle, stance_vel, swing_angle, swing_vel]

        Returns:
            st_foot: Stance foot position [x,y]
            sw_foot: Swing foot position [x,y] 
            hip: Hip position [x,y]
        """
        # Get the other 2 state variables if needed
        if len(x) != 4:
            x = np.array([x[0], x[1], 2*x[0], (1-np.cos(2*x[0]))*x[1]])

        # Reshape x to handle multiple states
        states = x.reshape(4, -1)

        # Stance foot is on the ground at the start
        xst0 = self.stance_x[-1]
        yst0 = 0
        size_states = states.shape[1]
        xst = xst0
        yst = yst0

        # Initialize arrays for positions
        xhip = np.zeros(size_states)
        yhip = np.zeros(size_states)
        xsw = np.zeros(size_states)
        ysw = np.zeros(size_states)

        for i in range(size_states):
            # Extract states for this point
            theta = states[0, i]  # stance angle
            phi = states[2, i]    # swing angle

            # Position of hip
            xhip[i] = xst0 - self.L * np.sin(theta - self.GAMMA)
            yhip[i] = yst0 + self.L * np.cos(theta - self.GAMMA)

            # Position of swing foot
            xsw[i] = xhip[i] - self.L * np.sin(phi - theta + self.GAMMA)
            ysw[i] = yhip[i] - self.L * np.cos(phi - theta + self.GAMMA)

            # Stance foot
            xst = np.append(xst, xst0)
            yst = np.append(yst, yst0)

        st_foot = np.column_stack((xst, yst))
        sw_foot = np.column_stack((xsw, ysw))
        hip = np.column_stack((xhip, yhip))

        return st_foot, sw_foot, hip
    
    def animate(self, x):
        """Animate the walker's motion.

        Args:
            x: State vector trajectories containing angles and angular velocities over time

        Returns:
            list: Animation frames
        """
        # Get trajectory data
        st_foot, sw_foot, hip = self.get_trajectory(x)
        xst = st_foot[:, 0]
        yst = st_foot[:, 1]
        xsw = sw_foot[:, 0]
        ysw = sw_foot[:, 1]
        xh = hip[:, 0]
        yh = hip[:, 1]

        # Create figure if it doesn't exist
        if self.fig is None:
            self.fig = plt.figure()
            ax = plt.gca()
            # Remove all axis elements
            ax.axis('off')
            plt.tick_params(
                axis='y',          # changes apply to both x and y axes
                which='major',         # both major and minor ticks are affected
                bottom=False,         # ticks along the bottom edge are off
                top=False,            # ticks along the top edge are off
                left=False,           # ticks along the left edge are off
                right=False,          # ticks along the right edge are off
                labelbottom=False,    # labels along the bottom edge are off
                labelleft=False)      # labels along the left edge are off

        # Plot ground slope
        slope_x = [-1000, 1000]
        slope_y = [0, 0]
        plt.plot(slope_x, slope_y, color='k', linewidth=0.1)

        frames = []
        for i in range(len(xh)):
            plt.cla()  # Clear current axis

            # Set axis limits
            plt.axis([xst[i]-5, xst[i]+5, -1, 5])

            # Plot stance leg (red)
            plt.plot([xst[i], xh[i]], [yst[i], yh[i]], 'r-', linewidth=2)

            # Plot swing leg (blue)
            plt.plot([xsw[i], xh[i]], [ysw[i], yh[i]], 'b-', linewidth=2)

            # Plot ground slope
            plt.plot(slope_x, slope_y, 'k-', linewidth=0.1)

            plt.draw()
            plt.pause(0.001)

            # Capture frame
            frames.append(plt.gcf())

        return frames