import numpy as np
import matplotlib.pyplot as plt


def plot_eigenvalues(data, target_step_length, target_step_frequency, plot=False):
    """Plot contours of eigenvalues from linear analysis results.
    
    Args:
        data: Dictionary containing linear analysis results
        target_step_length: Array of target step lengths
        target_step_frequency: Array of target step frequencies
        plot: Whether to show the plot
    """
    # Extract eigenvalue data
    eig_vals_real = data['eig_real']
    eig_max = np.max(np.abs(eig_vals_real), axis=2)
    
    blue_colors = plt.cm.Blues(np.linspace(0.1, 0.9, 100))
    # Create the plot
    fig = plt.figure()
    
    # Plot real part
    cs1 = plt.contour(target_step_frequency, target_step_length, eig_max, levels=np.linspace(0, 1, 100),colors=blue_colors)
    plt.clabel(cs1, inline=True)
    plt.xlabel('Target Step Frequency')
    plt.ylabel('Target Step Length')
    plt.title('Maximum Eigenvalue')
    plt.tight_layout()
    if plot:
        plt.show()

def plot_gains(data, target_step_length, target_step_frequency, plot=False):
    """Plot contours of gain matrices from linear analysis results.
    
    Args:
        data: Dictionary containing linear analysis results
        target_step_length: Array of target step lengths
        target_step_frequency: Array of target step frequencies
        plot: Whether to show the plot
    """
    # Extract gain matrices
    k_pp = data['K_pp']
    k_dlqr = data['K_lqr']

    blue_colors = plt.cm.Blues(np.linspace(0.1, 0.9, 100))
    # Create the plot for PP gains
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(8, 8))
    fig1.suptitle('PP Gains')
    
    # Plot PP gains
    cs1 = ax1.contour(target_step_frequency, target_step_length, k_pp[:,:,0,0], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax1.clabel(cs1, inline=True)
    ax1.set_title('K11')
    ax1.set_aspect('equal')

    cs2 = ax2.contour(target_step_frequency, target_step_length, k_pp[:,:,0,1], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax2.clabel(cs2, inline=True)
    ax2.set_title('K12')
    ax2.set_aspect('equal')

    cs3 = ax3.contour(target_step_frequency, target_step_length, k_pp[:,:,1,0], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax3.clabel(cs3, inline=True)
    ax3.set_title('K21')
    ax3.set_aspect('equal')
    ax3.set_ylabel('Step Length')

    cs4 = ax4.contour(target_step_frequency, target_step_length, k_pp[:,:,1,1], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax4.clabel(cs4, inline=True)
    ax4.set_title('K22')
    ax4.set_aspect('equal')

    cs5 = ax5.contour(target_step_frequency, target_step_length, k_pp[:,:,2,0], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax5.clabel(cs5, inline=True)
    ax5.set_title('K31')
    ax5.set_aspect('equal')

    cs6 = ax6.contour(target_step_frequency, target_step_length, k_pp[:,:,2,1], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax6.clabel(cs6, inline=True)
    ax6.set_title('K32')
    ax6.set_aspect('equal')
    ax6.set_xlabel('Step Frequency')

    plt.tight_layout()
    if plot:
        plt.show()

    # Create the plot for LQR gains
    fig2, ((ax7, ax8), (ax9, ax10), (ax11, ax12)) = plt.subplots(3, 2, figsize=(8, 8))
    fig2.suptitle('LQR Gains')

    # Plot LQR gains
    cs7 = ax7.contour(target_step_frequency, target_step_length, k_dlqr[:,:,0,0],levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax7.clabel(cs7, inline=True)
    ax7.set_title('K11')
    ax7.set_aspect('equal')

    cs8 = ax8.contour(target_step_frequency, target_step_length, k_dlqr[:,:,0,1],levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax8.clabel(cs8, inline=True)
    ax8.set_title('K12')
    ax8.set_aspect('equal')

    cs9 = ax9.contour(target_step_frequency, target_step_length, k_dlqr[:,:,1,0],levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax9.clabel(cs9, inline=True)
    ax9.set_title('K21')
    ax9.set_aspect('equal')

    cs10 = ax10.contour(target_step_frequency, target_step_length, k_dlqr[:,:,1,1],levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax10.clabel(cs10, inline=True)
    ax10.set_title('K22')    
    ax10.set_aspect('equal')

    cs11 = ax11.contour(target_step_frequency, target_step_length, k_dlqr[:,:,2,0],levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax11.clabel(cs11, inline=True)
    ax11.set_title('K31')
    ax11.set_aspect('equal')

    cs12 = ax12.contour(target_step_frequency, target_step_length, k_dlqr[:,:,2,1],levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax12.clabel(cs12, inline=True)
    ax12.set_title('K32')
    ax12.set_aspect('equal')

    plt.tight_layout()
    if plot:
        plt.show()

def plot_AB(data, target_step_length, target_step_frequency, plot=False):
    """Plot contours of A and B matrices from linear analysis results.
    
    Args:
        data: Dictionary containing linear analysis results
        target_step_length: Array of target step lengths
        target_step_frequency: Array of target step frequencies
        plot: Whether to show the plot
    """
    # Extract state matrices
    a_matrix = data['A']
    b_matrix = data['B']

    blue_colors = plt.cm.Blues(np.linspace(0.1, 0.9, 100))

    # Create the plot
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
    fig1.suptitle('A Matrice')
    
    # Plot A matrix elements
    cs1 = ax1.contour(target_step_frequency, target_step_length, a_matrix[:,:,0,0], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax1.clabel(cs1, inline=True)
    ax1.set_title('A11')
    
    cs2 = ax2.contour(target_step_frequency, target_step_length, a_matrix[:,:,0,1], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax2.clabel(cs2, inline=True)
    ax2.set_title('A12')

    # Plot B matrix elements
    cs3 = ax3.contour(target_step_frequency, target_step_length, a_matrix[:,:,1,0], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax3.clabel(cs3, inline=True)
    ax3.set_xlabel('Target Step Length')
    ax3.set_ylabel('Target Step Frequency')
    ax3.set_title('A21')

    cs4 = ax4.contour(target_step_frequency, target_step_length, a_matrix[:,:,1,1], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax4.clabel(cs4, inline=True)
    ax4.set_xlabel('Target Step Length')
    ax4.set_ylabel('Target Step Frequency')
    ax4.set_title('A22')

    fig2, ((ax5, ax6, ax7), (ax8, ax9, ax10)) = plt.subplots(2, 3, figsize=(8, 12))
    fig2.suptitle('B Matrice')

    # Plot B matrix elements
    cs5 = ax5.contour(target_step_frequency, target_step_length, b_matrix[:,:,0,0], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax5.clabel(cs5, inline=True)
    ax5.set_title('B11')
    ax5.set_aspect('equal')

    cs6 = ax6.contour(target_step_frequency, target_step_length, b_matrix[:,:,0,1], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax6.clabel(cs6, inline=True)
    ax6.set_title('B12')
    ax6.set_aspect('equal')

    cs7 = ax7.contour(target_step_frequency, target_step_length, b_matrix[:,:,0,2], levels=np.linspace(-0.3, 2, 100),colors=blue_colors)
    ax7.clabel(cs7, inline=True)
    ax7.set_title('B13')
    ax7.set_aspect('equal')

    cs8 = ax8.contour(target_step_frequency, target_step_length, b_matrix[:,:,1,0], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax8.clabel(cs8, inline=True)
    ax8.set_title('B21')
    ax8.set_aspect('equal')

    cs9 = ax9.contour(target_step_frequency, target_step_length, b_matrix[:,:,1,1], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax9.clabel(cs9, inline=True)
    ax9.set_title('B22')    
    ax9.set_aspect('equal')

    cs10 = ax10.contour(target_step_frequency, target_step_length, b_matrix[:,:,1,2], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax10.clabel(cs10, inline=True)
    ax10.set_title('B23')
    ax10.set_aspect('equal')
    
    plt.tight_layout()
    if plot:
        plt.show()

def plot_CD(data, target_step_length, target_step_frequency, plot=False):
    """Plot contours of C and D matrices from linear analysis results.
    
    Args:
        data: Dictionary containing linear analysis results
        target_step_length: Array of target step lengths
        target_step_frequency: Array of target step frequencies
        plot: Whether to show the plot
    """
    # Extract output matrices
    c_matrix = data['C']
    d_matrix = data['D']

    blue_colors = plt.cm.Blues(np.linspace(0.1, 0.9, 100))
    # Create the plot
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
    fig1.suptitle('C Matrice')

    # Plot C matrix elements
    cs1 = ax1.contour(target_step_frequency, target_step_length, c_matrix[:,:,0,0], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax1.clabel(cs1, inline=True)
    ax1.set_title('C11')
    ax1.set_aspect('equal')
    ax1.set_xlabel('Step Frequency')
    ax1.set_ylabel('Step Length')

    cs2 = ax2.contour(target_step_frequency, target_step_length, c_matrix[:,:,0,1], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax2.clabel(cs2, inline=True)
    ax2.set_title('C12')
    ax2.set_aspect('equal')

    cs3 = ax3.contour(target_step_frequency, target_step_length, c_matrix[:,:,1,0], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax3.clabel(cs3, inline=True)
    ax3.set_title('C21')
    ax3.set_aspect('equal')

    cs4 = ax4.contour(target_step_frequency, target_step_length, c_matrix[:,:,1,1], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax4.clabel(cs4, inline=True)
    ax4.set_title('C22')
    ax4.set_aspect('equal')

    # Plot D matrix elements
    fig2, ((ax5, ax6, ax7), (ax8, ax9, ax10)) = plt.subplots(2, 3, figsize=(8, 8))
    fig2.suptitle('D Matrice')

    cs5 = ax5.contour(target_step_frequency, target_step_length, d_matrix[:,:,0,0], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax5.clabel(cs5, inline=True)
    ax5.set_xlabel('Step Frequency')
    ax5.set_ylabel('Step Length')
    ax5.set_title('D11')
    ax5.set_aspect('equal')

    cs6 = ax6.contour(target_step_frequency, target_step_length, d_matrix[:,:,0,1], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax6.clabel(cs6, inline=True)
    ax6.set_xlabel('Step Frequency')
    ax6.set_ylabel('Step Length')
    ax6.set_title('D12')
    ax6.set_aspect('equal')

    cs7 = ax7.contour(target_step_frequency, target_step_length, d_matrix[:,:,0,2], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax7.clabel(cs7, inline=True)
    ax7.set_title('D13')
    ax7.set_aspect('equal')

    cs8 = ax8.contour(target_step_frequency, target_step_length, d_matrix[:,:,1,0], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax8.clabel(cs8, inline=True)
    ax8.set_title('D21')
    ax8.set_aspect('equal')

    cs9 = ax9.contour(target_step_frequency, target_step_length, d_matrix[:,:,1,1], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax9.clabel(cs9, inline=True)
    ax9.set_title('D22')
    ax9.set_aspect('equal')

    cs10 = ax10.contour(target_step_frequency, target_step_length, d_matrix[:,:,1,2], levels=np.linspace(-100, 1, 100),colors=blue_colors)
    ax10.clabel(cs10, inline=True)
    ax10.set_title('D23')
    ax10.set_aspect('equal')

    plt.tight_layout()
    if plot:
        plt.show()

def plot_states(data, target_step_length, target_step_frequency, plot=False):
    """Plot contours of state variables from limit cycle solutions.
    
    Args:
        data: Dictionary containing limit cycle solutions
        target_step_length: Array of target step lengths
        target_step_frequency: Array of target step frequencies
        plot: Whether to show the plot
    """
    # Extract state variables (first two columns)
    # Extract state variables from each array in the sequence
    theta = np.zeros((101, 101))
    theta_dot = np.zeros((101, 101))
    for i in range(len(data['arr_0'][0])):
        theta[:,i] = data['arr_0'][0][i][:,0]
        theta_dot[:,i] = data['arr_0'][0][i][:,1]
    
    blue_colors = plt.cm.Blues(np.linspace(0.1, 0.9, 100))
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
    
    min_theta = np.min(theta)
    max_theta = np.max(theta)
    min_theta_dot = np.min(theta_dot)
    max_theta_dot = np.max(theta_dot)
    # Plot first state variable
    cs1 = ax1.contour(target_step_frequency, target_step_length, theta, levels=np.linspace(min_theta, max_theta, 100), colors=blue_colors)
    ax1.clabel(cs1, inline=True)
    ax1.set_xlabel('Step Frequency')
    ax1.set_ylabel('Step Length')
    ax1.set_title('theta')
    ax1.set_aspect('equal')
    
    # Plot second state variable
    cs2 = ax2.contour(target_step_frequency, target_step_length, theta_dot, levels=np.linspace(min_theta_dot, max_theta_dot, 100), colors=blue_colors)
    ax2.clabel(cs2, inline=True)
    ax2.set_xlabel('Step Frequency')
    ax2.set_ylabel('Step Length')
    ax2.set_title('theta dot')
    ax2.set_aspect('equal')

    plt.tight_layout()
    if plot:
        plt.show()

def plot_controls(data, target_step_length, target_step_frequency, plot=False):
    """Plot contours of control variables from limit cycle solutions.
    
    Args:
        data: Dictionary containing limit cycle solutions
        target_step_length: Array of target step lengths
        target_step_frequency: Array of target step frequencies
        plot: Whether to show the plot
    """
    # Extract control variables (third and fourth columns)
    push_off = np.zeros((101, 101))
    spring_stiffness = np.zeros((101, 101))
    for i in range(len(data['arr_0'][0])):
        push_off[:,i] = data['arr_0'][0][i][:,2]
        spring_stiffness[:,i] = data['arr_0'][0][i][:,3]
    
    blue_colors = plt.cm.Blues(np.linspace(0.1, 0.9, 100))
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
    
    min_push_off = np.min(push_off)
    max_push_off = np.max(push_off)
    min_spring_stiffness = np.min(spring_stiffness)
    max_spring_stiffness = np.max(spring_stiffness)
    # Plot push off control
    cs1 = ax1.contour(target_step_frequency, target_step_length, push_off, levels=np.linspace(min_push_off, max_push_off, 100), colors=blue_colors)
    ax1.clabel(cs1, inline=True)
    ax1.set_xlabel('Step Frequency')
    ax1.set_ylabel('Step Length')
    ax1.set_title('Push Off')
    ax1.set_aspect('equal')
    
    # Plot spring stiffness control
    cs2 = ax2.contour(target_step_frequency, target_step_length, spring_stiffness, levels=np.linspace(min_spring_stiffness, max_spring_stiffness, 100), colors=blue_colors)
    ax2.clabel(cs2, inline=True)
    ax2.set_xlabel('Step Frequency')
    ax2.set_ylabel('Step Length')
    ax2.set_title('Spring Stiffness')
    ax2.set_aspect('equal')

    plt.tight_layout()
    if plot:
        plt.show()

if __name__ == "__main__":
    # Only call the function we want to plot
    plot_controls(plot=True)
    # Uncomment other functions as needed
    # plot_eigenvalues(plot=False)
    # plot_gains(plot=False)
    # plot_AB(plot=False)
    # plot_CD(plot=False)
    # plot_states(plot=False)
