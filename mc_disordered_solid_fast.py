import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar
from numba import jit  # For just-in-time compilation
from CalculateScattering import CrystalStructure, Grid
from numpy import real, imag, angle, amax

#-------- Initialize Lattice -----
def initialize_lattice(size=(50, 50)):
    """
    Initialize a 2D disordered solid with Cu (+1) and Au (-1) atoms.
    
    Parameters:
    -----------
    size : tuple
        Size of the 2D grid (height, width)
    
    Returns:
    --------
    lattice : numpy.ndarray
        2D array with +1 (Cu) and -1 (Au) values
    """
    # Create a 50/50 mixture of Cu (+1) and Au (-1) atoms
    num_sites = size[0] * size[1]
    half_sites = num_sites // 2
    
    # Create array with exactly 50% +1 and 50% -1
    flat_lattice = np.ones(num_sites, dtype=np.int8)
    flat_lattice[half_sites:] = -1
    
    # Shuffle the array and reshape to 2D
    np.random.shuffle(flat_lattice)
    return flat_lattice.reshape(size)

#-------- Calculate Energy -----
@jit(nopython=True)
def calculate_energy_numba(lattice, interactions_array):
    """
    Calculate the total energy of the system based on Ising interactions.
    Optimized with Numba for performance.
    """
    energy = 0.0
    height, width = lattice.shape
    
    for i in range(height):
        for j in range(width):
            spin = lattice[i, j]
            
            # Calculate interaction energy with neighbors
            for k in range(len(interactions_array)):
                # Convert to int for indexing
                dx = int(interactions_array[k, 0])
                dy = int(interactions_array[k, 1])
                J = interactions_array[k, 2]
                
                # Apply periodic boundary conditions
                ni, nj = (i + dy) % height, (j + dx) % width
                neighbor_spin = lattice[ni, nj]
                
                # Ising interaction: -J*s_i*s_j
                energy -= J * spin * neighbor_spin
                
    # We've counted each interaction twice
    return energy / 2.0

def calculate_energy(lattice, interactions):
    """
    Wrapper for the Numba-optimized energy calculation function.
    
    Parameters:
    -----------
    lattice : numpy.ndarray
        2D array with +1 (Cu) and -1 (Au) values
    interactions : list
        List of interactions in the form [(dx, dy, J), ...]
    
    Returns:
    --------
    energy : float
        Total energy of the system
    """
    # Convert interactions list to numpy array with correct data types
    interactions_array = np.array(interactions, dtype=np.float64)
    return calculate_energy_numba(lattice, interactions_array)

@jit(nopython=True)
def calculate_swap_energy_change(lattice, i1, j1, i2, j2, interactions_array, height, width):
    """
    Calculate energy change for swapping two sites.
    Optimized with Numba for performance.
    """
    # Create temporary lattice with the swap applied
    temp_lattice = lattice.copy()
    temp_spin1 = temp_lattice[i1, j1]
    temp_lattice[i1, j1] = temp_lattice[i2, j2]
    temp_lattice[i2, j2] = temp_spin1
    
    energy_diff = 0.0
    
    # Calculate energy change for site 1 and its neighbors
    for k in range(len(interactions_array)):
        dx = int(interactions_array[k, 0])
        dy = int(interactions_array[k, 1])
        J = interactions_array[k, 2]
        
        ni1 = (i1 + dy) % height
        nj1 = (j1 + dx) % width
        
        # Skip if the neighbor is site 2 (we'll handle this separately)
        if ni1 == i2 and nj1 == j2:
            continue
            
        # Calculate energy difference
        old_contrib = -J * lattice[i1, j1] * lattice[ni1, nj1]
        new_contrib = -J * temp_lattice[i1, j1] * lattice[ni1, nj1]
        energy_diff += new_contrib - old_contrib
    
    # Calculate energy change for site 2 and its neighbors
    for k in range(len(interactions_array)):
        dx = int(interactions_array[k, 0])
        dy = int(interactions_array[k, 1])
        J = interactions_array[k, 2]
        
        ni2 = (i2 + dy) % height
        nj2 = (j2 + dx) % width
        
        # Skip if the neighbor is site 1 (would be double-counting)
        if ni2 == i1 and nj2 == j1:
            continue
            
        # Calculate energy difference
        old_contrib = -J * lattice[i2, j2] * lattice[ni2, nj2]
        new_contrib = -J * temp_lattice[i2, j2] * lattice[ni2, nj2]
        energy_diff += new_contrib - old_contrib
    
    # Handle direct interaction between sites 1 and 2 if they are neighbors
    for k in range(len(interactions_array)):
        dx = int(interactions_array[k, 0])
        dy = int(interactions_array[k, 1])
        J = interactions_array[k, 2]
        
        # Check if site 2 is a neighbor of site 1 through this interaction
        if ((i1 + dy) % height == i2 and (j1 + dx) % width == j2) or \
           ((i2 + dy) % height == i1 and (j2 + dx) % width == j1):
            # The direct interaction energy doesn't change when swapping
            # because s1*s2 = s2*s1, so we don't need to add anything
            pass
    
    return energy_diff

#-------- Monte Carlo Simulation -----
@jit(nopython=True)
def run_mc_until_accepted(lattice, interactions_array, n_accepted, temperature, height, width):
    """
    Run Monte Carlo until reaching exactly N accepted moves.
    
    Parameters:
    -----------
    lattice : numpy.ndarray
        2D array with +1 (Cu) and -1 (Au) values
    interactions_array : numpy.ndarray
        Array of interaction parameters
    n_accepted : int
        Number of accepted moves to reach
    temperature : float
        Temperature for Metropolis acceptance
    height, width : int
        Dimensions of the lattice
        
    Returns:
    --------
    lattice : numpy.ndarray
        Updated lattice configuration
    accepted_good : int
        Number of accepted energy-lowering moves
    accepted_bad : int
        Number of accepted energy-raising moves
    rejected : int
        Number of rejected moves
    """
    accepted_good = 0
    accepted_bad = 0
    rejected = 0
    total_accepted = 0
    
    while total_accepted < n_accepted:
        # Find two sites with different occupancies
        while True:
            i1, j1 = np.random.randint(0, height), np.random.randint(0, width)
            i2, j2 = np.random.randint(0, height), np.random.randint(0, width)
            
            if lattice[i1, j1] != lattice[i2, j2]:
                break
        
        # Calculate energy change
        delta_energy = calculate_swap_energy_change(
            lattice, i1, j1, i2, j2, interactions_array, height, width
        )
        
        # Metropolis acceptance criterion
        if delta_energy <= 0:
            # Accept the move: swap the spins
            tmp = lattice[i1, j1]
            lattice[i1, j1] = lattice[i2, j2]
            lattice[i2, j2] = tmp
            accepted_good += 1
            total_accepted += 1
        elif np.random.random() < np.exp(-delta_energy/temperature):
            # Accept the move despite energy increase
            tmp = lattice[i1, j1]
            lattice[i1, j1] = lattice[i2, j2]
            lattice[i2, j2] = tmp
            accepted_bad += 1
            total_accepted += 1
        else:
            # Reject the move
            rejected += 1
    
    return lattice, accepted_good, accepted_bad, rejected

def run_monte_carlo(lattice, interactions, total_accepted_moves, equilibration_steps, temperature, save_every):
    """
    Run Monte Carlo simulation with swap moves.
    
    Parameters:
    -----------
    lattice : numpy.ndarray
        2D array with +1 (Cu) and -1 (Au) values
    interactions : list
        List of interactions in the form [(dx, dy, J), ...]
    total_accepted_moves : int
        Total number of accepted moves after equilibration
    equilibration_steps : int
        Number of accepted moves to perform before recording any configurations
    temperature : float
        Temperature for Metropolis acceptance
    save_every : int, optional
        Save configuration every save_every accepted moves
        
    Returns:
    --------
    history : list
        List of lattice configurations at saved steps
    final_lattice : numpy.ndarray
        Final configuration of the lattice
    stats : dict
        Statistics of the simulation (good/bad/rejected moves)
    """
    height, width = lattice.shape
    
    # Make a copy of the input lattice to avoid modifying the original
    current_lattice = lattice.copy()
    
    # Convert interactions list to numpy array for Numba
    interactions_array = np.array(interactions, dtype=np.float64)
    
    # Initialize history
    history = []
    
    # Initialize counters
    accepted_good = 0
    accepted_bad = 0
    rejected_moves = 0
    
    # Run equilibration
    if equilibration_steps > 0:
        print(f"Running {equilibration_steps} equilibration steps...")
        current_lattice, eq_good, eq_bad, eq_rejected = run_mc_until_accepted(
            current_lattice, interactions_array, equilibration_steps, 
            temperature, height, width
        )
        
        # Update statistics
        accepted_good += eq_good
        accepted_bad += eq_bad
        rejected_moves += eq_rejected
        
        print(f"Equilibration complete.")
    
    # Save initial (post-equilibration) configuration
    if save_every is not None:
        history.append(current_lattice.copy())
    
    # Number of saves we'll make
    if save_every is not None:
        num_saves = total_accepted_moves // save_every
        print(f"Will save {num_saves} configurations during production run")
    else:
        num_saves = 0
    
    # Run production with progress bar
    print(f"Running production: {total_accepted_moves} accepted moves...")
    remaining = total_accepted_moves
    
    with tqdm(total=total_accepted_moves) as pbar:
        while remaining > 0:
            # Determine how many moves to accept before next save
            moves_to_accept = min(remaining, save_every if save_every is not None else remaining)
            
            # Run Monte Carlo until exactly moves_to_accept moves are accepted
            current_lattice, good, bad, rej = run_mc_until_accepted(
                current_lattice, interactions_array, moves_to_accept, 
                temperature, height, width
            )
            
            # Update statistics
            accepted_good += good
            accepted_bad += bad
            rejected_moves += rej
            remaining -= moves_to_accept
            
            # Update progress bar
            pbar.update(moves_to_accept)
            
            # Save configuration if needed
            if save_every is not None and moves_to_accept == save_every:
                history.append(current_lattice.copy())
    
    # Calculate total moves
    total_moves = accepted_good + accepted_bad + rejected_moves
    
    # Calculate statistics
    stats = {
        "accepted_good": accepted_good,
        "accepted_bad": accepted_bad,
        "rejected": rejected_moves,
        "total_moves": total_moves,
        "accepted_ratio": (accepted_good + accepted_bad) / total_moves
    }
    
    print(f"Move statistics:")
    print(f"  Accepted (energy lowering): {accepted_good}/{total_moves} ({100*accepted_good/total_moves:.2f}%)")
    print(f"  Accepted (energy raising): {accepted_bad}/{total_moves} ({100*accepted_bad/total_moves:.2f}%)")
    print(f"  Rejected: {rejected_moves}/{total_moves} ({100*rejected_moves/total_moves:.2f}%)")
    print(f"  Total acceptance ratio: {100*(accepted_good+accepted_bad)/total_moves:.2f}%")
    
    return history, current_lattice, stats

#-------- Visualization -----

def visualize_lattice(lattice):
    """
    Visualize the lattice configuration.
    
    Parameters:
    -----------
    lattice : numpy.ndarray
        2D array with +1 (Cu) and -1 (Au) values
    """
    plt.figure(figsize=(8, 8))
    # Use a colormap: blue for Cu (+1), red for Au (-1)
    plt.imshow(lattice, cmap='bwr', vmin=-1, vmax=1)
    plt.colorbar(ticks=[-1, 1], label='Atom Type (Cu: +1, Au: -1)')
    plt.title("Disordered Solid Configuration")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def visualize_evolution(history, num_frames=4):
    """
    Visualize the evolution of the system over time.
    
    Parameters:
    -----------
    history : list
        List of lattice configurations at saved steps
    num_frames : int
        Number of frames to show from the history
    """
    if not history:
        print("No history available.")
        return
    
    frames = np.linspace(0, len(history)-1, num_frames, dtype=int)
    
    fig, axes = plt.subplots(1, num_frames, figsize=(16, 4))
    for i, step in enumerate(frames):
        ax = axes[i]
        im = ax.imshow(history[step], cmap='bwr', vmin=-1, vmax=1)
        ax.set_title(f"Step {step}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.colorbar(im, ax=axes, ticks=[-1, 1], label='Atom Type (Cu: +1, Au: -1)')
    plt.tight_layout()
    plt.show()

def convert_lattice_to_crystal_structure(lattice, a=3):
    """
    Convert a 2D lattice of +1 (Cu) and -1 (Au) into a 3D CrystalStructure
    
    Parameters:
    -----------
    lattice: 2D numpy array
        Array of +1 (Cu) and -1 (Au) values
    a: float
        Lattice parameter in Å
    
    Returns:
    --------
    CrystalStructure
        3D crystal structure for calculating scattering
    """
    height, width = lattice.shape
    
    # Create atoms list (element_type, x, y, z)
    atoms = []
    for x in range(height):
        for y in range(width):
            # Convert +1 to Cu, -1 to Au
            element = "Cu" if lattice[x, y] == 1 else "Au"
            atoms.append((element, x, y, 0))
    
    # Create crystal structure with a simple cubic unit cell
    # We set the supercell to the lattice dimensions 
    crystal = CrystalStructure(
        cell_parameters=(a, a, a, 90, 90, 90),  # Simple cubic cell
        atoms=atoms,
        supercell=(width, height, 1)  # 2D structure with thickness of 1
    )
    
    return crystal

def calculate_diffuse_scattering(lattice, grid, z_layer=1, blur=0.01):
    """
    Calculate diffuse scattering from a 2D lattice
    
    Parameters:
    -----------
    lattice: 2D numpy array
        Array of +1 (Cu) and -1 (Au) values
    grid: Grid object
        Grid for scattering calculation
    z_layer: int
        Which z-layer to extract for 2D visualization
    blur: float
        Blur parameter for scattering calculation
        
    Returns:
    --------
    tuple
        (crystal, sf, intensities) - crystal structure, structure factor, intensities
    """
    # Convert lattice to crystal structure
    crystal = convert_lattice_to_crystal_structure(lattice)
    
    # Calculate scattering
    sf = crystal.calculate_scattering(grid, blur=blur)
    
    # Get intensities
    intensities = sf.get_intensities()[:,:,z_layer]
    
    # Scale them
    intensities = intensities/amax(intensities)
    
    return crystal, sf, intensities

def calculate_average_diffuse_scattering(history_lattices, grid, z_layer=1, blur=0.01):
    """
    Calculate and average diffuse scattering from a history of lattice configurations
    
    Parameters:
    -----------
    history_lattices: list of 2D numpy arrays
        List of lattice configurations to average
    grid: Grid object
        Grid for scattering calculation
    z_layer: int
        Which z-layer to extract for 2D visualization
    blur: float
        Blur parameter for scattering calculation
        
    Returns:
    --------
    avg_intensities: numpy array
        Average intensities
    """
    if not history_lattices:
        print("No history provided to average.")
        return None
    
    print(f"Calculating average diffuse scattering from {len(history_lattices)} configurations...")
    
    # Get the first calculation to set up the grid and dimensions
    _, first_sf, first_intensities = calculate_diffuse_scattering(
        history_lattices[0], grid, z_layer, blur
    )
    
    # Initialize average intensities array with the first calculation
    avg_intensities = first_intensities.copy()
    
    # Process remaining lattices in history
    for i, lattice in enumerate(tqdm(history_lattices[1:], desc="Processing configurations")):
        _, sf, intensities = calculate_diffuse_scattering(
            lattice, grid, z_layer, blur)
        
        # Accumulate intensities
        avg_intensities += intensities
    
    # Divide by number of configurations to get average
    avg_intensities /= len(history_lattices)
    
    # Scale the final result
    avg_intensities /= amax(avg_intensities)
    
    return avg_intensities