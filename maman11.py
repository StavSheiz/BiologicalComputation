import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, colors

# --- CONFIGURATION ---

GRID_SIZE = 100
DAYS_IN_YEAR = 365
# Increased interval slightly to 200ms (5 frames/sec) to ensure smoother rendering.
FRAME_INTERVAL_MS = 200

# Constants
QUIVER_STEP = 10  # Not used in this version, but kept for reference if wind is re-added
X_Q, Y_Q = np.mgrid[0:GRID_SIZE:QUIVER_STEP, 0:GRID_SIZE:QUIVER_STEP]

# Land Type Definitions
LAND_TYPES = {1: 'Sea', 2: 'Land', 3: 'Forest', 4: 'City', 5: 'Glacier'}
LAND_CODES = {v: k for k, v in LAND_TYPES.items()}
# Wind Directions: 1:N(-1, 0), 2:S(1, 0), 3:E(0, 1), 4:W(0, -1), 0:No Wind(0, 0)
WIND_VECTORS = {1: (-1, 0), 2: (1, 0), 3: (0, 1), 4: (0, -1), 0: (0, 0)}

# Matplotlib Color mapping for Land Types (Categorical Map)
LAND_BOUNDS = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
LAND_COLORS = ['#4682b4', '#cd853f', '#228b22', '#b22222', '#ffffff']  # Sea, Land, Forest, City, Glacier
LAND_CMAP = colors.ListedColormap(LAND_COLORS)
LAND_NORM = colors.BoundaryNorm(LAND_BOUNDS, LAND_CMAP.N)

# Cloud Color mapping (0: None, 1: Cloud, 2: Rain Cloud)
CLOUD_BOUNDS = [-0.5, 0.5, 1.5, 2.5]
CLOUD_COLORS = ['#ffffff', '#D3D3D3', '#778899']  # Transparent, LightGray, SlateGray
CLOUD_CMAP = colors.ListedColormap(CLOUD_COLORS)
CLOUD_NORM = colors.BoundaryNorm(CLOUD_BOUNDS, CLOUD_CMAP.N)
CLOUD_TICKS = [0, 1, 2]
CLOUD_LABELS = ['No Cloud', 'Cloud', 'Rain Cloud']

# Simulation Parameters
INITIAL_PARAMS = {
    'Glacier_Start_T': 5, 'City_Pollution_Start': 30, 'Global_T_Base': 15,
    'Glacier_T_Threshold': 30,  # RULE 1
    'Pollution_T_Threshold': 60,  # RULE 2
    'Pollution_Spread_T': 20,  # RULE 3
}

# --- GLOBAL STATE (Initialized by setup_plot) ---
current_grid = None
current_day = 0
current_artists = {}


# --- UTILITY FUNCTIONS ---

def initialize_grid(size, params):
    """Initializes the multi-layered grid for the CA state using NumPy."""
    L = np.zeros((size, size), dtype=np.int8)
    T = np.full((size, size), params['Global_T_Base'], dtype=np.int8)
    H = np.random.choice(np.arange(-500, 3100, 100), size=(size, size)).astype(np.int16)
    P = np.zeros((size, size), dtype=np.int8)
    C = np.zeros((size, size), dtype=np.int8)
    W_dir = np.random.randint(1, 5, size=(size, size), dtype=np.int8)
    W_int = np.random.randint(1, 11, size=(size, size), dtype=np.int8)

    L[:] = LAND_CODES['Land']
    sea_mask = (H < -100) | (np.random.rand(size, size) < 0.2)
    L[sea_mask] = LAND_CODES['Sea']

    glacier_mask = (H > 2000) & (L != LAND_CODES['Sea'])
    L[glacier_mask] = LAND_CODES['Glacier']
    T[glacier_mask] = params['Glacier_Start_T']

    land_mask = (L == LAND_CODES['Land'])
    city_mask = land_mask & (np.random.rand(size, size) < 0.005)
    L[city_mask] = LAND_CODES['City']
    P[city_mask] = params['City_Pollution_Start']

    forest_mask = land_mask & (~city_mask) & (np.random.rand(size, size) < 0.5)
    L[forest_mask] = LAND_CODES['Forest']

    C[np.random.rand(size, size) < 0.05] = 1
    C[np.random.rand(size, size) < 0.05] = 2

    T = np.clip(T, -20, 50)
    P = np.clip(P, 0, 100)

    return {'L': L, 'T': T, 'H': H, 'P': P, 'C': C, 'W_dir': W_dir, 'W_int': W_int}


def simulate_step(current_grid, params):
    """Executes one day of simulation using the user's strict CA rule order."""
    R, C = current_grid['L'].shape

    # next_grid starts as a copy of current_grid for all layers
    next_grid = {key: arr.copy() for key, arr in current_grid.items()}

    # P_out and C_out are used to collect incoming values (R3, R4)
    P_out = next_grid['P'].copy()
    C_out = np.zeros((R, C), dtype=np.int8)

    # --- Step 1: LOCAL STATE CHANGES (Rules 1, 2, 6, 8, 9) ---
    # Rule 11 (High Altitude Rain) removed here.

    # R1: Melting Glaciers: If T > 30 and L = Glacier -> L = Land
    melt_mask = (current_grid['T'] > params['Glacier_T_Threshold']) & (current_grid['L'] == LAND_CODES['Glacier'])
    next_grid['L'][melt_mask] = LAND_CODES['Land']

    # R2: Pollution Increases Temperature: If P > 60 -> T = T + 1
    temp_up_mask = current_grid['P'] > params['Pollution_T_Threshold']
    next_grid['T'][temp_up_mask] = np.minimum(50, next_grid['T'][temp_up_mask] + 1)

    # R6: Rain Cools: If C = 2 -> T = T - 1
    rain_cool_mask = current_grid['C'] == 2
    next_grid['T'][rain_cool_mask] = np.maximum(-20, next_grid['T'][rain_cool_mask] - 1)

    # R8: City Generates Pollution: If L = 4 -> P = P + 1
    city_mask = current_grid['L'] == LAND_CODES['City']
    next_grid['P'][city_mask] = np.minimum(100, next_grid['P'][city_mask] + 1)

    # R9: Forest Decreases Pollution: If L = 3 -> P = P - 1
    forest_mask = current_grid['L'] == LAND_CODES['Forest']
    next_grid['P'][forest_mask] = np.maximum(0, next_grid['P'][forest_mask] - 1)

    # R11 logic removed: High Altitude Makes Cloud Rainy: If C = 1 and H > 2000 -> C = 2

    # --- Step 2: SPATIAL PROPAGATION (Rules 3, 4 - Backward Check) ---

    # Mask used for R5: Tracks cells that are sources for wind-driven movement.
    wind_cleared_cloud_mask = (current_grid['C'] >= 1) & (current_grid['W_int'] > 0)

    for r in range(R):
        for c in range(C):

            for direction in range(1, 5):
                # Calculate the neighbor coordinates (r_n, c_n) which is the source cell
                dr, dc = WIND_VECTORS[direction]
                r_n, c_n = r + dr, c + dc
                r_n, c_n = r_n % R, c_n % C

                # If the neighbor's wind points back to the current cell (r, c)
                opposite_dir = (direction % 4) + 1

                neighbor_wind_dir = current_grid['W_dir'][r_n, c_n]
                neighbor_wind_int = current_grid['W_int'][r_n, c_n]
                neighbor_P = current_grid['P'][r_n, c_n]
                neighbor_C = current_grid['C'][r_n, c_n]

                if neighbor_wind_dir == opposite_dir:

                    # R3: Wind Spreads Pollution
                    if neighbor_wind_int > 0 and neighbor_P > params['Pollution_Spread_T']:
                        # The user's rule update: pollution_transfer = round(neighbor_P * 0.1)
                        pollution_transfer = round(neighbor_P * 0.05)
                        # R3 adds pollution. P_out starts with R8/R9 changes.
                        P_out[r, c] = np.minimum(100, P_out[r, c] + pollution_transfer)

                    # R4: Wind Moves Cloud
                    if neighbor_wind_int > 0 and neighbor_C in [1, 2]:
                        # C.C = neighbour.C. Take the highest cloud state.
                        C_out[r, c] = np.maximum(C_out[r, c], neighbor_C)

    # --- Step 3: MERGE, R5, R7 ---
    # Rule 10 (Cloud After Rain) removed here.

    # R3 Merge: P_out now contains R8/R9 changes + incoming R3.
    next_grid['P'] = P_out

    # R4/R11 Merge: Cloud state is determined solely by R4 (incoming).
    # Since R11 is gone, next_grid['C'] should be zero or initial C values,
    # but we initialize it with R4 (C_out) to be safe.
    next_grid['C'] = C_out

    # R5: Cloud Moved (Source Clearing: C.C = 0, C.W = 0)
    next_grid['W_dir'][wind_cleared_cloud_mask] = 0
    next_grid['W_int'][wind_cleared_cloud_mask] = 0

    # R10 logic removed: Cloud After Rain: If C.C = 2 -> C.C = 1

    # R7: Random Wind for Next Day (Runs last, overriding R5 C.W=0 as per sequence)
    next_grid['W_dir'][:] = np.random.randint(1, 5, size=(R, C))
    next_grid['W_int'][:] = np.random.randint(1, 11, size=(R, C))

    # Final Clipping (Essential for array integrity)
    next_grid['T'] = np.clip(next_grid['T'], -20, 50)
    next_grid['P'] = np.clip(next_grid['P'], 0, 100)
    next_grid['C'] = np.clip(next_grid['C'], 0, 2)

    return next_grid


# --- MATPLOTLIB ANIMATION (Updated for White Background Visibility) ---

def setup_plot():
    """Initializes the Matplotlib figure, subplots, and colorbars."""
    global current_grid, current_artists

    current_grid = initialize_grid(GRID_SIZE, INITIAL_PARAMS)

    # Setup Figure and Subplots (2 rows, 2 columns)
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()  # Flatten for easier indexing (0, 1, 2, 3)
    # REMOVED: plt.style.use('dark_background') - Now uses default light theme

    # Subplot 1 (Top Left): Land Type
    ax1 = axes[0]
    ax1.set_title('1. Land Type (Categorical)')
    img1 = ax1.imshow(current_grid['L'], cmap=LAND_CMAP, norm=LAND_NORM, interpolation='none')
    ax1.set_xticks([]);
    ax1.set_yticks([])
    # Increased pad to push the colorbar ticks away from the plot
    cbar1 = plt.colorbar(img1, ax=ax1, fraction=0.046, pad=0.08, ticks=np.array(LAND_BOUNDS[:-1]) + 0.5)
    # Removed explicit color settings; text defaults to dark
    cbar1.ax.set_yticklabels(LAND_TYPES.values(), **{'fontsize': 10})
    # Removed explicit color setting
    cbar1.set_label('Land Classification', rotation=270, labelpad=40, fontsize=12)

    # Subplot 2 (Top Right): Temperature
    ax2 = axes[1]
    ax2.set_title('2. Temperature ($\degree C$)')
    img2 = ax2.imshow(current_grid['T'], cmap='coolwarm', vmin=-20, vmax=50, interpolation='none')
    ax2.set_xticks([]);
    ax2.set_yticks([])
    # Set ticks to show only min/max values [-20, 50]
    cbar2 = plt.colorbar(img2, ax=ax2, ticks=[-20, 50])
    # Removed explicit color setting
    cbar2.set_label('Temperature ($\degree C$)', rotation=270, labelpad=15, fontsize=12)
    # Removed explicit color setting
    cbar2.ax.tick_params(labelsize=10)

    # Subplot 3 (Bottom Left): Pollution (No Cloud Overlay)
    ax3 = axes[2]
    ax3.set_title('3. Pollution (%)')
    img3_P = ax3.imshow(current_grid['P'], cmap='inferno', vmin=0, vmax=100, interpolation='none')
    ax3.set_xticks([]);
    ax3.set_yticks([])
    # Set ticks to show only min/max values [0, 100]
    cbar3 = plt.colorbar(img3_P, ax=ax3, ticks=[0, 100])
    # Removed explicit color setting
    cbar3.set_label('Pollution (%)', rotation=270, labelpad=15, fontsize=12)
    # Removed explicit color setting
    cbar3.ax.tick_params(labelsize=10)

    # Subplot 4 (Bottom Right): Clouds (Dedicated Plot)
    ax4 = axes[3]
    ax4.set_title('4. Cloud/Rain State')
    # Use the defined Cloud color map
    img4_C = ax4.imshow(current_grid['C'], cmap=CLOUD_CMAP, norm=CLOUD_NORM, interpolation='none')
    ax4.set_xticks([]);
    ax4.set_yticks([])
    # Increased pad to push the colorbar ticks away from the plot
    cbar4 = plt.colorbar(img4_C, ax=ax4, fraction=0.046, pad=0.08, ticks=CLOUD_TICKS)
    # Removed explicit color setting; text defaults to dark
    cbar4.ax.set_yticklabels(CLOUD_LABELS, **{'fontsize': 10})
    # Removed explicit color setting
    cbar4.set_label('Cloud State Key', rotation=270, labelpad=40, fontsize=12)

    current_artists = {'L': img1, 'T': img2, 'P': img3_P, 'C': img4_C}
    # Removed explicit color setting
    fig.suptitle('Climate CA Simulation (Day: 0 | Avg T: -- | Avg P: -- | Ice Area: --)',
                 fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjusted layout for suptitle space
    return fig


def update_plot(frame_number):
    """Called by FuncAnimation to advance the simulation and update the plot."""
    global current_grid, current_day

    if current_day < DAYS_IN_YEAR:
        current_grid = simulate_step(current_grid, INITIAL_PARAMS)
        current_day += 1

    # Get global metrics
    R, C = current_grid['L'].shape
    total_cells = R * C
    avg_t = np.mean(current_grid['T'])
    avg_p = np.mean(current_grid['P'])
    ice_area = np.sum(current_grid['L'] == LAND_CODES['Glacier']) / total_cells * 100

    # --- Update Artists ---
    current_artists['L'].set_data(current_grid['L'])
    current_artists['T'].set_data(current_grid['T'])
    current_artists['P'].set_data(current_grid['P'])  # Pollution plot update
    current_artists['C'].set_data(current_grid['C'])  # Cloud plot update

    # Update dynamic title
    # Removed explicit color setting
    plt.suptitle(
        f"Climate CA Simulation (Day: {current_day} of {DAYS_IN_YEAR} | Avg T: {avg_t:.1f}°C | Avg P: {avg_p:.1f}% | Ice Area: {ice_area:.2f}%)",
        fontsize=16)

    # We must return all artists that were updated.
    return list(current_artists.values())


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    fig = setup_plot()

    print(f"Starting Matplotlib animation for {DAYS_IN_YEAR} days...")

    anim = animation.FuncAnimation(
        fig,
        update_plot,
        frames=DAYS_IN_YEAR,
        interval=FRAME_INTERVAL_MS,
        blit=False,
        repeat=False
    )

    plt.show()

    print("Simulation complete or window closed.")
