import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, colors

# --- CONFIGURATION ---

GRID_SIZE = 100
DAYS_IN_YEAR = 365
# For animating the panels
FRAME_INTERVAL_MS = 200

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
# History of global metrics over the year
temp_history = []
pollution_history = []
ice_area_history = []
# Global figure title artist for reliable access in update_plot
fig_subtitle = None
initial_grid_state = None


# --- UTILITY FUNCTIONS ---

def initialize_grid(size, params):
    """Initializes the multi-layered grid for the state using NumPy."""
    L = np.zeros((size, size), dtype=np.int8) # Land type
    T = np.full((size, size), params['Global_T_Base'], dtype=np.int8) # Temperature
    H = np.random.choice(np.arange(-500, 3100, 100), size=(size, size)).astype(np.int16) # Height
    P = np.zeros((size, size), dtype=np.int8) # Pollution
    C = np.zeros((size, size), dtype=np.int8)  # Clouds
    W_dir = np.random.randint(1, 5, size=(size, size), dtype=np.int8) # Wind direction
    W_int = np.random.randint(1, 11, size=(size, size), dtype=np.int8) # Wind intensity

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
    """Executes one day of simulation."""
    R, C = current_grid['L'].shape

    # next_grid starts as a copy of current_grid for all layers
    next_grid = {key: arr.copy() for key, arr in current_grid.items()}

    # P_out and C_out are used to collect incoming values (R3, R4)
    P_out = next_grid['P'].copy()
    C_out = np.zeros((R, C), dtype=np.int8)

    # --- Step 1: LOCAL STATE CHANGES (Rules 1, 2, 6, 8, 9) ---

    # R1: Melting Glaciers: If T > 30 and L = Glacier -> L = Land
    melt_mask = (current_grid['T'] > params['Glacier_T_Threshold']) & (current_grid['L'] == LAND_CODES['Glacier'])
    next_grid['L'][melt_mask] = LAND_CODES['Land']

    # R2: Pollution Increases Temperature: If P > 60 -> T = T + 1
    temp_up_mask = current_grid['P'] > params['Pollution_T_Threshold']
    next_grid['T'][temp_up_mask] = np.minimum(50, current_grid['T'][temp_up_mask] + 1)

    # R6: Rain Cools: If C = 2 -> T = T - 1
    rain_cool_mask = current_grid['C'] == 2
    next_grid['T'][rain_cool_mask] = np.maximum(-20, current_grid['T'][rain_cool_mask] - 1)

    # R8: City Generates Pollution: If L = 4 -> P = P + 1
    city_mask = current_grid['L'] == LAND_CODES['City']
    P_out[city_mask] = np.minimum(100, current_grid['P'][city_mask] + 1)

    # R9: Forest Decreases Pollution: If L = 3 -> P = P - 1
    forest_mask = current_grid['L'] == LAND_CODES['Forest']
    P_out[forest_mask] = np.maximum(0, current_grid['P'][forest_mask] - 1)

    # --- Step 2: SPATIAL PROPAGATION (Rules 3, 4 - Go over neighbours) ---

    # Mask used for R5: Tracks cells that are sources for wind-driven movement.
    wind_cleared_cloud_mask = (current_grid['C'] >= 1) & (current_grid['W_int'] > 0)

    # for each cell
    for r in range(R):
        for c in range(C):

            # for each neighbour
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
                        pollution_transfer = round(neighbor_P * 0.05)
                        # R3 adds pollution. P_out starts with R8/R9 changes.
                        P_out[r, c] = np.minimum(100, P_out[r, c] + pollution_transfer)

                    # R4: Wind Moves Cloud
                    if neighbor_wind_int > 0 and neighbor_C in [1, 2]:
                        # C.C = neighbour.C. Take the highest cloud state.
                        C_out[r, c] = np.maximum(C_out[r, c], neighbor_C)

    # --- Step 3: MERGE, R5, R7 ---
    # R3 Merge: P_out now contains R8/R9 changes + incoming R3.
    next_grid['P'] = P_out

    # R4 Merge: Cloud state is determined by R4 calculated above.
    next_grid['C'] = C_out

    # R5: Cloud Moved (Source Clearing: C.C = 0, C.W = 0)
    next_grid['W_dir'][wind_cleared_cloud_mask] = 0
    next_grid['W_int'][wind_cleared_cloud_mask] = 0


    # R7: Random Wind for Next Day (Runs last, overriding R5 C.W=0 as per sequence)
    next_grid['W_dir'][:] = np.random.randint(1, 5, size=(R, C))
    next_grid['W_int'][:] = np.random.randint(1, 11, size=(R, C))

    # Final Clipping (Essential for array integrity)
    next_grid['T'] = np.clip(next_grid['T'], -20, 50)
    next_grid['P'] = np.clip(next_grid['P'], 0, 100)
    next_grid['C'] = np.clip(next_grid['C'], 0, 2)

    return next_grid


# --- FIGURE CREATION FOR SNAPSHOTS AND ANIMATION ---

def create_4_panel_figure(grid, day_number, is_snapshot=True):
    """
    Creates or re-initializes the 4-panel Matplotlib figure structure.
    Returns: fig, artists, fig_subtitle
    """
    R, C = grid['L'].shape
    total_cells = R * C
    avg_t = np.mean(grid['T'])
    avg_p = np.mean(grid['P'])
    ice_area = np.sum(grid['L'] == LAND_CODES['Glacier']) / total_cells * 100

    # Setup Figure and Subplots (2 rows, 2 columns)
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()  # Flatten for easier indexing (0, 1, 2, 3)

    # Subplot 1 (Top Left): Land Type
    ax1 = axes[0]
    ax1.set_title('1. Land Type (Categorical)')
    img1 = ax1.imshow(grid['L'], cmap=LAND_CMAP, norm=LAND_NORM, interpolation='none')
    ax1.set_xticks([]);
    ax1.set_yticks([])
    cbar1 = plt.colorbar(img1, ax=ax1, fraction=0.046, pad=0.08, ticks=np.array(LAND_BOUNDS[:-1]) + 0.5)
    cbar1.ax.set_yticklabels(LAND_TYPES.values(), **{'fontsize': 10})
    cbar1.set_label('Land Classification', rotation=270, labelpad=40, fontsize=12)

    # Subplot 2 (Top Right): Temperature
    ax2 = axes[1]
    ax2.set_title('2. Temperature ($\degree C$)')
    img2 = ax2.imshow(grid['T'], cmap='coolwarm', vmin=-20, vmax=50, interpolation='none')
    ax2.set_xticks([]);
    ax2.set_yticks([])
    cbar2 = plt.colorbar(img2, ax=ax2, ticks=[-20, 50])
    cbar2.set_label('Temperature ($\degree C$)', rotation=270, labelpad=15, fontsize=12)
    cbar2.ax.tick_params(labelsize=10)

    # Subplot 3 (Bottom Left): Pollution
    ax3 = axes[2]
    ax3.set_title('3. Pollution (%)')
    img3_P = ax3.imshow(grid['P'], cmap='inferno', vmin=0, vmax=100, interpolation='none')
    ax3.set_xticks([]);
    ax3.set_yticks([])
    cbar3 = plt.colorbar(img3_P, ax=ax3, ticks=[0, 100])
    cbar3.set_label('Pollution (%)', rotation=270, labelpad=15, fontsize=12)
    cbar3.ax.tick_params(labelsize=10)

    # Subplot 4 (Bottom Right): Clouds
    ax4 = axes[3]
    ax4.set_title('4. Cloud/Rain State')
    img4_C = ax4.imshow(grid['C'], cmap=CLOUD_CMAP, norm=CLOUD_NORM, interpolation='none')
    ax4.set_xticks([]);
    ax4.set_yticks([])
    cbar4 = plt.colorbar(img4_C, ax=ax4, fraction=0.046, pad=0.08, ticks=CLOUD_TICKS)
    cbar4.ax.set_yticklabels(CLOUD_LABELS, **{'fontsize': 10})
    cbar4.set_label('Cloud State Key', rotation=270, labelpad=40, fontsize=12)

    artists = {'L': img1, 'T': img2, 'P': img3_P, 'C': img4_C}

    # Store the subtitle
    title_text = f"Climate CA Simulation {'Snapshot' if is_snapshot else ''} (Day: {day_number} | Avg T: {avg_t:.1f}°C | Avg P: {avg_p:.1f}% | Ice Area: {ice_area:.2f}%)"
    fig_subtitle_artist = fig.suptitle(title_text, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjusted layout for subtitle space

    return fig, artists, fig_subtitle_artist


def setup_animation_figure():
    """Initializes the Matplotlib figure for the animation."""
    global current_grid, current_artists, fig_subtitle, current_day, initial_grid_state

    # Initialize simulation state
    current_day = 0
    current_grid = initialize_grid(GRID_SIZE, INITIAL_PARAMS)
    # Store a deep copy of the initial state for the Day 1 snapshot after the animation finishes
    initial_grid_state = {key: arr.copy() for key, arr in current_grid.items()}

    # Setup figure using the reusable function, but suppress colorbars for clean animation
    fig, artists, suptitle_artist = create_4_panel_figure(current_grid, 0, is_snapshot=False)

    current_artists = artists
    fig_subtitle = suptitle_artist

    return fig


def update_plot(frame_number):
    """Called by FuncAnimation to advance the simulation and update the plot."""
    global current_grid, current_day, temp_history, pollution_history, ice_area_history, fig_subtitle

    if current_day < DAYS_IN_YEAR:
        current_grid = simulate_step(current_grid, INITIAL_PARAMS)
        current_day += 1

    # Get global metrics
    R, C = current_grid['L'].shape
    total_cells = R * C
    avg_t = np.mean(current_grid['T'])
    avg_p = np.mean(current_grid['P'])
    ice_area = np.sum(current_grid['L'] == LAND_CODES['Glacier']) / total_cells * 100

    # --- COLLECT DATA ---
    temp_history.append(avg_t)
    pollution_history.append(avg_p)
    ice_area_history.append(ice_area)
    # --------------------

    current_artists['L'].set_data(current_grid['L'])
    current_artists['T'].set_data(current_grid['T'])
    current_artists['P'].set_data(current_grid['P'])
    current_artists['C'].set_data(current_grid['C'])

    # Update dynamic title using the global reference
    if fig_subtitle:
        fig_subtitle.set_text(
            f"Climate CA Simulation (Day: {current_day} of {DAYS_IN_YEAR} | Avg T: {avg_t:.1f}°C | Avg P: {avg_p:.1f}% | Ice Area: {ice_area:.2f}%)"
        )

    # Return all artists that were modified, including the title
    return list(current_artists.values()) + [fig_subtitle]


def analyze_and_plot_results(t_hist, p_hist, i_hist):
    """
    Performs statistical analysis and generates a normalized time series plot,
    saving the result to a file.
    """
    # 1) Measurement and Printing
    print("\n--- Statistical Analysis of Global Parameters (Over One Year) ---")

    data_map = {
        "Average Temperature (°C)": np.array(t_hist),
        "Average Pollution (%)": np.array(p_hist),
        "Glacier Area (%)": np.array(i_hist)
    }

    print("\n| Parameter | Average | Std Dev | Range |")
    print("|---|---|---|---|")

    normalized_data = {}

    for name, data in data_map.items():
        if len(data) < 2:
            print(f"| {name} | {data[0]:.2f} | 0.00 | 0.00 |")
            normalized_data[name] = np.full_like(data, 0.5)
            continue

        mean_val = np.mean(data)
        std_dev = np.std(data)
        range_val = np.max(data) - np.min(data)

        # Printing the data
        print(f"| {name} | {mean_val:.2f} | {std_dev:.2f} | {range_val:.2f} |")

        # 2) Data Normalization (Min-Max Scaling)
        # Normalize to the range [0, 1]
        min_val = np.min(data)
        max_val = np.max(data)
        if (max_val - min_val) > 1e-6:  # Check if range is effectively non-zero
            normalized_data[name] = (data - min_val) / (max_val - min_val)
        else:
            # If all values are identical (range 0), normalize to 0.5
            normalized_data[name] = np.full_like(data, 0.5)

    # 2) Visualization
    fig_analysis, ax_analysis = plt.subplots(figsize=(12, 6))
    days = np.arange(1, len(t_hist) + 1)

    ax_analysis.plot(days, normalized_data["Average Temperature (°C)"], label='Average Temperature (Normalized)',
                     color='r')
    ax_analysis.plot(days, normalized_data["Average Pollution (%)"], label='Average Pollution (Normalized)', color='g')
    ax_analysis.plot(days, normalized_data["Glacier Area (%)"], label='Glacier Area (Normalized)', color='b')

    ax_analysis.set_title('System Metrics Tracking Over 365 Days (Min-Max Normalization)')
    ax_analysis.set_xlabel('Simulation Day')
    ax_analysis.set_ylabel('Normalized Value (0 to 1)')
    ax_analysis.legend()
    ax_analysis.grid(True)

    # Save the figure to a file
    filename = 'analytics_time_series.png'
    fig_analysis.savefig(filename, dpi=150)
    plt.close(fig_analysis)
    print(f"Saved analysis plot to {filename}")


def save_snapshot(grid, day_number, filename):
    """Generates and saves a static snapshot of the 4-panel figure."""
    print(f"Generating snapshot for Day {day_number}...")
    fig, _, _ = create_4_panel_figure(grid, day_number, is_snapshot=True)
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved snapshot to {filename}")


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # 1. Setup the animation figure
    fig_anim = setup_animation_figure()

    # 2. Save Day 1 Snapshot using the stored initial state
    save_snapshot(initial_grid_state, 1, 'day_1_snapshot.png')

    print(f"Starting Matplotlib animation for {DAYS_IN_YEAR} days...")
    print("System running simulation with selected stable parameters.")

    # 3. Animation setup and run
    anim = animation.FuncAnimation(
        fig_anim,
        update_plot,
        frames=DAYS_IN_YEAR,
        interval=FRAME_INTERVAL_MS,
        # We need blit=False because the subtitle is being updated.
        blit=False,
        repeat=False
    )

    plt.show()

    # 4. Post-animation processing
    print("\n--- Simulation Complete. Running Statistical Analysis and Plot Generation ---")

    # Save Day 365 Snapshot using the final state of the global grid
    save_snapshot(current_grid, DAYS_IN_YEAR, 'day_365_snapshot.png')

    # Run analysis and save the results
    analyze_and_plot_results(temp_history, pollution_history, ice_area_history)


    print("Simulation complete or window closed.")
