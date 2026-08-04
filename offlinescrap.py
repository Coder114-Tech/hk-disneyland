import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import heapq
import random
from folium import plugins
import math
import time

# --- Configuration & Park Layout ---

CSV_FILE = "hkdl_june15_timeline.csv"
RIDE_DURATION = 8  

RIDE_LOCATIONS = {
    "Hyperspace Mountain": "Tomorrowland",
    "Iron Man Experience - Presented by AIA": "Tomorrowland",
    "Frozen Ever After - Presented by Blue Cross": "World of Frozen",
    "Wandering Oaken's Sliding Sleighs": "World of Frozen",
    "RC Racer": "Toy Story Land",
    "Toy Soldier Parachute Drop": "Toy Story Land",
    "Mystic Manor": "Mystic Point",
    "Big Grizzly Mountain Runaway Mine Cars": "Grizzly Gulch"
}

DELAYED_RIDES = [
    "RC Racer", "Toy Soldier Parachute Drop", 
    "Mystic Manor", "Big Grizzly Mountain Runaway Mine Cars"
]

# Real-world approximate GPS coordinates for the map
LAND_COORDS = {
    "Entrance": [22.313003078183442, 114.0432931733108],
    "Tomorrowland": [22.313377592523448, 114.04183504593641],
    "World of Frozen": [22.31247483846683, 114.03882636467279],
    "Toy Story Land": [22.310474700112668, 114.03967888153315],
    "Mystic Point": [22.310063990309636, 114.04097960803556],
    "Grizzly Gulch": [22.310292281405896, 114.04208467812052]
}

WALK_MATRIX = {
    "Entrance": {"Entrance": 0, "Tomorrowland": 5, "World of Frozen": 12, "Toy Story Land": 15, "Mystic Point": 13, "Grizzly Gulch": 12},
    "Tomorrowland": {"Entrance": 5, "Tomorrowland": 3, "World of Frozen": 10, "Toy Story Land": 15, "Mystic Point": 15, "Grizzly Gulch": 15},
    "World of Frozen": {"Entrance": 12, "Tomorrowland": 10, "World of Frozen": 3, "Toy Story Land": 12, "Mystic Point": 15, "Grizzly Gulch": 15},
    "Toy Story Land": {"Entrance": 15, "Tomorrowland": 15, "World of Frozen": 12, "Toy Story Land": 3, "Mystic Point": 3, "Grizzly Gulch": 5},
    "Mystic Point": {"Entrance": 13, "Tomorrowland": 15, "World of Frozen": 15, "Toy Story Land": 3, "Mystic Point": 3, "Grizzly Gulch": 3},
    "Grizzly Gulch": {"Entrance": 12, "Tomorrowland": 15, "World of Frozen": 15, "Toy Story Land": 5, "Mystic Point": 3, "Grizzly Gulch": 3}
}

# --- Helper Functions ---

def estimate_execution_time_precise(target_rides):
    """
    Calculates the exact state-space complexity N * 2^N and computes 
    an estimated execution time down to millisecond precision.
    """
    N = len(target_rides)
    if N == 0:
        return "0.00 ms", 0, "No rides selected"
        
    unique_rides = len(set(target_rides))
    theoretical_states = N * (2 ** N)
    pruning_factor = 0.20 if N > 8 else 0.40
    estimated_operations = int(theoretical_states * pruning_factor)
    
    # Python CPU processing rate: ~1,200,000 operations/sec on modern hardware
    ops_per_millisecond = 1200
    est_ms = estimated_operations / ops_per_millisecond

    if est_ms < 1.0:
        time_str = f"{est_ms:.2f} ms"
        label = "< 1 ms (Ultra Instant)"
    elif est_ms < 1000:
        time_str = f"{est_ms:.1f} ms"
        label = f"~{int(est_ms)} ms"
    else:
        sec = est_ms / 1000
        time_str = f"{sec:.2f} sec"
        label = f"~{sec:.1f}s"
        
    return time_str, estimated_operations, label

def mins_to_time(minutes):
    h = int(minutes // 60)
    m = int(minutes % 60)
    return f"{h:02d}:{m:02d}"

def time_to_mins(time_str):
    h, m = map(int, time_str.split(':'))
    return h * 60 + m

@st.cache_data 
def load_data():
    try:
        df = pd.read_csv(CSV_FILE, dtype=str)
        df.set_index("Time", inplace=True)
        return df.to_dict()
    except FileNotFoundError:
        return None

def get_wait_time(data_dict, ride, current_mins):
    time_str = mins_to_time(current_mins)
    ride_data = data_dict.get(ride, {})
    wait = ride_data.get(time_str, None)
    
    if wait is None or str(wait).strip() in ("", "nan"):
        return 5
        
    try:
        return int(float(wait))
    except ValueError:
        return 5 

# --- Main App UI ---

st.set_page_config(page_title="HKDL Route AI", layout="centered")

# Beautiful Centered Header
st.markdown("<h1 style='text-align: center;'>Hong Kong Disneyland Route Optimizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; margin-bottom: 2rem;'>Plan your perfect Disney day using historical queue data (June 15, 2026) and advanced pathfinding AI.</p>", unsafe_allow_html=True)

data_dict = load_data()
if data_dict is None:
    st.error(f"Could not find `{CSV_FILE}`. Please ensure it is in the same folder.")
    st.stop()

# --- Sidebar for Settings ---
with st.sidebar:
    st.header("⚙️ Advanced Settings")
    st.markdown("Adjust park parameters before calculating.")
    choice = st.radio("Back of Park Opening Time:", ["10:30 AM", "11:00 AM"])
    delayed_open_mins = time_to_mins("10:30") if choice == "10:30 AM" else time_to_mins("11:00 AM")
    st.caption("Certain lands (like Grizzly Gulch and Toy Story Land) may open later than the main entrance.")

# --- Ride Selection Container ---
with st.container(border=True):
    st.subheader("Select Your Rides")
    st.markdown("Choose how many times you'd like to ride each attraction.")
    
    target_rides = []
    col1, col2 = st.columns(2)
    
    for i, ride in enumerate(RIDE_LOCATIONS.keys()):
        with col1 if i % 2 == 0 else col2:
            count = st.number_input(ride.split(" -")[0], min_value=0, max_value=5, value=1, step=1, key=ride)
            target_rides.extend([ride] * count)

st.write("") # Spacer

# --- Precise Estimate Calculation Container ---
est_time_str, est_ops, speed_label = estimate_execution_time_precise(target_rides)

with st.container(border=True):
    st.subheader("Computation Estimates")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric(label="Total Rides", value=len(target_rides))
    with m_col2:
        st.metric(label="Est. Search Space", value=f"{est_ops:,} states")
    with m_col3:
        st.metric(label="Est. Execution Time", value=est_time_str)

st.divider()

# --- Calculation Trigger ---
if st.button("Calculate Fastest Route", type="primary", use_container_width=True):
    if not target_rides:
        st.warning("Please select at least one ride!")
    else:
        start_cpu_time = time.perf_counter()  # Millisecond-accurate timer
        
        with st.spinner(f"Evaluating optimal state paths... (Est: {est_time_str})"):
            indexed_rides = []
            ride_counts = {}
            for ride in target_rides:
                c = ride_counts.get(ride, 0)
                indexed_rides.append((ride, c))
                ride_counts[ride] = c + 1
                
            num_rides = len(indexed_rides)
            ALL_VISITED_MASK = (1 << num_rides) - 1
            
            start_time = time_to_mins("10:00")
            counter = 0
            pq = [(start_time, counter, "Entrance", 0, [], ["Entrance"])]
            best_visited = {}
            optimal_result = None
            
            while pq:
                curr_time, _, curr_loc, mask, current_log, loc_path = heapq.heappop(pq)
                
                if mask == ALL_VISITED_MASK:
                    optimal_result = [curr_time, current_log, loc_path]
                    break
                    
                state_key = (curr_loc, mask)
                if state_key in best_visited and best_visited[state_key] <= curr_time:
                    continue
                best_visited[state_key] = curr_time
                
                tried_names = set()
                for idx, (ride_name, copy_id) in enumerate(indexed_rides):
                    if mask & (1 << idx):
                        continue
                    if ride_name in tried_names:
                        continue
                        
                    tried_names.add(ride_name)
                    land = RIDE_LOCATIONS[ride_name]
                    walk = WALK_MATRIX[curr_loc][land]
                    arr_time = curr_time + walk
                    wait_open = max(0, delayed_open_mins - arr_time) if ride_name in DELAYED_RIDES else 0
                    actual_arr = arr_time + wait_open
                    q_wait = get_wait_time(data_dict, ride_name, actual_arr)
                    finish = actual_arr + q_wait + RIDE_DURATION
                    
                    step_log = {
                        "Time": mins_to_time(actual_arr),
                        "Action / Ride": ("Wait: " if wait_open > 0 else "") + ride_name.split(" -")[0],
                        "Walk": f"{walk}m",
                        "Queue": f"{q_wait}m"
                    }
                    next_mask = mask | (1 << idx)
                    counter += 1
                    heapq.heappush(pq, (finish, counter, land, next_mask, current_log + [step_log], loc_path + [land]))
                    
            # Stop timer
            elapsed_ms = (time.perf_counter() - start_cpu_time) * 1000
            
            # Save results & execution stats to session
            st.session_state['best_state'] = optimal_result if optimal_result else [float('inf'), [], []]
            st.session_state['total_rides'] = len(target_rides)
            st.session_state['elapsed_ms'] = elapsed_ms

# --- Persistent Display Block ---
if 'best_state' in st.session_state:
    best_state = st.session_state['best_state']
    total_rides = st.session_state['total_rides']
    elapsed_ms = st.session_state.get('elapsed_ms', 0)
    
    if best_state[0] == float('inf'):
        st.error("Could not calculate a route.")
    else:
        st.success(
            f"**Optimal Route Found!** Finish all {total_rides} rides by **{mins_to_time(best_state[0])}** "
            f"*(Calculated in **{elapsed_ms:.2f} ms**)*"
        )
        
        # UI TABS FOR CLEANER LAYOUT
        tab1, tab2 = st.tabs(["Interactive Map", "Itinerary Schedule"])
        
        # --- Map Generation (Tab 1) ---
        with tab1:
            st.markdown("#### Your Visual Route")
            m = folium.Map(location=[22.3125, 114.0435], zoom_start=16, tiles="CartoDB positron")
            path_coords = []
            
            for i, loc in enumerate(best_state[2]):
                base_lat, base_lon = LAND_COORDS[loc]
                if i == 0:
                    j_lat, j_lon = base_lat, base_lon
                else:
                    j_lat = base_lat + random.uniform(-0.0003, 0.0003)
                    j_lon = base_lon + random.uniform(-0.0003, 0.0003)
                path_coords.append([j_lat, j_lon])
                
                # Custom html badge labels
                label_text = f"Start: {loc}" if i == 0 else f"{i}. {loc}"
                bg_color = "#d9534f" if i == 0 else "#0275d8"
                custom_icon = folium.DivIcon(
                    icon_size=(150, 36),
                    icon_anchor=(0, 0),
                    html=f'''
                        <div style="
                            font-size: 11px; 
                            font-weight: bold; 
                            color: white; 
                            background-color: {bg_color}; 
                            border: 1px solid white;
                            border-radius: 4px; 
                            padding: 3px 7px; 
                            box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
                            white-space: nowrap;
                            display: inline-block;
                        ">
                            {label_text}
                        </div>
                    '''
                )
                folium.Marker(location=[j_lat, j_lon], icon=custom_icon).add_to(m)
                
            plugins.AntPath(
                locations=path_coords, 
                dash_array=[10, 20],
                delay=800,
                color='green',
                pulse_color='black',
                weight=5,
                opacity=0.8
            ).add_to(m)
            
            st_folium(m, width=700, height=400, returned_objects=[])

        # --- Table Generation (Tab 2) ---
        with tab2:
            st.markdown("#### Step-by-Step Plan")
            st.table(best_state[1])
            st.caption(f"💡 *Assumes 10:00 AM entry & {RIDE_DURATION} min duration per ride.*")

