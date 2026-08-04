import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import heapq
import random
from folium import plugins

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
        # Convert DataFrame to dictionary for ultra-fast O(1) memory lookups
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

st.title("HKDL Route Optimizer")
st.markdown("Plan your perfect Disney day using historical queue data and advanced pathfinding AI.")

data_dict = load_data()
if data_dict is None:
    st.error(f"Could not find `{CSV_FILE}`. Please ensure it is in the same folder.")
    st.stop()

with st.expander("Advanced Settings", expanded=False):
    choice = st.radio("Back of Park Opening Time:", ["10:30 AM", "11:00 AM"])
    delayed_open_mins = time_to_mins("10:30") if choice == "10:30 AM" else time_to_mins("11:00 AM")

st.subheader("Select Your Rides")

target_rides = []
col1, col2 = st.columns(2)
for i, ride in enumerate(RIDE_LOCATIONS.keys()):
    with col1 if i % 2 == 0 else col2:
        count = st.number_input(ride.split(" -")[0], min_value=0, max_value=5, value=1, step=1, key=ride)
        target_rides.extend([ride] * count)

if st.button("Calculate Fastest Route", type="primary", use_container_width=True):
    if not target_rides:
        st.warning("Please select at least one ride!")
    else:
        with st.spinner("Finding optimal route in milliseconds..."):
            
            # Index duplicate ride instances
            indexed_rides = []
            ride_counts = {}
            for ride in target_rides:
                c = ride_counts.get(ride, 0)
                indexed_rides.append((ride, c))
                ride_counts[ride] = c + 1
            
            num_rides = len(indexed_rides)
            ALL_VISITED_MASK = (1 << num_rides) - 1

            start_time = time_to_mins("10:00")
            # Heap Queue format: (current_time, curr_loc, visited_mask, step_logs, location_path)
            pq = [(start_time, "Entrance", 0, [], ["Entrance"])]
            best_visited = {}
            optimal_result = None

            # --- Ultra-Fast Dijkstra Algorithm ---
            while pq:
                curr_time, curr_loc, mask, current_log, loc_path = heapq.heappop(pq)

                # First time reaching ALL_VISITED_MASK is guaranteed to be optimal
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
                    
                    # Avoid branching on identical rides at the same decision node
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
                    heapq.heappush(pq, (finish, land, next_mask, current_log + [step_log], loc_path + [land]))

            # Save results to session
            st.session_state['best_state'] = optimal_result if optimal_result else [float('inf'), [], []]
            st.session_state['total_rides'] = len(target_rides)


# --- Persistent Display Block ---
if 'best_state' in st.session_state:
    best_state = st.session_state['best_state']
    total_rides = st.session_state['total_rides']

    if best_state[0] == float('inf'):
        st.error("Could not calculate a route.")
    else:
        st.success(f"**Optimal Route Found!** Finish all {total_rides} rides by **{mins_to_time(best_state[0])}**")
        
        # --- Map Generation ---
        st.subheader("Interactive Route Map")
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
            
            # Custom html badge labels that stay open permanently
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
            
            folium.Marker(
                location=[j_lat, j_lon],
                icon=custom_icon
            ).add_to(m)
        
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

        # --- Table Generation ---
        st.subheader("Itinerary")
        st.table(best_state[1])
        st.caption(f"Assumes 10:00 AM entry & {RIDE_DURATION} min duration per ride.")
