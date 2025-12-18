import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import warnings
# Filter out the specific deprecation warning from pkg_resources
warnings.filterwarnings("ignore", category=UserWarning, module='pygame.pkgdata')


class UrbanCausalIntersectionExtendedEnv(gym.Env):
    """
    Extended urban intersection environment with:
    - Larger map (1600x1600)
    - Multiple intersections in a grid
    - Traffic lights at each intersection
    - NPC cars that follow traffic rules
    - Pedestrians that respect traffic lights
    - More complex traffic rules
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, observation_type: str = "kinematic"):
        super().__init__()
        
        # Window setup - larger map
        self.map_size = 1600  # 1600x1600 grid
        self.window_size = min(1600, 1200)  # Limit window size for display (can be scaled)
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.view_offset = np.array([0.0, 0.0])  # Camera offset for following agent
        self.road_width = 80  # Wider roads
        self.intersection_size = 160  # Larger intersections
        self.block_size = 400  # Size of each city block
        self.observation_type = observation_type
        
        # Calculate grid layout
        # Number of intersections: 3x3 grid = 9 intersections
        self.num_intersections_x = 3
        self.num_intersections_y = 3
        
        # Default context
        self.context = {
            "friction": 0.98,  # Slight friction
            "traffic_light_duration": 60,  # Traffic light cycle duration
            "pedestrian_spawn_rate": 0.03,
            "num_pedestrians": 15,
            "jaywalk_probability": 0.1,
            "npc_car_spawn_rate": 0.02,
            "num_npc_cars": 20,
            "temperature": 20,
            "roughness": 0.0,
            "traffic_density": "medium",
            "pedestrian_density": "medium",
            "driver_impatience": 0.5,
            "npc_color": "random",
            "npc_size": "random"
        }
        
        # --- Define Spaces ---
        # Observation: configurable
        if self.observation_type == "pixel":
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(84, 84, 3),
                dtype=np.uint8
            )
        else:
            # kinematic (55 dims): 5 agent + 16 lidar + 20 cars + 10 peds + 4 light
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(55,),
                dtype=np.float32
            )
        
        # Actions: continuous [acceleration, steering], each in [-1, 1]
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                                       high=np.array([1.0, 1.0], dtype=np.float32),
                                       dtype=np.float32)
        
        # --- State Variables ---
        self._agent_location = None
        self._agent_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self._agent_heading = None
        
        # Physics constants
        self.max_speed = 8.0
        self.acceleration = 0.5
        self.angular_velocity = 0.10
        self.car_length = 25
        self.car_width = 12
        self.safety_buffer = float(self.car_width + 20)
        
        # NPC cars
        self.npc_cars = []
        self.npc_car_speed = 4.0
        
        # Pedestrian system
        self.pedestrians = []
        self.pedestrian_radius = 6
        self.pedestrian_speed = 0.6
        
        # Traffic lights - one per intersection
        self.traffic_lights = {}  # Key: (intersection_x, intersection_y), Value: {"state": "red"/"green", "timer": int, "phase": 0-3}
        self.traffic_light_timer = 0
        
        # Intersection positions (grid)
        self.intersections = []
        self._initialize_intersections()
        
        # Roads - horizontal and vertical road segments
        self.horizontal_roads = []
        self.vertical_roads = []
        self._initialize_roads()
        
        # Zebra crossings at each intersection
        self.zebra_crossings = []
        self._initialize_zebra_crossings()
        
        # Spawn points for cars and pedestrians
        self.car_spawn_points = []
        self.pedestrian_spawn_points = []
        self.jaywalker_spawn_points = []
        self._initialize_spawn_points()

    def _initialize_intersections(self):
        """Initialize intersection positions in a grid."""
        spacing = self.block_size + self.road_width
        start_x = self.road_width + 200
        start_y = self.road_width + 200
        
        for i in range(self.num_intersections_y):
            for j in range(self.num_intersections_x):
                x = start_x + j * spacing
                y = start_y + i * spacing
                self.intersections.append({
                    "pos": np.array([x, y]),
                    "grid": (j, i)
                })
                # Initialize directional traffic lights for this intersection
                # 4 lights per intersection: north, south, east, west
                # Phases: 0=NS green/EW red, 1=NS yellow/EW red, 2=NS red/EW green, 3=NS red/EW yellow
                initial_phase = (i + j) % 4
                directions = {}
                if initial_phase == 0:  # NS green, EW red
                    directions = {"north": "green", "south": "green", "east": "red", "west": "red"}
                elif initial_phase == 1:  # NS yellow, EW red
                    directions = {"north": "yellow", "south": "yellow", "east": "red", "west": "red"}
                elif initial_phase == 2:  # NS red, EW green
                    directions = {"north": "red", "south": "red", "east": "green", "west": "green"}
                else:  # phase 3: NS red, EW yellow
                    directions = {"north": "red", "south": "red", "east": "yellow", "west": "yellow"}
                
                self.traffic_lights[(j, i)] = {
                    "timer": 0,
                    "phase": initial_phase,
                    "directions": directions
                }

    def _initialize_roads(self):
        """Initialize road segments."""
        spacing = self.block_size + self.road_width
        
        # Horizontal roads
        for i in range(self.num_intersections_y + 1):
            y = self.road_width + 200 + i * spacing
            self.horizontal_roads.append({
                "y": y,
                "x_start": 0,
                "x_end": self.map_size
            })
        
        # Vertical roads
        for j in range(self.num_intersections_x + 1):
            x = self.road_width + 200 + j * spacing
            self.vertical_roads.append({
                "x": x,
                "y_start": 0,
                "y_end": self.map_size
            })

    def _initialize_zebra_crossings(self):
        """Initialize zebra crossings at each intersection."""
        for intersection in self.intersections:
            pos = intersection["pos"]
            # 4 crossings per intersection (north, south, east, west)
            offset = self.intersection_size // 2 - 20
            
            self.zebra_crossings.extend([
                # North crossing
                {"start": np.array([pos[0] - offset, pos[1] - offset]), 
                 "end": np.array([pos[0] + offset, pos[1] - offset]), 
                 "direction": "horizontal",
                 "intersection": intersection["grid"]},
                # South crossing
                {"start": np.array([pos[0] - offset, pos[1] + offset]), 
                 "end": np.array([pos[0] + offset, pos[1] + offset]), 
                 "direction": "horizontal",
                 "intersection": intersection["grid"]},
                # East crossing
                {"start": np.array([pos[0] + offset, pos[1] - offset]), 
                 "end": np.array([pos[0] + offset, pos[1] + offset]), 
                 "direction": "vertical",
                 "intersection": intersection["grid"]},
                # West crossing
                {"start": np.array([pos[0] - offset, pos[1] - offset]), 
                 "end": np.array([pos[0] - offset, pos[1] + offset]), 
                 "direction": "vertical",
                 "intersection": intersection["grid"]},
            ])

    def _initialize_spawn_points(self):
        """Initialize spawn points for cars and pedestrians."""
        spacing = self.block_size + self.road_width
        
        # Car spawn points on roads - RIGHT SIDE ONLY to prevent head-on collisions
        # Right-side driving rules:
        # - Horizontal roads: cars going EAST (right) drive on BOTTOM side (south), cars going WEST (left) drive on TOP side (north)
        # - Vertical roads: cars going NORTH (up) drive on RIGHT side (east), cars going SOUTH (down) drive on LEFT side (west)
        
        for road in self.horizontal_roads:
            for x in range(300, self.map_size - 300, 300):
                # Determine direction: alternate segments go east vs west
                going_east = (x // 300) % 2 == 0
                if going_east:
                    # Going east: right side is BOTTOM (south) of road
                    right_y_offset = self.road_width // 4  # Bottom lane
                    heading = 0.0  # East (right)
                else:
                    # Going west: right side is TOP (north) of road  
                    right_y_offset = -self.road_width // 4  # Top lane
                    heading = np.pi  # West (left)
                
                self.car_spawn_points.append({
                    "pos": np.array([float(x), road["y"] + right_y_offset]),
                    "heading": heading,
                    "type": "horizontal",
                    "direction": "east" if going_east else "west"
                })
        
        for road in self.vertical_roads:
            for y in range(300, self.map_size - 300, 300):
                # Determine direction: alternate segments go north vs south
                going_north = (y // 300) % 2 == 0
                if going_north:
                    # Going north: right side is RIGHT (east) of road
                    right_x_offset = self.road_width // 4  # Right lane
                    heading = np.pi / 2  # North (up)
                else:
                    # Going south: right side is LEFT (west) of road
                    right_x_offset = -self.road_width // 4  # Left lane
                    heading = -np.pi / 2  # South (down)
                
                self.car_spawn_points.append({
                    "pos": np.array([road["x"] + right_x_offset, float(y)]),
                    "heading": heading,
                    "type": "vertical",
                    "direction": "north" if going_north else "south"
                })
        
        # Pedestrian spawn points at intersection corners (for zebra crossing users)
        # Spawn at the four corners of each intersection
        for intersection in self.intersections:
            pos = intersection["pos"]
            corner_offset = self.intersection_size // 2 + 30  # Offset to corner positions
            # Four corners of the intersection
            self.pedestrian_spawn_points.extend([
                np.array([pos[0] - corner_offset, pos[1] - corner_offset]),  # Northwest corner
                np.array([pos[0] + corner_offset, pos[1] - corner_offset]),  # Northeast corner
                np.array([pos[0] - corner_offset, pos[1] + corner_offset]),  # Southwest corner
                np.array([pos[0] + corner_offset, pos[1] + corner_offset]),  # Southeast corner
            ])
        
        # Jaywalker spawn points - on the sides of roads (not at intersections)
        # Spawn on sidewalks along roads, between intersections
        for road in self.horizontal_roads:
            # Spawn points on the sides (sidewalks) of horizontal roads
            for x in range(250, self.map_size - 250, 250):
                spawn_pos = np.array([float(x), road["y"]])
                # Check if this is far enough from intersections
                too_close = False
                for intersection in self.intersections:
                    if np.linalg.norm(spawn_pos - intersection["pos"]) < 250:
                        too_close = True
                        break
                if not too_close and 50 <= spawn_pos[0] < self.map_size - 50:
                    # Spawn on both sides of the road (top and bottom sidewalks)
                    # Top side (north sidewalk)
                    self.jaywalker_spawn_points.append(spawn_pos + np.array([0.0, -self.road_width // 2 - 15]))
                    # Bottom side (south sidewalk)
                    self.jaywalker_spawn_points.append(spawn_pos + np.array([0.0, self.road_width // 2 + 15]))
        
        for road in self.vertical_roads:
            # Spawn points on the sides (sidewalks) of vertical roads
            for y in range(250, self.map_size - 250, 250):
                spawn_pos = np.array([road["x"], float(y)])
                # Check if this is far enough from intersections
                too_close = False
                for intersection in self.intersections:
                    if np.linalg.norm(spawn_pos - intersection["pos"]) < 250:
                        too_close = True
                        break
                if not too_close and 50 <= spawn_pos[1] < self.map_size - 50:
                    # Spawn on both sides of the road (left and right sidewalks)
                    # Left side (west sidewalk)
                    self.jaywalker_spawn_points.append(spawn_pos + np.array([-self.road_width // 2 - 15, 0.0]))
                    # Right side (east sidewalk)
                    self.jaywalker_spawn_points.append(spawn_pos + np.array([self.road_width // 2 + 15, 0.0]))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Update context if options provided
        if options is not None and isinstance(options, dict):
            self.context.update(options)
        
        # Reset agent (user car) - spawn on a random road (right side only)
        spawn_idx = self.np_random.integers(0, len(self.car_spawn_points))
        spawn = self.car_spawn_points[spawn_idx]
        self._agent_location = spawn["pos"].copy().astype(np.float32)
        self._agent_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self._agent_heading = spawn["heading"]
        # Store agent direction for traffic light checking (if needed)
        self._agent_direction = spawn.get("direction", "east")
        
        self._agent_direction = spawn.get("direction", "east")
        
        # --- Causal RL: Domain Randomization & Interventions ---
        # 1. Temperature
        if options and "temperature" in options:
            self.temperature = options["temperature"]
        else:
            temps = [-10, 0, 10, 20, 30]
            self.temperature = int(self.np_random.choice(temps))
        
        # Map temp to roughness
        self.roughness = (self.temperature - (-10)) / (30 - (-10))
        
        # 2. Traffic Density
        if options and "traffic_density" in options:
            self.traffic_density = options["traffic_density"]
        else:
            self.traffic_density = self.np_random.choice(["low", "medium", "high"])
            
        # Map traffic density to spawn rates
        if self.traffic_density == "low":
            self.context["npc_car_spawn_rate"] = 0.005
            self.context["num_npc_cars"] = 10
        elif self.traffic_density == "medium":
            self.context["npc_car_spawn_rate"] = 0.02
            self.context["num_npc_cars"] = 20
        else: # high
            self.context["npc_car_spawn_rate"] = 0.05
            self.context["num_npc_cars"] = 40
            
        # 3. Pedestrian Density
        if options and "pedestrian_density" in options:
            self.pedestrian_density = options["pedestrian_density"]
        else:
            self.pedestrian_density = self.np_random.choice(["low", "medium", "high"])
            
        # Map pedestrian density to spawn rates
        if self.pedestrian_density == "low":
            self.context["pedestrian_spawn_rate"] = 0.01
            self.context["num_pedestrians"] = 5
        elif self.pedestrian_density == "medium":
            self.context["pedestrian_spawn_rate"] = 0.03
            self.context["num_pedestrians"] = 15
        else: # high
            self.context["pedestrian_spawn_rate"] = 0.06
            self.context["num_pedestrians"] = 30
            
        # 4. Driver Impatience
        if options and "driver_impatience" in options:
            self.driver_impatience = float(options["driver_impatience"])
        else:
            self.driver_impatience = float(self.np_random.uniform(0.0, 1.0))
            
        # 5. NPC Color
        if options and "npc_color" in options:
            self.npc_color = options["npc_color"]
        else:
            colors = ["random", "red", "blue", "green", "yellow", "white", "black"]
            self.npc_color = self.np_random.choice(colors)
            
        # 6. NPC Size
        if options and "npc_size" in options:
            self.npc_size = options["npc_size"]
        else:
            sizes = ["random", "small", "medium", "large"]
            self.npc_size = self.np_random.choice(sizes)
        
        # Update context
        self.context["temperature"] = self.temperature
        self.context["roughness"] = self.roughness
        self.context["traffic_density"] = self.traffic_density
        self.context["pedestrian_density"] = self.pedestrian_density
        self.context["driver_impatience"] = self.driver_impatience
        self.context["npc_color"] = self.npc_color
        self.context["npc_size"] = self.npc_size
        
        # Physics modifiers
        # Friction: REMOVED dependency on temp for agent (kept at default 0.98)
        
        # NPC Speed Factor
        self.npc_speed_factor = 0.6 + (0.4 * self.roughness)
        
        # NPC Brake Factor Adjustment
        self.npc_brake_mod = 0.05 * (1.0 - self.roughness)
        
        # Initialize view offset
        self.view_offset = self._agent_location - np.array([self.window_size / 2, self.window_size / 2])
        self.view_offset = np.clip(self.view_offset, 0, max(0, self.map_size - self.window_size))
        
        # Reset NPC cars
        self.npc_cars = []
        
        # Reset pedestrians
        self.pedestrians = []
        
        # Reset traffic lights
        for key in self.traffic_lights:
            self.traffic_lights[key]["timer"] = 0
        
        self.traffic_light_timer = 0
        
        observation = self._get_obs()
        info = {
            "causal_vars": {
                "temperature": self.temperature,
                "traffic_density": self.traffic_density,
                "pedestrian_density": self.pedestrian_density,
                "driver_impatience": self.driver_impatience,
                "npc_color": self.npc_color,
                "npc_size": self.npc_size,
                "roughness": self.roughness
            }
        }
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info

    def _get_obs(self):
        """Get observation based on configured observation_type."""
        if self.observation_type == "pixel":
            # Render full frame and downscale to 84x84
            frame = self._render_frame()
            if frame is None or frame.size == 0:
                return np.zeros((84, 84, 3), dtype=np.uint8)
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            scaled = pygame.transform.smoothscale(surf, (84, 84))
            arr = np.transpose(pygame.surfarray.pixels3d(scaled), (1, 0, 2)).copy()
            return arr.astype(np.uint8)
        
        # Kinematic observation (55 dims)
        obs_parts = []
        # Agent state (5)
        obs_parts.extend([
            float(self._agent_location[0]),
            float(self._agent_location[1]),
            float(self._agent_velocity[0]),
            float(self._agent_velocity[1]),
            float(self._agent_heading),
        ])
        # LIDAR (16 rays)
        obs_parts.extend(self._compute_lidar(num_rays=16, max_range=200.0))
        # Nearest 5 cars (relative pos/vel: 4 each -> 20)
        obs_parts.extend(self._nearest_cars_features(k=5))
        # Nearest 5 pedestrians (relative pos: 2 each -> 10)
        obs_parts.extend(self._nearest_ped_features(k=5))
        # Next traffic light (one-hot + time to change)
        red, yellow, green, ttc = self._next_traffic_light_features()
        obs_parts.extend([red, yellow, green, ttc])
        return np.array(obs_parts, dtype=np.float32)

    def _compute_lidar(self, num_rays: int = 16, max_range: float = 200.0):
        """Simple LIDAR: distance to nearest pedestrian or car along ray directions."""
        angles = self._agent_heading + np.linspace(-np.pi, np.pi, num_rays, endpoint=False)
        dists = np.full(num_rays, max_range, dtype=np.float32)
        obstacles = []
        for car in self.npc_cars:
            obstacles.append((car["pos"], self.car_width / 2))
        for ped in self.pedestrians:
            obstacles.append((ped["pos"], self.pedestrian_radius))
        agent_pos = self._agent_location
        for i, ang in enumerate(angles):
            ray_dir = np.array([np.cos(ang), -np.sin(ang)], dtype=np.float32)
            # Sample along the ray
            for r in np.linspace(5, max_range, num=25):
                p = agent_pos + ray_dir * r
                # bounds
                if p[0] < 0 or p[0] > self.map_size or p[1] < 0 or p[1] > self.map_size:
                    dists[i] = min(dists[i], r)
                    break
                # obstacles
                for (op, rad) in obstacles:
                    if np.linalg.norm(p - op) <= rad + 2:
                        dists[i] = min(dists[i], r)
                        break
                if dists[i] < r:
                    break
        return dists.tolist()

    def _nearest_cars_features(self, k: int = 5):
        """Return k nearest NPC cars as [rel_x, rel_y, rel_vx, rel_vy] each."""
        feats = []
        rels = []
        for car in self.npc_cars:
            rel_pos = car["pos"] - self._agent_location
            rels.append((np.linalg.norm(rel_pos), rel_pos, car["velocity"]))
        rels.sort(key=lambda x: x[0])
        for i in range(k):
            if i < len(rels):
                _, rel_pos, rel_vel = rels[i]
                feats.extend([float(rel_pos[0]), float(rel_pos[1]), float(rel_vel[0]), float(rel_vel[1])])
            else:
                feats.extend([0.0, 0.0, 0.0, 0.0])
        return feats

    def _nearest_ped_features(self, k: int = 5):
        """Return k nearest pedestrians as [rel_x, rel_y] each."""
        feats = []
        rels = []
        for ped in self.pedestrians:
            rel_pos = ped["pos"] - self._agent_location
            rels.append((np.linalg.norm(rel_pos), rel_pos))
        rels.sort(key=lambda x: x[0])
        for i in range(k):
            if i < len(rels):
                _, rel_pos = rels[i]
                feats.extend([float(rel_pos[0]), float(rel_pos[1])])
            else:
                feats.extend([0.0, 0.0])
        return feats

    def _next_traffic_light_features(self):
        """Return next light state (red,yellow,green one-hot) and time-to-change normalized [0,1]."""
        inter = self._get_nearest_intersection(self._agent_location)
        if inter is None:
            return 0.0, 0.0, 1.0, 0.0  # default green
        direction = self._agent_travel_direction()
        grid = inter["grid"]
        light = self.traffic_lights.get(grid)
        if not light:
            return 0.0, 0.0, 1.0, 0.0
        state = light["directions"].get(direction, "green")
        red = 1.0 if state == "red" else 0.0
        yellow = 1.0 if state == "yellow" else 0.0
        green = 1.0 if state == "green" else 0.0
        ttc = light["timer"] / max(1, self.context.get("traffic_light_duration", 60))
        return red, yellow, green, float(ttc)

    def _agent_travel_direction(self):
        """Approximate agent cardinal travel direction by heading."""
        if abs(np.cos(self._agent_heading)) > abs(np.sin(self._agent_heading)):
            return "east" if np.cos(self._agent_heading) > 0 else "west"
        return "south" if np.sin(self._agent_heading) < 0 else "north"

    def _get_nearest_intersection(self, pos):
        """Get the nearest intersection to a position."""
        min_dist = float('inf')
        nearest = None
        for intersection in self.intersections:
            dist = np.linalg.norm(pos - intersection["pos"])
            if dist < min_dist:
                min_dist = dist
                nearest = intersection
        return nearest

    def _get_traffic_light_state(self, intersection_grid, direction):
        """Get traffic light state for a given intersection and direction."""
        if intersection_grid not in self.traffic_lights:
            return "green"  # Default to green if intersection not found
        
        light = self.traffic_lights[intersection_grid]
        
        # Return the state for the specific direction
        if direction in light["directions"]:
            return light["directions"][direction]
        return "green"  # Default

    def _update_traffic_lights(self):
        """Update traffic light states based on timer with proper phases."""
        self.traffic_light_timer += 1
        light_duration = self.context.get("traffic_light_duration", 60)
        
        # Phases:
        # 0: NS green, EW red
        # 1: NS yellow, EW red  
        # 2: NS red, EW green
        # 3: NS red, EW yellow
        
        # Update each traffic light
        for key, light in self.traffic_lights.items():
            light["timer"] += 1
            
            if light["timer"] >= light_duration:
                light["timer"] = 0
                light["phase"] = (light["phase"] + 1) % 4
                
                # Update directional states based on phase
                phase = light["phase"]
                if phase == 0:  # NS green, EW red
                    light["directions"]["north"] = "green"
                    light["directions"]["south"] = "green"
                    light["directions"]["east"] = "red"
                    light["directions"]["west"] = "red"
                elif phase == 1:  # NS yellow, EW red
                    light["directions"]["north"] = "yellow"
                    light["directions"]["south"] = "yellow"
                    light["directions"]["east"] = "red"
                    light["directions"]["west"] = "red"
                elif phase == 2:  # NS red, EW green
                    light["directions"]["north"] = "red"
                    light["directions"]["south"] = "red"
                    light["directions"]["east"] = "green"
                    light["directions"]["west"] = "green"
                elif phase == 3:  # NS red, EW yellow
                    light["directions"]["north"] = "red"
                    light["directions"]["south"] = "red"
                    light["directions"]["east"] = "yellow"
                    light["directions"]["west"] = "yellow"

    def _spawn_npc_car(self):
        """Spawn a new NPC car."""
        if len(self.npc_cars) >= self.context.get("num_npc_cars", 20):
            return
        
        spawn_idx = self.np_random.integers(0, len(self.car_spawn_points))
        spawn = self.car_spawn_points[spawn_idx]
        
        # Determine Size
        global_size = self.context.get("npc_size", "random")
        if global_size != "random":
            if global_size == "small":
                length, width = 30, 15
            elif global_size == "medium":
                length, width = 40, 20
            else: # large
                length, width = 50, 25
        else:
            length = 30 + self.np_random.random() * 20
            width = 15 + self.np_random.random() * 10
            
        # Visual variety: per-car size and color (rainbow shades)
        # car_len = float(18 + self.np_random.random()*20)  # [18..38]
        rainbow = [
            (255, 0, 0), (255, 127, 0), (255, 255, 0),
            (0, 200, 0), (0, 120, 255), (75, 0, 130), (148, 0, 211)
        ]
        # Determine Color
        global_color = self.context.get("npc_color", "random")
        if global_color != "random":
            color_map = {
                "red": (200, 50, 50),
                "blue": (50, 50, 200),
                "green": (50, 200, 50),
                "yellow": (200, 200, 50),
                "white": (240, 240, 240),
                "black": (50, 50, 50)
            }
            color = color_map.get(global_color, (50, 50, 200))
        else:
            base = rainbow[int(self.np_random.integers(0, len(rainbow)))]
            shade = 0.7 + float(self.np_random.random())*0.3
            color = (int(base[0]*shade), int(base[1]*shade), int(base[2]*shade))
            
        # Determine Size
        global_size = self.context.get("npc_size", "random")
        if global_size != "random":
            if global_size == "small":
                length, width = 30, 15
            elif global_size == "medium":
                length, width = 40, 20
            else: # large
                length, width = 50, 25
        else:
            length = 30 + self.np_random.random() * 20
            width = 15 + self.np_random.random() * 10
            
        # Apply Driver Impatience
        impatience = self.context.get("driver_impatience", 0.5)
        impatience_speed_mod = 0.9 + (0.3 * impatience)
        
        npc_car = {
            "pos": spawn["pos"].copy().astype(np.float32),
            "heading": spawn["heading"],
            "velocity": np.array([0.0, 0.0], dtype=np.float32),
            "speed": self.npc_car_speed * (0.8 + self.np_random.random() * 0.4) * getattr(self, "npc_speed_factor", 1.0) * impatience_speed_mod,
            "target_speed": 0.0,
            "state": "driving",  # driving, stopped_at_light, slowing_for_yellow, turning
            "road_type": spawn["type"],
            "direction": spawn.get("direction", "east"),  # Store direction for traffic light checking
            "direction": spawn.get("direction", "east"),  # Store direction for traffic light checking
            "turn_decision": self.np_random.choice(["straight", "left", "right"], p=[0.6, 0.2, 0.2]),
            "has_turned": False,  # Track if car has completed turn at current intersection
            "lateral_offset": 0.0, # For pedestrian avoidance
            "length": length,
            "width": width,
            "color": color
        }
        
        self.npc_cars.append(npc_car)

    def _update_npc_cars(self):
        """Update NPC car positions and behaviors."""
        cars_to_remove = []
        
        for i, car in enumerate(self.npc_cars):
            # Initialize: assume front is clear, car should keep going
            # This prevents deadlocks - cars will continue unless obstacle detected
            car["target_speed"] = car["speed"]
            car["state"] = "driving"
            
            # Check if near intersection
            nearest_intersection = self._get_nearest_intersection(car["pos"])
            if nearest_intersection:
                # Determine direction of travel from heading
                # Use the car's stored direction if available, otherwise calculate
                if "direction" in car:
                    direction = car["direction"]
                else:
                    # Calculate from heading
                    if abs(np.cos(car["heading"])) > abs(np.sin(car["heading"])):
                        direction = "east" if np.cos(car["heading"]) > 0 else "west"
                    else:
                        direction = "south" if np.sin(car["heading"]) < 0 else "north"
                
                # Calculate distance to intersection edge along the direction of travel
                # Intersection edge is at intersection_size/2 from center
                # Stop BEFORE the intersection edge (before zebra crossing)
                intersection_pos = nearest_intersection["pos"]
                intersection_edge_dist = self.intersection_size // 2
                stopping_distance = 60  # Stop 60 pixels before intersection edge (before zebra crossing)
                
                # Check if car is inside the intersection
                dist_to_center = np.linalg.norm(car["pos"] - intersection_pos)
                is_inside_intersection = dist_to_center < intersection_edge_dist
                
                # Check if car is approaching the intersection along its direction
                approaching = False
                dist_to_edge = 0.0
                should_stop = False
                
                if direction == "north":  # Coming from south, going north
                    if car["pos"][1] > intersection_pos[1]:  # Car is south of intersection
                        dist_to_edge = car["pos"][1] - (intersection_pos[1] + intersection_edge_dist)
                        approaching = dist_to_edge < (stopping_distance + 30) and dist_to_edge > -intersection_edge_dist
                        should_stop = dist_to_edge < stopping_distance and dist_to_edge > 0
                elif direction == "south":  # Coming from north, going south
                    if car["pos"][1] < intersection_pos[1]:  # Car is north of intersection
                        dist_to_edge = (intersection_pos[1] - intersection_edge_dist) - car["pos"][1]
                        approaching = dist_to_edge < (stopping_distance + 30) and dist_to_edge > -intersection_edge_dist
                        should_stop = dist_to_edge < stopping_distance and dist_to_edge > 0
                elif direction == "east":  # Coming from west, going east
                    if car["pos"][0] < intersection_pos[0]:  # Car is west of intersection
                        dist_to_edge = (intersection_pos[0] - intersection_edge_dist) - car["pos"][0]
                        approaching = dist_to_edge < (stopping_distance + 30) and dist_to_edge > -intersection_edge_dist
                        should_stop = dist_to_edge < stopping_distance and dist_to_edge > 0
                elif direction == "west":  # Coming from east, going west
                    if car["pos"][0] > intersection_pos[0]:  # Car is east of intersection
                        dist_to_edge = car["pos"][0] - (intersection_pos[0] + intersection_edge_dist)
                        approaching = dist_to_edge < (stopping_distance + 30) and dist_to_edge > -intersection_edge_dist
                        should_stop = dist_to_edge < stopping_distance and dist_to_edge > 0
                
                # Decide on turn direction when approaching intersection (before entering)
                # Reset turn decision when far from intersection
                if dist_to_edge > stopping_distance + 100:
                    car["has_turned"] = False
                    car["turn_decision"] = None
                elif approaching and car["turn_decision"] is None and not car.get("has_turned", False):
                    # Randomly decide: straight (50%), left (25%), right (25%)
                    rand = self.np_random.random()
                    if rand < 0.5:
                        car["turn_decision"] = "straight"
                    elif rand < 0.75:
                        car["turn_decision"] = "left"
                    else:
                        car["turn_decision"] = "right"
                
                # Check traffic light for this direction
                light_state = self._get_traffic_light_state(nearest_intersection["grid"], direction)
                
                # IMPORTANT: If car is inside intersection, it should clear it regardless of light
                # Only stop for pedestrians, not for traffic lights
                if is_inside_intersection:
                    # Car is in the middle of intersection - must clear it
                    # Ignore traffic light, but still respect pedestrians (handled later)
                    car["target_speed"] = car["speed"]
                    car["state"] = "clearing_intersection"
                    
                    # Handle turning at intersection when inside
                    if car["turn_decision"] and not car.get("has_turned", False):
                        # Turn when inside intersection
                        self._execute_npc_turn(car, nearest_intersection, direction)
                        car["has_turned"] = True
                else:
                    # Car is outside intersection - normal traffic light behavior
                    # Stop at red or yellow light BEFORE intersection (before zebra crossing)
                    if should_stop and (light_state == "red" or light_state == "yellow"):
                        if light_state == "red":
                            car["target_speed"] = 0.0
                            car["state"] = "stopped_at_light"
                            # If car is too close to intersection, move it back slightly
                            if dist_to_edge < 10:
                                if direction == "north":
                                    car["pos"][1] = intersection_pos[1] + intersection_edge_dist + stopping_distance
                                elif direction == "south":
                                    car["pos"][1] = intersection_pos[1] - intersection_edge_dist - stopping_distance
                                elif direction == "east":
                                    car["pos"][0] = intersection_pos[0] - intersection_edge_dist - stopping_distance
                                elif direction == "west":
                                    car["pos"][0] = intersection_pos[0] + intersection_edge_dist + stopping_distance
                        else:  # yellow
                            car["target_speed"] = car["speed"] * 0.3  # Slow down for yellow
                            car["state"] = "slowing_for_yellow"
                    elif light_state == "green" or not should_stop:
                        # Green light or not at intersection - can drive
                        # Set initial target speed (pedestrian avoidance will adjust if needed)
                        car["target_speed"] = car["speed"]
                        car["state"] = "driving"
                        
                        # Handle turning at intersection when entering with green light
                        if car["turn_decision"] and not car.get("has_turned", False):
                            # Turn when close to intersection center (within intersection or just entering)
                            if dist_to_center < intersection_edge_dist + 20:  # Entering or inside intersection
                                self._execute_npc_turn(car, nearest_intersection, direction)
                                car["has_turned"] = True
                    else:
                        car["target_speed"] = car["speed"]
                        car["state"] = "driving"
            
            # Track obstacles in front to prevent deadlocks
            obstacle_in_front = False
            
            # HARD CONSTRAINT: Avoid collisions with other NPC cars
            # Cars only stop if they are FACING the other car (heading aligned with direction to obstacle)
            # Ignore cars to the side or behind - keep driving
            min_car_dist = float('inf')
            closest_car_ahead = None
            
            for other_car in self.npc_cars:
                if other_car is car:
                    continue
                
                # Calculate distance to other car
                dist = np.linalg.norm(car["pos"] - other_car["pos"])
                safe_distance = self.car_width + 15  # Minimum safe distance (car width + buffer)
                
                if dist < safe_distance + 30:  # Within detection range
                    # Calculate direction to other car
                    to_other = other_car["pos"] - car["pos"]
                    if np.linalg.norm(to_other) < 0.1:
                        continue  # Skip if cars are too close (handled by separation logic)
                    
                    to_other_normalized = to_other / np.linalg.norm(to_other)
                    
                    # Calculate car's forward direction vector
                    forward_vec = np.array([np.cos(car["heading"]), -np.sin(car["heading"])])
                    
                    # HARD CONSTRAINT: Check if car is FACING the other car
                    # Calculate angle between car's heading and direction to other car
                    # Dot product gives cosine of angle: 1 = same direction, 0 = perpendicular, -1 = opposite
                    cos_angle = np.dot(forward_vec, to_other_normalized)
                    
                    # Car is "facing" the obstacle if angle is small (within ~60 degrees)
                    # cos(60°) ≈ 0.5, so we check if cos_angle > 0.5 (angle < 60°)
                    # Also check that obstacle is ahead (dist_along > 0)
                    dist_along_path = np.dot(to_other, forward_vec)
                    is_facing = cos_angle > 0.5 and dist_along_path > 10  # Facing AND ahead
                    
                    if is_facing and dist < safe_distance + 40:
                        obstacle_in_front = True
                        if dist < min_car_dist:
                            min_car_dist = dist
                            closest_car_ahead = {
                                "dist": dist,
                                "dist_along": dist_along_path,
                                "cos_angle": cos_angle,
                                "other_pos": other_car["pos"]
                            }
            
            # React to closest car ahead to avoid collision
            # HARD CONSTRAINT: Maintain safe following distance - cars CANNOT hit each other
            if closest_car_ahead:
                dist = closest_car_ahead["dist"]
                dist_along = closest_car_ahead["dist_along"]
                safe_distance = self.car_width + 20  # Safe following distance
                
                # HARD CONSTRAINT: Prevent collision by maintaining safe distance
                if dist < safe_distance:  # Too close - stop immediately to prevent collision
                    car["target_speed"] = 0.0
                    # Emergency brake
                    car["velocity"][0] *= 0.7
                    car["velocity"][1] *= 0.7
                elif dist_along > 0 and dist < safe_distance + 15:  # Very close ahead - stop
                    car["target_speed"] = 0.0
                    car["velocity"][0] *= 0.85
                    car["velocity"][1] *= 0.85
                elif dist_along > 0 and dist < safe_distance + 25:  # Close ahead - slow significantly
                    # Match speed of car ahead or stop
                    car["target_speed"] = min(car["target_speed"], car["speed"] * 0.1)
                elif dist_along > 0 and dist < safe_distance + 40:  # Nearby ahead - reduce speed
                    car["target_speed"] = min(car["target_speed"], car["speed"] * 0.3)
            
            # Avoid the agent car (treat agent as an obstacle) - ensure omnidirectional margin too
            forward_vec = np.array([np.cos(car["heading"]), -np.sin(car["heading"])])
            vec_to_agent = self._agent_location - car["pos"]
            dist_to_agent = np.linalg.norm(vec_to_agent)
            if dist_to_agent > 1e-3:
                dir_to_agent = vec_to_agent / dist_to_agent
                cos_angle = float(np.dot(forward_vec, dir_to_agent))
                dist_along = float(np.dot(vec_to_agent, forward_vec))
                if cos_angle > 0.5 and dist_along > 10:
                    obstacle_in_front = True
                    if dist_to_agent < (self.car_width + 12):
                        car["target_speed"] = 0.0
                        car["velocity"][0] *= 0.8
                        car["velocity"][1] *= 0.8
                    elif dist_to_agent < (self.car_width + 40):
                        car["target_speed"] = min(car["target_speed"], car["speed"] * 0.15)
                    elif dist_to_agent < (self.car_width + 70):
                        car["target_speed"] = min(car["target_speed"], car["speed"] * 0.4)
                # Omnidirectional margin: stop if within buffer regardless of angle
                if dist_to_agent < (self.car_width + 8):
                    car["target_speed"] = 0.0
                    car["velocity"][0] *= 0.85
                    car["velocity"][1] *= 0.85
            
            # Pedestrian avoidance - give pedestrians space
            # HARD CONSTRAINT: Cars only stop if they are FACING the pedestrian (heading aligned)
            # For cars inside intersection: always check for pedestrians (clear intersection safely)
            # For cars outside intersection: only if not stopped at traffic light
            # Cars clearing intersection should still stop for pedestrians they are facing
            nearby_pedestrians = []  # Initialize before check
            if car["state"] != "stopped_at_light":  # Avoid pedestrians unless stopped at red light
                # Check for pedestrians in the car's vicinity
                forward_vec = np.array([np.cos(car["heading"]), -np.sin(car["heading"])])
                
                for ped in self.pedestrians:
                    # Calculate distance to pedestrian
                    dist = np.linalg.norm(car["pos"] - ped["pos"])
                    
                    # Check pedestrians within detection range (90 pixel radius)
                    if dist < 90 and dist > 0.1:  # Avoid division by zero
                        # Calculate direction to pedestrian
                        to_ped = ped["pos"] - car["pos"]
                        to_ped_normalized = to_ped / np.linalg.norm(to_ped)
                        
                        # HARD CONSTRAINT: Check if car is FACING the pedestrian
                        # Calculate angle between car's heading and direction to pedestrian
                        cos_angle = np.dot(forward_vec, to_ped_normalized)
                        dist_along_path = np.dot(to_ped, forward_vec)
                        
                        # Car is "facing" the pedestrian if angle is small (within ~60 degrees)
                        # cos(60°) ≈ 0.5, so we check if cos_angle > 0.5 (angle < 60°)
                        # Also check that pedestrian is ahead (dist_along > 10)
                        is_facing = cos_angle > 0.5 and dist_along_path > 10
                        
                        if is_facing:  # Only react if car is facing the pedestrian
                            obstacle_in_front = True
                            nearby_pedestrians.append({
                                "dist": dist,
                                "dist_along": dist_along_path,
                                "cos_angle": cos_angle
                            })
                
                # React to nearby pedestrians based on proximity and position
                # Only react to pedestrians IN FRONT - ignore side/behind pedestrians
                if nearby_pedestrians:
                    # Sort by distance
                    nearby_pedestrians.sort(key=lambda p: p["dist"])
                    closest = nearby_pedestrians[0]
                    
                    # All pedestrians in this list are already confirmed to be ahead (dist_along_path > 10)
                    # If pedestrian is very close ahead (within 35 pixels), stop completely
                    if closest["dist"] < 35:
                        car["target_speed"] = 0.0
                        # Emergency brake
                        car["velocity"][0] *= 0.8
                        car["velocity"][1] *= 0.8
                    # If pedestrian is close ahead (within 50 pixels ahead), slow down significantly
                    elif closest["dist_along"] < 50 and closest["dist"] < 70:
                        car["target_speed"] = min(car["target_speed"], car["speed"] * 0.3)  # Slow to 30% speed
                    # If pedestrian is nearby ahead (within 70 pixels ahead), reduce speed
                    elif closest["dist_along"] < 70 and closest["dist"] < 90:
                        car["target_speed"] = min(car["target_speed"], car["speed"] * 0.6)  # Slow to 60% speed
            
            # If no obstacles detected in front and not stopped at light, ensure car continues at cruise speed
            # This prevents deadlocks - cars will keep going when front is clear
            if (not obstacle_in_front and 
                car["state"] != "stopped_at_light" and 
                car["state"] != "slowing_for_yellow"):
                # No obstacles in front - continue at cruise speed
                car["target_speed"] = car["speed"]
                car["state"] = "driving"
            
            # Update velocity towards target speed (respecting collision avoidance above)
            current_speed = np.linalg.norm(car["velocity"])
            if current_speed < car["target_speed"]:
                # Accelerate (but respect target speed set by collision/pedestrian avoidance or traffic lights)
                accel = 0.3
                car["velocity"][0] += accel * np.cos(car["heading"])
                car["velocity"][1] -= accel * np.sin(car["heading"])
            elif current_speed > car["target_speed"]:
                # Decelerate to target speed
                brake = 0.9
                car["velocity"][0] *= brake
                car["velocity"][1] *= brake
            
            # Limit speed
            speed = np.linalg.norm(car["velocity"])
            if speed > car["speed"]:
                car["velocity"] = (car["velocity"] / speed) * car["speed"]
            
            # Align velocity with heading when moving (prevent sliding, like agent)
            if speed > 0.01:
                car["velocity"][0] = speed * np.cos(car["heading"])
                car["velocity"][1] = -speed * np.sin(car["heading"])
            
            # Apply friction
            car["velocity"][0] *= 0.99
            car["velocity"][1] *= 0.99
            
            # Update position
            car["pos"][0] += car["velocity"][0]
            car["pos"][1] += car["velocity"][1]
            
            # HARD CONSTRAINT: Final collision check - if cars are too close, separate them immediately
            for other_car in self.npc_cars:
                if other_car is car:
                    continue
                dist = np.linalg.norm(car["pos"] - other_car["pos"])
                min_separation = self.car_width + 2  # Minimum separation distance
                if dist < min_separation:  # Too close - separate cars immediately
                    # Calculate separation vector
                    separation_vec = car["pos"] - other_car["pos"]
                    if np.linalg.norm(separation_vec) < 0.1:
                        # Cars are exactly on top of each other - separate randomly
                        separation_vec = np.array([1.0, 0.0]) if self.np_random.random() > 0.5 else np.array([0.0, 1.0])
                    else:
                        separation_vec = separation_vec / np.linalg.norm(separation_vec)
                    
                    # Move cars apart to maintain minimum separation
                    move_dist = (min_separation - dist) / 2 + 1
                    car["pos"] += separation_vec * move_dist
                    other_car["pos"] -= separation_vec * move_dist
                    
                    # Stop both cars to prevent further collision
                    car["velocity"] = np.array([0.0, 0.0], dtype=np.float32)
                    car["target_speed"] = 0.0
                    other_car["velocity"] = np.array([0.0, 0.0], dtype=np.float32)
                    other_car["target_speed"] = 0.0
            
            # --- Pedestrian Avoidance (Lateral Offset) ---
            # Decay offset to return to center
            car["lateral_offset"] = car.get("lateral_offset", 0.0) * 0.9
            
            # Check for pedestrians to steer around
            for ped in self.pedestrians:
                vec_to_ped = ped["pos"] - car["pos"]
                dist_to_ped = float(np.linalg.norm(vec_to_ped))
                if dist_to_ped < 80.0:
                    # Project onto car's local frame
                    fwd = np.array([np.cos(car["heading"]), -np.sin(car["heading"])])
                    right = np.array([np.sin(car["heading"]), np.cos(car["heading"])])
                    
                    forward_dist = float(np.dot(vec_to_ped, fwd))
                    lateral_dist = float(np.dot(vec_to_ped, right))
                    
                    # If pedestrian is ahead and slightly to side
                    if 0 < forward_dist < 60.0 and abs(lateral_dist) < 20.0:
                        # Steer away
                        # If ped is right (lat > 0), steer left (offset -)
                        # If ped is left (lat < 0), steer right (offset +)
                        push = -2.0 if lateral_dist > 0 else 2.0
                        car["lateral_offset"] += push
            
            # Clamp offset
            car["lateral_offset"] = np.clip(car["lateral_offset"], -20.0, 20.0)

            # Clamp offset
            car["lateral_offset"] = np.clip(car["lateral_offset"], -20.0, 20.0)

            # --- Lane Keeping Control (Heading Adjustment) ---
            # Instead of forcing position, we adjust heading to steer towards the target lane position
            
            steering_correction = 0.0
            target_h = car["heading"]
            
            if car["road_type"] == "horizontal":
                # Find nearest road center
                nearest_road_y = min(self.horizontal_roads, 
                                    key=lambda r: abs(r["y"] - car["pos"][1]))["y"]
                direction = car.get("direction", "east")
                
                if direction == "east":
                    # Target Y = Road Center + Lane Offset (Right/South) + Avoidance Offset
                    base_y = nearest_road_y + self.road_width // 4
                    target_y = base_y + car["lateral_offset"]
                    base_heading = 0.0
                    
                    # P-Controller
                    error = target_y - car["pos"][1]
                    # Error > 0 (Need Down) -> Turn Right (Increase Heading)
                    steering_correction = np.clip(error * 0.01, -0.2, 0.2)
                    target_h = base_heading + steering_correction
                    
                else:  # west
                    # Target Y = Road Center - Lane Offset (Top/North) - Avoidance Offset
                    base_y = nearest_road_y - self.road_width // 4
                    target_y = base_y - car["lateral_offset"]
                    base_heading = np.pi
                    
                    # P-Controller
                    error = target_y - car["pos"][1]
                    # Error > 0 (Need Down) -> Turn Left (relative to West) -> Increase Heading
                    steering_correction = np.clip(error * 0.01, -0.2, 0.2)
                    target_h = base_heading + steering_correction
                    
            else:  # vertical
                # Find nearest road center
                nearest_road_x = min(self.vertical_roads,
                                    key=lambda r: abs(r["x"] - car["pos"][0]))["x"]
                direction = car.get("direction", "north")
                
                if direction == "north":
                    # Target X = Road Center + Lane Offset (Right/East) + Avoidance Offset
                    base_x = nearest_road_x + self.road_width // 4
                    target_x = base_x + car["lateral_offset"]
                    base_heading = np.pi / 2
                    
                    # P-Controller
                    error = target_x - car["pos"][0]
                    # Error > 0 (Need Right) -> Turn Right (relative to North) -> Decrease Heading
                    steering_correction = np.clip(error * 0.01, -0.2, 0.2)
                    target_h = base_heading - steering_correction
                    
                else:  # south
                    # Target X = Road Center - Lane Offset (Left/West) - Avoidance Offset
                    base_x = nearest_road_x - self.road_width // 4
                    target_x = base_x - car["lateral_offset"]
                    base_heading = -np.pi / 2
                    
                    # P-Controller
                    error = target_x - car["pos"][0]
                    # Error > 0 (Need Right) -> Turn Left (relative to South) -> Increase Heading
                    steering_correction = np.clip(error * 0.01, -0.2, 0.2)
                    target_h = base_heading + steering_correction

            # Apply Steering with Non-Holonomic Constraint
            # Only change heading if moving
            speed = np.linalg.norm(car["velocity"])
            if speed > 0.1:
                # Smoothly interpolate to target heading
                diff = target_h - car["heading"]
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
                # Limit turn rate based on speed
                max_turn = 0.05 * (speed / 5.0) + 0.01
                step = np.clip(diff, -max_turn, max_turn)
                car["heading"] += step
            
            # Remove if off screen
            if (car["pos"][0] < -200 or car["pos"][0] > self.map_size + 200 or
                car["pos"][1] < -200 or car["pos"][1] > self.map_size + 200):
                cars_to_remove.append(i)
        
        # Remove cars
        for i in sorted(cars_to_remove, reverse=True):
            self.npc_cars.pop(i)
        # Final collision resolver: only front/back for cars, front-only for pedestrians
        # NPC-to-NPC: only if aligned front/back
        for i in range(len(self.npc_cars)):
            for j in range(i + 1, len(self.npc_cars)):
                a = self.npc_cars[i]
                b = self.npc_cars[j]
                vec_ab = b["pos"] - a["pos"]
                d = np.linalg.norm(vec_ab)
                if d < 1e-6:
                    continue
                forward_a = np.array([np.cos(a["heading"]), -np.sin(a["heading"])])
                cos_angle = float(np.dot(forward_a, vec_ab / d))
                dist_along = float(np.dot(vec_ab, forward_a))
                # Only resolve if aligned front/back (not sides)
                if abs(cos_angle) > 0.5 and abs(dist_along) > 10:
                    la = float(a.get("length", self.car_length))
                    wa = float(a.get("width", self.car_width))
                    lb = float(b.get("length", self.car_length))
                    wb = float(b.get("width", self.car_width))
                    ra = max(la, wa) / 2.0
                    rb = max(lb, wb) / 2.0
                    min_sep = ra + rb + 4.0
                    if d < min_sep:
                        dir_vec = vec_ab / d
                        move = (min_sep - d) + 0.5
                        # Check if cars are stopped - don't push stopped cars
                        speed_a = np.linalg.norm(a["velocity"])
                        speed_b = np.linalg.norm(b["velocity"])
                        is_a_stopped = speed_a < 0.1 and a.get("target_speed", 0) < 0.1
                        is_b_stopped = speed_b < 0.1 and b.get("target_speed", 0) < 0.1
                        
                        if is_a_stopped and not is_b_stopped:
                            # Only move b (a is stopped, don't push it)
                            b["pos"] += dir_vec * move
                            b["target_speed"] = 0.0
                            b["velocity"] *= 0.8
                        elif is_b_stopped and not is_a_stopped:
                            # Only move a (b is stopped, don't push it)
                            a["pos"] -= dir_vec * move
                            a["target_speed"] = 0.0
                            a["velocity"] *= 0.8
                        elif not is_a_stopped and not is_b_stopped:
                            # Both moving - move both apart
                            a["pos"] -= dir_vec * (move / 2.0)
                            b["pos"] += dir_vec * (move / 2.0)
                            a["velocity"] *= 0.8
                            b["velocity"] *= 0.8
                            a["target_speed"] = 0.0
                            b["target_speed"] = 0.0
                        # If both stopped, don't move either (they stay stationary)
        # NPC-to-Agent: only if aligned front/back
        for car in self.npc_cars:
            vec_to_agent = self._agent_location - car["pos"]
            d = np.linalg.norm(vec_to_agent)
            if d < 1e-6:
                continue
            forward_car = np.array([np.cos(car["heading"]), -np.sin(car["heading"])])
            cos_angle = float(np.dot(forward_car, vec_to_agent / d))
            dist_along = float(np.dot(vec_to_agent, forward_car))
            # Only resolve if aligned front/back (not sides)
            if abs(cos_angle) > 0.5 and abs(dist_along) > 10:
                l = float(car.get("length", self.car_length))
                w = float(car.get("width", self.car_width))
                ra = max(l, w) / 2.0
                r_agent = max(self.car_length, self.car_width) / 2.0
                min_sep = ra + r_agent + 4.0
                if d < min_sep:
                    dir_vec = vec_to_agent / d
                    move = (min_sep - d) + 0.5
                    car["pos"] -= dir_vec * move
                    car["velocity"] *= 0.8
                    car["target_speed"] = 0.0
        # NPC-to-pedestrian: only if pedestrian is in front
        for car in self.npc_cars:
            forward_car = np.array([np.cos(car["heading"]), -np.sin(car["heading"])])
            for ped in self.pedestrians:
                vec_to_ped = ped["pos"] - car["pos"]
                d = np.linalg.norm(vec_to_ped)
                if d < 1e-6:
                    continue
                cos_angle = float(np.dot(forward_car, vec_to_ped / d))
                dist_along = float(np.dot(vec_to_ped, forward_car))
                # Only resolve if pedestrian is in FRONT (not side or back)
                if cos_angle > 0.5 and dist_along > 10:
                    l = float(car.get("length", self.car_length))
                    w = float(car.get("width", self.car_width))
                    ra = max(l, w) / 2.0
                    min_sep = ra + self.pedestrian_radius + 4.0
                    if d < min_sep:
                        dir_vec = vec_to_ped / d
                        move = (min_sep - d) + 0.5
                        car["pos"] -= dir_vec * move
                        car["velocity"] *= 0.8
                        car["target_speed"] = 0.0

    def _execute_npc_turn(self, car, intersection, current_direction):
        """Execute a turn for an NPC car at an intersection."""
        if not car["turn_decision"] or car["turn_decision"] == "straight":
            return  # No turn needed
        
        # Turn left or right based on decision
        if car["turn_decision"] == "left":
            # Turn left: -90 degrees
            if current_direction == "north":
                new_direction = "west"
                new_heading = np.pi
                new_road_type = "horizontal"
            elif current_direction == "south":
                new_direction = "east"
                new_heading = 0.0
                new_road_type = "horizontal"
            elif current_direction == "east":
                new_direction = "north"
                new_heading = np.pi / 2
                new_road_type = "vertical"
            else:  # west
                new_direction = "south"
                new_heading = -np.pi / 2
                new_road_type = "vertical"
        else:  # right
            # Turn right: +90 degrees
            if current_direction == "north":
                new_direction = "east"
                new_heading = 0.0
                new_road_type = "horizontal"
            elif current_direction == "south":
                new_direction = "west"
                new_heading = np.pi
                new_road_type = "horizontal"
            elif current_direction == "east":
                new_direction = "south"
                new_heading = -np.pi / 2
                new_road_type = "vertical"
            else:  # west
                new_direction = "north"
                new_heading = np.pi / 2
                new_road_type = "vertical"
        
        # Update car properties
        car["direction"] = new_direction
        car["heading"] = new_heading
        car["road_type"] = new_road_type
        # Align velocity with new heading
        speed = np.linalg.norm(car["velocity"])
        if speed > 0.01:
            car["velocity"][0] = speed * np.cos(new_heading)
            car["velocity"][1] = -speed * np.sin(new_heading)

    def _spawn_pedestrian(self):
        """Spawn a new pedestrian."""
        if len(self.pedestrians) >= self.context.get("num_pedestrians", 15):
            return
        
        # Determine if jaywalking
        is_jaywalking = self.np_random.random() < self.context.get("jaywalk_probability", 0.1)
        
        if is_jaywalking:
            # Jaywalkers spawn on streets (not at intersections) and cross directly across the street
            if len(self.jaywalker_spawn_points) == 0:
                return  # No jaywalker spawn points available
            spawn_idx = self.np_random.integers(0, len(self.jaywalker_spawn_points))
            spawn_pos = self.jaywalker_spawn_points[spawn_idx].copy()
            
            # Determine which road they're on by finding nearest road
            on_horizontal = False
            nearest_road = None
            min_dist = float('inf')
            
            for road in self.horizontal_roads:
                dist = abs(spawn_pos[1] - road["y"])
                if dist < min_dist:
                    min_dist = dist
                    nearest_road = road
                    on_horizontal = True
            
            for road in self.vertical_roads:
                dist = abs(spawn_pos[0] - road["x"])
                if dist < min_dist:
                    min_dist = dist
                    nearest_road = road
                    on_horizontal = False
            
            # Set target to opposite side of the road (directly across, perpendicular)
            target = spawn_pos.copy()
            if on_horizontal:
                # On horizontal road: cross vertically (perpendicular to road)
                # Determine which side of road they're on, cross to opposite
                road_y = nearest_road["y"]
                if spawn_pos[1] <= road_y:
                    # On top/north side, cross to bottom/south side
                    target[1] = road_y + self.road_width // 2 + 10
                else:
                    # On bottom/south side, cross to top/north side
                    target[1] = road_y - self.road_width // 2 - 10
                # Keep x the same (cross perpendicular)
            else:
                # On vertical road: cross horizontally (perpendicular to road)
                # Determine which side of road they're on, cross to opposite
                road_x = nearest_road["x"]
                if spawn_pos[0] <= road_x:
                    # On left/west side, cross to right/east side
                    target[0] = road_x + self.road_width // 2 + 10
                else:
                    # On right/east side, cross to left/west side
                    target[0] = road_x - self.road_width // 2 - 10
                # Keep y the same (cross perpendicular)
            
            pedestrian = {
                "pos": spawn_pos.astype(np.float32),
                "target": target.astype(np.float32),
                "speed": self.pedestrian_speed,
                "is_jaywalking": True,
                "phase": "direct",
                "intersection_grid": None,  # Jaywalkers not associated with intersection
                "waiting_for_light": False
            }
        else:
            # Regular pedestrians ALWAYS spawn at intersections and use zebra crossings ONLY
            if len(self.pedestrian_spawn_points) == 0:
                return
            spawn_idx = self.np_random.integers(0, len(self.pedestrian_spawn_points))
            spawn_pos = self.pedestrian_spawn_points[spawn_idx].copy()
            
            # Find nearest intersection
            nearest_intersection = self._get_nearest_intersection(spawn_pos)
            if not nearest_intersection:
                return
            
            # ALWAYS use a crossing - find the crossing closest to spawn point
            crossing = None
            min_crossing_dist = float('inf')
            for zc in self.zebra_crossings:
                if zc["intersection"] == nearest_intersection["grid"]:
                    # Calculate distance to crossing start
                    dist_to_start = np.linalg.norm(spawn_pos - zc["start"])
                    if dist_to_start < min_crossing_dist:
                        min_crossing_dist = dist_to_start
                        crossing = zc
            
            if not crossing:
                return  # Must have a crossing, don't spawn without one
            
            # Set target to crossing start (sidewalk to crossing)
            target = crossing["start"].copy()
            
            pedestrian = {
                "pos": spawn_pos.astype(np.float32),
                "target": target.astype(np.float32),
                "speed": self.pedestrian_speed,
                "is_jaywalking": False,
                "phase": "to_crossing_start",
                "intersection_grid": nearest_intersection["grid"],
                "waiting_for_light": False,
                "has_started_crossing": False,  # Track if pedestrian has started crossing
                "crossing": None  # Will be set when reaching crossing
            }
        
        self.pedestrians.append(pedestrian)

    def _update_pedestrians(self):
        """Update pedestrian positions."""
        pedestrians_to_remove = []
        
        for i, ped in enumerate(self.pedestrians):
            # Check traffic light if near intersection (only for pedestrians using crossings)
            if not ped["is_jaywalking"] and ped["intersection_grid"] in self.traffic_lights:
                # HARD CONSTRAINT: If pedestrian has started crossing, they continue regardless of light change
                # Once on the crossing or moving to destination, they always continue
                if ped.get("phase") in ["on_crossing", "to_destination"]:
                    # Pedestrian has started crossing - continue even if light changes to green
                    ped["waiting_for_light"] = False
                    # Mark that they've started crossing so they can always continue
                    ped["has_started_crossing"] = True
                elif ped.get("has_started_crossing", False):
                    # Even if phase changed but they've started crossing, continue
                    ped["waiting_for_light"] = False
                else:
                    # Not yet started crossing - check if it's safe to cross
                    dist_to_intersection = np.linalg.norm(ped["pos"] - self.intersections[
                        next(j for j, inter in enumerate(self.intersections) 
                             if inter["grid"] == ped["intersection_grid"])
                    ]["pos"])
                    
                    light = self.traffic_lights[ped["intersection_grid"]]
                    
                    # Pedestrians cross when perpendicular traffic is RED (cars stopped)
                    # Logic: When cars perpendicular to crossing direction have RED light, pedestrians can cross
                    # Phase 0: NS green (NS cars GO), EW red (EW cars STOP) → Pedestrians crossing EW can GO
                    # Phase 1: NS yellow (NS cars slowing), EW red (EW cars STOP) → Pedestrians crossing EW can GO
                    # Phase 2: NS red (NS cars STOP), EW green (EW cars GO) → Pedestrians crossing NS can GO
                    # Phase 3: NS red (NS cars STOP), EW yellow (EW cars slowing) → Pedestrians crossing NS can GO
                    
                    if dist_to_intersection < 100:  # Near intersection
                        # Determine which direction pedestrian wants to cross
                        to_crossing = ped.get("target", ped["pos"]) - ped["pos"]
                        if abs(to_crossing[0]) > abs(to_crossing[1]):  # Horizontal crossing (EW)
                            # Crossing east-west: check if NS traffic is RED (then EW cars stopped, safe to cross)
                            # NS traffic is RED in phases 2,3
                            if light["phase"] in [2, 3]:  # NS red → EW cars stopped → can cross EW
                                ped["waiting_for_light"] = False
                                # Mark that they're starting to cross
                                ped["has_started_crossing"] = True
                            else:  # NS green/yellow → NS cars moving, wait
                                ped["waiting_for_light"] = True
                                continue
                        else:  # Vertical crossing (NS)
                            # Crossing north-south: check if EW traffic is RED (then NS cars stopped, safe to cross)
                            # EW traffic is RED in phases 0,1
                            if light["phase"] in [0, 1]:  # EW red → NS cars stopped → can cross NS
                                ped["waiting_for_light"] = False
                                # Mark that they're starting to cross
                                ped["has_started_crossing"] = True
                            else:  # EW green/yellow → EW cars moving, wait
                                ped["waiting_for_light"] = True
                                continue
                    else:
                        ped["waiting_for_light"] = False
            else:
                ped["waiting_for_light"] = False
            
            # Only move if not waiting for light (only for zebra crossing users who haven't started)
            if not ped["is_jaywalking"] and ped.get("waiting_for_light", False):
                continue  # Skip movement this step
            
            # Handle zebra crossing users - they MUST walk only on the crossing
            if not ped["is_jaywalking"]:
                # Check if pedestrian should be on crossing
                if ped.get("phase") == "on_crossing":
                    # MUST be on crossing - ensure crossing is set
                    if "crossing" not in ped or ped["crossing"] is None:
                        # Find the crossing they should be on
                        best_crossing = None
                        min_dist = float('inf')
                        for zc in self.zebra_crossings:
                            if zc["intersection"] == ped["intersection_grid"]:
                                dist_to_start = np.linalg.norm(ped["pos"] - zc["start"])
                                if dist_to_start < min_dist:
                                    min_dist = dist_to_start
                                    best_crossing = zc
                        if best_crossing:
                            ped["crossing"] = best_crossing
                            ped["target"] = best_crossing["end"].copy()
                    
                    # HARD CONSTRAINT: Move ONLY along the crossing line (never leave it)
                    if "crossing" in ped and ped["crossing"] is not None:
                        crossing = ped["crossing"]
                        
                        # Calculate movement along the crossing line
                        direction_to_end = crossing["end"] - ped["pos"]
                        distance_to_end = np.linalg.norm(direction_to_end)
                        
                        # HARD CONSTRAINT: Force position to stay exactly on crossing line
                        if crossing["direction"] == "horizontal":
                            # Horizontal crossing: MUST keep y fixed to crossing line
                            ped["pos"][1] = crossing["start"][1]
                            # Only move in x direction
                            if distance_to_end < ped["speed"]:
                                # Reached end, move to exact end position
                                ped["pos"][0] = crossing["end"][0]
                                ped["phase"] = "to_destination"
                            else:
                                # Move along crossing line in x direction only
                                step_size = min(ped["speed"], distance_to_end)
                                if direction_to_end[0] > 0:
                                    ped["pos"][0] += step_size
                                else:
                                    ped["pos"][0] -= step_size
                                # HARD CONSTRAINT: Ensure y stays on line (in case of any drift)
                                ped["pos"][1] = crossing["start"][1]
                        else:
                            # Vertical crossing: MUST keep x fixed to crossing line
                            ped["pos"][0] = crossing["start"][0]
                            # Only move in y direction
                            if distance_to_end < ped["speed"]:
                                # Reached end, move to exact end position
                                ped["pos"][1] = crossing["end"][1]
                                ped["phase"] = "to_destination"
                            else:
                                # Move along crossing line in y direction only
                                step_size = min(ped["speed"], distance_to_end)
                                if direction_to_end[1] > 0:
                                    ped["pos"][1] += step_size
                                else:
                                    ped["pos"][1] -= step_size
                                # HARD CONSTRAINT: Ensure x stays on line (in case of any drift)
                                ped["pos"][0] = crossing["start"][0]
                        
                        # Update destination if phase changed
                        if ped.get("phase") == "to_destination":
                            # Set target to opposite sidewalk (corner)
                            intersection_pos = self.intersections[
                                next(j for j, inter in enumerate(self.intersections)
                                     if inter["grid"] == ped["intersection_grid"])
                            ]["pos"]
                            # Find opposite corner
                            corner_offset = self.intersection_size // 2 + 30
                            if crossing["direction"] == "horizontal":
                                # Horizontal crossing - target is opposite corner (north or south)
                                if ped["pos"][1] < intersection_pos[1]:
                                    # Currently on north side, go to south corner
                                    ped["target"] = np.array([intersection_pos[0] - corner_offset, intersection_pos[1] + corner_offset])
                                else:
                                    # Currently on south side, go to north corner
                                    ped["target"] = np.array([intersection_pos[0] - corner_offset, intersection_pos[1] - corner_offset])
                            else:
                                # Vertical crossing - target is opposite corner (east or west)
                                if ped["pos"][0] < intersection_pos[0]:
                                    # Currently on west side, go to east corner
                                    ped["target"] = np.array([intersection_pos[0] + corner_offset, intersection_pos[1] - corner_offset])
                                else:
                                    # Currently on east side, go to west corner
                                    ped["target"] = np.array([intersection_pos[0] - corner_offset, intersection_pos[1] - corner_offset])
                        
                        continue  # Skip normal movement logic
                
                # Handle transition to crossing phase
                if ped.get("phase") == "to_crossing_start":
                    # Find the crossing closest to current position
                    min_dist = float('inf')
                    best_crossing = None
                    for zc in self.zebra_crossings:
                        if zc["intersection"] == ped["intersection_grid"]:
                            dist_to_start = np.linalg.norm(ped["pos"] - zc["start"])
                            if dist_to_start < min_dist:
                                min_dist = dist_to_start
                                best_crossing = zc
                    
                    # Move towards crossing start
                    direction = ped["target"] - ped["pos"]
                    distance = np.linalg.norm(direction)
                    
                    if distance < ped["speed"] or (best_crossing and min_dist < 15):
                        # Reached crossing start - transition to on_crossing
                        ped["phase"] = "on_crossing"
                        if best_crossing:
                            ped["crossing"] = best_crossing
                            ped["target"] = best_crossing["end"].copy()
                            # Move to exact crossing start position
                            if best_crossing["direction"] == "horizontal":
                                ped["pos"][1] = best_crossing["start"][1]
                                # Position on crossing start (closest point on line)
                                ped["pos"][0] = best_crossing["start"][0]
                            else:
                                ped["pos"][0] = best_crossing["start"][0]
                                # Position on crossing start (closest point on line)
                                ped["pos"][1] = best_crossing["start"][1]
                    else:
                        # Move towards crossing start (normal movement)
                        direction_normalized = direction / distance
                        ped["pos"] += direction_normalized * ped["speed"]
                    continue
                
                # Handle to_destination phase (from crossing to corner)
                if ped.get("phase") == "to_destination":
                    direction = ped["target"] - ped["pos"]
                    distance = np.linalg.norm(direction)
                    
                    if distance < ped["speed"]:
                        pedestrians_to_remove.append(i)
                    else:
                        direction_normalized = direction / distance
                        ped["pos"] += direction_normalized * ped["speed"]
                    continue
            
            # Handle jaywalkers - cross directly across road
            if ped["is_jaywalking"]:
                direction = ped["target"] - ped["pos"]
                distance = np.linalg.norm(direction)
                
                # Ensure perpendicular movement only
                abs_dx = abs(direction[0])
                abs_dy = abs(direction[1])
                if abs_dx > 0.1 and abs_dy > 0.1:
                    # Diagonal movement detected - prioritize the larger component
                    if abs_dx > abs_dy:
                        direction[1] = 0  # Keep only horizontal movement
                    else:
                        direction[0] = 0  # Keep only vertical movement
                    distance = np.linalg.norm(direction)
                
                if distance < ped["speed"]:
                    pedestrians_to_remove.append(i)
                else:
                    direction_normalized = direction / distance
                    ped["pos"] += direction_normalized * ped["speed"]
            
            # Remove if off screen
            if (ped["pos"][0] < -100 or ped["pos"][0] > self.map_size + 100 or
                ped["pos"][1] < -100 or ped["pos"][1] > self.map_size + 100):
                pedestrians_to_remove.append(i)
        
        for i in sorted(pedestrians_to_remove, reverse=True):
            self.pedestrians.pop(i)

    def _check_collision(self, pos1, radius1, pos2, radius2):
        """Check collision between two circular objects."""
        distance = np.linalg.norm(pos1 - pos2)
        return distance < (radius1 + radius2)

    def step(self, action):
        """Execute one step in the environment."""
        # --- Agent Physics Update (continuous control) ---
        # Action expected as [accel in [-1,1], steer in [-1,1]]
        action = np.asarray(action, dtype=np.float32)
        if action.shape == ():
            # compatibility
            action = np.array([0.0, 0.0], dtype=np.float32)
        accel_cmd = float(np.clip(action[0], -1.0, 1.0))
        steer_cmd = float(np.clip(action[1], -1.0, 1.0))

        # Acceleration/braking along heading
        # If braking (accel_cmd < 0), only brake if moving forward (no reverse)
        current_speed = np.linalg.norm(self._agent_velocity)
        if accel_cmd < 0 and current_speed > 0.01:
            # Braking: apply negative acceleration but don't allow reverse
            # Scale braking efficiency by roughness (icy = less braking power)
            # Roughness 0.0 (-10C) -> 40% braking power
            # Roughness 1.0 (30C) -> 100% braking power
            braking_efficiency = 0.4 + (0.6 * self.context.get("roughness", 1.0))
            accel = self.acceleration * accel_cmd * braking_efficiency
            
            self._agent_velocity[0] += accel * np.cos(self._agent_heading)
            self._agent_velocity[1] += accel * -np.sin(self._agent_heading)
            # Clamp to prevent reverse (velocity should not go negative along heading)
            speed_after_brake = np.linalg.norm(self._agent_velocity)
            forward_vec = np.array([np.cos(self._agent_heading), -np.sin(self._agent_heading)])
            vel_along_heading = np.dot(self._agent_velocity, forward_vec)
            if vel_along_heading < 0:
                # Prevent reverse - set velocity to zero
                self._agent_velocity = np.array([0.0, 0.0], dtype=np.float32)
        elif accel_cmd > 0:
            # Accelerating forward
            accel = self.acceleration * accel_cmd
            self._agent_velocity[0] += accel * np.cos(self._agent_heading)
            self._agent_velocity[1] += accel * -np.sin(self._agent_heading)

        # Steering modifies heading
        self._agent_heading += self.angular_velocity * steer_cmd

        # Align velocity with heading when moving (no-slide)
        speed = np.linalg.norm(self._agent_velocity)
        if speed > 0.01:
            self._agent_velocity[0] = speed * np.cos(self._agent_heading)
            self._agent_velocity[1] = -speed * np.sin(self._agent_heading)
        
        # Apply friction
        speed = np.linalg.norm(self._agent_velocity)
        if speed > 0.01:
            friction = self.context.get("friction", 0.98)
            self._agent_velocity[0] *= friction
            self._agent_velocity[1] *= friction
            
            new_speed = np.linalg.norm(self._agent_velocity)
            if new_speed < 0.01:
                self._agent_velocity = np.array([0.0, 0.0], dtype=np.float32)
            elif new_speed > self.max_speed:
                self._agent_velocity = (self._agent_velocity / new_speed) * self.max_speed
        
        # Update position
        self._agent_location[0] += self._agent_velocity[0]
        self._agent_location[1] += self._agent_velocity[1]
        
        # --- Update Systems ---
        # Update traffic lights
        self._update_traffic_lights()
        
        # Spawn NPC cars
        if self.np_random.random() < self.context.get("npc_car_spawn_rate", 0.02):
            self._spawn_npc_car()
        
        # Update NPC cars
        self._update_npc_cars()
        
        # Spawn pedestrians
        if self.np_random.random() < self.context.get("pedestrian_spawn_rate", 0.03):
            self._spawn_pedestrian()
        
        # Update pedestrians
        self._update_pedestrians()
        
        # --- Compute Rewards & Termination ---
        terminated = False
        reward = 0.0
        
        # Determine if agent left the map
        off_screen = (
            self._agent_location[0] < 0 or self._agent_location[0] > self.map_size or
            self._agent_location[1] < 0 or self._agent_location[1] > self.map_size
        )
        
        # Check for pedestrian collision first
        collision_with_pedestrian = False
        for ped in self.pedestrians:
            if self._check_collision(self._agent_location, self.car_width / 2, 
                                     ped["pos"], self.pedestrian_radius):
                collision_with_pedestrian = True
                break
        
        # --- Reward Structure (Standardized) ---
        # 1. Time penalty (encourage efficiency)
        reward -= 0.05
        
        # 2. Speed Reward
        speed = np.linalg.norm(self._agent_velocity)
        if speed > 0.1:
            reward += 0.01 * speed
    
        # 3. Red Light Penalty
        if self._agent_runs_red_light():
            reward -= 0.5
            
        # 4. Termination Conditions
        if collision_with_pedestrian:
            terminated = True
            reward = -100.0  # Collision (Car/Pedestrian)
        elif off_screen:
            terminated = True
            
            # Check if exit was valid (on road)
            # Extended env has multiple roads. Check if agent is within any road's width.
            valid_exit = False
            x, y = self._agent_location
            
            # Check horizontal roads
            for road in self.horizontal_roads:
                # Check if y is within road width
                if abs(y - road["y"]) <= self.road_width / 2 + 5: # +5 tolerance
                    valid_exit = True
                    break
                    
            # Check vertical roads
            if not valid_exit:
                for road in self.vertical_roads:
                    # Check if x is within road width
                    if abs(x - road["x"]) <= self.road_width / 2 + 5: # +5 tolerance
                        valid_exit = True
                        break
            
            if valid_exit:
                reward = 100.0  # Success (Valid Exit)
            else:
                reward = -10.0 # Off-Road Exit
            
        observation = self._get_obs()
        info = {
            "num_pedestrians": len(self.pedestrians),
            "num_npc_cars": len(self.npc_cars),
            "collision_with_pedestrian": collision_with_pedestrian,
            "agent_speed": float(np.linalg.norm(self._agent_velocity)),
            "causal_vars": {
                "temperature": self.temperature,
                "traffic_density": self.traffic_density,
                "pedestrian_density": self.pedestrian_density,
                "driver_impatience": self.driver_impatience,
                "roughness": self.roughness
            }
        }
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, False, info

    def _agent_runs_red_light(self):
        """Detect if agent is crossing an intersection when its directional light is red."""
        inter = self._get_nearest_intersection(self._agent_location)
        if inter is None:
            return False
        pos = inter["pos"]
        edge = self.intersection_size / 2.0
        # inside or crossing near the edge
        near = abs(self._agent_location[0] - pos[0]) < edge + 10 and abs(self._agent_location[1] - pos[1]) < edge + 10
        if not near:
            return False
        direction = self._agent_travel_direction()
        state = self._get_traffic_light_state(inter["grid"], direction)
        return state == "red"

    def _agent_on_active_crossing(self):
        """Check if agent is on a zebra crossing that currently has pedestrians on it."""
        for zc in self.zebra_crossings:
            if zc["direction"] == "horizontal":
                y = zc["start"][1]
                x0, x1 = min(zc["start"][0], zc["end"][0]), max(zc["start"][0], zc["end"][0])
                if abs(self._agent_location[1] - y) < 6 and x0 - 5 <= self._agent_location[0] <= x1 + 5:
                    # any pedestrian on this crossing?
                    for ped in self.pedestrians:
                        if not ped.get("is_jaywalking", False) and ped.get("phase") == "on_crossing" and ped.get("crossing") is not None:
                            if ped["crossing"] is zc:
                                return True
            else:
                x = zc["start"][0]
                y0, y1 = min(zc["start"][1], zc["end"][1]), max(zc["start"][1], zc["end"][1])
                if abs(self._agent_location[0] - x) < 6 and y0 - 5 <= self._agent_location[1] <= y1 + 5:
                    for ped in self.pedestrians:
                        if not ped.get("is_jaywalking", False) and ped.get("phase") == "on_crossing" and ped.get("crossing") is not None:
                            if ped["crossing"] is zc:
                                return True
        return False

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """Render a single frame using pygame."""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        # Update view offset to follow agent (centered view)
        view_center = np.array([self.window_size / 2, self.window_size / 2])
        self.view_offset = self._agent_location - view_center
        # Clip to map bounds
        self.view_offset[0] = np.clip(self.view_offset[0], 0, max(0, self.map_size - self.window_size))
        self.view_offset[1] = np.clip(self.view_offset[1], 0, max(0, self.map_size - self.window_size))
        
        # Create canvas (full map size)
        canvas = pygame.Surface((self.map_size, self.map_size))
        canvas.fill((100, 100, 100))  # Darker gray background
        
        # Draw roads
        for road in self.horizontal_roads:
            pygame.draw.rect(canvas, (60, 60, 60), 
                           pygame.Rect(0, road["y"] - self.road_width // 2, 
                                      self.map_size, self.road_width))
        
        for road in self.vertical_roads:
            pygame.draw.rect(canvas, (60, 60, 60),
                           pygame.Rect(road["x"] - self.road_width // 2, 0,
                                      self.road_width, self.map_size))
                                      
        # Draw dashed center lines
        # Horizontal roads
        for road in self.horizontal_roads:
            y = road["y"]
            for x in range(0, self.map_size, 40):
                pygame.draw.line(canvas, (255, 255, 255), (x, y), (x + 20, y), 2)
                
        # Vertical roads
        for road in self.vertical_roads:
            x = road["x"]
            for y in range(0, self.map_size, 40):
                pygame.draw.line(canvas, (255, 255, 255), (x, y), (x, y + 20), 2)
        
        # Draw intersections
        for intersection in self.intersections:
            pos = intersection["pos"]
            grid = intersection["grid"]
            light = self.traffic_lights[grid]
            
            # Draw intersection area
            pygame.draw.rect(canvas, (50, 50, 50),
                           pygame.Rect(pos[0] - self.intersection_size // 2,
                                      pos[1] - self.intersection_size // 2,
                                      self.intersection_size, self.intersection_size))
            
            # Draw 4 directional traffic lights (one for each direction)
            light_offset = self.intersection_size // 2 - 15
            light_size = 6
            
            # North light (top)
            north_state = light["directions"]["north"]
            north_color = (0, 255, 0) if north_state == "green" else (255, 255, 0) if north_state == "yellow" else (255, 0, 0)
            pygame.draw.circle(canvas, north_color, (int(pos[0]), int(pos[1] - light_offset)), light_size)
            
            # South light (bottom)
            south_state = light["directions"]["south"]
            south_color = (0, 255, 0) if south_state == "green" else (255, 255, 0) if south_state == "yellow" else (255, 0, 0)
            pygame.draw.circle(canvas, south_color, (int(pos[0]), int(pos[1] + light_offset)), light_size)
            
            # East light (right)
            east_state = light["directions"]["east"]
            east_color = (0, 255, 0) if east_state == "green" else (255, 255, 0) if east_state == "yellow" else (255, 0, 0)
            pygame.draw.circle(canvas, east_color, (int(pos[0] + light_offset), int(pos[1])), light_size)
            
            # West light (left)
            west_state = light["directions"]["west"]
            west_color = (0, 255, 0) if west_state == "green" else (255, 255, 0) if west_state == "yellow" else (255, 0, 0)
            pygame.draw.circle(canvas, west_color, (int(pos[0] - light_offset), int(pos[1])), light_size)
        
        # Draw zebra crossings
        stripe_width = 10
        stripe_spacing = 5
        for crossing in self.zebra_crossings:
            if crossing["direction"] == "horizontal":
                start_x = int(crossing["start"][0])
                end_x = int(crossing["end"][0])
                y = int(crossing["start"][1])
                x = start_x
                while x < end_x:
                    pygame.draw.rect(canvas, (255, 255, 255),
                                   pygame.Rect(x, y - 4, stripe_width, 8))
                    x += stripe_width + stripe_spacing
            else:
                start_y = int(crossing["start"][1])
                end_y = int(crossing["end"][1])
                x = int(crossing["start"][0])
                y = start_y
                while y < end_y:
                    pygame.draw.rect(canvas, (255, 255, 255),
                                   pygame.Rect(x - 4, y, 8, stripe_width))
                    y += stripe_width + stripe_spacing
        
        # Draw NPC cars (rainbow shades)
        for npc_car in self.npc_cars:
            car_pos = npc_car["pos"].astype(int)
            cos_h = np.cos(npc_car["heading"])
            sin_h = np.sin(npc_car["heading"])
            
            length = float(npc_car.get("length", self.car_length))
            width = float(npc_car.get("width", self.car_width))
            corners = np.array([
                [-length / 2, -width / 2],
                [ length / 2, -width / 2],
                [ length / 2,  width / 2],
                [-length / 2,  width / 2]
            ])
            
            rotation_matrix = np.array([
                [cos_h, sin_h],
                [-sin_h, cos_h]
            ])
            
            rotated_corners = (rotation_matrix @ corners.T).T + car_pos
            color = npc_car.get("color", (0, 100, 255))
            pygame.draw.polygon(canvas, color, rotated_corners)
        
        # Draw pedestrians
        for ped in self.pedestrians:
            ped_pos = ped["pos"].astype(int)
            # Blue for zebra crossing users, Red for jaywalkers
            if ped["is_jaywalking"]:
                color = (255, 100, 100)  # Red for jaywalkers
            else:
                color = (100, 100, 255)  # Blue for zebra crossing users
                if ped.get("waiting_for_light", False):
                    color = (150, 150, 255)  # Lighter blue when waiting at light
            pygame.draw.circle(canvas, color, ped_pos, self.pedestrian_radius)
        
        # Draw agent (red car)
        agent_pos = self._agent_location.astype(int)
        cos_h = np.cos(self._agent_heading)
        sin_h = np.sin(self._agent_heading)
        
        corners = np.array([
            [-self.car_length / 2, -self.car_width / 2],
            [self.car_length / 2, -self.car_width / 2],
            [self.car_length / 2, self.car_width / 2],
            [-self.car_length / 2, self.car_width / 2]
        ])
        
        rotation_matrix = np.array([
            [cos_h, sin_h],
            [-sin_h, cos_h]
        ])
        
        rotated_corners = (rotation_matrix @ corners.T).T + agent_pos
        pygame.draw.polygon(canvas, (255, 0, 0), rotated_corners)
        front_offset = np.array([self.car_length / 2 * cos_h, -self.car_length / 2 * sin_h])
        front_pos = agent_pos + front_offset
        pygame.draw.circle(canvas, (200, 0, 0), front_pos.astype(int), 3)
        
        if self.render_mode == "human":
            # Draw only the visible portion (centered on agent)
            view_rect = pygame.Rect(self.view_offset[0], self.view_offset[1], 
                                   self.window_size, self.window_size)
            self.window.blit(canvas, (0, 0), view_rect)
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        """Clean up resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _render_frame(self):
        """Render a single frame using pygame."""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Create canvas (full map size)
        canvas = pygame.Surface((self.map_size, self.map_size))
        canvas.fill((128, 128, 128))  # Gray background
        
        # Draw roads
        for road in self.horizontal_roads:
            pygame.draw.rect(canvas, (64, 64, 64), 
                           pygame.Rect(road["x_start"], road["y"] - self.road_width/2, 
                                     road["x_end"] - road["x_start"], self.road_width))
        
        for road in self.vertical_roads:
            pygame.draw.rect(canvas, (64, 64, 64), 
                           pygame.Rect(road["x"] - self.road_width/2, road["y_start"], 
                                     self.road_width, road["y_end"] - road["y_start"]))
            
        # Draw intersections
        for inter in self.intersections:
            pos = inter["pos"]
            pygame.draw.rect(canvas, (80, 80, 80),
                           pygame.Rect(pos[0]-self.intersection_size/2, pos[1]-self.intersection_size/2,
                                       self.intersection_size, self.intersection_size))
            
            # Draw traffic lights
            light = self.traffic_lights[inter["grid"]]
            light_offset = self.intersection_size//2 - 15
            size = 6
            
            # Draw lights for all directions
            # North
            color = (0,255,0) if light["directions"]["north"]=="green" else (255,255,0) if light["directions"]["north"]=="yellow" else (255,0,0)
            pygame.draw.circle(canvas, color, (int(pos[0]), int(pos[1]-light_offset)), size)
            # South
            color = (0,255,0) if light["directions"]["south"]=="green" else (255,255,0) if light["directions"]["south"]=="yellow" else (255,0,0)
            pygame.draw.circle(canvas, color, (int(pos[0]), int(pos[1]+light_offset)), size)
            # East
            color = (0,255,0) if light["directions"]["east"]=="green" else (255,255,0) if light["directions"]["east"]=="yellow" else (255,0,0)
            pygame.draw.circle(canvas, color, (int(pos[0]+light_offset), int(pos[1])), size)
            # West
            color = (0,255,0) if light["directions"]["west"]=="green" else (255,255,0) if light["directions"]["west"]=="yellow" else (255,0,0)
            pygame.draw.circle(canvas, color, (int(pos[0]-light_offset), int(pos[1])), size)

        # Draw zebra crossings
        stripe_width = 10
        stripe_spacing = 6
        for crossing in self.zebra_crossings:
            if crossing["direction"] == "horizontal":
                start_x = int(crossing["start"][0])
                end_x = int(crossing["end"][0])
                y = int(crossing["start"][1])
                x = start_x
                while x < end_x:
                    pygame.draw.rect(canvas, (255, 255, 255), 
                                   pygame.Rect(x, y - 4, stripe_width, 8))
                    x += stripe_width + stripe_spacing
            else:
                start_y = int(crossing["start"][1])
                end_y = int(crossing["end"][1])
                x = int(crossing["start"][0])
                y = start_y
                while y < end_y:
                    pygame.draw.rect(canvas, (255, 255, 255), 
                                   pygame.Rect(x - 4, y, 8, stripe_width))
                    y += stripe_width + stripe_spacing

        # Draw NPC cars
        for car in self.npc_cars:
            car_pos = car["pos"].astype(int)
            cos_h = np.cos(car["heading"])
            sin_h = np.sin(car["heading"])
            length = car.get("length", self.car_length)
            width = car.get("width", self.car_width)
            corners = np.array([[-length / 2, -width / 2],
                                [ length / 2, -width / 2],
                                [ length / 2,  width / 2],
                                [-length / 2,  width / 2]])
            rotation_matrix = np.array([[cos_h, sin_h],[-sin_h, cos_h]])
            rotated_corners = (rotation_matrix @ corners.T).T + car_pos
            color = car.get("color", (0, 100, 255))
            pygame.draw.polygon(canvas, color, rotated_corners)

        # Draw pedestrians
        for ped in self.pedestrians:
            ped_pos = ped["pos"].astype(int)
            color = (255, 100, 100) if ped.get("is_jaywalking", False) else (100, 100, 255)
            pygame.draw.circle(canvas, color, ped_pos, self.pedestrian_radius)
            # Direction indicator
            if "target" in ped:
                direction = ped["target"] - ped["pos"]
                if np.linalg.norm(direction) > 0:
                    direction_normalized = direction / np.linalg.norm(direction)
                    front_pos = ped_pos + (direction_normalized * self.pedestrian_radius * 1.5).astype(int)
                    pygame.draw.circle(canvas, (255, 255, 255), front_pos, 2)

        # Draw Agent
        agent_pos = self._agent_location.astype(int)
        cos_h = np.cos(self._agent_heading)
        sin_h = np.sin(self._agent_heading)
        corners = np.array([
            [-self.car_length / 2, -self.car_width / 2],
            [self.car_length / 2, -self.car_width / 2],
            [self.car_length / 2, self.car_width / 2],
            [-self.car_length / 2, self.car_width / 2]
        ])
        rotation_matrix = np.array([[cos_h, sin_h], [-sin_h, cos_h]])
        rotated_corners = (rotation_matrix @ corners.T).T + agent_pos
        pygame.draw.polygon(canvas, (255, 0, 0), rotated_corners)
        front_offset = np.array([self.car_length / 2 * cos_h, -self.car_length / 2 * sin_h])
        front_pos = agent_pos + front_offset
        pygame.draw.circle(canvas, (200, 0, 0), front_pos.astype(int), 3)

        # Crop to view window (camera follows agent)
        # Update view offset to keep agent centered
        target_offset = self._agent_location - np.array([self.window_size / 2, self.window_size / 2])
        # Smooth camera movement
        self.view_offset += (target_offset - self.view_offset) * 0.1
        # Clamp to map bounds
        self.view_offset = np.clip(self.view_offset, 0, max(0, self.map_size - self.window_size))
        
        view_rect = pygame.Rect(int(self.view_offset[0]), int(self.view_offset[1]), 
                              self.window_size, self.window_size)
        view_surface = canvas.subsurface(view_rect)
        
        # Draw HUD on the view surface (Temp/Roughness)
        if pygame.font:
            font = pygame.font.Font(None, 36)
            temp_text = font.render(f"Temp: {self.context.get('temperature', 20)} C", True, (255, 255, 255))
            rough_text = font.render(f"Roughness: {self.context.get('roughness', 0.0):.2f}", True, (255, 255, 255))
            view_surface.blit(temp_text, (10, 10))
            view_surface.blit(rough_text, (10, 50))

        if self.render_mode == "human":
            self.window.blit(view_surface, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(view_surface)), axes=(1, 0, 2)
            )

    def close(self):
        """Clean up resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

