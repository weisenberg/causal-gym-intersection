import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import warnings
# Filter out the specific deprecation warning from pkg_resources
warnings.filterwarnings("ignore", category=UserWarning, module='pygame.pkgdata')


class UrbanCausalIntersectionEnv(gym.Env):
    """
    A custom Gymnasium environment representing a 2D top-down urban intersection.
    
    The agent (a car) must navigate while obeying traffic rules.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, observation_type: str = "kinematic", max_npcs: int = None, max_pedestrians: int = None):
        super().__init__()
        
        # Window setup
        self.window_size = 600  # 600x600 grid
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.observation_type = observation_type
        
        # Intersection center
        self.intersection_center = np.array([300.0, 300.0])
        
        # Default context
        self.context = {
            "friction": 1.0,
            "traffic_light_duration": 300, # 10 seconds at 30 FPS
            "pedestrian_spawn_rate": 0.05,  # Probability of spawning a pedestrian per step (increased)
            "num_pedestrians": 2 if max_pedestrians is None else max_pedestrians,  # Maximum number of pedestrians
            "jaywalk_probability": 0.15,  # Probability that a pedestrian will jaywalk (reduced - more use crossings)
            "temperature": 20,  # Default temperature
            "roughness": 0.0,   # Default roughness
            "traffic_density": "medium", # low, medium, high
            "pedestrian_density": "medium", # low, medium, high
            "driver_impatience": 0.5, # 0.0 to 1.0
            "npc_color": "random", # random, red, blue, green, yellow, white, black
            "npc_size": "random", # random, small, medium, large
        }
        
        # --- Define Spaces ---
        if self.observation_type == "pixel":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
            )
        else:
            # 55-dim (now includes nearest cars since we add simple NPC traffic)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(55,), dtype=np.float32
            )
        
        # Actions: continuous [acceleration, steering] in [-1, 1]
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                                       high=np.array([1.0, 1.0], dtype=np.float32),
                                       dtype=np.float32)
        
        # --- State Variables ---
        # The car (red car) - only vehicle in the environment
        # Roads are two-way, car spawns on the right side
        # Road width is 100 pixels (250-350), center at 300
        # For right-side driving:
        #   - Vertical road going north: right side is x = 320 (east of center)
        #   - Vertical road going south: right side is x = 320 (still east of center, right lane)
        #   - Horizontal road going east: right side is y = 270 (north of center, right lane)
        #   - Horizontal road going west: right side is y = 330 (south of center, right lane)
        self._agent_location = None  # Will be set in reset
        self._agent_velocity = np.array([0.0, 0.0], dtype=np.float32)  # Velocity vector (vx, vy)
        self._agent_heading = None  # Will be set in reset
        
        # Physics constants
        self.max_speed = 5.0
        self.acceleration = 0.3
        self.angular_velocity = 0.1  # Radians per step for turning
        self.car_length = 20
        self.car_width = 10
        self.safety_buffer = float(self.car_width + 20)
        self.road_width = 100
        self.intersection_size = 120
        
        # Traffic lights per intersection with directional states
        self.traffic_lights = {}  # id -> {timer, phase, directions}
        self.traffic_light_timer = 0
        
        # NPC cars (lightweight traffic)
        self.npc_cars = []
        self.npc_car_speed = 3.5
        self.max_npcs = 2 if max_npcs is None else max_npcs

        # Time tracking for success criterion (59 seconds)
        self.step_count = 0
        self.success_after_steps = int(59 * self.metadata["render_fps"])
        
        # Intersection crossing tracking for exploration rewards
        self.intersections_crossed = set()
        self.was_in_intersection = False
        self.minimum_steps_for_exit = int(30 * self.metadata["render_fps"])
        
        # Episode tracking for display
        self.episode_count = 0
        self.episode_reward = 0.0
        
        # Pedestrian system
        self.pedestrians = []
        self.pedestrian_radius = 5
        self.pedestrian_speed = 0.5
        
        # Initialize Layout (calls _generate_layout -> build_crossings, etc)
        # This will set intersections, crossings, spawn points using default (center) layout
        self._generate_layout(randomize=False)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Update context if options provided
        if options is not None and isinstance(options, dict):
            self.context.update(options)
            
        # GENERATE LAYOUT (Randomized)
        self._generate_layout(randomize=True)
        
        # Reset state - spawn car on right side of a random road, facing intersection
        # _generate_layout has repopulated _car_spawn_points for this episode
        spawn_idx = self.np_random.integers(0, len(self._car_spawn_points))
        car_spawn = self._car_spawn_points[spawn_idx]
        self._agent_location = car_spawn["pos"].copy().astype(np.float32)
        # Add random noise to heading (+/- 0.5 radians approx +/- 28 deg)
        heading_noise = self.np_random.uniform(-0.5, 0.5)
        self._agent_heading = float(car_spawn["heading"] + heading_noise)
        self._agent_velocity = np.array([0.0, 0.0], dtype=np.float32)
        
        # --- Causal RL: Domain Randomization & Interventions ---
        # 1. Temperature (Intervention check)
        if options and "temperature" in options:
            self.temperature = options["temperature"]
        else:
            temps = [-10, 0, 10, 20, 30]
            self.temperature = int(self.np_random.choice(temps))
            
        # Map temp to roughness
        self.roughness = (self.temperature - (-10)) / (30 - (-10))
        
        # 2. Traffic Density (Intervention check)
        if options and "traffic_density" in options:
            self.traffic_density = options["traffic_density"]
        else:
            self.traffic_density = self.np_random.choice(["low", "medium", "high"])
            
        # Map traffic density to spawn rates
        if self.traffic_density == "low":
            self.context["npc_car_spawn_rate"] = 0.01
            self.context["num_npc_cars"] = 3
        elif self.traffic_density == "medium":
            self.context["npc_car_spawn_rate"] = 0.05
            self.context["num_npc_cars"] = 8
        else: # high
            self.context["npc_car_spawn_rate"] = 0.1
            self.context["num_npc_cars"] = 15
            
        # 3. Pedestrian Density (Intervention check)
        if options and "pedestrian_density" in options:
            self.pedestrian_density = options["pedestrian_density"]
        else:
            self.pedestrian_density = self.np_random.choice(["low", "medium", "high"])
            
        # Map pedestrian density to spawn rates
        if self.pedestrian_density == "low":
            self.context["pedestrian_spawn_rate"] = 0.01
            self.context["num_pedestrians"] = 3
        elif self.pedestrian_density == "medium":
            self.context["pedestrian_spawn_rate"] = 0.05
            self.context["num_pedestrians"] = 8
        else: # high
            self.context["pedestrian_spawn_rate"] = 0.1
            self.context["num_pedestrians"] = 15
            
        # 4. Driver Impatience (Intervention check)
        if options and "driver_impatience" in options:
            self.driver_impatience = float(options["driver_impatience"])
        else:
            self.driver_impatience = float(self.np_random.uniform(0.0, 1.0))
            
        # 5. NPC Color (Visual Confounder)
        if options and "npc_color" in options:
            self.npc_color = options["npc_color"]
        else:
            # Randomly choose a mode: mixed ("random") or one of the specific colors
            # This ensures episodes vary between "Rainbow Traffic" and "Uniform Color Traffic"
            colors = ["random", "red", "blue", "green", "yellow", "white", "black"]
            self.npc_color = self.np_random.choice(colors)
            
        # 6. NPC Size (Visual Confounder)
        if options and "npc_size" in options:
            self.npc_size = options["npc_size"]
        else:
            # Randomly choose a mode
            sizes = ["random", "small", "medium", "large"]
            self.npc_size = self.np_random.choice(sizes)
        
        # Update context with all causal vars
        self.context["temperature"] = self.temperature
        self.context["roughness"] = self.roughness
        self.context["traffic_density"] = self.traffic_density
        self.context["pedestrian_density"] = self.pedestrian_density
        self.context["driver_impatience"] = self.driver_impatience
        self.context["npc_color"] = self.npc_color
        self.context["npc_size"] = self.npc_size
        
        # Physics modifiers based on roughness
        # Friction: REMOVED dependency on temp for agent (kept at default 1.0)
        # Agent should be able to accelerate to max speed independent of temp
        
        # NPC Speed Factor: 0.6 (icy/cautious) to 1.0 (normal)
        self.npc_speed_factor = 0.6 + (0.4 * self.roughness)
        
        # NPC Brake Factor Adjustment:
        self.npc_brake_mod = 0.05 * (1.0 - self.roughness)
        
        # Reset pedestrians
        self.pedestrians = []
        # Reset NPC cars
        self.npc_cars = []
        
        # Spawn initial NPCs and Pedestrians to match "random reset"
        # Spawn NPCs
        for _ in range(self.context.get("num_npc_cars", 5)):
             self._spawn_npc_car()
             
        # Spawn Pedestrians
        for _ in range(self.context.get("num_pedestrians", 5)):
             self._spawn_pedestrian()
             
        # Reset time and intersection tracking
        self.step_count = 0
        self.intersections_crossed = set()
        self.was_in_intersection = False
        
        # Reset episode tracking
        self.episode_count += 1
        self.episode_reward = 0.0

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
    
    def _spawn_pedestrian(self):
        """Spawn a new pedestrian at a random spawn point with random behavior."""
        if len(self.pedestrians) >= self.context.get("num_pedestrians", 5):
            return
        
        # Choose random spawn point
        spawn_idx = self.np_random.integers(0, len(self.spawn_points))
        spawn_pos = self.spawn_points[spawn_idx].copy()
        
        # Determine if pedestrian will jaywalk or use crossing
        is_jaywalking = self.np_random.random() < self.context.get("jaywalk_probability", 0.3)
        
        # Get the corresponding crossing for this spawn point
        crossing = self.zebra_crossings[spawn_idx]
        
        # Determine target destination and movement phases
        if is_jaywalking:
            # Jaywalking: walk anywhere on the road (random point on opposite side or anywhere in road area)
            # Choose a random target point anywhere in the road area (250-350 range) or on opposite sidewalk
            if spawn_idx < 2:  # North or South sidewalk (horizontal roads)
                # Can cross anywhere horizontally - choose random x in road area or opposite sidewalk
                if self.np_random.random() < 0.5:
                    # Target somewhere on the opposite sidewalk (random x position)
                    target = np.array([
                        self.np_random.uniform(200.0, 400.0),  # Random x on opposite side
                        self.destination_points[spawn_idx][1]  # y position of opposite sidewalk
                    ])
                else:
                    # Target somewhere in the road/intersection area (random position)
                    target = np.array([
                        self.np_random.uniform(200.0, 400.0),  # Random x
                        self.np_random.uniform(200.0, 400.0)   # Random y in road area
                    ])
            else:  # East or West sidewalk (vertical roads)
                # Can cross anywhere vertically - choose random y in road area or opposite sidewalk
                if self.np_random.random() < 0.5:
                    # Target somewhere on the opposite sidewalk (random y position)
                    target = np.array([
                        self.destination_points[spawn_idx][0],  # x position of opposite sidewalk
                        self.np_random.uniform(200.0, 400.0)   # Random y on opposite side
                    ])
                else:
                    # Target somewhere in the road/intersection area (random position)
                    target = np.array([
                        self.np_random.uniform(200.0, 400.0),  # Random x in road area
                        self.np_random.uniform(200.0, 400.0)   # Random y
                    ])
            phase = "direct"  # Only one phase for jaywalking
        else:
            # Use zebra crossing: 3 phases
            # Phase 1: Walk from sidewalk to crossing START
            # Phase 2: Walk along crossing from START to END (walking ON the crossing)
            # Phase 3: Walk from crossing END to destination sidewalk
            target = crossing["start"].copy()  # First target: start of crossing
            phase = "to_crossing_start"
        
        pedestrian = {
            "pos": spawn_pos,
            "target": target,
            "speed": self.pedestrian_speed,
            "is_jaywalking": is_jaywalking,
            "phase": phase,  # Movement phase: "to_crossing_start", "on_crossing", "to_destination", "direct"
            "crossing_idx": spawn_idx if not is_jaywalking else None
        }
        
        self.pedestrians.append(pedestrian)
    
    def _update_pedestrians(self):
        """Update pedestrian positions."""
        pedestrians_to_remove = []
        
        for i, ped in enumerate(self.pedestrians):
            # Calculate direction to target
            direction = ped["target"] - ped["pos"]
            distance = np.linalg.norm(direction)
            
            if distance < ped["speed"]:
                # Reached target - move to next phase
                if ped["is_jaywalking"]:
                    # Jaywalking: reached destination, remove
                    # Or choose a new random target to continue wandering
                    if self.np_random.random() < 0.7:  # 70% chance to remove, 30% to wander more
                        pedestrians_to_remove.append(i)
                    else:
                        # Choose a new random target anywhere in the road area
                        ped["target"] = np.array([
                            self.np_random.uniform(150.0, 450.0),  # Wider range for wandering
                            self.np_random.uniform(150.0, 450.0)
                        ])
                else:
                    # Pedestrian using crossing - handle phases
                    crossing_idx = ped["crossing_idx"]
                    crossing = self.zebra_crossings[crossing_idx]
                    
                    if ped["phase"] == "to_crossing_start":
                        # Reached crossing start, now walk along the crossing
                        ped["phase"] = "on_crossing"
                        ped["target"] = crossing["end"].copy()  # Walk to crossing end
                    elif ped["phase"] == "on_crossing":
                        # Finished walking on crossing, now go to destination sidewalk
                        ped["phase"] = "to_destination"
                        ped["target"] = self.destination_points[crossing_idx].copy()
                    elif ped["phase"] == "to_destination":
                        # Reached destination sidewalk, remove pedestrian
                        pedestrians_to_remove.append(i)
            else:
                # Move towards target
                direction_normalized = direction / distance
                ped["pos"] += direction_normalized * ped["speed"]
            
            # Remove pedestrians that go off-screen
            if (ped["pos"][0] < -50 or ped["pos"][0] > 650 or
                ped["pos"][1] < -50 or ped["pos"][1] > 650):
                pedestrians_to_remove.append(i)
        
        # Remove pedestrians that have left the screen (in reverse order to maintain indices)
        for i in sorted(pedestrians_to_remove, reverse=True):
            self.pedestrians.pop(i)
    
    def _check_pedestrian_collision(self):
        """Check if agent collides with any pedestrian."""
        for ped in self.pedestrians:
            distance = np.linalg.norm(self._agent_location - ped["pos"])
            if distance < (self.car_width / 2 + self.pedestrian_radius):
                return True
        return False
    
    def _check_obb_collision(self, pos1, heading1, length1, width1, pos2, heading2, length2, width2):
        """Check collision between two oriented bounding boxes using Separating Axis Theorem."""
        # Get corners of both rectangles
        def get_corners(pos, heading, length, width):
            cos_h = np.cos(heading)
            sin_h = np.sin(heading)
            half_l = length / 2.0
            half_w = width / 2.0
            corners = np.array([
                [-half_l, -half_w],
                [half_l, -half_w],
                [half_l, half_w],
                [-half_l, half_w]
            ])
            rotation = np.array([[cos_h, sin_h], [-sin_h, cos_h]])
            return (rotation @ corners.T).T + pos
        
        corners1 = get_corners(pos1, heading1, length1, width1)
        corners2 = get_corners(pos2, heading2, length2, width2)
        
        # Test axes from both rectangles
        axes = [
            np.array([np.cos(heading1), -np.sin(heading1)]),  # Box 1 forward
            np.array([-np.sin(heading1), -np.cos(heading1)]), # Box 1 side
            np.array([np.cos(heading2), -np.sin(heading2)]),  # Box 2 forward
            np.array([-np.sin(heading2), -np.cos(heading2)])  # Box 2 side
        ]
        
        for axis in axes:
            # Project all corners onto axis
            proj1 = np.dot(corners1, axis)
            proj2 = np.dot(corners2, axis)
            
            # Check for separation
            if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                return False  # Separating axis found, no collision
        
        return True  # No separating axis found, collision detected

    def _get_obs(self):
        """Get observation either kinematic (55) or pixel (84x84x3)."""
        if self.observation_type == "pixel":
            frame = self._render_frame()
            if frame is None or frame.size == 0:
                return np.zeros((84, 84, 3), dtype=np.uint8)
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            scaled = pygame.transform.smoothscale(surf, (84, 84))
            arr = np.transpose(pygame.surfarray.pixels3d(scaled), (1, 0, 2)).copy()
            return arr.astype(np.uint8)
        # Kinematic: agent (5) + lidar16 + cars(20) + peds(10) + light(4)
        obs = []
        obs.extend([
            float(self._agent_location[0]),
            float(self._agent_location[1]),
            float(self._agent_velocity[0]),
            float(self._agent_velocity[1]),
            float(self._agent_heading),
        ])
        obs.extend(self._compute_lidar(num_rays=16, max_range=200.0))
        obs.extend(self._nearest_cars_features(k=5))
        obs.extend(self._nearest_ped_features(k=5))
        r, y, g, ttc = self._next_traffic_light_features()
        obs.extend([r, y, g, ttc])
        return np.array(obs, dtype=np.float32)

    def _compute_lidar(self, num_rays: int = 16, max_range: float = 200.0):
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
            for r in np.linspace(5, max_range, num=25):
                p = agent_pos + ray_dir * r
                if p[0] < 0 or p[0] > self.window_size or p[1] < 0 or p[1] > self.window_size:
                    dists[i] = min(dists[i], r)
                    break
                for (op, rad) in obstacles:
                    if np.linalg.norm(p - op) <= rad + 2:
                        dists[i] = min(dists[i], r)
                        break
                if dists[i] < r:
                    break
        return dists.tolist()

    def _nearest_cars_features(self, k: int = 5):
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

    def _initialize_traffic_lights(self):
        # 4-phase like extended env
        for inter in self.intersections:
            phase = inter["id"] % 4
            dirs = {"north": "red", "south": "red", "east": "red", "west": "red"}
            if phase == 0:
                dirs.update({"north": "green", "south": "green"})
            elif phase == 1:
                dirs.update({"north": "yellow", "south": "yellow"})
            elif phase == 2:
                dirs.update({"east": "green", "west": "green"})
            else:
                dirs.update({"east": "yellow", "west": "yellow"})
            # Remove states for non-existing approaches
            for d in list(dirs.keys()):
                if d not in inter["approaches"]:
                    dirs[d] = "red"
            self.traffic_lights[inter["id"]] = {"timer": 0, "phase": phase, "directions": dirs}

    def _update_traffic_lights(self):
        self.traffic_light_timer += 1
        duration = self.context.get("traffic_light_duration", 30)
        for inter_id, light in self.traffic_lights.items():
            light["timer"] += 1
            if light["timer"] >= duration:
                light["timer"] = 0
                light["phase"] = (light["phase"] + 1) % 4
                phase = light["phase"]
                dirs = {"north": "red", "south": "red", "east": "red", "west": "red"}
                if phase == 0:
                    dirs.update({"north": "green", "south": "green"})
                elif phase == 1:
                    dirs.update({"north": "yellow", "south": "yellow"})
                elif phase == 2:
                    dirs.update({"east": "green", "west": "green"})
                else:
                    dirs.update({"east": "yellow", "west": "yellow"})
                # Disable missing approaches
                inter = next(i for i in self.intersections if i["id"] == inter_id)
                for d in list(dirs.keys()):
                    if d not in inter["approaches"]:
                        dirs[d] = "red"
                light["directions"] = dirs

    def _build_crossings(self):
        crossings = []
        offset = self.intersection_size // 2 - 10
        for inter in self.intersections:
            pos = inter["pos"]
            if "north" in inter["approaches"]:
                crossings.append({"start": np.array([pos[0] - offset, pos[1] - offset]), "end": np.array([pos[0] + offset, pos[1] - offset]), "direction": "horizontal", "inter_id": inter["id"]})
            if "south" in inter["approaches"]:
                crossings.append({"start": np.array([pos[0] - offset, pos[1] + offset]), "end": np.array([pos[0] + offset, pos[1] + offset]), "direction": "horizontal", "inter_id": inter["id"]})
            if "east" in inter["approaches"]:
                crossings.append({"start": np.array([pos[0] + offset, pos[1] - offset]), "end": np.array([pos[0] + offset, pos[1] + offset]), "direction": "vertical", "inter_id": inter["id"]})
            if "west" in inter["approaches"]:
                crossings.append({"start": np.array([pos[0] - offset, pos[1] - offset]), "end": np.array([pos[0] - offset, pos[1] + offset]), "direction": "vertical", "inter_id": inter["id"]})
        return crossings

    def _build_car_spawns(self):
        spawns = []
        # Spawn OUTSIDE the map so cars drive into view for single cross roads
        # Horizontal main road at y=300 (right-side mapping):
        # left->right (eastbound) = bottom side (y=330), right->left (westbound) = upper side (y=270)
        for x in [-120.0, -60.0]:
            spawns.append({"pos": np.array([x, 330.0]), "heading": 0.0, "direction": "east", "type": "horizontal"})
        for x in [660.0, 720.0]:
            spawns.append({"pos": np.array([x, 270.0]), "heading": np.pi, "direction": "west", "type": "horizontal"})
        # Vertical road at x=300
        # Right-side: northbound (east side x=320), southbound (west side x=280)
        for y in [660.0, 720.0]:
            spawns.append({"pos": np.array([320.0, y]), "heading": np.pi/2, "direction": "north", "type": "vertical"})
        for y in [-120.0, -60.0]:
            spawns.append({"pos": np.array([280.0, y]), "heading": -np.pi/2, "direction": "south", "type": "vertical"})
        return spawns

    def _get_nearest_intersection(self, pos):
        min_d = 1e9
        nearest = None
        for inter in self.intersections:
            d = np.linalg.norm(pos - inter["pos"])
            if d < min_d:
                min_d = d
                nearest = inter
        return nearest

    def _get_traffic_light_state(self, inter_id, direction):
        light = self.traffic_lights.get(inter_id)
        if not light:
            return "green"
        return light["directions"].get(direction, "red")

    def _next_traffic_light_features(self):
        inter = self._get_nearest_intersection(self._agent_location)
        if inter is None:
            return 0.0, 0.0, 1.0, 0.0
        # Determine travel direction by heading
        if abs(np.cos(self._agent_heading)) > abs(np.sin(self._agent_heading)):
            direction = "east" if np.cos(self._agent_heading) > 0 else "west"
        else:
            direction = "south" if np.sin(self._agent_heading) < 0 else "north"
        state = self._get_traffic_light_state(inter["id"], direction)
        red = 1.0 if state == "red" else 0.0
        yellow = 1.0 if state == "yellow" else 0.0
        green = 1.0 if state == "green" else 0.0
        ttc = self.traffic_lights[inter["id"]]["timer"] / max(1, self.context.get("traffic_light_duration", 30))
        return red, yellow, green, float(ttc)

    def _spawn_npc_car(self):
        if len(self.npc_cars) >= 8:
            return
        spawn_idx = int(self.np_random.integers(0, len(self.car_spawn_points)))
        spawn = self.car_spawn_points[spawn_idx]
        # Per-car variability
        # Apply Temperature Physics
        speed_factor = getattr(self, "npc_speed_factor", 1.0)
        brake_mod = getattr(self, "npc_brake_mod", 0.0)
        
        # Apply Driver Impatience
        # Impatience (0.0 to 1.0) increases acceleration and max speed, decreases stopping distance buffer
        impatience = self.context.get("driver_impatience", 0.5)
        
        # Base cruise speed modified by impatience (more impatient = faster)
        # Impatience 0.0 -> 0.9x speed, Impatience 1.0 -> 1.2x speed
        impatience_speed_mod = 0.9 + (0.3 * impatience)
        
        cruise_speed = float(self.npc_car_speed * (0.7 + self.np_random.random()*0.8)) * speed_factor * impatience_speed_mod
        max_speed = cruise_speed * (1.1 + self.np_random.random()*0.3)
        
        # Accel modified by impatience (more impatient = harder accel)
        # Impatience 0.0 -> 0.8x accel, Impatience 1.0 -> 1.5x accel
        impatience_accel_mod = 0.8 + (0.7 * impatience)
        accel = float(0.18 + self.np_random.random()*0.22) * speed_factor * impatience_accel_mod
        
        base_brake = float(0.82 + self.np_random.random()*0.12)
        brake_factor = min(0.99, base_brake + brake_mod) # Higher factor = weaker braking
        
        stopping_distance = float(30 + self.np_random.random()*40)
        # Impatience reduces stopping distance (tailgating)
        # Impatience 1.0 -> 0.8x stopping distance
        stopping_distance *= (1.0 - (0.2 * impatience))
        
        # Increase stopping distance on ice (visual behavior logic)
        if self.roughness < 0.5:
            stopping_distance *= 1.5
        # Determine Size (Length/Width)
        global_size = self.context.get("npc_size", "random")
        if global_size != "random":
            if global_size == "small":
                car_len, car_wid = 30, 15
            elif global_size == "medium":
                car_len, car_wid = 40, 20
            else: # large
                car_len, car_wid = 50, 25
        else:
            # Random size per car
            car_len = float(18 + self.np_random.random()*20)  # [18..38]
            car_wid = float(10 + self.np_random.random()*16)  # [10..26]
        # Colors: random rainbow base with shading
        rainbow = [
            (255, 0, 0),      # red
            (255, 127, 0),    # orange
            (255, 255, 0),    # yellow
            (0, 200, 0),      # green
            (0, 120, 255),    # blue
            (75, 0, 130),     # indigo
            (148, 0, 211),    # violet
        ]
        base = rainbow[int(self.np_random.integers(0, len(rainbow)))]
        # Apply shade variation (multiply by factor 0.7..1.0)
        shade = 0.7 + float(self.np_random.random())*0.3
        color = (int(base[0]*shade), int(base[1]*shade), int(base[2]*shade))
        # Determine Color
        global_color = self.context.get("npc_color", "random")
        if global_color != "random":
            # Use the forced global color
            color_map = {
                "red": (200, 50, 50),
                "blue": (50, 50, 200),
                "green": (50, 200, 50),
                "yellow": (200, 200, 50),
                "white": (240, 240, 240),
                "black": (50, 50, 50)
            }
            color = color_map.get(global_color, (50, 50, 200)) # Default to blue if unknown
        else:
            # Random color per car
            color = (
                self.np_random.integers(50, 255),
                self.np_random.integers(50, 255),
                self.np_random.integers(50, 255),
            )
            
        # Determine Size (Length/Width)
        global_size = self.context.get("npc_size", "random")
        if global_size != "random":
            if global_size == "small":
                length, width = 30, 15
            elif global_size == "medium":
                length, width = 40, 20
            else: # large
                length, width = 50, 25
        else:
            # Random size per car
            length = 30 + self.np_random.random() * 20
            width = 15 + self.np_random.random() * 10

        car = {
            "pos": spawn["pos"].copy().astype(np.float32),
            "heading": spawn["heading"],
            "velocity": np.array([0.0, 0.0], dtype=np.float32),
            "cruise_speed": cruise_speed,
            "max_speed": max_speed,
            "accel": accel,
            "brake_factor": brake_factor,
            "stopping_distance": stopping_distance,
            "target_speed": cruise_speed,
            "type": spawn["type"],
            "direction": spawn["direction"],
            "state": "driving",
            "length": length,
            "width": width,
            "width": width,
            "color": color,
            "width": width,
            "color": color,

        }
        self.npc_cars.append(car)

    def _update_npc_cars(self):
        cars_to_remove = []
        for i, car in enumerate(self.npc_cars):
            # Initialize: assume front is clear, car should keep going
            car["target_speed"] = car["cruise_speed"]
            car["state"] = "driving"
            
            # Simple traffic light adherence near nearest intersection
            inter = self._get_nearest_intersection(car["pos"])
            if inter is not None:
                inter_pos = inter["pos"]
                edge = self.intersection_size/2
                stopping_distance = car.get("stopping_distance", 40.0)
                # compute direction
                direction = car["direction"]
                # distance to edge along approach
                should_stop = False
                dist_to_center = np.linalg.norm(car["pos"] - inter_pos)
                inside = dist_to_center < edge
                if not inside:
                    if direction == "east" and car["pos"][0] < inter_pos[0]:
                        dist = (inter_pos[0] - edge) - car["pos"][0]
                        light = self._get_traffic_light_state(inter["id"], "east")
                        should_stop = 0 < dist < stopping_distance and (light in ["red", "yellow"])
                    if direction == "west" and car["pos"][0] > inter_pos[0]:
                        dist = car["pos"][0] - (inter_pos[0] + edge)
                        light = self._get_traffic_light_state(inter["id"], "west")
                        should_stop = 0 < dist < stopping_distance and (light in ["red", "yellow"])
                    if direction == "north" and car["pos"][1] > inter_pos[1]:
                        dist = car["pos"][1] - (inter_pos[1] + edge)
                        light = self._get_traffic_light_state(inter["id"], "north")
                        should_stop = 0 < dist < stopping_distance and (light in ["red", "yellow"])
                    if direction == "south" and car["pos"][1] < inter_pos[1]:
                        dist = (inter_pos[1] - edge) - car["pos"][1]
                        light = self._get_traffic_light_state(inter["id"], "south")
                        should_stop = 0 < dist < stopping_distance and (light in ["red", "yellow"])
                if should_stop:
                    car["target_speed"] = 0.0
                    car["state"] = "stopped_at_light"
            
            # Global collision avoidance: only check obstacles in FRONT
            forward_vec = np.array([np.cos(car["heading"]), -np.sin(car["heading"])], dtype=np.float32)
            min_margin = max(self.car_width, car.get("width", self.car_width)) + 6.0
            obstacle_in_front = False
            
            # Check agent (only if in front)
            vec_to_agent = self._agent_location - car["pos"]
            dist_to_agent = float(np.linalg.norm(vec_to_agent))
            if dist_to_agent > 1e-3:
                dir_to_agent = vec_to_agent / dist_to_agent
                cos_angle = float(np.dot(forward_vec, dir_to_agent))
                dist_along = float(np.dot(vec_to_agent, forward_vec))
                # Only react if agent is in FRONT (not side or back)
                if cos_angle > 0.5 and dist_along > 10:
                    obstacle_in_front = True
                    if dist_to_agent < min_margin:
                        car["target_speed"] = 0.0
                        car["velocity"] *= car.get("brake_factor", 0.88)
                    elif dist_to_agent < car.get("stopping_distance", 40.0) * 0.6:
                        car["target_speed"] = min(car["target_speed"], car["cruise_speed"] * 0.2)
                    elif dist_to_agent < car.get("stopping_distance", 40.0):
                        car["target_speed"] = min(car["target_speed"], car["cruise_speed"] * 0.5)
            # Check pedestrians (only if in front)
            for ped in self.pedestrians:
                vec_to_ped = ped["pos"] - car["pos"]
                dist_to_ped = float(np.linalg.norm(vec_to_ped))
                if dist_to_ped <= 1e-3:
                    continue
                dir_to_ped = vec_to_ped / dist_to_ped
                cos_angle = float(np.dot(forward_vec, dir_to_ped))
                dist_along = float(np.dot(vec_to_ped, forward_vec))
                # Only react to pedestrians in FRONT (not side or back)
                if cos_angle > 0.5 and dist_along > 10:  # Ahead and within 60° cone
                    obstacle_in_front = True
                    ped_margin = self.pedestrian_radius + car.get("width", self.car_width)/2 + 8.0
                    if dist_to_ped < ped_margin:
                        car["target_speed"] = 0.0
                        car["velocity"] *= car.get("brake_factor", 0.9)
                    elif dist_to_ped < car.get("stopping_distance", 40.0) * 0.6:
                        car["target_speed"] = min(car["target_speed"], car["cruise_speed"] * 0.2)
                    elif dist_to_ped < car.get("stopping_distance", 40.0):
                        car["target_speed"] = min(car["target_speed"], car["cruise_speed"] * 0.5)


            # Check other NPC cars (only if in front)
            for j, other in enumerate(self.npc_cars):
                if j == i:
                    continue
                vec_to_car = other["pos"] - car["pos"]
                dist_to_car = float(np.linalg.norm(vec_to_car))
                if dist_to_car <= 1e-3:
                    continue
                
                dir_to_car = vec_to_car / dist_to_car
                cos_angle = float(np.dot(forward_vec, dir_to_car))
                dist_along = float(np.dot(vec_to_car, forward_vec))
                
                # Only react to cars in FRONT (not sides or back)
                if cos_angle > 0.5 and dist_along > 10:  # Ahead and within 60° cone
                    obstacle_in_front = True
                    car_margin = (car.get("width", self.car_width) + other.get("width", self.car_width))/2 + 6.0
                    if dist_to_car < car_margin:
                        car["target_speed"] = 0.0
                        car["velocity"] *= car.get("brake_factor", 0.9)
                    elif dist_to_car < car.get("stopping_distance", 40.0) * 0.6:
                        car["target_speed"] = min(car["target_speed"], car["cruise_speed"] * 0.2)
                    elif dist_to_car < car.get("stopping_distance", 40.0):
                        car["target_speed"] = min(car["target_speed"], car["cruise_speed"] * 0.5)
            
            # If no obstacles in front and not stopped at light, ensure car continues at cruise speed
            if not obstacle_in_front and car["state"] != "stopped_at_light":
                car["target_speed"] = car["cruise_speed"]
                car["state"] = "driving"
            # accelerate/decelerate to target
            cur = np.linalg.norm(car["velocity"])
            if cur < car["target_speed"]:
                acc = car.get("accel", 0.25)
                car["velocity"][0] += acc * np.cos(car["heading"])
                car["velocity"][1] += -acc * np.sin(car["heading"])
            elif cur > car["target_speed"]:
                car["velocity"] *= car.get("brake_factor", 0.9)
            # align and limit
            spd = np.linalg.norm(car["velocity"])
            if spd > 0.01:
                car["velocity"][0] = spd * np.cos(car["heading"])
                car["velocity"][1] = -spd * np.sin(car["heading"])
            if spd > car.get("max_speed", 5.0):
                car["velocity"] = (car["velocity"]/spd) * car["max_speed"]
            # move
            car["pos"] += car["velocity"]
            # Keep in lane (Local Frame Logic)
            # Transform to local
            local_pos = self._world_to_local(car["pos"])
            
            if car.get("type") == "horizontal":
                # Horizontal Road (along X)
                # Eastbound (+x): Right side is bottom (+y in screen coords) -> y = 20
                # Westbound (-x): Right side is top (-y in screen coords) -> y = -20
                target_y = 20.0 if car["direction"] == "east" else -20.0
                # Soft correct or snap? Snap is safer for simple rail-like movement
                local_pos[1] = target_y
                
            elif car.get("type") == "vertical":
                # Vertical Road (along Y)
                # Southbound (+y): Right side is left (-x in screen coords) -> x = -20
                # Northbound (-y): Right side is right (+x in screen coords) -> x = 20
                target_x = 20.0 if car["direction"] == "north" else -20.0
                local_pos[0] = target_x
                
            # Transform back to world
            car["pos"] = self._local_to_world(local_pos)

            # Remove off-screen (Distance based check)
            # Layout center is at self.layout_center. Window/Sim area is ~600x600.
            # Spawns are at 800. We need to allow them to exist until they pass through.
            # 1200 radius is safe.
            if np.linalg.norm(car["pos"] - self.layout_center) > 1200.0:
                 cars_to_remove.append(i)
        
        for i in sorted(cars_to_remove, reverse=True):
            self.npc_cars.pop(i)

    def step(self, action):
        """Execute one step in the environment."""
        # --- Physics Update (continuous control) ---
        action = np.asarray(action, dtype=np.float32)
        if action.shape == ():
            action = np.array([0.0, 0.0], dtype=np.float32)
        accel_cmd = float(np.clip(action[0], -1.0, 1.0))
        steer_cmd = float(np.clip(action[1], -1.0, 1.0))
        # Accel along heading
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
        # Steering (realistic: can only turn while moving)
        speed = np.linalg.norm(self._agent_velocity)
        # Scale turning by speed: no turn at speed=0, reduced turn at low speed, full turn at speed>=0.5
        turning_factor = min(1.0, speed / 0.5) if speed > 0.01 else 0.0
        self._agent_heading += self.angular_velocity * steer_cmd * turning_factor
        # No slide
        speed = np.linalg.norm(self._agent_velocity)
        if speed > 0.01:
            self._agent_velocity[0] = speed * np.cos(self._agent_heading)
            self._agent_velocity[1] = -speed * np.sin(self._agent_heading)
        
        # Apply friction (only if not stationary - car should stay stationary until user acts)
        speed = np.linalg.norm(self._agent_velocity)
        if speed > 0.01:  # Only apply friction if moving (avoid floating point drift)
            friction = self.context.get("friction", 1.0)
            self._agent_velocity[0] *= friction
            self._agent_velocity[1] *= friction
            
            # If velocity becomes very small, stop completely (car stays stationary)
            new_speed = np.linalg.norm(self._agent_velocity)
            if new_speed < 0.01:
                self._agent_velocity = np.array([0.0, 0.0], dtype=np.float32)
            elif new_speed > self.max_speed:
                self._agent_velocity = (self._agent_velocity / new_speed) * self.max_speed
        
        # Update position
        self._agent_location[0] += self._agent_velocity[0]
        self._agent_location[1] += self._agent_velocity[1]
        
        # --- Pedestrian System ---
        # Update traffic lights
        self._update_traffic_lights()
        
        # Spawn new cars if needed
        if len(self.npc_cars) < self.max_npcs:
            if np.random.rand() < 0.05:  # 5% chance to spawn a car per step if under limit
                self._spawn_npc_car()
        # Update NPC cars
        self._update_npc_cars()
        
        # Spawn new pedestrians
        if self.np_random.random() < self.context.get("pedestrian_spawn_rate", 0.02):
            self._spawn_pedestrian()
        
        # Update pedestrian positions
        self._update_pedestrians()
        
        # --- Compute Rewards & Termination ---
        terminated = False
        reward = 0.0
        
        # Check boundaries
        off_screen = (
            self._agent_location[0] < 0 or self._agent_location[0] > 600 or
            self._agent_location[1] < 0 or self._agent_location[1] > 600
        )
        
        # Check if exit was valid (on road)
        valid_exit = False
        if off_screen:
            valid_exit = self._check_valid_exit()
        
        # Check for pedestrian collision
        collision = self._check_pedestrian_collision()
        
        # Check for NPC car collision (using oriented bounding box collision)
        npc_collision = False
        for car in self.npc_cars:
            if self._check_obb_collision(
                self._agent_location, self._agent_heading, self.car_length, self.car_width,
                car["pos"], car["heading"], car.get("length", self.car_length), car.get("width", self.car_width)
            ):
                npc_collision = True
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
        if collision or npc_collision:
            terminated = True
            reward = -100.0
        elif off_screen:
            terminated = True
            if valid_exit:
                reward = 100.0
            else:
                reward = -10.0 # Off-road exit
        
        # Combine collision flags
        collision = collision or npc_collision
        
        # Shaped rewards
        # + survival
        reward += 0.1
        # + progress/efficiency
        agent_speed = float(np.linalg.norm(self._agent_velocity))
        reward += 1.0 * (agent_speed / max(1e-6, self.max_speed))
        
        # - stationary penalty (encourages exploration)
        if agent_speed < 0.1:
            reward -= 0.5  # Penalty for not moving
        
        # + intersection crossing bonus (encourages exploration)
        current_in_intersection = False
        for inter in self.intersections:
            inter_pos = inter["pos"]
            edge = self.intersection_size / 2.0
            dist_to_center = np.linalg.norm(self._agent_location - inter_pos)
            if dist_to_center < edge:
                current_in_intersection = True
                # If we just entered this intersection (weren't in any intersection before)
                if not self.was_in_intersection and inter["id"] not in self.intersections_crossed:
                    reward += 50.0  # Bonus for crossing a new intersection
                    self.intersections_crossed.add(inter["id"])
                break
        self.was_in_intersection = current_in_intersection
        
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
        if collision or npc_collision:
            terminated = True
            reward = -100.0
        elif off_screen:
            terminated = True
            if valid_exit:
                reward = 100.0
            else:
                reward = -10.0 # Off-road exit
        
        # Terminations and success
        info = {
            "num_pedestrians": len(self.pedestrians),
            "collision": collision,
            "num_npc_cars": len(self.npc_cars),
            "success": valid_exit and off_screen,
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
        # Time limit (Truncation)
        self.step_count += 1
        if self.step_count >= self.success_after_steps:
            truncated = True
            info["success"] = True  # success signal if survived

        observation = self._get_obs()
        
        # Track episode reward
        self.episode_reward += reward

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _agent_runs_red_light(self):
        """Detect if agent is entering/inside intersection while its direction has red."""
        inter = self._get_nearest_intersection(self._agent_location)
        if inter is None:
            return False
            
        # Check proximity to intersection center (World Coords)
        pos = inter["pos"]
        edge = self.intersection_size / 2.0
        # Use simpler distance check instead of box for rotated intersection
        dist = np.linalg.norm(self._agent_location - pos)
        # 85 is approx sqrt(edge^2 + edge^2) + margin, slightly loose
        if dist > edge * 1.5: 
            return False
            
        # Determine "logical" cardinal travel direction relative to layout
        # Global Heading - Layout Rotation
        rel_heading = self._agent_heading - self.layout_rotation
        # Normalize to -pi to pi
        rel_heading = (rel_heading + np.pi) % (2 * np.pi) - np.pi
        
        # Directions in local frame: 
        # West: pi (or -pi), East: 0, South: pi/2, North: -pi/2
        # (Based on standard unit circle where 0 is Right/+X)
        
        if abs(np.cos(rel_heading)) > abs(np.sin(rel_heading)):
            direction = "east" if np.cos(rel_heading) > 0 else "west"
        else:
            direction = "south" if np.sin(rel_heading) > 0 else "north" # sin(pi/2)=1 (South)
            
        state = self._get_traffic_light_state(inter["id"], direction)
        return state == "red"

    def _agent_on_active_crossing(self):
        """Penalty condition: agent overlaps a zebra crossing while a pedestrian is on it."""
        for idx, zc in enumerate(self.zebra_crossings):
            # Check distance from agent to crossing segment
            p = self._agent_location
            a = zc["start"]
            b = zc["end"]
            
            # Segment vector
            ab = b - a
            ap = p - a
            
            # Project p onto ab
            t = np.dot(ap, ab) / np.dot(ab, ab)
            
            # Check if within segment bounds (0 <= t <= 1)
            # and lateral distance is small (crossing width approx 6-10px)
            if 0 <= t <= 1:
                closest_point = a + t * ab
                dist = np.linalg.norm(p - closest_point)
                
                if dist < 10.0: # Agent is ON the crossing
                     # any ped currently on this crossing?
                    for ped in self.pedestrians:
                        if (not ped["is_jaywalking"]) and ped.get("crossing_idx", -1) == idx and ped.get("phase") == "on_crossing":
                            return True
        return False

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """Render a single frame using pygame with Rotated Roads."""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((128, 128, 128))  # Gray background
        
        # Helper to draw rotated rect
        def draw_local_rect(color, x, y, w, h):
            # Define 4 corners in local
            corners = np.array([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ])
            world_corners = [self._local_to_world(c) for c in corners]
            pygame.draw.polygon(canvas, color, world_corners)

        def draw_local_line(color, p1, p2, width=2):
            wp1 = self._local_to_world(np.array(p1))
            wp2 = self._local_to_world(np.array(p2))
            pygame.draw.line(canvas, color, wp1, wp2, width)

        # Draw Roads
        # Horizontal (-300 to 300, width 100 centered at 0 means -50 to 50)
        draw_local_rect((64, 64, 64), -300, -50, 600, 100) # "Horizontal" road
        draw_local_rect((64, 64, 64), -50, -300, 100, 600) # "Vertical" road
        
        # Draw Center Lines
        # Horizontal
        for x in range(-300, 300, 40):
            if not (-50 <= x <= 50): # Skip intersection
                draw_local_line((255, 255, 255), (x, 0), (x+20, 0))
        # Vertical
        for y in range(-300, 300, 40):
            if not (-50 <= y <= 50):
                draw_local_line((255, 255, 255), (0, y), (0, y+20))
                
        # Draw Intersection (Center)
        draw_local_rect((80, 80, 80), -60, -60, 120, 120)
        
        # Draw Traffic Lights
        for inter in self.intersections:
            # We assume inter is at 0,0 local (since we only support 1 for now in this rendering logic)
            light = self.traffic_lights[inter["id"]]
            offset = 50
            size = 5
            
            # Helper for light circle
            def draw_light(direction, color_str):
                c = (255,0,0)
                if color_str == "green": c = (0,255,0)
                elif color_str == "yellow": c = (255,255,0)
                
                if direction == "north": p = (0, -offset)
                elif direction == "south": p = (0, offset)
                elif direction == "east": p = (offset, 0)
                elif direction == "west": p = (-offset, 0)
                
                wp = self._local_to_world(np.array(p))
                pygame.draw.circle(canvas, c, (int(wp[0]), int(wp[1])), size)
                
            if "north" in inter["approaches"]: draw_light("north", light["directions"]["north"])
            if "south" in inter["approaches"]: draw_light("south", light["directions"]["south"])
            if "east" in inter["approaches"]: draw_light("east", light["directions"]["east"])
            if "west" in inter["approaches"]: draw_light("west", light["directions"]["west"])

        # Draw Zebra Crossings
        for crossing in self.zebra_crossings:
            # They are already world coords
            pygame.draw.line(canvas, (255, 255, 255), crossing["start"], crossing["end"], 4)

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
            # Color: red for jaywalking, blue for using crossing
            color = (255, 100, 100) if ped["is_jaywalking"] else (100, 100, 255)
            pygame.draw.circle(canvas, color, ped_pos, self.pedestrian_radius)
            # Draw a small circle to show direction
            direction = ped["target"] - ped["pos"]
            if np.linalg.norm(direction) > 0:
                direction_normalized = direction / np.linalg.norm(direction)
                front_pos = ped_pos + (direction_normalized * self.pedestrian_radius * 1.5).astype(int)
                pygame.draw.circle(canvas, (255, 255, 255), front_pos, 2)
        
        # Draw the car (red car - only vehicle) that rotates based on heading
        agent_pos = self._agent_location.astype(int)
        
        # Calculate rectangle corners based on heading
        # In pygame, y increases downward, so we need to adjust the rotation
        # heading: 0 = right, π/2 = up (negative y in pygame), etc.
        cos_h = np.cos(self._agent_heading)
        sin_h = np.sin(self._agent_heading)
        
        # Rectangle corners relative to center (before rotation)
        # Car initially points along positive x-axis (right)
        corners = np.array([
            [-self.car_length / 2, -self.car_width / 2],
            [self.car_length / 2, -self.car_width / 2],
            [self.car_length / 2, self.car_width / 2],
            [-self.car_length / 2, self.car_width / 2]
        ])
        
        # Rotate corners
        # In pygame, y increases downward. Our heading convention: 0=right, π/2=up (negative y in pygame)
        # So we need to account for the y-flip in the rotation
        rotation_matrix = np.array([
            [cos_h, sin_h],    # For pygame: flipped y component
            [-sin_h, cos_h]
        ])
        
        rotated_corners = (rotation_matrix @ corners.T).T + agent_pos
        
        # Draw the rotated rectangle (red car - only vehicle in environment)
        pygame.draw.polygon(canvas, (255, 0, 0), rotated_corners)
        # Draw a small circle at the front to show direction
        # Front is in the direction of heading (accounting for pygame's y-down coordinate system)
        front_offset = np.array([self.car_length / 2 * cos_h, -self.car_length / 2 * sin_h])
        front_pos = agent_pos + front_offset
        pygame.draw.circle(canvas, (200, 0, 0), front_pos.astype(int), 3)

        # Render info overlay
        if pygame.font:
            font = pygame.font.Font(None, 36)
            # Environment info
            temp_text = font.render(f"Temp: {self.context.get('temperature', 20)} C", True, (255, 255, 255))
            rough_text = font.render(f"Roughness: {self.context.get('roughness', 0.0):.2f}", True, (255, 255, 255))
            # Episode info
            episode_text = font.render(f"Episode: {self.episode_count}", True, (255, 255, 255))
            reward_text = font.render(f"Total Reward: {self.episode_reward:.1f}", True, (255, 255, 255))
            step_text = font.render(f"Step: {self.step_count}", True, (255, 255, 255))
            
            canvas.blit(temp_text, (10, 10))
            canvas.blit(rough_text, (10, 50))
            canvas.blit(episode_text, (10, 90))
            canvas.blit(reward_text, (10, 130))
            canvas.blit(step_text, (10, 170))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
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

    def _local_to_world(self, local_pos):
        """Transform local coordinates (relative to intersection center, 0-rotation) to world coordinates."""
        # Rotate
        c, s = np.cos(self.layout_rotation), np.sin(self.layout_rotation)
        rot_x = local_pos[0] * c - local_pos[1] * s
        rot_y = local_pos[0] * s + local_pos[1] * c
        # Translate
        return np.array([rot_x + self.layout_center[0], rot_y + self.layout_center[1]])

    def _world_to_local(self, world_pos):
        """Transform world coordinates to local coordinates (relative to intersection center, 0-rotation)."""
        # Translate
        tx = world_pos[0] - self.layout_center[0]
        ty = world_pos[1] - self.layout_center[1]
        # Rotate (inverse)
        c, s = np.cos(-self.layout_rotation), np.sin(-self.layout_rotation)
        rot_x = tx * c - ty * s
        rot_y = tx * s + ty * c
        return np.array([rot_x, rot_y])

    def _generate_layout(self, randomize=False):
        """Generate the road layout, potentially randomized."""
        if randomize:
            # Random center between 200 and 400 (window is 600x600)
            self.layout_center = self.np_random.uniform(200.0, 400.0, size=2)
            # Random rotation (0 to 2pi)
            self.layout_rotation = self.np_random.uniform(0, 2 * np.pi)
        else:
            self.layout_center = np.array([300.0, 300.0])
            self.layout_rotation = 0.0

        self.intersection_center = self.layout_center

        # Rebuild Intersections (World Coords)
        self.intersections = [
            {"pos": self.layout_center.copy(), "id": 0, "approaches": ["north", "south", "east", "west"]},
        ]
        
        # Rebuild Traffic Lights (if they rely on approaches/phases, init them)
        self._initialize_traffic_lights()

        # Rebuild Zebra Crossings (Local -> World)
        self.zebra_crossings = self._build_crossings()

        # Rebuild Car Spawn Points (Local -> World)
        self.car_spawn_points = self._build_car_spawns()
        # Default agent spawns are the same as car spawns for now
        self._car_spawn_points = self.car_spawn_points 

        # Rebuild Pedestrian Spawn/Dest Points (Local -> World)
        self._build_ped_points()

    def _build_ped_points(self):
        # Define in LOCAL coordinates 
        local_spawns = [
            np.array([-50.0, -100.0]), # North sidewalk
            np.array([-50.0, 100.0]),  # South sidewalk
            np.array([100.0, -50.0]),  # East sidewalk
            np.array([-100.0, -50.0]), # West sidewalk
        ]
        self.spawn_points = [self._local_to_world(p) for p in local_spawns]
        
        local_dests = [
            np.array([-50.0, 100.0]), # Opposites
            np.array([-50.0, -100.0]), 
            np.array([-100.0, -50.0]), 
            np.array([100.0, -50.0])
        ]
        self.destination_points = [self._local_to_world(p) for p in local_dests]

    def _build_car_spawns(self):
        # Define in LOCAL coordinates
        spawns = []
        # South road (bottom, +y local), going North (-y local, -pi/2 heading)
        # Right side: x > 0.
        h_up = -np.pi / 2 + self.layout_rotation
        p_south = self._local_to_world(np.array([20.0, 250.0]))
        spawns.append({"pos": p_south, "heading": h_up, "direction": "north", "type": "vertical"})

        # North road (top, -y local), going South (+y local, +pi/2 heading)
        # Right side: x < 0.
        h_down = np.pi / 2 + self.layout_rotation
        p_north = self._local_to_world(np.array([-20.0, -250.0]))
        spawns.append({"pos": p_north, "heading": h_down, "direction": "south", "type": "vertical"})

        # East road (right, +x local), going West (-x local, pi heading)
        # Right side: y < 0.
        h_left = np.pi + self.layout_rotation
        p_east = self._local_to_world(np.array([250.0, -20.0]))
        spawns.append({"pos": p_east, "heading": h_left, "direction": "west", "type": "horizontal"})

        # West road (left, -x local), going East (+x local, 0 heading)
        # Right side: y > 0.
        h_right = 0.0 + self.layout_rotation
        p_west = self._local_to_world(np.array([-250.0, 20.0]))
        spawns.append({"pos": p_west, "heading": h_right, "direction": "east", "type": "horizontal"})
        
        return spawns

    def _build_crossings(self):
        crossings = []
        # North
        c1_start = self._local_to_world(np.array([-50.0, -50.0]))
        c1_end = self._local_to_world(np.array([50.0, -50.0]))
        crossings.append({"start": c1_start, "end": c1_end, "direction": "horizontal", "inter_id": 0})
        # South
        c2_start = self._local_to_world(np.array([-50.0, 50.0]))
        c2_end = self._local_to_world(np.array([50.0, 50.0]))
        crossings.append({"start": c2_start, "end": c2_end, "direction": "horizontal", "inter_id": 0})
        # East
        c3_start = self._local_to_world(np.array([50.0, -50.0]))
        c3_end = self._local_to_world(np.array([50.0, 50.0]))
        crossings.append({"start": c3_start, "end": c3_end, "direction": "vertical", "inter_id": 0})
        # West
        c4_start = self._local_to_world(np.array([-50.0, -50.0]))
        c4_end = self._local_to_world(np.array([-50.0, 50.0]))
        crossings.append({"start": c4_start, "end": c4_end, "direction": "vertical", "inter_id": 0})
        return crossings

    def _check_valid_exit(self):
        # Convert agent pos to local
        local_pos = self._world_to_local(self._agent_location)
        x, y = local_pos[0], local_pos[1]
        
        # Valid exit if on road (abs(coord) < width/2)
        in_horizontal_lane = abs(y) <= 50
        in_vertical_lane = abs(x) <= 50
        return in_horizontal_lane or in_vertical_lane