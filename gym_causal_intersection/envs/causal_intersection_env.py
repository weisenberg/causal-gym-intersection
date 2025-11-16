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

    def __init__(self, render_mode=None, observation_type: str = "kinematic"):
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
            "traffic_light_duration": 30,
            "pedestrian_spawn_rate": 0.05,  # Probability of spawning a pedestrian per step (increased)
            "num_pedestrians": 8,  # Maximum number of pedestrians (increased)
            "jaywalk_probability": 0.15  # Probability that a pedestrian will jaywalk (reduced - more use crossings)
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
        
        # Single intersection at center (4-way)
        self.intersections = [
            {"pos": np.array([300.0, 300.0]), "id": 0, "approaches": ["north", "south", "east", "west"]},
        ]
        # Traffic lights per intersection with directional states
        self.traffic_lights = {}  # id -> {timer, phase, directions}
        self.traffic_light_timer = 0
        self._initialize_traffic_lights()
        
        # Zebra crossings for each intersection (4 sides if approach exists)
        self.zebra_crossings = self._build_crossings()
        
        # NPC cars (lightweight traffic)
        self.npc_cars = []
        self.npc_car_speed = 3.5  # legacy default; per-car overrides are created at spawn
        self.car_spawn_points = self._build_car_spawns()
        
        # Time tracking for success criterion (59 seconds)
        self.step_count = 0
        self.success_after_steps = int(59 * self.metadata["render_fps"])
        
        # Car spawn points on the right side of roads, facing towards intersection
        # Road center is at 300, road spans 250-350 (100px wide)
        # Right side offset: ~20 pixels from center towards the right
        self._car_spawn_points = [
            # South road (bottom), going north towards intersection, right side
            {"pos": np.array([320.0, 550.0]), "heading": np.pi / 2},  # π/2 = up (north)
            # North road (top), going south towards intersection, right side  
            {"pos": np.array([320.0, 50.0]), "heading": -np.pi / 2},  # -π/2 = down (south)
            # East road (right), going west towards intersection, right side
            {"pos": np.array([550.0, 270.0]), "heading": np.pi},  # π = left (west)
            # West road (left), going east towards intersection, right side
            {"pos": np.array([50.0, 330.0]), "heading": 0.0},  # 0 = right (east)
        ]
        
        # Pedestrian system
        self.pedestrians = []  # List of pedestrian dicts: {pos, target, speed, is_jaywalking, crossed}
        self.pedestrian_radius = 5
        self.pedestrian_speed = 0.5  # Pixels per step
        
        # Legacy crossing references retained for compatibility (not used)
        
        # Pedestrian spawn points - on sidewalks, one side of each road
        # Pedestrians will cross to the opposite side using zebra crossings
        # Roads are 100px wide (250-350), so sidewalks are outside this range
        # Mapping: spawn_idx -> crossing_idx
        # 0: North sidewalk -> uses North crossing (horizontal, east-west)
        # 1: South sidewalk -> uses South crossing (horizontal, east-west)
        # 2: East sidewalk -> uses East crossing (vertical, north-south)
        # 3: West sidewalk -> uses West crossing (vertical, north-south)
        self.spawn_points = [
            # North sidewalk (top) - uses north crossing (horizontal)
            np.array([250.0, 200.0]),   # West side of north sidewalk
            # South sidewalk (bottom) - uses south crossing (horizontal)
            np.array([250.0, 400.0]),  # West side of south sidewalk
            # East sidewalk (right) - uses east crossing (vertical)
            np.array([400.0, 250.0]),  # North side of east sidewalk
            # West sidewalk (left) - uses west crossing (vertical)
            np.array([200.0, 250.0]),   # North side of west sidewalk
        ]
        
        # Destination points for pedestrians after crossing (opposite sidewalks)
        self.destination_points = [
            np.array([350.0, 400.0]),   # From north -> to south (east side)
            np.array([350.0, 200.0]),  # From south -> to north (east side)
            np.array([200.0, 350.0]),  # From east -> to west (south side)
            np.array([400.0, 350.0]),   # From west -> to east (south side)
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Update context if options provided
        if options is not None and isinstance(options, dict):
            self.context.update(options)
        
        # Reset state - spawn car on right side of a random road, facing intersection
        spawn_idx = self.np_random.integers(0, len(self._car_spawn_points))
        car_spawn = self._car_spawn_points[spawn_idx]
        self._agent_location = car_spawn["pos"].copy().astype(np.float32)
        self._agent_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self._agent_heading = car_spawn["heading"]
        
        # Reset pedestrians
        self.pedestrians = []
        # Reset time and NPCs
        self.step_count = 0

        observation = self._get_obs()
        info = {}

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
        cruise_speed = float(self.npc_car_speed * (0.7 + self.np_random.random()*0.8))  # ~[2.45..6.3]
        max_speed = cruise_speed * (1.1 + self.np_random.random()*0.3)  # slightly higher than cruise
        accel = float(0.18 + self.np_random.random()*0.22)  # [0.18..0.40]
        brake_factor = float(0.82 + self.np_random.random()*0.12)  # [0.82..0.94]
        stopping_distance = float(30 + self.np_random.random()*40)  # [30..70]
        # Car sizes (allow wider bodies)
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
            "length": car_len,
            "width": car_wid,
            "color": color
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
            # keep in lane (right-side) for single vertical/horizontal roads
            if car["type"] == "horizontal":
                # Horizontal right-side: eastbound = bottom (y=330), westbound = upper (y=270)
                car["pos"][1] = 330.0 if car["direction"] == "east" else 270.0
            else:
                # Vertical road right-side: northbound at x=320 (east side), southbound at x=280 (west side)
                car["pos"][0] = 320.0 if car["direction"] == "north" else 280.0
            # remove off-screen (allow room for off-map spawns to enter)
            if car["pos"][0] < -200 or car["pos"][0] > 800 or car["pos"][1] < -200 or car["pos"][1] > 800:
                cars_to_remove.append(i)
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
                            b["velocity"] *= b.get("brake_factor", 0.9)
                        elif is_b_stopped and not is_a_stopped:
                            # Only move a (b is stopped, don't push it)
                            a["pos"] -= dir_vec * move
                            a["target_speed"] = 0.0
                            a["velocity"] *= a.get("brake_factor", 0.9)
                        elif not is_a_stopped and not is_b_stopped:
                            # Both moving - move both apart
                            a["pos"] -= dir_vec * (move / 2.0)
                            b["pos"] += dir_vec * (move / 2.0)
                            a["target_speed"] = 0.0
                            b["target_speed"] = 0.0
                            a["velocity"] *= a.get("brake_factor", 0.9)
                            b["velocity"] *= b.get("brake_factor", 0.9)
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
                    car["target_speed"] = 0.0
                    car["velocity"] *= car.get("brake_factor", 0.85)
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
                        car["target_speed"] = 0.0
                        car["velocity"] *= car.get("brake_factor", 0.9)

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
            accel = self.acceleration * accel_cmd
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
        # Steering
        self._agent_heading += self.angular_velocity * steer_cmd
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
        
        # Spawn new NPC cars (low rate)
        if self.np_random.random() < 0.05:
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
        
        # Check for pedestrian collision
        collision = self._check_pedestrian_collision()
        
        # Shaped rewards
        # + survival
        reward += 0.1
        # + progress/efficiency
        agent_speed = float(np.linalg.norm(self._agent_velocity))
        reward += 1.0 * (agent_speed / max(1e-6, self.max_speed))
        # - running a red light
        if self._agent_runs_red_light():
            reward -= 0.5
        # - safety buffer proximity to NPC cars
        for car in self.npc_cars:
            if np.linalg.norm(car["pos"] - self._agent_location) < self.safety_buffer:
                reward -= 0.1
                break
        # - zebra crossing with pedestrian on it
        if self._agent_on_active_crossing():
            reward -= 2.0
        
        # Terminations and success
        info = {
            "num_pedestrians": len(self.pedestrians),
            "collision": collision,
            "num_npc_cars": len(self.npc_cars),
            "success": False
        }
        if collision:
            terminated = True
            reward -= 100.0  # crash penalty
        elif off_screen:
            terminated = True
            reward += 10.0  # existing off-screen success bonus
        else:
            # step-based success if alive for 59 seconds
            self.step_count += 1
            if self.step_count >= self.success_after_steps:
                terminated = True
                info["success"] = True  # success signal; no extra reward requested

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _agent_runs_red_light(self):
        """Detect if agent is entering/inside intersection while its direction has red."""
        inter = self._get_nearest_intersection(self._agent_location)
        if inter is None:
            return False
        pos = inter["pos"]
        edge = self.intersection_size / 2.0
        near = abs(self._agent_location[0] - pos[0]) < edge + 8 and abs(self._agent_location[1] - pos[1]) < edge + 8
        if not near:
            return False
        # Determine cardinal travel direction
        if abs(np.cos(self._agent_heading)) > abs(np.sin(self._agent_heading)):
            direction = "east" if np.cos(self._agent_heading) > 0 else "west"
        else:
            direction = "south" if np.sin(self._agent_heading) < 0 else "north"
        state = self._get_traffic_light_state(inter["id"], direction)
        return state == "red"

    def _agent_on_active_crossing(self):
        """Penalty condition: agent overlaps a zebra crossing while a pedestrian is on it."""
        for idx, zc in enumerate(self.zebra_crossings):
            if zc["direction"] == "horizontal":
                y = zc["start"][1]
                x0, x1 = min(zc["start"][0], zc["end"][0]), max(zc["start"][0], zc["end"][0])
                if abs(self._agent_location[1] - y) < 6 and x0 - 5 <= self._agent_location[0] <= x1 + 5:
                    # any ped currently on this crossing?
                    for ped in self.pedestrians:
                        if (not ped["is_jaywalking"]) and ped.get("crossing_idx", -1) == idx and ped.get("phase") == "on_crossing":
                            return True
            else:
                x = zc["start"][0]
                y0, y1 = min(zc["start"][1], zc["end"][1]), max(zc["start"][1], zc["end"][1])
                if abs(self._agent_location[0] - x) < 6 and y0 - 5 <= self._agent_location[1] <= y1 + 5:
                    for ped in self.pedestrians:
                        if (not ped["is_jaywalking"]) and ped.get("crossing_idx", -1) == idx and ped.get("phase") == "on_crossing":
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

        # Create canvas
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((128, 128, 128))  # Gray background
        
        # Draw roads: single vertical and horizontal cross
        pygame.draw.rect(canvas, (64, 64, 64), pygame.Rect(0, 250, 600, 100))  # horizontal
        pygame.draw.rect(canvas, (64, 64, 64), pygame.Rect(250, 0, 100, 600))  # vertical
        
        # Draw intersections and traffic lights
        for inter in self.intersections:
            pos = inter["pos"]
            pygame.draw.rect(canvas, (80, 80, 80),
                             pygame.Rect(pos[0]-self.intersection_size/2, pos[1]-self.intersection_size/2,
                                         self.intersection_size, self.intersection_size))
            light = self.traffic_lights[inter["id"]]
            light_offset = self.intersection_size//2 - 12
            size = 5
            # north
            if "north" in inter["approaches"]:
                color = (0,255,0) if light["directions"]["north"]=="green" else (255,255,0) if light["directions"]["north"]=="yellow" else (255,0,0)
                pygame.draw.circle(canvas, color, (int(pos[0]), int(pos[1]-light_offset)), size)
            # south
            if "south" in inter["approaches"]:
                color = (0,255,0) if light["directions"]["south"]=="green" else (255,255,0) if light["directions"]["south"]=="yellow" else (255,0,0)
                pygame.draw.circle(canvas, color, (int(pos[0]), int(pos[1]+light_offset)), size)
            # east
            if "east" in inter["approaches"]:
                color = (0,255,0) if light["directions"]["east"]=="green" else (255,255,0) if light["directions"]["east"]=="yellow" else (255,0,0)
                pygame.draw.circle(canvas, color, (int(pos[0]+light_offset), int(pos[1])), size)
            # west
            if "west" in inter["approaches"]:
                color = (0,255,0) if light["directions"]["west"]=="green" else (255,255,0) if light["directions"]["west"]=="yellow" else (255,0,0)
                pygame.draw.circle(canvas, color, (int(pos[0]-light_offset), int(pos[1])), size)
        
        # Draw zebra crossings (white stripes)
        stripe_width = 8
        stripe_spacing = 4
        for crossing in self.zebra_crossings:
            if crossing["direction"] == "horizontal":
                # Horizontal crossing: draw vertical stripes
                start_x = int(crossing["start"][0])
                end_x = int(crossing["end"][0])
                y = int(crossing["start"][1])
                x = start_x
                while x < end_x:
                    pygame.draw.rect(canvas, (255, 255, 255), 
                                   pygame.Rect(x, y - 3, stripe_width, 6))
                    x += stripe_width + stripe_spacing
            else:
                # Vertical crossing: draw horizontal stripes
                start_y = int(crossing["start"][1])
                end_y = int(crossing["end"][1])
                x = int(crossing["start"][0])
                y = start_y
                while y < end_y:
                    pygame.draw.rect(canvas, (255, 255, 255), 
                                   pygame.Rect(x - 3, y, 6, stripe_width))
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