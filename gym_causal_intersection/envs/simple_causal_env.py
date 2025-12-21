import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from gym_causal_intersection.envs.causal_intersection_env import UrbanCausalIntersectionEnv

class SimpleCausalIntersectionEnv(UrbanCausalIntersectionEnv):
    """
    A simplified version of UrbanCausalIntersectionEnv with:
    - Single vertical road (no intersection)
    - One zebra crossing with traffic light
    - Reduced Lidar (4 directions)
    - Discrete Action Space
    - Discretized Observation Space
    """
    def __init__(self, render_mode=None, max_npcs=2, max_pedestrians=2):
        # Initialize parent with reduced defaults
        super().__init__(render_mode=render_mode, max_npcs=max_npcs, max_pedestrians=max_pedestrians)
        self.max_pedestrians = 30 # Increased default for lively demo
        
        # Override context for simple env
        self.context["traffic_light_duration"] = 300 # 10 seconds at 30 FPS
        
        # --- Discrete Action Space ---
        # 0: Idle (maintain speed)
        # 1: Accelerate
        # 2: Brake
        # 3: Steer Left (small)
        # 4: Steer Right (small)
        self.action_space = spaces.Discrete(5)
        
        # --- Full State Observation Space ---
        # Agent (6) + Road (13) + NPCs (5*7=35) + Peds (30*5=150) + Light (2) = 206
        # Round up to 210 for safety/padding
        self.obs_max_npcs = 5
        self.obs_max_peds = 30
        self.obs_dim = 6 + 13 + (self.obs_max_npcs * 7) + (self.obs_max_peds * 5) + 2
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # Override intersection to be just a single "virtual" intersection for the light
        # This will be handled in _generate_layout
        
        # Override spawn points for the single crossing
        # This will be handled in _generate_layout -> _build_ped_points
        

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        # Correct last_waypoint_index based on actual spawn
        # Otherwise _check_off_road and Lidar will fail in first step
        if self._agent_location is not None:
            min_d = float('inf')
            best_idx = 0
            # Search all waypoints
            for i, p in enumerate(self.track_waypoints):
                d = np.linalg.norm(self._agent_location - p)
                if d < min_d:
                    min_d = d
                    best_idx = i
            self.last_waypoint_index = best_idx
            
            # --- VALIDATION: Check for overlapping NPCs ---
            # Remove any NPC that is too close to the agent
            safe_distance = 25.0 # meters
            valid_npcs = []
            for car in self.npc_cars:
                d = np.linalg.norm(car["pos"] - self._agent_location)
                if d > safe_distance:
                    valid_npcs.append(car)
                else:
                    pass # print(f"DEBUG: Removed colliding NPC at {d:.2f}m")
            self.npc_cars = valid_npcs
            # -----------------------------------------------
            
        # Re-compute observation with correct state
        obs = self._get_obs()
        return obs, info

    def _build_agent_spawns(self):
        spawns = []
        # Fixed Index 20 (Safe Start)
        idx = 20
        if idx < len(self.track_data):
            d = self.track_data[idx]
            # Center of Right Lane (-normal * 10)
            pos = d["pos"] - d["normal"] * 10.0
            heading = np.arctan2(d["tangent"][1], d["tangent"][0])
            spawns.append({
                "pos": pos, 
                "heading": heading, 
                "direction": "north", 
                "type": "vertical"
            })
        return spawns

    def _generate_layout(self, randomize=False):
        """Generate the simple road layout, potentially randomized."""
        if randomize:
            self.layout_center = self.np_random.uniform(200.0, 400.0, size=2)
            self.layout_rotation = self.np_random.uniform(0, 2 * np.pi)
        else:
            self.layout_center = np.array([300.0, 300.0])
            self.layout_rotation = 0.0

        self.intersection_center = self.layout_center

        # Rebuild Intersections (World Coords) - Only North/South approaches
        self.intersections = [
            {"pos": self.layout_center.copy(), "id": 0, "approaches": ["north", "south"]},
        ]
        
        self.track_width = 80.0
        # Generate full track data (points, normals, left/right edges)
        self.track_data = self._generate_spline_road()
        self.track_waypoints = [p["pos"] for p in self.track_data] 
        self.last_waypoint_index = 0
        
        self._initialize_traffic_lights()
        
        # Crossing strictly on the road (e.g. at index 50)
        self.zebra_crossings = self._build_crossings()
        
        # Dynamic Spawns based on track
        self.npc_spawn_points = self._build_car_spawns()
        self.agent_spawn_points = self._build_agent_spawns()
        
        self.car_spawn_points = self.npc_spawn_points
        self._car_spawn_points = self.car_spawn_points
        
        self._build_ped_points()

    def _generate_spline_road(self):
        # Generate points
        num_points = 400
        # Fit within enlarged window (-1000 to 1000)
        # Local Coords: -1000 to 1000 -> World: Relative to layout center
        y_points = np.linspace(-1000, 1000, num_points)
        x_points = np.zeros(num_points)
        
        freq = self.np_random.uniform(0.001, 0.003) # Very low freq
        amp = self.np_random.uniform(20, 50)        # Low amplitude
        phase = self.np_random.uniform(0, 2*np.pi)
        x_points = amp * np.sin(freq * y_points + phase)
        
        # Second weak layer
        freq2 = self.np_random.uniform(0.01, 0.02)
        amp2 = self.np_random.uniform(5, 10)
        x_points += amp2 * np.sin(freq2 * y_points)
        
        # Calculate raw points
        raw_points = []
        for i in range(num_points):
            local_p = np.array([x_points[i], y_points[i]])
            world_p = self._local_to_world(local_p)
            raw_points.append(world_p)
            
        # Calculate Normals (Miter Joint style to prevent gaps)
        track_data = []
        half_w = self.track_width / 2.0
        
        for i in range(num_points):
            p = raw_points[i]
            
            # Estimate tangent
            if i == 0:
                tangent = raw_points[i+1] - p
            elif i == num_points - 1:
                tangent = p - raw_points[i-1]
            else:
                # Average tangent
                tangent = raw_points[i+1] - raw_points[i-1]
                
            # Normalize tangent
            t_norm = np.linalg.norm(tangent)
            if t_norm > 0: tangent /= t_norm
            else: tangent = np.array([0, 1]) # fallback
            
            # Normal is perpendicular to tangent
            normal = np.array([-tangent[1], tangent[0]])
            
            # Vertices
            left_v = p + normal * half_w
            right_v = p - normal * half_w
            
            track_data.append({
                "pos": p,
                "normal": normal,
                "tangent": tangent,
                "left": left_v,
                "right": right_v
            })
            
        return track_data


    def _build_car_spawns(self):
        # Spawn NPCs along the track with spacing to prevent overlap
        spawns = []
        # Strictly Index 50+ (Agent is at 20)
        start_idx = 50
        step = 40 # 40 indices ~ 80 meters spacing
        
        indices = range(start_idx, len(self.track_data), step)
        
        for idx in indices:
            if idx >= len(self.track_data): continue
            
            data = self.track_data[idx]
            pos = data["pos"]
            
            # Simple Logic: One car per slot, alternating lanes?
            # Or just spawn in North lane for chasing?
            # Let's put them in BOTH lanes for traffic.
            
            # Lane 1 (North)
            p1 = pos - data["normal"] * 10.0
            h1 = np.arctan2(data["tangent"][1], data["tangent"][0])
            spawns.append({
                "pos": p1, "heading": h1, "direction": "north", "type": "vertical"
            })
            
            # Lane 2 (South) - careful, they drive opposite
            # p2 = pos + data["normal"] * 10.0
            # h2 = np.arctan2(-data["tangent"][1], -data["tangent"][0])
            # spawns.append({
            #    "pos": p2, "heading": h2, "direction": "south", "type": "vertical"
            # })
            
        return spawns
        


    def _build_crossings(self):
        # Place crossing exactly at index 50
        idx = 50
        data = self.track_data[idx]
        p = data["pos"]
        n = data["normal"] # Points Left
        w = self.track_width / 2.0
        
        # Start (Left) to End (Right) or vice versa
        # Let's say Start is Left (p + n*w)
        c_start = p + n * w
        c_end = p - n * w
        
        return [{
            "start": c_start, 
            "end": c_end, 
            "direction": "horizontal", 
            "inter_id": 0
        }]

    def _build_ped_points(self):
        # Make sidewalks match the crossing
        crossing = self.zebra_crossings[0]
        start = crossing["start"]
        end = crossing["end"]
        # Just spawn slightly off-road at crossing endpoints
        self.spawn_points = [start - (start-end)*0.1] 
        self.destination_points = [end + (end-start)*0.1]

    def _spawn_pedestrian(self):
        # Override to safely handle the single crossing environment + Jaywalking
        if len(self.pedestrians) >= self.max_pedestrians:
            return

        # Simple logic: Spawn at start or end of the single crossing OR random spot
        
        # 50% chance of jaywalking
        if self.np_random.random() < 0.5:
             # Jaywalking at random spot
             # Pick safe index
             idx = self.np_random.integers(10, len(self.track_data)-10)
             if abs(idx - 50) < 5: idx = 60 # Avoid zebra area overlap
             
             data = self.track_data[idx]
             p = data["pos"]
             n = data["normal"]
             w = self.track_width / 2.0 + 5.0
             
             if self.np_random.random() < 0.5:
                 pos = p + n * w
                 target = p - n * w
             else:
                 pos = p - n * w
                 target = p + n * w
                 
             crossing_idx = -1 # No crossing
             is_jaywalking = True
             
        else:
             # Zebra Crossing (Legal)
             crossing_idx = 0
             if not self.zebra_crossings:
                 return
             crossing = self.zebra_crossings[crossing_idx]
            
             # Pick start or end
             if self.np_random.random() < 0.5:
                 pos = crossing["start"].copy()
                 target = crossing["end"].copy()
             else:
                 pos = crossing["end"].copy()
                 target = crossing["start"].copy()
             
             is_jaywalking = False
            
        ped = {
            "pos": pos.astype(np.float32),
            "target": target.astype(np.float32),
            "state": "walking", 
            "waiting_time": 0,
            "crossing_idx": crossing_idx,
            "speed": 2.0,
            "is_jaywalking": is_jaywalking,
            "phase": "crossing_road" # Simplified
        }
        self.pedestrians.append(ped)

    def _update_npc_cars(self):
        # NPC Control Loop for Spline Following
        dt = 1.0 # 1 step
        
        for i, car in enumerate(self.npc_cars):
            # 1. Find nearest waypoint index
            # Optimization: search near last known index if we stored it
            # For now, global search is okay for small N
            # self.track_waypoints matches self.track_data indices
            dists = np.linalg.norm(np.array(self.track_waypoints) - car["pos"], axis=1)
            closest_idx = np.argmin(dists)
            
            # 2. Determine Lane Target
            # We want them to drive on the right side relative to road direction
            
            target_speed = car["max_speed"]
            
            # Simple Logic: Follow the spline in direction of current heading
            # Check alignment with tangent
            tangent = self.track_data[closest_idx]["tangent"]
            heading_vec = np.array([np.cos(car["heading"]), np.sin(car["heading"])])
            dot = np.dot(tangent, heading_vec)
            
            moving_forward = dot > 0
            
            # Lookahead
            lookahead = 5
            if moving_forward:
                target_idx = min(len(self.track_data)-1, closest_idx + lookahead)
                lane_offset_vec = -self.track_data[target_idx]["normal"] * 20.0 # Right side
            else:
                target_idx = max(0, closest_idx - lookahead)
                lane_offset_vec = self.track_data[target_idx]["normal"] * 20.0 # Left side (which is Right for oncoming)
                
            target_pos = self.track_data[target_idx]["pos"] + lane_offset_vec
            
            # Steering Control
            desired_heading = np.arctan2(target_pos[1] - car["pos"][1], target_pos[0] - car["pos"][0])
            
            # Normalize angle diff
            angle_diff = (desired_heading - car["heading"] + np.pi) % (2 * np.pi) - np.pi
            
            # Apply turn limit
            max_turn = 0.3
            turn = np.clip(angle_diff, -max_turn, max_turn)
            car["heading"] += turn
            
            # Collision Avoidance (Braking)
            # Check for car in front
            dist_to_front = float('inf')
            car_len = car.get("length", 40)
            
            for other in self.npc_cars:
                if other is car: continue
                # Only care about cars in same lane/direction
                if other.get("direction") != car.get("direction"):
                    continue
                    
                d = np.linalg.norm(other["pos"] - car["pos"])
                other_len = other.get("length", 40)
                
                # Check if it's effectively in range
                if d < 100.0:
                    to_other = other["pos"] - car["pos"]
                    # Check if in front (dot product)
                    if np.dot(to_other, heading_vec) > 0:
                        # Angle check to ensure it's actually ahead in lane, not just roughly ahead
                        # Normalized dot?
                        # If strict lane following, just distance is mostly enough if we filtered by direction
                        
                        # Surface distance (Boundary to Boundary)
                        surf_dist = d - (car_len/2.0 + other_len/2.0)
                        dist_to_front = min(dist_to_front, surf_dist)
                        
            # Check Agent (Agent might interfere)
            if self._agent_location is not None:
                d = np.linalg.norm(self._agent_location - car["pos"])
                if d < 100.0:
                    to_agent = self._agent_location - car["pos"]
                    if np.dot(to_agent, heading_vec) > 0:
                        # Rough check for agent size (assume 40)
                        surf_dist = d - (car_len/2.0 + 20.0) 
                        dist_to_front = min(dist_to_front, surf_dist)
                        
            # Braking Logic based on Surface Distance
            if dist_to_front < 10.0: # Very close (bumper to bumper)
                target_speed = 0.0
            elif dist_to_front < 40.0: # Moderate distance
                target_speed *= 0.3 # Slow down
            elif dist_to_front < 80.0: # Long range
                target_speed *= 0.8 # Slight adjustment
                
            # Physics Update
            # Accel/Decel
            current_speed = np.linalg.norm(car["velocity"])
            if current_speed < target_speed:
                current_speed += car["accel"]
            else:
                current_speed -= car["accel"]
            
            # Apply velocity
            car["velocity"] = np.array([np.cos(car["heading"]), np.sin(car["heading"])]) * current_speed
            car["pos"] += car["velocity"]

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        if not pygame.font.get_init():
            pygame.font.init()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 200, 0))  # Grass
        
        # --- Camera Transform ---
        # Center the agent on screen
        # Screen Center
        cx, cy = self.window_size / 2.0, self.window_size / 2.0
        
        # Camera Focus
        if self._agent_location is not None:
             camera_x, camera_y = self._agent_location
        else:
             camera_x, camera_y = self.layout_center
             
        # Offset to apply to all world coords
        offset_x = cx - camera_x
        offset_y = cy - camera_y
        
        def to_screen(pos):
             return (pos[0] + offset_x, pos[1] + offset_y)
        
        # --- Draw Continuous Road Strip ---
        if len(self.track_data) > 1:
            # Draw quads joining i to i+1
            for i in range(len(self.track_data) - 1):
                d1 = self.track_data[i]
                d2 = self.track_data[i+1]
                
                # Transform to screen
                p1l = to_screen(d1["left"])
                p2l = to_screen(d2["left"])
                p2r = to_screen(d2["right"])
                p1r = to_screen(d1["right"])
                
                poly = [p1l, p2l, p2r, p1r]
                pygame.draw.polygon(canvas, (64, 64, 64), poly)
                
                # Center Line (White)
                if i % 2 == 0:
                    pygame.draw.line(canvas, (255, 255, 255), to_screen(d1["pos"]), to_screen(d2["pos"]), 2)
            
            # Draw Borders (Red/White curb)
            left_points = [to_screen(d["left"]) for d in self.track_data]
            right_points = [to_screen(d["right"]) for d in self.track_data]
            pygame.draw.lines(canvas, (200, 0, 0), False, left_points, 3)
            pygame.draw.lines(canvas, (200, 0, 0), False, right_points, 3)
            
            # --- Draw Finish Line ---
            # At last index - 2 (near very end)
            f_idx = len(self.track_data) - 2
            f_data = self.track_data[f_idx]
            f_p1 = to_screen(f_data["left"])
            f_p2 = to_screen(f_data["right"])
            # Draw Checkerboard pattern or simple Black/White line
            pygame.draw.line(canvas, (255, 255, 255), f_p1, f_p2, 10)
            # Add checkers
            mid = ((f_p1[0]+f_p2[0])/2, (f_p1[1]+f_p2[1])/2)
            pygame.draw.line(canvas, (0, 0, 0), f_p1, mid, 10)
            
        # --- Draw Traffic Lights ---
        if self.intersections and self.traffic_lights:
            light = self.traffic_lights[self.intersections[0]["id"]]
            is_car_green = light["directions"]["north"] == "green"
        
            idx = 50
            data = self.track_data[idx]
            p = data["pos"]
            t = data["tangent"]
            n = data["normal"] 
            
            w = self.track_width / 2.0 + 10
            car_light_pos = p - n*w - t*20
            ped_light_pos = p - n*w + t*20
            
            car_col = (0, 255, 0) if is_car_green else (255, 0, 0)
            ped_col = (255, 0, 0) if is_car_green else (0, 255, 0)
            
            pygame.draw.circle(canvas, car_col, np.array(to_screen(car_light_pos)).astype(int), 10)
            pygame.draw.circle(canvas, ped_col, np.array(to_screen(ped_light_pos)).astype(int), 8)
        
        # --- Draw Zebra Crossings ---
        for crossing in self.zebra_crossings:
            pygame.draw.line(canvas, (255, 255, 255), to_screen(crossing["start"]), to_screen(crossing["end"]), 6)

        # --- Draw NPC Cars ---
        for car in self.npc_cars:
            car_pos = np.array(to_screen(car["pos"])) # Screen coords
            # Use raw subheading for rotation
            cos_h = np.cos(car["heading"])
            sin_h = np.sin(car["heading"])
            length, width = car.get("length", 40), car.get("width", 20)
            
            # Local corners
            corners = np.array([
                [-length / 2, -width / 2],
                [length / 2, -width / 2],
                [length / 2, width / 2],
                [-length / 2, width / 2]
            ])
            rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
            rotated = (rot @ corners.T).T + car_pos
            pygame.draw.polygon(canvas, car["color"], rotated)
            
        # --- Draw Pedestrians ---
        for ped in self.pedestrians:
            ped_pos = np.array(to_screen(ped["pos"])).astype(int)
            pygame.draw.circle(canvas, (255, 255, 0), ped_pos, 5)
            # Draw walking direction hint
            direction = ped["target"] - ped["pos"]
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction_normalized = direction / norm
                front_pos = ped_pos + (direction_normalized * 10).astype(int)
                pygame.draw.line(canvas, (255, 255, 0), ped_pos, front_pos, 2)

        # --- Draw Agent ---
        if self._agent_location is not None:
            # Should be exactly at center (cx, cy)
            agent_pos = np.array(to_screen(self._agent_location)).astype(int)
            
            cos_h = np.cos(self._agent_heading)
            sin_h = np.sin(self._agent_heading)
            corners = np.array([
                [-self.car_length / 2, -self.car_width / 2],
                [self.car_length / 2, -self.car_width / 2],
                [self.car_length / 2, self.car_width / 2],
                [-self.car_length / 2, self.car_width / 2]
            ])
            rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
            r_corners = (rot @ corners.T).T + agent_pos
            pygame.draw.polygon(canvas, (255, 0, 0), r_corners)
            # Front indicator
            front_offset = np.array([self.car_length / 2 * cos_h, self.car_length / 2 * sin_h])
            front_pos = agent_pos + front_offset
            pygame.draw.circle(canvas, (200, 0, 0), front_pos.astype(int), 3)
        
        # --- Info Text ---
        if pygame.font:
            font = pygame.font.Font(None, 36)
            # Environment info
            temp_text = font.render(f"Temp: {self.context.get('temperature', 20)} C", True, (255, 255, 255))
            rough_text = font.render(f"Roughness: {self.context.get('roughness', 0.0):.2f}", True, (255, 255, 255))
            step_text = font.render(f"Step: {self.step_count}", True, (255, 255, 255))
            reward_text = font.render(f"Reward: {self.episode_reward:.1f}", True, (255, 255, 255))
            
            canvas.blit(temp_text, (10, 10))
            canvas.blit(rough_text, (10, 50))
            canvas.blit(step_text, (10, 90))
            canvas.blit(reward_text, (10, 130))
            
            # Light Timer
            if self.intersections:
                light = self.traffic_lights[self.intersections[0]["id"]]
                duration = self.context.get("traffic_light_duration", 300)
                timer = light["timer"]
                timer_text = font.render(f"Light: {duration - timer}", True, (255, 255, 255))
                canvas.blit(timer_text, (10, 170))
        
        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def _render_frame_old(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        if not pygame.font.get_init():
            pygame.font.init()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 200, 0))  # Grass
        
        # --- Draw Continuous Road Strip ---
        if len(self.track_data) > 1:
            # Draw quads joining i to i+1
            for i in range(len(self.track_data) - 1):
                d1 = self.track_data[i]
                d2 = self.track_data[i+1]
                
                # Polygon: L1 -> L2 -> R2 -> R1
                poly = [d1["left"], d2["left"], d2["right"], d1["right"]]
                pygame.draw.polygon(canvas, (64, 64, 64), poly)
                
                # Center Line (White)
                if i % 2 == 0:
                    pygame.draw.line(canvas, (255, 255, 255), d1["pos"], d2["pos"], 2)
            
            # Draw Borders (Red/White curb)
            # Lines along left and right edges
            left_points = [d["left"] for d in self.track_data]
            right_points = [d["right"] for d in self.track_data]
            pygame.draw.lines(canvas, (200, 0, 0), False, left_points, 3)
            pygame.draw.lines(canvas, (200, 0, 0), False, right_points, 3)
            
        # --- Draw Traffic Lights ---
        if self.intersections and self.traffic_lights:
            light = self.traffic_lights[self.intersections[0]["id"]]
            is_car_green = light["directions"]["north"] == "green"
        
            # Place lights near crossing index 50
            # Offset along tangent?
            idx = 50
            data = self.track_data[idx]
            p = data["pos"]
            t = data["tangent"]
            n = data["normal"] 
            
            # Car Light: Right side, before crossing (backwards along T)
            # Pos = p - n*w - t*20
            w = self.track_width / 2.0 + 10
            car_light_pos = p - n*w - t*20
            
            car_light_color = (0, 255, 0) if is_car_green else (255, 0, 0)
            pygame.draw.circle(canvas, car_light_color, car_light_pos.astype(int), 10)
            
            # Ped Light: Right side, aligned
            ped_light_pos = p - n*w + t*20
            ped_light_color = (255, 0, 0) if is_car_green else (0, 255, 0)
            pygame.draw.circle(canvas, ped_light_color, ped_light_pos.astype(int), 8)
        
        # --- Draw Zebra Crossings ---
        for crossing in self.zebra_crossings:
            pygame.draw.line(canvas, (255, 255, 255), crossing["start"], crossing["end"], 6)

        # --- Draw NPC Cars ---
        for car in self.npc_cars:
            car_pos = car["pos"].astype(int)
            cos_h = np.cos(car["heading"])
            sin_h = np.sin(car["heading"])
            length = car.get("length", 40)
            width = car.get("width", 20)
            corners = np.array([
                [-length / 2, -width / 2],
                [length / 2, -width / 2],
                [length / 2, width / 2],
                [-length / 2, width / 2]
            ])
            rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
            rotated_corners = (rotation_matrix @ corners.T).T + car_pos
            pygame.draw.polygon(canvas, car["color"], rotated_corners)
        
        # --- Draw Pedestrians ---
        for ped in self.pedestrians:
            ped_pos = ped["pos"].astype(int)
            pygame.draw.circle(canvas, (255, 255, 0), ped_pos, 5)
            # Draw walking direction hint
            direction = ped["target"] - ped["pos"]
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction_normalized = direction / norm
                front_pos = ped_pos + (direction_normalized * 10).astype(int)
                pygame.draw.line(canvas, (255, 255, 0), ped_pos, front_pos, 2)

        # --- Draw Agent ---
        if self._agent_location is not None:
            agent_pos = self._agent_location.astype(int)
            cos_h = np.cos(self._agent_heading)
            sin_h = np.sin(self._agent_heading)
            corners = np.array([
                [-self.car_length / 2, -self.car_width / 2],
                [self.car_length / 2, -self.car_width / 2],
                [self.car_length / 2, self.car_width / 2],
                [-self.car_length / 2, self.car_width / 2]
            ])
            rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
            rotated_corners = (rotation_matrix @ corners.T).T + agent_pos
            pygame.draw.polygon(canvas, (255, 0, 0), rotated_corners)
            # Front indicator
            front_offset = np.array([self.car_length / 2 * cos_h, self.car_length / 2 * sin_h])
            front_pos = agent_pos + front_offset
            pygame.draw.circle(canvas, (200, 0, 0), front_pos.astype(int), 3)
        
        # --- Info Text ---
        if pygame.font:
            font = pygame.font.Font(None, 36)
            # Environment info
            temp_text = font.render(f"Temp: {self.context.get('temperature', 20)} C", True, (255, 255, 255))
            rough_text = font.render(f"Roughness: {self.context.get('roughness', 0.0):.2f}", True, (255, 255, 255))
            step_text = font.render(f"Step: {self.step_count}", True, (255, 255, 255))
            reward_text = font.render(f"Reward: {self.episode_reward:.1f}", True, (255, 255, 255))
            
            canvas.blit(temp_text, (10, 10))
            canvas.blit(rough_text, (10, 50))
            canvas.blit(step_text, (10, 90))
            canvas.blit(reward_text, (10, 130))
            
            # Light Timer
            if self.intersections:
                light = self.traffic_lights[self.intersections[0]["id"]]
                duration = self.context.get("traffic_light_duration", 300)
                timer = light["timer"]
                timer_text = font.render(f"Light: {duration - timer}", True, (255, 255, 255))
                canvas.blit(timer_text, (10, 170))

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))


    def _get_track_distance(self, pos, window=50):
        # Determine distance from pos to the nearest point on the spline center line
        min_dist = float('inf')
        
        # Search window
        start = max(0, self.last_waypoint_index - window)
        end = min(len(self.track_waypoints) - 1, self.last_waypoint_index + window)
        
        # If window is too small or index reset, search wider? 
        # For now assume tracking is good.
        
        for i in range(start, end):
             p1 = self.track_waypoints[i]
             if i < len(self.track_waypoints)-1:
                 p2 = self.track_waypoints[i+1]
                 l2 = np.sum((p1 - p2)**2)
                 if l2 == 0: 
                     dist = np.linalg.norm(pos - p1)
                 else:
                     t = max(0, min(1, np.dot(pos - p1, p2 - p1) / l2))
                     projection = p1 + t * (p2 - p1)
                     dist = np.linalg.norm(pos - projection)
             else:
                 dist = np.linalg.norm(pos - p1)
                 
             if dist < min_dist:
                 min_dist = dist
        return min_dist

    def _check_off_road(self):
        # Get distance to center
        dist = self._get_track_distance(self._agent_location, window=50)
        
        # Track width is 80, so half width is 40. Buffer 5.
        if dist > (self.track_width / 2.0 + 5.0):
            return True
        return False
        
    def _update_waypoint_index(self):
        # Search forward/backward locally to update progress
        start_idx = max(0, self.last_waypoint_index - 10)
        end_idx = min(len(self.track_waypoints), self.last_waypoint_index + 50)
        
        min_d = float('inf')
        best_idx = self.last_waypoint_index
        
        for i in range(start_idx, end_idx):
            p = self.track_waypoints[i]
            d = np.linalg.norm(p - self._agent_location)
            if d < min_d:
                min_d = d
                best_idx = i
        
        self.last_waypoint_index = best_idx

    def step(self, action):
        # Map discrete action to continuous control
        # Actions: continuous [acceleration, steering] in [-1, 1]
        
        # Default: maintain
        accel = 0.0
        steer = 0.0
        
        if action == 1: # Accelerate
            accel = 1.0
        elif action == 2: # Brake
            accel = -1.0
        elif action == 3: # Left
            steer = 0.5
        elif action == 4: # Right
            steer = -0.5
            
        continuous_action = np.array([accel, steer], dtype=np.float32)
        
        # 1. Apply action
        self._apply_action(continuous_action)
        
        # 2. Update entities
        self._update_npc_cars()
        self._update_pedestrians()
        
        # Spawn new pedestrians dynamically
        if self.np_random.random() < 0.05: # 5% chance per step ~ 1 per 20 steps
            self._spawn_pedestrian()
            
        self._update_traffic_lights()
        
        # 3. Check collisions/termination
        terminated = False
        truncated = False
        reward = 0.0
        
        # --- unified Waypoint Update & Reward ---
        old_index = self.last_waypoint_index
        self._update_waypoint_index() # Updates self.last_waypoint_index
        new_index = self.last_waypoint_index
        
        # Calculate Progress Reward
        # reward = (new_index - old_index) * scale
        progress = new_index - old_index
        if progress > 0:
            reward += progress * 10.0 # BOOST: +10.0 per Waypoint (> cost of time)
            
        # Penalize time (existence) to encourage speed
        reward -= 0.05 # Reduced from 0.1
        
        # --- Penalties and Termination ---
        # Search forward from last index (window search) -> REMOVED double call
        # self._update_waypoint_index() 
        
        # Off-road check
        if self._check_off_road():
            terminated = True
            reward = -100.0
        elif self._check_collision():
             terminated = True
             reward = -100.0
        elif self._check_pedestrian_collision():
            terminated = True
            reward -= 100.0
        elif self._agent_runs_red_light():
            reward -= 20.0 # Red light penalty
            
        # Success check (reached end of track)
        if self.last_waypoint_index >= len(self.track_waypoints) - 2:
            terminated = True
            reward += 100.0 # Completion bonus
                        
        # Update episode reward for display
        self.episode_reward += reward
        
        # Explicit early termination on success
        if self.episode_reward > 1000.0:
            terminated = True
        
        # Time limit
        self.step_count += 1
        if self.step_count >= self.context.get("max_steps", 1000):
             truncated = True
        if self.step_count >= self.success_after_steps:
            truncated = True
            
        # Get Obs
        obs = self._get_obs()
        info = { 
            "temperature": self.context.get("temperature", 20),
            "progress_index": self.last_waypoint_index
        }
        
        if self.render_mode == "human":
            self._render_frame()
        

        return obs, reward, terminated, truncated, info

    def _apply_action(self, action):
        # Copied/Simplified from parent
        accel_cmd = np.clip(action[0], -1.0, 1.0)
        steer_cmd = np.clip(action[1], -1.0, 1.0)
        
        # Update velocity
        current_speed = np.linalg.norm(self._agent_velocity)
        target_speed = current_speed + accel_cmd * self.acceleration
        target_speed = np.clip(target_speed, -self.max_speed, self.max_speed)
        
        # Update heading (only if moving)
        if current_speed > 0.1:
            self._agent_heading += steer_cmd * self.angular_velocity
            
        # Update position
        self._agent_velocity = np.array([
            np.cos(self._agent_heading) * target_speed,
            np.sin(self._agent_heading) * target_speed
        ])
        self._agent_location += self._agent_velocity

    def _check_collision(self):
        # Check NPC collisions
        agent_poly = (self._agent_location, self._agent_heading, self.car_length, self.car_width)
        for car in self.npc_cars:
            car_poly = (car["pos"], car["heading"], car["length"], car["width"])
            if self._check_obb_collision(*agent_poly, *car_poly):
                return True
        return False

    def _compute_lidar(self, max_range=200.0):
        # 8 rays: Front, Front-Left, Left, Back-Left, Back, Back-Right, Right, Front-Right
        # Angles relative to heading
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4]
        dists = []
        
        obstacles = []
        for car in self.npc_cars:
            obstacles.append((car["pos"], self.car_width / 2))
        for ped in self.pedestrians:
            obstacles.append((ped["pos"], self.pedestrian_radius))
            
        track_half_width = self.track_width / 2.0
            
        for angle_offset in angles:
            angle = self._agent_heading + angle_offset
            ray_dir = np.array([np.cos(angle), -np.sin(angle)])
            
            min_d = max_range
            # Raycast
            for r in np.linspace(5, max_range, 20): # increased resolution
                p = self._agent_location + ray_dir * r
                
                # 1. Wall check (Map Bounds) -> REMOVED for Infinite Road
                # if p[0] < 0 or p[0] > 600 or p[1] < 0 or p[1] > 600:
                #     min_d = min(min_d, r)
                #     break

                    
                # 2. Obstacle check
                for (op, rad) in obstacles:
                    if np.linalg.norm(p - op) <= rad + 5:
                        min_d = min(min_d, r)
                        # print(f"DEBUG: Obs Hit at {r}, p={p}, op={op}")
                        break
                
                # 3. Road Edge Check
                # If distance to center > half_width, it's a "wall"
                rdist = self._get_track_distance(p, window=50)
                if rdist > track_half_width:
                     min_d = min(min_d, r)
                     break
                     
                if min_d < max_range:
                    break
            dists.append(min_d)
            
        return dists

    def _get_obs(self):
        # Return full state vector
        state = []
        
        # 1. Agent State (6)
        # Pos (x,y), Vel (vx,vy), Heading (cos, sin)
        # Normalize Pos to map scale approx (0-600) -> 0-1
        state.extend(self._agent_location / 600.0)
        state.extend(self._agent_velocity / 5.0) # approx max speed
        state.append(np.cos(self._agent_heading))
        state.append(np.sin(self._agent_heading))
        
        # 2. Road State (13)
        # Dist to center, Heading Error, Next 5 Waypoints (relative)
        # Track State
        dist = self._get_track_distance(self._agent_location)
        state.append(dist / 40.0) # Width normalized
        
        # Heading error relative to track tangent
        if self.last_waypoint_index < len(self.track_data):
            tangent = self.track_data[self.last_waypoint_index]["tangent"]
            # Dot product for alignment
            agent_dir = np.array([np.cos(self._agent_heading), np.sin(self._agent_heading)])
            alignment = np.dot(tangent, agent_dir)
            state.append(alignment)
        else:
            state.append(0.0)
            
        # Parametric Progress (0-1)
        state.append(self.last_waypoint_index / len(self.track_data))
        
        # Next 5 Waypoints (Relative to Agent)
        # If running out of track, repeat last
        for i in range(1, 6):
            idx = min(len(self.track_data)-1, self.last_waypoint_index + i * 5) # spacing
            wp = self.track_data[idx]["pos"]
            rel_wp = (wp - self._agent_location) / 600.0
            state.extend(rel_wp)
            
        # 3. NPC State (Max 5 * 7 = 35)
        # [Present, RelX, RelY, RelVX, RelVY, CosH, SinH]
        cars_sorted = sorted(self.npc_cars, key=lambda c: np.linalg.norm(c["pos"] - self._agent_location))
        
        for i in range(self.obs_max_npcs):
            if i < len(cars_sorted):
                c = cars_sorted[i]
                rel_pos = (c["pos"] - self._agent_location) / 600.0
                rel_vel = (c["velocity"] - self._agent_velocity) / 5.0
                state.append(1.0) # Present
                state.extend(rel_pos)
                state.extend(rel_vel)
                state.append(np.cos(c["heading"]))
                state.append(np.sin(c["heading"]))
            else:
                state.extend([0.0]*7)
                
        # 4. Pedestrian State (Max 30 * 5 = 150)
        # [Present, RelX, RelY, VX, VY] (Peds usually don't have heading var, just target)
        peds_sorted = sorted(self.pedestrians, key=lambda p: np.linalg.norm(p["pos"] - self._agent_location))
        
        for i in range(self.obs_max_peds):
            if i < len(peds_sorted):
                p = peds_sorted[i]
                rel_pos = (p["pos"] - self._agent_location) / 600.0
                rel_target = (p["target"] - self._agent_location) / 600.0
                state.append(1.0)
                state.extend(rel_pos)
                state.extend(rel_target)
            else:
                state.extend([0.0]*5)
                
        # 5. Traffic Light (2)
        r, y, g, ttc = self._next_traffic_light_features()
        if r > 0.5: l_state = -1.0
        elif y > 0.5: l_state = 0.0
        else: l_state = 1.0
        state.append(l_state)
        state.append(ttc)
        
        return np.array(state, dtype=np.float32)

    def _update_pedestrians(self):
        # Override to handle simple walking + jaywalking
        alive_peds = []
        for ped in self.pedestrians:
             # Move towards target
             dx = ped["target"][0] - ped["pos"][0]
             dy = ped["target"][1] - ped["pos"][1]
             dist = np.sqrt(dx*dx + dy*dy)
             
             if dist < 5.0:
                 # Reached target, remove (don't add to alive_peds)
                 continue
                 
             # Normalize
             speed = ped["speed"]
             
             # Check Light if NOT Jaywalking
             if not ped.get("is_jaywalking", False):
                  # Check traffic light at intersection 0
                  # If light is Green for Cars (North/South), Pedestrians should STOP
                  if self.intersections and self.traffic_lights:
                      # Assuming intersection ID 0
                      light = self.traffic_lights.get(0)
                      if light and light["directions"]["north"] == "green":
                          # Cars green -> Peds Red -> Stop
                          speed = 0.0
             
             if speed > 0:
                 move_x = (dx/dist) * speed
                 move_y = (dy/dist) * speed
                 ped["pos"][0] += move_x
                 ped["pos"][1] += move_y
            
             alive_peds.append(ped)
             
        self.pedestrians = alive_peds
