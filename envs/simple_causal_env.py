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
        
        # --- Physics Buffs (Arcade-ify) ---
        # 1. Faster Steering
        self.max_steer_change = 0.2 # Was 0.05 or similar? 
        # 2. Super Brakes
        self.brake_accel = 2.0 # Was likely < 1.0. Double it.
        # 3. Grip (Tire Friction) which reduces sliding
        self.friction = 0.8 # Was 0.5?
        
        # --- Discrete Action Space ---
        # 0: Idle (maintain speed)
        # 1: Accelerate
        # 2: Brake
        # 3: Steer Left
        # 4: Steer Right 
        # 5: Panic Brake (New)
        self.action_space = spaces.Discrete(6)



        

        # --- Full State Observation Space (Refactored) ---
        # --- Full State Observation Space (Refactored) ---
        # Agent (7): Centering, Lateral, Heading, Speed, Steer, LaneID, Angle
        # Lidar (5): 5 Rays
        # Semantic (4): Type, RelVX, RelVY, Width
        # Light (4): Green, Yellow, Red, Dist
        # Road (10): 5 * [RelX, RelY]
        # NPCs (20): 5 * [RelX, RelY, RelVX, RelVY]
        # Peds (60): 30 * [RelX, RelY]
        # Total: 7 + 5 + 4 + 4 + 10 + 20 + 60 = 110
        self.obs_max_npcs = 5
        self.obs_max_peds = 30
        self.obs_dim = 110
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # Override intersection to be just a single "virtual" intersection for the light
        # This will be handled in _generate_layout
        
        # Override spawn points for the single crossing
        # This will be handled in _generate_layout -> _build_ped_points
        

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        # Endless Runner State
        self.total_distance = 0.0
        self.last_milestone = 0.0
        self.consecutive_idle_steps = 0 # Anti-Camping Counter
        self.last_steering = 0.0 # for steering stability penalty
        
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
        
        if pygame.font:
            font = pygame.font.Font(None, 28) # Smaller font for better fit
            
            # --- Top Left: Agent Stats ---
            # REMOVED: Managed by OverlayWrapper in train scripts
            # step_text = font.render(f"Step: {self.step_count}", True, (255, 255, 255))
            # reward_text = font.render(f"Reward: {self.episode_reward:.1f}", True, (255, 255, 255))
            # canvas.blit(step_text, (10, 10))
            # canvas.blit(reward_text, (10, 40))
            
            # --- Top Right: Env Info & Traffic Light ---
            # Align from right edge (window_size)
            xr = self.window_size - 10
            
            # 1. Traffic Light (Circle)
            # Center at (xr - 30, 30)
            cx, cy = xr - 30, 30
            
            color = (0, 255, 0)
            if self.traffic_light_state == 1: color = (255, 255, 0)
            elif self.traffic_light_state == 2: color = (255, 0, 0)
            
            pygame.draw.circle(canvas, color, (int(cx), int(cy)), 15)
            
            # 2. Distance to Light
            _, dist = self._get_upcoming_light()
            dist_txt = font.render(f"Light Dist: {dist:.1f}m", True, (255, 255, 255))
            # Right align text
            rect = dist_txt.get_rect()
            rect.topright = (xr - 60, 20)
            canvas.blit(dist_txt, rect)
            
            # 3. Env Context (Temp, Roughness) -> Below Light
            temp = self.context.get("temperature", 20)
            rough = self.context.get("roughness", 0.0)
            
            temp_txt = font.render(f"Temp: {temp:.1f} C", True, (200, 200, 255))
            rough_txt = font.render(f"Rough: {rough:.2f}", True, (200, 200, 255))
            
            rect_t = temp_txt.get_rect()
            rect_t.topright = (xr, 60)
            canvas.blit(temp_txt, rect_t)
            
            rect_r = rough_txt.get_rect()
            rect_r.topright = (xr, 90)
            canvas.blit(rough_txt, rect_r)
        
        # --- Draw Target Lane Center (Cyan) ---
        if len(self.track_data) > 1:
             points = []
             # Draw segment around agent
             start_i = max(0, self.last_waypoint_index - 50)
             end_i = min(len(self.track_data), self.last_waypoint_index + 100)
             
             for i in range(start_i, end_i):
                 d = self.track_data[i]
                 # Right lane center: -20.0 (Center of 0 to -40 strip)
                 p = d["pos"] - d["normal"] * 20.0
                 sp = to_screen(p)
                 points.append((int(sp[0]), int(sp[1])))
                 
             if len(points) > 1:
                 pygame.draw.lines(canvas, (0, 255, 255), False, points, 3)
                 
        # --- Draw Lidar Rays (Yellow/Red) ---
        if hasattr(self, "latest_lidar") and self.latest_lidar and self._agent_location is not None:
             angles = np.radians([-60, -30, 0, 30, 60])
             sp = to_screen(self._agent_location)
             start_pos = (int(sp[0]), int(sp[1]))
             
             for i, dist_norm in enumerate(self.latest_lidar):
                 angle = self._agent_heading + angles[i]
                 # Match logic range (150.0)
                 dist_world = dist_norm * 150.0 
                 
                 end_world = self._agent_location + np.array([np.cos(angle), np.sin(angle)]) * dist_world
                 ep = to_screen(end_world)
                 end_pos = (int(ep[0]), int(ep[1]))
                 
                 color = (255, 255, 0)
                 if dist_norm < 0.2: color = (255, 0, 0) # Close warning
                 
                 pygame.draw.line(canvas, color, start_pos, end_pos, 3)

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

    def _initialize_traffic_lights(self):
        # Multiple Traffic Lights at fixed waypoints
        # e.g. indices 100, 300 on the 400-point track
        self.traffic_light_indices = [100, 300]
        
        # Shared State Machine
        # 0=Green, 1=Yellow, 2=Red
        self.traffic_light_state = 0 
        self.traffic_light_timer = 0
        
        # Durations (steps). Assuming 30FPS equivalent?
        # Red must be > RoadWidth/PedSpeed. Width=80, Speed=2 -> 40 steps.
        # Let's be safe: 200
        self.light_durations = {
             0: 300, # Green (Car Go)
             1: 50,  # Yellow (Warning)
             2: 200  # Red (Car Stop, Ped Go)
        }
        
    def _update_traffic_lights(self):
        self.traffic_light_timer += 1
        current_max = self.light_durations[self.traffic_light_state]
        
        if self.traffic_light_timer >= current_max:
            self.traffic_light_timer = 0
            # Cycle: 0->1->2->0
            self.traffic_light_state = (self.traffic_light_state + 1) % 3

    def _get_upcoming_light(self):
        # Find nearest upcoming light index
        closest_dist = float('inf')
        closest_idx = -1
        
        for idx in self.traffic_light_indices:
            # Distance in terms of indices (cyclic check)
            diff = idx - self.last_waypoint_index
            
            if diff < -10: 
                continue
                
            dist = 0
            if diff < 0: dist = 0 # On it
            else: dist = diff
            
            if dist <closest_dist:
                closest_dist = dist
                closest_idx = idx
                
        # If no light directly ahead, target first (loop)
        if closest_idx == -1 and len(self.traffic_light_indices) > 0:
             idx = self.traffic_light_indices[0]
             dist = (400 - self.last_waypoint_index) + idx
             closest_dist = dist
             closest_idx = idx
             
        return closest_idx, closest_dist # Dist in Waypoint units

    def _initialize_npcs(self):
        self.npc_cars = []
        # Try to spawn max_npcs cars
        # Max NPCs = self.obs_max_npcs (5) or from init args
        count = 5 
        if not self.npc_spawn_points: return
        
        # Shuffle points
        points = list(self.npc_spawn_points)
        self.np_random.shuffle(points)
        
        for i in range(min(count, len(points))):
            sp = points[i]
            # Random Speed 2.0 - 4.0
            speed = self.np_random.uniform(2.0, 4.0)
            
            car = {
                "pos": sp["pos"].copy(),
                "heading": sp["heading"],
                "velocity": np.array([np.cos(sp["heading"]), np.sin(sp["heading"])]) * speed,
                "color": (0, 0, 255), # Blue
                "length": 40.0,
                "width": 20.0,
                "speed": speed,
                "max_speed": speed + 1.0, # Slightly higher than init
                "accel": 0.1,
                "direction": sp["direction"]
            }
            self.npc_cars.append(car)

    def step(self, action):
        # Map discrete action to continuous control
        # Actions: continuous [acceleration, steering] in [-1, 1]
        
        # Default: maintain
        accel = 0.0
        steer = 0.0
        
        if action == 1: # Accelerate
            accel = 1.0
        elif action == 2: # Brake (Normal)
            accel = -0.5 # Half braking
        elif action == 3: # Left
            steer = 1.0 # Max steering (Buffed)
        elif action == 4: # Right
            steer = -1.0 # Max steering (Buffed)
        elif action == 5: # Panic Brake (New)
            accel = -1.0 # Full braking force
            
        continuous_action = np.array([accel, steer], dtype=np.float32)
        
        # 1. Apply action
        self._apply_action(continuous_action)
        
        # 2. Update entities
        self._update_npc_cars()
        self._update_pedestrians()
        
        # Spawn new pedestrians Logic
        # Sync with Red Light: Only spawn if Red Phase
        spawn_chance = 0.05
        # Also ensure we are relatively close to a crossing (Light Index)?
        # Or just allow random jaywalking only on RED?
        # User said "Sync... Only spawn or move... when RED".
        
        if self.traffic_light_state == 2: # RED
             if self.np_random.random() < spawn_chance: 
                 self._spawn_pedestrian()
                 
        # Update existing peds (they stop if car is GREEN? Or just keep walking?)
        # "Walking peds should wait... or not spawn". 
        # _update_pedestrians will move them. 
        # Let's assume once spawned they commit to crossing.
            
        self._update_traffic_lights()
        
        # 3. Check collisions/termination
        terminated = False
        truncated = False
        reward = 0.0
        
        # Check Red Light Run
        # If we cross the "Stop Line" (Light Index) while RED
        # Logic: Check if we jumped OVER a light index in this step
        # old_index vs last_waypoint_index
        
        # --- unified Waypoint Update & Reward ---
        old_index = self.last_waypoint_index
        self._update_waypoint_index() # Updates self.last_waypoint_index
        new_index = self.last_waypoint_index
        
        # Check Red Light Violation
        # Did we cross a light index?
        for light_idx in self.traffic_light_indices:
            # If old < light <= new (Simple crossing check)
            if old_index < light_idx and new_index >= light_idx:
                if self.traffic_light_state == 2: # RED
                     # Massive Penalty
                     # User suggested -50.0.
                     reward -= 50.0
                     # Optional: Terminal? "You can treat... as terminal"
                     # Let's make it terminal to enforce hardness.
                     terminated = True
                     # print("Ran Red Light! Terminating.")
        
        # --- Infinite Loop Logic ---
        # If we reach end of track, teleport back to start (Index 20)
        # This creates "Endless Runner"
        if self.last_waypoint_index >= len(self.track_data) - 10:
             # Teleport to start
             start_idx = 20
             start_data = self.track_data[start_idx]
             
             # Actually, let's just use the spawn point logic again
             d = start_data
             self._agent_location = d["pos"] - d["normal"] * 10.0
             self._agent_heading = np.arctan2(d["tangent"][1], d["tangent"][0])
             # Preserve velocity direction relative to track? Or just magnitude?
             # Magnitude is safest.
             self._agent_velocity = np.array([np.cos(self._agent_heading), np.sin(self._agent_heading)]) * np.linalg.norm(self._agent_velocity)
             self.last_waypoint_index = start_idx
             
             # Respawn NPCs to avoid "ghosts"
             self.npc_cars = [] 
             self.npc_spawn_points = self._build_car_spawns()
             self._initialize_npcs() 
             # Remove ones too close to new start
             self.npc_cars = [c for c in self.npc_cars if np.linalg.norm(c["pos"] - self._agent_location) > 50.0]
        
        # Update Total Distance
        step_dist = np.linalg.norm(self._agent_velocity)
        self.total_distance += step_dist
        
        # --- Rewards ---
        
        # --- Multi-Ray Lidar & Safety ---
        lidar_dists, semantic_info = self._compute_multiray_lidar()
        
        # --- Flow or Fail Reward Logic v2 ---
        
        # 1. Cost of Living (Crucial Change)
        # Constant penalty to force action.
        reward -= 0.1 
        
        # 2. Determine Context
        upcoming_idx, dist_to_light = self._get_upcoming_light() # Dist in Waypoint units
        is_red_light = (self.traffic_light_state == 2) and (dist_to_light < 40.0)
        
        # Blocked by obstacle?
        min_lidar = min(lidar_dists)
        is_blocked = (min_lidar < 0.2) # ~30m
        # Also strictly check for obstacles < 10m (0.06 approx)
        is_hard_blocked = (min_lidar < 0.1) or is_red_light
        
        should_stop = is_red_light or is_blocked
        
        current_speed = np.linalg.norm(self._agent_velocity)
        max_speed = 5.0
        norm_speed = current_speed / max_speed
        
        # 3. Conditional Speed Reward
        if not should_stop:
             # SAFE Context (Green light, no obstacle)
             # Reward Speed heavily.
             # Must be > 0.1 to be net positive.
             reward += 1.0 * norm_speed
        else:
             # BLOCKED Context (Red light, Obstacle)
             # Reward Stopping.
             # If speed is 0, reward is 1.0. If speed is max, reward is 0.0.
             reward += 1.0 * (1.0 - norm_speed)
             
        # 4. Anti-Camping Termination (Move or Die)
        # Logic: If speed < 2.0 (and not blocked) for > 50 steps -> Terminate with -50.0
        # "The agent is currently gaming the system by waiting."
        if current_speed < 2.0 and not should_stop:
            self.consecutive_idle_steps += 1
        else:
            self.consecutive_idle_steps = 0
            
        if self.consecutive_idle_steps > 50: 
            reward -= 50.0
            terminated = True
            # print("Terminated due to laziness (Move or Die).")
            
        # 5. Steering Stability Reward
        # Fix wobbling by penalizing rapid steering changes.
        # Track last steering in self.last_steering (initialized in reset)
        steer_diff = abs(continuous_action[1] - self.last_steering)
        reward -= 0.1 * steer_diff
        self.last_steering = continuous_action[1] # Update for next step

        # 6. Milestone Bonus (+10 every 100m) - Keep this as progress incentive

        if self.total_distance - self.last_milestone >= 100.0:
            reward += 10.0
            self.last_milestone = self.total_distance

        # Penalize turning hard
        if action == 3 or action == 4:
            reward -= 0.05 * 0.5
            
        # 6. Safety Penalties (Critical)
        
        # --- PEDESTRIAN HORROR (Pre-Crash Fear) ---
        # "Don't even get CLOSE to them at high speed."
        # If dist_to_pedestrian < 5.0 meters AND speed > 2.0, apply a penalty of -1.0 per step.
        if semantic_info and semantic_info.get("type") == 1.0: # Pedestrian
            # Dist is normalized by 60.0 (max_dist)
            dist_m = semantic_info["dist"] * 60.0 
            if dist_m < 5.0 and current_speed > 2.0:
                reward -= 1.0

        if min_lidar < 0.05 and current_speed > 2.0: # Very close impact
            reward -= 5.0
            
        if semantic_info["type"] == 1.0 and semantic_info["dist"] < 0.2:
            reward -= 5.0 # Threatening pedestrian
            
        # Lane Centering Reward (New Right-Lane Strategy)
        # 1.0 - abs(cross_track_error_from_target / tolerance)
        # Use our new helper
        lateral, centering_error, _ = self._calculate_local_lane_info()
        
        # Tolerance: 15.0m
        centering_reward = 1.0 - (centering_error / 15.0)
        centering_reward = max(0.0, centering_reward) # Clip >= 0
        
        reward += centering_reward 
        
        # Oncoming Traffic Penalty (Left Lane is Lava)
        # If lateral < 0 (Left Side), massive penalty scaling with distance into danger.
        if lateral < 0.0:
            reward -= 0.5 * abs(lateral)
            # Example: At -10 (Left Center): -5.0 penalty per step.
            # This is huge. It will force right lane quickly.
            
        # --- Penalties and Termination ---
            
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
            
        # Success check (reached end of track) -> REMOVED (Endless)
        # if self.last_waypoint_index >= len(self.track_waypoints) - 2:
        #    terminated = True
        #    reward += 100.0 # Completion bonus
                        
        # Update episode reward for display
        self.episode_reward += reward
        
        # Explicit early termination on success -> REMOVED
        # if self.episode_reward > 1000.0:
        #     terminated = True
        
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

    def _get_lookahead_cte(self, lookahead_dist=10.0):
        # Pure Pursuit Logic: Find curvature error relative to a point ahead
        # 1. Project current pos to find base index
        # 2. Look ahead by 'lookahead_dist' along the track
        # 3. Calculate lateral error to THAT point's tangent/normal?
        # Actually standard CTE is fine, but we want the ERROR relative to the Lookahead Point
        # Ideally: Vector from Agent to Lookahead Point. 
        # Calculate angle difference between Agent Heading and Vector to Point.
        # But user asked for "Cross Track Error... relative to a point 5-10 meters ahead".
        # This usually means: What is the distance from the Lookahead Point to the line defined by the Agent's Heading? No.
        # It likely means: What is the distance of the AGENT from the path defined at the lookahead point?
        # Standard Pure Pursuit uses "Steering Angle = arctan(2L sin(alpha) / k)".
        # Let's interpret "CTE to lookahead": 
        # Simply calculate the distance of the agent from the center line, but using the normal vector OF THE LOOKAHEAD POINT.
        # Or better: Project Agent Pos onto the Lookahead Point's Line.
        
        # Identify Lookahead Point
        current_idx = self.last_waypoint_index
        target_dist = self.track_waypoints[current_idx]
        
        # Traverse forward to find point at dist
        # Approx: 1 index ~ 2 meters? No. Spline is variable.
        # Let's just step forward
        search_idx = current_idx
        accum_dist = 0
        while accum_dist < lookahead_dist and search_idx < len(self.track_data) - 1:
            p1 = self.track_data[search_idx]["pos"]
            p2 = self.track_data[search_idx+1]["pos"]
            accum_dist += np.linalg.norm(p2 - p1)
            search_idx += 1
            
        target_point = self.track_data[search_idx]["pos"]
        target_tangent = self.track_data[search_idx]["tangent"]
        
        # Calculate vector from Agent to Target
        to_target = target_point - self._agent_location
        
        # Cross Product (2D) to find signed distance
        # "How far is the target to the left/right of my current heading?" - That's pure pursuit alpha.
        # User asked for "Cross Track Error".
        # Let's calculate the lateral distance of the Agent relative to the Target's Tangent.
        # i.e. Project (Agent - Target) onto Random Normal?
        # Let's stick to the Pure Pursuit interpretation which is robust for steering.
        # Return the Signed Angle to the target point?
        # Or just the distance to the Spline at the lookahead index?
        # "Calculate the error relative to a point 5-10 meters ahead"
        # Let's calculate the distance between Agent and Target Point? No.
        
        # Simplest Smoothing: Return the distance of the agent to the line passing through Target Point with Target Tangent.
        # projected_dist = dot(Agent - Target, Normal_at_Target)
        normal_at_target = self.track_data[search_idx]["normal"]
        cte = np.dot(self._agent_location - target_point, normal_at_target)
        
        return cte

    def _get_obs(self):
        # Return full state vector
        state = []
        
        # 1. Agent State (6)
        # Pos (x,y), Vel (vx,vy), Heading (cos, sin)
        state.extend(self._agent_location / 600.0)
        state.extend(self._agent_velocity / 5.0) # approx max speed
        state.append(np.cos(self._agent_heading))
        state.append(np.sin(self._agent_heading))
        
        # 2. Road State (13)
        # Dist to center (Smoothed Lookahead CTE)
        # User requested 5-10m. Let's use 10.0.
        cte_lookahead = self._get_lookahead_cte(lookahead_dist=10.0)
        state.append(cte_lookahead / 40.0) # Width normalized
        
    def _compute_multiray_lidar(self):
        # 5 Rays: -60, -30, 0, 30, 60
        angles = np.radians([-60, -30, 0, 30, 60])
        
        # Dynamic Range based on Temperature (Braking Distance)
        # Lower Temp -> Longer Range needed
        # Base: 150m at 20C.
        # -10C -> Needs ~200m?
        temp = self.context.get("temperature", 20.0)
        # Simple linear scaling: 
        # Range = 150 + (20 - temp) * 3.0
        # If 20C: 150
        # If -20C: 150 + 40*3 = 270m
        max_dist = 150.0 + (20.0 - temp) * 3.0
        max_dist = max(100.0, max_dist) # Min clamp
        
        lidar_readings = []
        
        # Track nearest object globally for semantic info
        nearest_obj_dist = float('inf')
        nearest_obj_data = {
            "type": 0.0, # 0=None, 0.5=Car, 1.0=Ped
            "rel_vx": 0.0,
            "rel_vy": 0.0,
            "width": 0.0,
            "dist": 1.0 # Normalized dist of nearest
        }
        
        for angle_offset in angles:
            heading = self._agent_heading + angle_offset
            dx = np.cos(heading)
            dy = np.sin(heading)
            ray_dir = np.array([dx, dy])
            
            min_d = max_dist
            ray_hit_type = 0 # 0=None, 1=Car, 2=Ped
            
            # Check NPCs
            for car in self.npc_cars:
                vec = car["pos"] - self._agent_location
                proj = np.dot(vec, ray_dir)
                if proj > 0 and proj < min_d:
                    dist_sq = np.sum(vec**2) # True distance
                    rejection = np.linalg.norm(vec - proj * ray_dir)
                    if rejection < (self.car_width + self.car_width)/2 + 1.0:
                        min_d = proj
                        ray_hit_type = 1 # Car
                        # Update global nearest
                        if dist_sq < nearest_obj_dist:
                            nearest_obj_dist = dist_sq
                            # Calculate Rel Velocity (Local Frame)
                            rel_v_world = car["velocity"] - self._agent_velocity
                            # Rotate to agent frame
                            c, s = np.cos(-self._agent_heading), np.sin(-self._agent_heading)
                            rx = rel_v_world[0]*c - rel_v_world[1]*s
                            ry = rel_v_world[0]*s + rel_v_world[1]*c
                            
                            nearest_obj_data = {
                                "type": 0.5, # Car
                                "rel_vx": np.clip(rx / 10.0, -1, 1),
                                "rel_vy": np.clip(ry / 10.0, -1, 1),
                                "width": self.car_width / 5.0, # Norm
                                "dist": min_d / max_dist
                            }

            # Check Peds
            for ped in self.pedestrians:
                vec = ped["pos"] - self._agent_location
                proj = np.dot(vec, ray_dir)
                if proj > 0 and proj < min_d:
                    dist_sq = np.sum(vec**2)
                    rejection = np.linalg.norm(vec - proj * ray_dir)
                    if rejection < (self.car_width/2 + 1.0):
                        min_d = proj
                        ray_hit_type = 2 # Ped
                        if dist_sq < nearest_obj_dist:
                            nearest_obj_dist = dist_sq
                            # Ped Velocity? (Construct from speed/target or use stored?)
                            # Peds don't have 'velocity' vector stored directly, usually just speed/target
                            # Let's approx 0 or calculate?
                            # _update_pedestrians modifies pos... let's assume raw walking speed 2.0 towards target
                            pv_x = 0; pv_y = 0
                            # Simply use 0 for now or assume previous logic? 
                            # Better: compute current velocity unit vec
                            t_dir = ped["target"] - ped["pos"]
                            d_t = np.linalg.norm(t_dir)
                            if d_t > 0:
                                pv = (t_dir / d_t) * ped["speed"]
                                pv_x, pv_y = pv[0], pv[1]
                                
                            rel_v_world = np.array([pv_x, pv_y]) - self._agent_velocity
                            c, s = np.cos(-self._agent_heading), np.sin(-self._agent_heading)
                            rx = rel_v_world[0]*c - rel_v_world[1]*s
                            ry = rel_v_world[0]*s + rel_v_world[1]*c
                            
                            nearest_obj_data = {
                                "type": 1.0, # Ped
                                "rel_vx": np.clip(rx / 10.0, -1, 1),
                                "rel_vy": np.clip(ry / 10.0, -1, 1),
                                "width": self.pedestrian_radius * 2 / 5.0, 
                                "dist": min_d / max_dist
                            }
                            
            val = min_d / max_dist
            if ray_hit_type == 2: val = -val
            lidar_readings.append(val)
            
        if nearest_obj_data is None:
             nearest_obj_data = {"type": 0.0, "rel_vx": 0.0, "rel_vy": 0.0, "width": 0.0, "dist": 1.0}
             
        self.latest_lidar = lidar_readings
        
        return lidar_readings, nearest_obj_data

    def _transform_to_ego(self, target_pos, target_vel=None):
        """
        Transform world coordinates to agent's egocentric frame.
        X+ = Forward, Y+ = Left
        """
        # Translation
        dx = target_pos[0] - self._agent_location[0]
        dy = target_pos[1] - self._agent_location[1]
        
        # Rotation (Agent Heading)
        # We want to rotate world vector by -heading
        c = np.cos(-self._agent_heading)
        s = np.sin(-self._agent_heading)
        
        rel_x = dx * c - dy * s
        rel_y = dx * s + dy * c
        
        if target_vel is not None:
             dvx = target_vel[0] - self._agent_velocity[0]
             dvy = target_vel[1] - self._agent_velocity[1]
             
             rel_vx = dvx * c - dvy * s
             rel_vy = dvx * s + dvy * c
             return [rel_x, rel_y, rel_vx, rel_vy]
             
        return [rel_x, rel_y]

    def _calculate_local_lane_info(self):
        """
        Calculates lateral (cross track) error and longitudinal progress.
        Sign Convention:
        - Lateral > 0: Right Side (Good)
        - Lateral < 0: Left Side (Oncoming/Lava)
        """
        # Find closest waypoint for local tangent reference
        if self.last_waypoint_index >= len(self.track_data):
            return 0.0, 0.0, 0.0 # Default
            
        d = self.track_data[self.last_waypoint_index]
        road_center = d["pos"]
        tangent = d["tangent"]
        
        # Vector from road center to car
        dx = self._agent_location[0] - road_center[0]
        dy = self._agent_location[1] - road_center[1]
        
        # Tangent (tx, ty)
        tx, ty = tangent
        
        # Longitudinal = Dot Product (Project onto tangent)
        longitudinal = dx * tx + dy * ty
        
        # Lateral = Cross Product (Z-component of D x T) 
        # Or Dot Product with Right-Facing Normal (ty, -tx)
        # Normal (Left) is (-ty, tx).
        # We want Positive = Right. So we project onto Right Normal (ty, -tx).
        lateral = dx * ty + dy * (-tx)
        
        # Verify:
        # If Road East (1, 0). Right Normal (0, -1).
        # If Car South (0, -10). Right Side.
        # lateral = 0*0 + (-10)*(-1) = 10. (Positive). Correct.
        # If Car North (0, 10). Left Side.
        # lateral = 0*0 + 10*(-1) = -10. (Negative). Correct.
        
        # Calculate Error relative to Right Lane Center (+20.0)
        # Lane width ~40? User said "Lane width / 2".
        # If track width is 80 (radius?). Lane width is 40. Center is 20.
        target_x = 20.0 
        centering_error = abs(lateral - target_x)
        
        # Road Angle
        road_angle = np.arctan2(ty, tx)
        
        return lateral, centering_error, road_angle

    def _get_obs(self):
        # Return Refactored Egocentric State (96 dim)
        state = []
        
        # 1. Agent State (7) 
        # CTE to Target, Heading Error, Speed, Steering, Lane Info
        
        lateral, centering_error, road_angle = self._calculate_local_lane_info()
        
        # 1. Centering Error (Dist to Right Lane Center)
        # Normalize to lane width (approx 20)
        state.append(np.clip(centering_error / 20.0, 0.0, 1.0))
        
        # 2. Signed Lateral Position (Crucial for knowing Left vs Right)
        # >0 Right, <0 Left. 
        state.append(np.clip(lateral / 20.0, -1.0, 1.0))
        
        # Lookahead CTE (Keep smoothness) -> Relative to Right Center?
        # Re-using the Lookahead logic:
        # We should probably update _get_lookahead_cte to target Right Lane too?
        # Or just rely on the new immediate CTE for now.
        # User requested "Ensure agent knows which lane...". Signed Lateral is enough.
        
        # 3. Heading Error
        diff = road_angle - self._agent_heading
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        state.append(np.clip(diff, -1.0, 1.0))
        
        # 4. Normalized Speed
        speed = np.linalg.norm(self._agent_velocity)
        state.append(np.clip(speed / 5.0, 0.0, 1.0))
        
        # 5. Steering (Keep 0.0 for now as placeholder)
        state.append(0.0) 
        
        # 6. Lane ID (Explicit Feature)
        # +1.0 for Right, -1.0 for Left
        state.append(1.0 if lateral > 0 else -1.0)
        
        # 7. Angle to Lane (Redundant with Heading Error but useful)
        state.append(np.sin(diff))
        
        # --- Multi-Ray Lidar (5) & Semantic (4) ---
        lidar_dists, semantic = self._compute_multiray_lidar()
        
        # 5 Rays
        state.extend(np.clip(lidar_dists, 0.0, 1.0))
        
        # Semantic (4)
        state.append(semantic["type"])
        state.append(semantic["rel_vx"])
        state.append(semantic["rel_vy"])
        state.append(semantic["width"])
        
        # Light (4): Green, Yellow, Red, Dist
        light_idx, light_dist = self._get_upcoming_light()
        
        # Encoding
        if self.traffic_light_state == 0: # Green
             state.extend([1.0, 0.0, 0.0])
        elif self.traffic_light_state == 1: # Yellow
             state.extend([0.0, 1.0, 0.0])
        else: # Red
             state.extend([0.0, 0.0, 1.0])
             
        # Normalize Dist (Max range 100?)
        # If > 100, clip to 1.0
        norm_dist = np.clip(light_dist / 100.0, 0.0, 1.0)
        state.append(norm_dist)
        
        # 2. Road Preview (5 Waypoints) -> Egocentric (Keep existing logic)
        for i in range(1, 6):
            idx = min(len(self.track_data)-1, self.last_waypoint_index + i * 5)
            wp = self.track_data[idx]["pos"]
            rel = self._transform_to_ego(wp)
            # Normalize distance? Divide by view range (e.g. 100m)
            state.append(np.clip(rel[0] / 100.0, -1.0, 1.0))
            state.append(np.clip(rel[1] / 100.0, -1.0, 1.0))
            
        # 3. NPCs (5 Cars) -> Egocentric
        # Sort by distance
        cars_sorted = sorted(self.npc_cars, key=lambda c: np.linalg.norm(c["pos"] - self._agent_location))
        
        for i in range(self.obs_max_npcs):
            if i < len(cars_sorted):
                c = cars_sorted[i]
                rel = self._transform_to_ego(c["pos"], c["velocity"])
                # Pos / 100, Vel / 10
                state.append(np.clip(rel[0] / 100.0, -1.0, 1.0))
                state.append(np.clip(rel[1] / 100.0, -1.0, 1.0))
                state.append(np.clip(rel[2] / 10.0, -1.0, 1.0))
                state.append(np.clip(rel[3] / 10.0, -1.0, 1.0))
            else:
                state.extend([0.0]*4)
                
        # 4. Peds (30 Peds) -> Egocentric
        peds_sorted = sorted(self.pedestrians, key=lambda p: np.linalg.norm(p["pos"] - self._agent_location))
        for i in range(self.obs_max_peds):
            if i < len(peds_sorted):
                p = peds_sorted[i]
                rel = self._transform_to_ego(p["pos"])
                state.append(np.clip(rel[0] / 100.0, -1.0, 1.0))
                state.append(np.clip(rel[1] / 100.0, -1.0, 1.0))
            else:
                state.extend([0.0]*2)
                

        
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
             # Check Light if NOT Jaywalking
             if not ped.get("is_jaywalking", False):
                   # Check traffic light state
                   # Cars Green (0) or Yellow (1) -> Peds Stop
                   if self.traffic_light_state == 0 or self.traffic_light_state == 1:
                       speed = 0.0
             
             if speed > 0:
                 move_x = (dx/dist) * speed
                 move_y = (dy/dist) * speed
                 ped["pos"][0] += move_x
                 ped["pos"][1] += move_y
            
             alive_peds.append(ped)
             
        self.pedestrians = alive_peds
