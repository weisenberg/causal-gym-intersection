
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
            
        return obs, info
