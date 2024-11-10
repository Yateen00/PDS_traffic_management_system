import numpy as np
from gymnasium import spaces
from .traffic_signal import TrafficSignal
from .observations import ObservationFunction


# send observation class name to environment
# reward function should be initlised and object should be passed to environment


class CustomObservationFunction(ObservationFunction):
    """Custom observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize custom observation function."""
        super().__init__(ts)
        self.i = 0

    def __call__(self) -> np.ndarray:
        """Return the custom observation."""
        observation = []
        on_phase = [
            1 if self.ts.green_phase == i else 0
            for i in range(0, self.ts.num_green_phases)
        ]
        min_green = [
            0
            if self.ts.time_since_last_phase_change
            < self.ts.min_green + self.ts.yellow_time
            else 1
        ]
        observation.extend(on_phase + min_green)

        # density of only waiting vehciles
        lanes_waiting_density = self.ts.get_lanes_queue()
        arr = lanes_waiting_density
        observation.extend([(arr[i] + arr[i + 1]) / 2 for i in range(0, len(arr), 2)])

        # based on vehcile that could fit
        lanes_density = self.ts.get_lanes_density()
        arr = lanes_density
        observation.extend([(arr[i] + arr[i + 1]) / 2 for i in range(0, len(arr), 2)])

        # total waiting time. extend normalised average waiting time
        halting = self.ts.get_halting_lanes_count()
        accumulated_waiting_time_per_lane = (
            self.ts.get_accumulated_waiting_time_per_lane()
        )
        arr = accumulated_waiting_time_per_lane
        observation.extend(
            [
                (arr[i] + arr[i + 1])
                / (
                    2
                    * 5
                    * self.ts.max_green
                    * (
                        halting[i] + halting[i + 1]
                        if halting[i] + halting[i + 1] != 0
                        else 1
                    )
                )
                for i in range(0, len(arr), 2)
            ]
        )
        self.i += 1
        timestamp = self.ts.env.sim_step / (
            self.ts.env.max_green * self.ts.env.max_steps
        )
        observation.extend([timestamp])

        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space, assuming normalized observation."""
        num_lanes = len(self.ts.lanes) // 2  # Number of lane pairs

        # Number of elements in the observation array
        num_on_phase = len(range(0, self.ts.num_green_phases))
        num_min_green = 1  # Single value indicating if min green time has passed
        num_lanes_waiting_density = num_lanes
        num_lanes_density = num_lanes
        num_accumulated_waiting_time = num_lanes
        num_timestamp = 1

        total_elements = (
            num_on_phase
            + num_min_green
            + num_lanes_waiting_density
            + num_lanes_density
            + num_accumulated_waiting_time
            + num_timestamp
        )

        low = np.zeros(total_elements, dtype=np.float32)
        high = np.ones(total_elements, dtype=np.float32)

        return spaces.Box(low=low, high=high, dtype=np.float32)


# _diff_waiting_time_reward


class CustomRewardFunction:
    def __init__(self, max_green_duration, min_green_duration):
        self.max_green_duration = max_green_duration
        self.min_green_duration = min_green_duration
        self.last_avg_waiting_time = 0
        self.last_density = 0
        self.cumulative = 0
        self.times = 0

    def __call__(self, ts: TrafficSignal, get_cum=False) -> float:
        """Number based reward function."""
        if get_cum:
            return self.get_cumulative_reward(ts)
        # Get the current total waiting time
        current_avg_waiting_time = (sum(ts.get_accumulated_waiting_time_per_lane())) / (
            ts.max_green
            * 5
            * (ts.get_total_queued() if ts.get_total_queued() != 0 else 1)
        )

        # Get the current vehicle densities
        current_vehicle_densities = ts.get_lanes_density()

        # Calculate the change in total waiting time
        total_waiting_time_change = (
            self.last_avg_waiting_time - current_avg_waiting_time
        ) / (ts.max_green + ts.yellow_time)

        # Calculate the average vehicle density
        average_vehicle_density = sum(current_vehicle_densities) / len(
            current_vehicle_densities
        )

        # Calculate the change in vehicle density
        density_change = self.last_density - average_vehicle_density

        # Define weights for each component
        weight_total_waiting_time_change = 1.2
        weight_density_change = 0.8
        weight_current_avg_waiting_time = 1.5
        weight_average_vehicle_density = 0.3

        # Calculate the weighted reward
        reward = (
            weight_total_waiting_time_change * total_waiting_time_change
            + weight_density_change * density_change
            - weight_current_avg_waiting_time * current_avg_waiting_time
            - weight_average_vehicle_density * average_vehicle_density
        ) / (
            weight_total_waiting_time_change
            + weight_density_change
            + weight_current_avg_waiting_time
            + weight_average_vehicle_density
        )
        self.cumulative += reward
        self.times += 1
        # Update the last total waiting time and density
        self.last_avg_waiting_time = current_avg_waiting_time
        self.last_density = average_vehicle_density
        return reward

    def get_cumulative_reward(self, ts: TrafficSignal):
        reward = self.cumulative / self.times
        self.cumulative = 0
        self.times = 0
        return reward
