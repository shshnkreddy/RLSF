"""
Driving environment based on:
- https://github.com/dsadigh/driving-preferences
- https://github.com/Stanford-ILIAD/easy-active-learning/
"""

import os
import sys
sys.path.append('/home/shashank/prefcrl/SIM-RL')
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from matplotlib.image import AxesImage, BboxImage
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from scipy.ndimage import rotate, zoom
import gymnasium as gym
from gymnasium import spaces
import safety_gymnasium 
import random
from Sources.wrapper import DriverVizWrapper

# Get the absolute path of the image folder
IMG_FOLDER = os.path.abspath('<abs path to /img>')

# Check if the directory exists
if not os.path.isdir(IMG_FOLDER):
    raise FileNotFoundError(f"Directory '{IMG_FOLDER}' not found.")

# Check if the file exists
file_path = os.path.join(IMG_FOLDER, "grass.png")
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File '{file_path}' not found.")

# Load the image
GRASS = np.tile(plt.imread(file_path), (5, 5, 1))

# Function to load and zoom the car images
def load_and_zoom_car(color):
    file_path = os.path.join(IMG_FOLDER, f"car-{color}.png")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")
    return zoom(np.array(plt.imread(file_path) * 255.0, dtype=np.uint8), [0.3, 0.3, 1.0])

# Dictionary comprehension to load car images for different colors
CAR = {
    color: load_and_zoom_car(color)
    for color in ["gray", "orange", "purple", "red", "white", "yellow"]
}

COLOR_AGENT = "orange"
COLOR_ROBOT = "white"

CAR_AGENT = CAR[COLOR_AGENT]
CAR_ROBOT = CAR[COLOR_ROBOT]
CAR_SCALE = 0.15 / max(list(CAR.values())[0].shape[:2])

LANE_SCALE = 10.0
LANE_COLOR = (0.4, 0.4, 0.4)  # 'gray'
LANE_BCOLOR = "white"

STEPS = 100
EP_LEN = 100

def set_image(
    obj,
    data,
    scale=CAR_SCALE,
    x=[0.0, 0.0, 0.0, 0.0],
):
    ox = x[0]
    oy = x[1]
    angle = x[2]
    img = rotate(data, np.rad2deg(angle))
    h, w = img.shape[0], img.shape[1]
    obj.set_data(img)
    obj.set_extent(
        [
            ox - scale * w * 0.5,
            ox + scale * w * 0.5,
            oy - scale * h * 0.5,
            oy + scale * h * 0.5,
        ]
    )


class Car:
    def __init__(self, initial_state, actions):
        self.initial_state = initial_state
        self.state = self.initial_state
        self.actions = actions
        self.action_i = 0

    def reset(self):
        self.state = self.initial_state
        self.action_i = 0

    def update(self, update_fct) -> None:
        u1, u2 = self.actions[self.action_i % len(self.actions)]
        self.state = update_fct(self.state, u1, u2)
        self.action_i += 1

    def gaussian(self, x, height=0.07, width=0.03):
        car_pos = np.asarray([self.state[0], self.state[1]])
        car_theta = self.state[2]
        car_heading = (np.cos(car_theta), np.sin(car_theta))
        pos = np.asarray([x[0], x[1]])
        d = car_pos - pos
        dh = np.dot(d, car_heading)
        dw = np.cross(d, car_heading)
        return np.exp(-0.5 * ((dh / height) ** 2 + (dw / width) ** 2))


class Lane:
    def __init__(
        self,
        start_pos,
        end_pos,
        width,
    ):
        self.start_pos = np.asarray(start_pos)
        self.end_pos = np.asarray(end_pos)
        self.width = width
        d = self.end_pos - self.start_pos
        self.dir = d / np.linalg.norm(d)
        self.perp = np.asarray([-self.dir[1], self.dir[0]])

    def gaussian(self, state, sigma=0.5):
        pos = np.asarray([state[0], state[1]])
        dist_perp = np.dot(pos - self.start_pos, self.perp)
        return np.exp(-0.5 * (dist_perp / (sigma * self.width / 2.0)) ** 2)

    def direction(self, x):
        return np.cos(x[2]) * self.dir[0] + np.sin(x[2]) * self.dir[1]

    def shifted(self, m):
        return Lane(
            self.start_pos + self.perp * self.width * m,
            self.end_pos + self.perp * self.width * m,
            self.width,
        )


def get_lane_x(lane, scenario):
    if(scenario=='blocked' or scenario=='changing_lane'):
        if lane == "left":
            return -0.17
        elif lane == "right":
            return 0.17
        elif lane == "middle":
            return 0
        else:
            raise Exception("Unknown lane:", lane)
    elif(scenario=='twolanes' or scenario=='stopping'):
        if lane == "left":
            return -0.17/2
        elif lane == "right":
            return 0.17/2


class Driver(gym.Env):
    def __init__(
        self,
        cars,
        starting_speed=0.41,
        constraint=False, 
        scenario="blocked"
    ):  
        self.scenario = scenario
        self.starting_speed = starting_speed
        if(scenario=='blocked'):
            self.target_v = 0.55
            starting_lane = random.choice(["left", "right"])
            initial_x = get_lane_x(starting_lane, scenario)
            self.initial_state = [initial_x, -0.1, np.pi / 2, starting_speed]
            lane = Lane([0.0, -1.0], [0.0, 1.0], 0.17)
            road = Lane([0.0, -1.0], [0.0, 1.0], 0.17 * 3)
            self.lanes = [lane.shifted(0), lane.shifted(-1), lane.shifted(1)]
            self.fences = [lane.shifted(2), lane.shifted(-2)]
            self.roads = [road]
        elif(scenario=='twolanes'):
            self.target_v = 0.45
            starting_lane = "left"
            initial_x = get_lane_x(starting_lane, scenario)
            self.initial_state = [initial_x, -0.1, np.pi / 2, starting_speed]
            lane = Lane([0.0, -1.0], [0.0, 1.0], 0.17)
            road = Lane([0.0, -1.0], [0.0, 1.0], 0.17 * 2)
            self.lanes = [lane.shifted(-0.5), lane.shifted(0.5)]
            self.fences = [lane.shifted(2), lane.shifted(-2)]
            self.roads = [road]
        elif(scenario=='changing_lane'):
            self.target_v = 0.5
            starting_lane = "middle"
            initial_x = get_lane_x(starting_lane, scenario)
            self.initial_state = [initial_x, -0.1, np.pi / 2, starting_speed]
            lane = Lane([0.0, -1.0], [0.0, 1.0], 0.17)
            road = Lane([0.0, -1.0], [0.0, 1.0], 0.17 * 3)
            self.lanes = [lane.shifted(0), lane.shifted(-1), lane.shifted(1)]
            self.fences = [lane.shifted(2), lane.shifted(-2)]
            self.roads = [road]
        elif(scenario=='stopping'):
            self.target_v = 0.5
            starting_lane = "left"
            initial_x = get_lane_x(starting_lane, scenario)
            self.initial_state = [initial_x, -0.1, np.pi / 2, starting_speed]
            lane = Lane([0.0, -1.0], [0.0, 1.0], 0.17)
            road = Lane([0.0, -1.0], [0.0, 1.0], 0.17 * 2)
            self.lanes = [lane.shifted(-0.5), lane.shifted(0.5)]
            self.fences = [lane.shifted(2), lane.shifted(-2)]
            self.roads = [road]
                
        self.state = self.initial_state
        self.prev_state = self.initial_state
        self.collision_flag = False

        #Different reward function
        self.constraint = constraint
        self.episode_length = 100
        self.dt = 0.2

        self.friction = 1
        self.vmax = 1
        self.xlim = (-0.7, 0.7)
        # self.ylim = (-0.2, 0.8)
        self.ylim = (-0.2, 5)
        
        self.cars = cars
        

        self.time = 0

        self.action_d = 2
        self.action_min = np.array([-1, -1])
        self.action_max = np.array([1, 1])

        self.n_obs_features = self._get_observation().shape[0]
        
        self.observation_space = spaces.Box(
            low=10.0, 
            high=10.0,
            shape=(self.n_obs_features,)
        )

        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0,
            shape=(self.action_d,)
        )

        self.history = []
        self._update_history()

    def _update_history(self):
        self.history.append((np.array(self.state), self._get_car_states()))

    def _get_car_states(self):
        return [np.array(car.state) for car in self.cars]

    def _update_state(self, state, u1, u2):
        x, y, theta, v = state
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = v * u1
        dv = u2 - self.friction * v
        new_v = max(min(v + dv * self.dt, self.vmax), -self.vmax)
        return [x + dx * self.dt, y + dy * self.dt, theta + dtheta * self.dt, new_v]

    def _get_reward_for_state(self, features):
        #Reach the top as soon as possible
        # r = features['target_location']
        x, y, theta, v = self.state
        x_prev, y_prev, theta_prev, v_prev = self.prev_state

        r = (y - y_prev)*10
        # r = 0

        #Stay on the road 
        r += -1*features['off_street']

        #Don't drive backwards
        r += -1*features['drive_backward']

        #Stay in the lane
        # if(self.scenario=='two_lanes'):
        #     r += -1*features['not_in_lane']/10
        # else:
        r += -1*features['not_in_lane']

        #reward = -1 if out of frame
        if(y < self.ylim[0] or x > self.xlim[1] or x < self.xlim[0]):
            # if(not self.constraint):
            r = -100

        #Reward for collision
        if(features['distance_to_other_car'] > 0.4):
            r = -100

        # if(y > self.ylim[1]):
        #     r += 100

        return r

    
    def _get_constraint_for_state(self, features):
        #Distance to other car
        c = False
        c = c or features['distance_to_other_car'] > 0.025

        #Velocity constraint
        c = c or features['too_fast']

        # #Stay on the road
        c = c or features['off_street']

        # #Don't drive backwards
        c = c or features['drive_backward']

        #Check if out of frame
        x, y, theta, v = self.state
        
        if(y < self.ylim[0] or x > self.xlim[1] or x < self.xlim[0]):
            c = True

        #Check for collision
        if(features['distance_to_other_car'] > 0.4):
            c = True
            self.collision_flag = True

        # if(self.collision_flag):
        #     c = True
        
        return float(c)
    
    def _is_done(self, features):
        done = False 
        if(not self.constraint):
            ##Terminate if there is a collision
            ##In the CMDP case, we don't terminate because the car intentionally crashes to achieve a cost of 1
            done = done or features['distance_to_other_car'] > 0.4
        
            x, y, theta, v = self.state
            done = done or (y > self.ylim[1] or y < self.ylim[0] or x > self.xlim[1] or x < self.xlim[0])

        if(self.constraint):
            x, y, theta, v = self.state
            done = done or (y > self.ylim[1])

        return done
    
    def _get_observation(self):
        #Concat states of all cars
        car_states = np.array([car.state for car in self.cars]).flatten()
        
        #Add collision flag
        # return np.concatenate([self.state, car_states, [float(self.collision_flag)]])
        return np.concatenate([self.state, car_states])

    def step(self, action):
        action = np.array(action)
        u1, u2 = action

        #Get r and c
        features_r = self._get_features()
        reward = self._get_reward_for_state(features_r)
        features_c = self._get_features(constraint=True)
        cost = self._get_constraint_for_state(features_c)
        
        done = bool(self.time >= self.episode_length)
        done = done or self._is_done(features_r)

        #Update state
        self.prev_state = self.state
        self.state = self._update_state(self.state, u1, u2)
        for car in self.cars:
            car.update(self._update_state)
        self._update_history()
        self.time += 1

        return self._get_observation(), reward, cost, done, done, dict()
    
    def _get_features(self, constraint=False):
        state = self.state
        
        x, y, theta, v = state

        off_street = int(np.abs(x) > self.roads[0].width / 2)

        b = 5000
        a = 10
        if(self.scenario=='blocked' or self.scenario=='changing_lane') :
            d_to_lane = np.min([(x - 0.17) ** 2, x**2, (x + 0.17) ** 2])
            # d_to_lane = (x - 0.17) ** 2
        elif(self.scenario=='twolanes'):
            d_to_lane = (x + 0.17/2) ** 2
        elif(self.scenario=='stopping'):
            d_to_lane = np.min([(x - 0.17/2) ** 2, (x + 0.17/2) ** 2])

        not_in_lane = 1 / (1 + np.exp(-b * d_to_lane + a)) 
            
        big_angle = np.abs(np.cos(theta))

        drive_backward = int(v < 0)
        too_fast = v > (self.target_v+0.2)

        distance_to_other_car = 0
        b = 30
        a = 0.01
        for car in self.cars:
            car_x, car_y, car_theta, car_v = car.state
            if(constraint):
                distance_to_other_car += np.exp(
                    -b * (10 * (x - car_x) ** 2 + 2 * (y - car_y) ** 2) + b * a
                )
            else:
                distance_to_other_car += np.exp(
                    -b * (10 * (x - car_x) ** 2 + 2 * (y - car_y) ** 2) + b * a
                )

        keeping_speed = -np.square(v - self.target_v)
        # target_location = -np.square(y - self.ylim[1])
        # target_location = y-self.ylim[1]
        target_location = y

        return {
            "keep_speed": keeping_speed,
            "target_location": target_location,
            "off_street": off_street,
            "not_in_lane": not_in_lane,
            "big_angle": big_angle,
            "drive_backward": drive_backward,
            "too_fast": too_fast,
            "distance_to_other_car": distance_to_other_car,
        }

    def reset(self):
        if(self.scenario=='blocked'):
            starting_lane = random.choice(["left","right"])
        elif(self.scenario=='twolanes'):
            starting_lane = "left"
        elif(self.scenario=='changing_lane'):
            starting_lane = "middle"
        elif(self.scenario=='stopping'):
            starting_lane = "left"
        # print("Starting lane:", starting_lane)
        self.collision_flag = False
        initial_x = get_lane_x(starting_lane, self.scenario)
        self.initial_state = [initial_x, -0.1, np.pi / 2, self.starting_speed]
        self.prev_state = self.initial_state
        self.state = self.initial_state
        self.time = 0
        for car in self.cars:
            car.reset()
        self.history = []
        self._update_history()
        return self._get_observation(), dict()

    def render(self, mode="rgb_array"):
        if mode not in ("human", "rgb_array", "human_static"):
            raise NotImplementedError("render mode {} not supported".format(mode))
        fig = plt.figure(figsize=(7, 7))

        ax = plt.gca()
        ax.set_xlim(self.xlim[0], self.xlim[1])
        ax.set_ylim(self.ylim[0], self.ylim[1])
        ax.set_aspect("equal")

        grass = BboxImage(ax.bbox, interpolation="bicubic", zorder=-1000)
        grass.set_data(GRASS)
        ax.add_artist(grass)

        for lane in self.lanes:
            path = Path(
                [
                    lane.start_pos
                    - LANE_SCALE * lane.dir
                    - lane.perp * lane.width * 0.5,
                    lane.start_pos
                    - LANE_SCALE * lane.dir
                    + lane.perp * lane.width * 0.5,
                    lane.end_pos + LANE_SCALE * lane.dir + lane.perp * lane.width * 0.5,
                    lane.end_pos + LANE_SCALE * lane.dir - lane.perp * lane.width * 0.5,
                    lane.start_pos
                    - LANE_SCALE * lane.dir
                    - lane.perp * lane.width * 0.5,
                ],
                [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
            )
            ax.add_artist(
                PathPatch(
                    path,
                    facecolor=LANE_COLOR,
                    lw=0.5,
                    edgecolor=LANE_BCOLOR,
                    zorder=-100,
                )
            )

        for car in self.cars:
            img = AxesImage(ax, interpolation="bicubic", zorder=20)
            set_image(img, CAR_ROBOT, x=car.state)
            ax.add_artist(img)

        human = AxesImage(ax, interpolation=None, zorder=100)
        set_image(human, CAR_AGENT, x=self.state)
        ax.add_artist(human)

        plt.axis("off")
        plt.tight_layout()
        if mode != "human_static":
            fig.canvas.draw()
            rgb = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            rgb = rgb.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            del fig
            if mode == "rgb_array":
                return rgb
            elif mode == "human":
                plt.imshow(rgb, origin="upper")
                plt.axis("off")
                plt.tight_layout()
                plt.pause(0.05)
                plt.clf()
        return None

    def close(self):
        pass
        # if self.viewer:
        #     self.viewer.close()
        #     self.viewer = None

    def plot_history(self):
        x_player = []
        y_player = []
        N_cars = len(self.cars)
        x_cars = [[] for _ in range(N_cars)]
        y_cars = [[] for _ in range(N_cars)]
        for player_state, car_states in self.history:
            x_player.append(player_state[0])
            y_player.append(player_state[1])
            for i in range(N_cars):
                x_cars[i].append(car_states[i][0])
                y_cars[i].append(car_states[i][1])

        self.reset()
        self.render(mode="human_static")
        plt.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # plt.tight_layout()
        for i in range(N_cars):
            plt.plot(
                x_cars[i],
                y_cars[i],
                zorder=10,
                linestyle="-",
                color=COLOR_ROBOT,
                linewidth=2.5,
                marker="o",
                markersize=8,
                markevery=1,
            )

        plt.plot(
            x_player,
            y_player,
            zorder=10,
            linestyle="-",
            color=COLOR_AGENT,
            linewidth=2.5,
            marker="o",
            markersize=8,
            markevery=1,
        )

def get_cars(cars_trajectory):
    scenario = cars_trajectory
    if cars_trajectory == "blocked":
        # three cars
        x1 = get_lane_x("left", scenario)
        y1 = 0.40
        s1 = 0.2
        speeds1 = np.random.normal(loc=s1, scale=0.05, size=(EP_LEN, ))
        actions1 = [(0, speeds1[i]) for i in range(EP_LEN)]
        car1 = Car([x1, y1, np.pi / 2.0, s1], actions1)
        x2 = get_lane_x("middle", scenario)
        y2 = 0.45
        s2 = 0.25
        speeds2 = np.random.normal(loc=s2, scale=0.05, size=(EP_LEN, ))
        actions2 = [(0, speeds2[i]) for i in range(EP_LEN)]
        car2 = Car([x2, y2, np.pi / 2.0, s2], actions2)
        x3 = get_lane_x("right", scenario)
        y3 = 0.50
        s3 = 0.3
        speeds3 = np.random.normal(loc=s3, scale=0.05, size=(EP_LEN, ))
        actions3 = [(0, speeds3[i]) for i in range(EP_LEN)]
        car3 = Car([x3, y3, np.pi / 2.0, s3], actions3)
        cars = [car1, car2, car3]
        
    elif cars_trajectory == "changing_lane":
        # car driving from right to middle lane
        car_x = get_lane_x("right", scenario)
        straight_speed = 0.528
        # straight_speed = 0.0
        car1 = Car(
            [car_x, 0, np.pi / 2.0, straight_speed],
            [(0, straight_speed)] * 5
            + [(1, straight_speed)] * 4
            + [(-1, straight_speed)] * 4
            + [(0, straight_speed)] * EP_LEN,
        )
        # car driving on the left lane
        x2 = get_lane_x("left", scenario)
        y2 = 1.0
        s2 = 0.2
        speeds2 = np.random.normal(loc=s2, scale=0.05, size=(EP_LEN, ))
        actions2 = [(0, speeds2[i]) for i in range(EP_LEN)]
        car2 = Car([x2, y2, np.pi / 2.0, s2], actions2)

        cars = [car1, car2]
    
    elif cars_trajectory == "twolanes":
        car1_x = get_lane_x("right", scenario)
        straight_speed = 0.228
        # straight_speed = 0.0
        car1 = Car(
            [car1_x, 5.0, -np.pi / 2.0, -0.168],
            [(0, straight_speed)] * EP_LEN
        )

        car2_x = get_lane_x("right", scenario)
        straight_speed = 0.228
        car2 = Car(
            [car2_x, 3.0, -np.pi / 2.0, -0.228],
            [(0, straight_speed)] * EP_LEN
        )

        car3_x = get_lane_x("left", scenario)
        straight_speed = 0.128
        car3 = Car(
            [car3_x, 1.5, np.pi / 2.0, 0.128],
            [(0, straight_speed)] * EP_LEN
        )

        cars = [car1, car2, car3]

    elif(cars_trajectory=='stopping'):
        car1_x = get_lane_x("left", scenario)
        straight_speed = 0.3
        car1 = Car(
            [car1_x, 1.0, np.pi / 2.0, 0.0],
            [(0, straight_speed)] * 20 +
            [(0, 0.0)] * EP_LEN 
        )

        car2_x = get_lane_x("right", scenario)
        straight_speed = 0.3
        car2 = Car(
            [car2_x, 1.2, np.pi / 2.0, 0.0],
            [(0, straight_speed)] * EP_LEN
        )

        car3_x = get_lane_x("right", scenario)
        straight_speed = 0.3
        car3 = Car(
            [car3_x, -0.5, np.pi / 2.0, 0.0],
            [(0, straight_speed)] * EP_LEN
        )
        cars = [car1, car2, car3]

    else:
        raise Exception("Unknown cars trajectory:", cars_trajectory)
    return cars



def get_driver(scenario, constraint=False, viz_obs=False):
    cars = get_cars(scenario)

    if scenario == 'changing_lane':
        starting_speed = 0.1
    else:
        starting_speed = 0.5

    env =  Driver(
        cars,
        starting_speed=starting_speed,
        constraint=constraint, 
        scenario=scenario
    )
    env = DriverVizWrapper(env) if viz_obs else env
    return env

