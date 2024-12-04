import pygame
import numpy as np
import pygame.rect

class DriverRender:
    def __init__(self, env, height, width):
        self.env = env
        # pygame.init()
        # pygame.display.init()
        # self.screen = pygame.display.set_mode((height, width))
        self.screen = pygame.Surface((height, width))
        self.screen_height = height
        self.screen_width = width

        self.xlim = [self.env.xlim[0]/1.5, self.env.xlim[1]/1.5]
        
        self.ylim = self.env.ylim

        
        PATH = '<abs path to ./img>'
        # self.grass_img = pygame.image.load(f"{PATH}/grass3.png")
        # #Rescale grass_img to screen size
        # self.grass_img = pygame.transform.scale(self.grass_img, (self.screen_height, self.screen_width))
        
        #Rescale car images
        self.car_height = 0.17
        self.car_width = 0.17/2
        self.ego_car_img = pygame.image.load(f"{PATH}/car-white.png")
        #simple rectangle for ego car
        # self.ego_car_img = pygame.Surface((self.screen_height//8, self.screen_width//8))
        #make car red
        self.ego_car_img.fill((255, 0, 0))
        rel_scale_y = (self.car_height/(self.ylim[1]-self.ylim[0]))
        rel_scale_x = (self.car_width/(self.xlim[1]-self.xlim[0]))
        self.ego_car_img = pygame.transform.scale(self.ego_car_img, (rel_scale_y*self.screen_height, rel_scale_x*self.screen_width))
        
        self.car_img = pygame.image.load(f"{PATH}/car-orange.png")
        # self.car_img = pygame.Surface((self.screen_height//8, self.screen_width//8))
        #make car green
        self.car_img.fill((0, 255, 0))
        self.car_img = pygame.transform.scale(self.car_img, (rel_scale_y*self.screen_height, rel_scale_x*self.screen_width))

        

    def convert_env_pos_to_screen_pos(self, x, y):
        x -= 0.17/2
        screen_x = (x - self.xlim[0]) * self.screen_width / (self.xlim[1] - self.xlim[0])
        screen_y = (y - self.ylim[0]) * self.screen_height / (self.ylim[1] - self.ylim[0])
        return screen_y, screen_x
    

    def render(self, mode="rgb_array"):
        self.screen.fill((0, 0, 0))

        min_lane_x = self.xlim[1]
        max_lane_x = self.xlim[0]

        #Make the grass image as background
        # self.screen.blit(self.grass_img, (0, 0))

        for lane in self.env.lanes:
            x_start = lane.start_pos[0] 
            min_lane_x = min(min_lane_x, x_start)
            y_start = self.ylim[0] 

            width = lane.width
            x_end = x_start+width
            max_lane_x = max(max_lane_x, x_end)
            
            y_end = self.ylim[1]

        #Draw road 
        screen_x_start, screen_y_start = self.convert_env_pos_to_screen_pos(min_lane_x, self.ylim[0])
        screen_x_end, screen_y_end = self.convert_env_pos_to_screen_pos(max_lane_x, self.ylim[1])
        pygame.draw.rect(self.screen, (0, 0, 0), (screen_x_start, screen_y_start, screen_x_end-screen_x_start, screen_y_end-screen_y_start))
    
        #Draw lanes
        for lane in self.env.lanes:
            x_start = lane.start_pos[0] 
            y_start = self.ylim[0] 

            width = lane.width
            x_end = x_start+width
            
            y_end = self.ylim[1]

            screen_x_start, screen_y_start = self.convert_env_pos_to_screen_pos(x_start, y_start)
            screen_x_end, screen_y_end = self.convert_env_pos_to_screen_pos(x_start, y_end)
            pygame.draw.line(self.screen, (255, 255, 255), (screen_x_start, screen_y_start), (screen_x_end, screen_y_end), self.screen_width//64)
            screen_x_start, screen_y_start = self.convert_env_pos_to_screen_pos(x_end, y_start)
            screen_x_end, screen_y_end = self.convert_env_pos_to_screen_pos(x_end, y_end)
            pygame.draw.line(self.screen, (255, 255, 255), (screen_x_start, screen_y_start), (screen_x_end, screen_y_end), self.screen_width//64)

        
        for car in self.env.cars:
            x, y, heading, _ = car.state.copy()
           
            x = x + 0.17/4
            if(heading < 0):
                x += 0.17/16
            
            # print(x, y)
            x, y = self.convert_env_pos_to_screen_pos(x, y)

            self.screen.blit(pygame.transform.rotate(self.car_img, ((heading-np.pi/2)/np.pi)*180.0), (x, y))
            

        x, y, heading, _ = self.env.state.copy()
        # print('Heading:', heading*180/np.pi)
        x = x + 0.17/4
        x, y = self.convert_env_pos_to_screen_pos(x, y)
        self.screen.blit(pygame.transform.rotate(self.ego_car_img, ((heading-np.pi/2)/np.pi)*180.0), (x, y))


        #rotate and mirror flip the image
        rgb = pygame.surfarray.array3d(pygame.transform.flip(pygame.transform.rotate(self.screen, 180), False, True))
        if mode == 'grayscale':
            return self.grayscale(rgb)
        elif mode == 'rgb_array':
            return rgb
        elif mode == 'human':
            raise NotImplementedError
        
    def grayscale(self, rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    def close(self):
        pass
        



    
    

    




