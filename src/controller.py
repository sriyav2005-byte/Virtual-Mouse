import pyautogui
import numpy as np
import time

class MouseController:
    def __init__(self, screen_w=1920, screen_h=1080):
        self.screen_w = screen_w
        self.screen_h = screen_h
        pyautogui.FAILSAFE = False
        
        self.prev_x, self.prev_y = 0, 0
        self.smooth_factor = 3.0
        
        self.is_dragging = False
        pyautogui.PAUSE = 0.0
        
        # Get actual screen size if available
        try:
            self.screen_w, self.screen_h = pyautogui.size()
        except:
            pass

    def get_screen_coords(self, cam_x, cam_y, pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        
        cam_x = np.clip(cam_x, x1, x2)
        cam_y = np.clip(cam_y, y1, y2)
        
        screen_x = np.interp(cam_x, (x1, x2), (0, self.screen_w))
        screen_y = np.interp(cam_y, (y1, y2), (0, self.screen_h))
        
        return int(screen_x), int(screen_y)

    def move(self, sx, sy):
        if self.prev_x == 0 and self.prev_y == 0:
            self.prev_x, self.prev_y = sx, sy
            
        curr_x = self.prev_x + (sx - self.prev_x) / self.smooth_factor
        curr_y = self.prev_y + (sy - self.prev_y) / self.smooth_factor
        
        pyautogui.moveTo(curr_x, curr_y)
        self.prev_x, self.prev_y = curr_x, curr_y
        
    def click(self, button='left'):
        pyautogui.click(button=button)
        time.sleep(0.2)
        
    def double_click(self):
        pyautogui.doubleClick()
        time.sleep(0.2)
        
    def scroll(self, amount):
        pyautogui.scroll(amount)
        
    def start_drag(self):
        if not self.is_dragging:
            pyautogui.mouseDown(button='left')
            self.is_dragging = True
            
    def stop_drag(self):
        if self.is_dragging:
            pyautogui.mouseUp(button='left')
            self.is_dragging = False
