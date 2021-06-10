import matplotlib.pyplot as plt 
import cv2
import os
import sys
from IPython import embed

class ImageMarker:

    def __init__(self, tasks=[]) -> None:
        super().__init__()
        self._fig = plt.figure()
        self._tasks = tasks
        self._task_index = -1
        self._current_task = None
        self._marker_set = False
        self._interrupt = False
        self._fig.canvas.mpl_connect('button_press_event', self._on_click_event)
        self._fig.canvas.mpl_connect('close_event', self._fig_close_event)
        self._fig.canvas.mpl_connect('key_press_event', self._key_press_event)

    def mark_movie(self, filename, frame_number=0):
        if not os.path.exists(filename):
            raise IOError("file %s does not exist!" % filename)
        video = cv2.VideoCapture()
        video.open(filename)
        frame_counter = 0
        success = True
        frame = None
        while success and frame_counter <= frame_number:
            print("Reading frame: %i" % frame_counter, end="\r")
            success, frame = video.read()
            frame_counter += 1

        if success:
           self._fig.gca().imshow(frame)
        else:
           print("Could not read frame number %i either failed to open movie or beyond maximum frame number!" % frame_number)
           return []
        plt.ion()
        plt.show(block=False)
        
        self._task_index = -1
        if len(self._tasks) > 0:
            self._next_task()
        
        while not self._tasks_done:
            plt.pause(0.250)
            if self._interrupt:
                return []

        self._fig.gca().set_title("All set and done!\n Window will close in 2s")
        self._fig.canvas.draw()
        plt.pause(2.0)
        return [t.marker_positions for t in self._tasks]

    def _key_press_event(self, event):
        print("Key pressed: %s!" % event.key)
    
    @property
    def _tasks_done(self):
        done = self._task_index == len(self._tasks) and self._current_task is not None and self._current_task.task_done
        return done
    
    def _next_task(self):
        if self._current_task is None:
            self._task_index += 1
            self._current_task = self._tasks[self._task_index]
     
        if self._current_task is not None and not self._current_task.task_done:
            self._fig.gca().set_title("%s: \n%s: %s" % (self._current_task.name, self._current_task.message, self._current_task.current_marker))
            self._fig.canvas.draw()
        elif self._current_task is not None and self._current_task.task_done:
            self._task_index += 1
            if self._task_index < len(self._tasks):
                self._current_task = self._tasks[self._task_index]
                self._fig.gca().set_title("%s: \n%s: %s" % (self._current_task.name, self._current_task.message, self._current_task.current_marker))
                self._fig.canvas.draw()
     
    def _on_click_event(self, event):
        self._fig.gca().scatter(event.xdata, event.ydata, marker=self._current_task.marker_symbol, color=self._current_task.marker_color, s=20)
        event.canvas.draw()
        self._current_task.set_position(self._current_task.current_marker, event.xdata, event.ydata)
        self._next_task()

    def _fig_close_event(self, even):
        self._interrupt = True

class MarkerTask():
    def __init__(self, name:str, marker_names=[], message="", marker="o", color="tab:blue") -> None:
        super().__init__()
        self._positions = {}
        self._marker_names = marker_names
        self._name = name
        self._message = message
        self._current_marker = marker_names[0] if len(marker_names) > 0 else None
        self._current_index = 0
        self._marker = marker
        self._marker_color = color
    
    @property
    def positions(self):
        return self._positions
    
    @property
    def name(self)->str:
        return self._name

    @property
    def message(self)->str:
        return self._message
    
    def set_position(self, marker_name, x, y):
        self._positions[marker_name] = (x, y)
        if not self.task_done:
            self._current_index += 1
            self._current_marker = self._marker_names[self._current_index]
    
    @property
    def marker_positions(self):
        return self._positions

    @property
    def task_done(self):
        return len(self._positions) == len(self._marker_names)
    
    @property
    def current_marker(self):
        return self._current_marker
    
    @property
    def marker_symbol(self):
        return self._marker
    
    @property
    def marker_color(self):
        return self._marker_color
    
    def __str__(self) -> str:
        return "MarkerTask %s with markers: %s" % (self.name, [mn for mn in self._marker_names])


if __name__ == "__main__":
    tank_task = MarkerTask("tank limits", ["bottom left corner", "top left corner", "top right corner", "bottom right corner"], "Mark tank corners")
    feeder_task = MarkerTask("Feeder positions", list(map(str, range(1, 2))), "Mark feeder positions")
    tasks = [tank_task, feeder_task]
    im = ImageMarker(tasks)
    # vid1 = "2020.12.11_lepto48DLC_resnet50_boldnessDec11shuffle1_200000_labeled.mp4"
    print(sys.argv[0])
    print (sys.argv[1])
    vid1 = sys.argv[1]
    marker_positions = im.mark_movie(vid1, 10)
    print(marker_positions)