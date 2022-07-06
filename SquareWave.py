# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:38:08 2020

@author: acubelic
"""
# Note, canvas coordinate system is flipped compared to cartesian plan

try:
    import tkinter
except ImportError:  # python 2
    import Tkinter as tkinter
import tkinter.colorchooser as colorchooser
import math
import numpy as np


class create_canvas:
    """Create canvas object."""

    def __init__(self, root_window, canvas_width, canvas_height, bgc):
        self.root_window = root_window
        self.width = canvas_width
        self.height = canvas_height
        self.bgc = bgc
        self.id = tkinter.Canvas(root_window, width=self.width,
                                 height=self.height, background=bgc,
                                 highlightthickness=0)

        # Adjusts the number of increments on each axis
        if self.height > self.width:
            self.Y_AXIS_INC = self.height / self.width * AXES_DIVISIONS
            self.X_AXIS_INC = AXES_DIVISIONS
        else:
            self.X_AXIS_INC = self.width / self.height * AXES_DIVISIONS
            self.Y_AXIS_INC = AXES_DIVISIONS

        # to keep track of mouse movement [x, y]
        self._origin = [self.width / 4, self.height / 2]
        self._drag_data = [0, 0]
        self._zoom_data = 0
        # add bindings for clicking, dragging and releasing canvas
        self.id.bind("<ButtonPress-1>", self.drag_start)
        self.id.bind("<ButtonRelease-1>", self.drag_stop)
        self.id.bind("<B1-Motion>", self.drag)
        self.id.bind("<MouseWheel>", self.mouse_wheel)

    def get_canvas_scaleFactors(self):
        """Return the x_scale and y_scale."""
        # Updates canvas & obtains canvas information
        # self.update()

        x_scale_factor = self.width / self.X_AXIS_INC
        y_scale_factor = self.height / self.Y_AXIS_INC

        return x_scale_factor, y_scale_factor

    def update_size(self):
        # Data from old origin
        x_old_origin, y_old_origin = self._origin
        x_ratio = x_old_origin / self.width
        y_ratio = y_old_origin / self.height
        # Updating new origin
        self.width = self.id.winfo_width()
        self.height = self.id.winfo_height()
        self._origin = [self.width * x_ratio, self.height * y_ratio]

        if self.height > self.width:
            self.Y_AXIS_INC = self.height / self.width * AXES_DIVISIONS
            self.X_AXIS_INC = AXES_DIVISIONS
        else:
            self.X_AXIS_INC = self.width / self.height * AXES_DIVISIONS
            self.Y_AXIS_INC = AXES_DIVISIONS

    def update_colour(self, bgc):
        self.bgc = bgc
        self.id.configure(background=bgc)

    def drag_start(self, event):
        """Begining drag of an object"""
        # record the item and its location
        x_pos, y_pos = 0, 1
        self._drag_data[x_pos] = event.x
        self._drag_data[y_pos] = event.y

    def drag_stop(self, event):
        """End drag of an object"""
        # reset the drag information
        x_pos, y_pos = 0, 1
        self._drag_data[x_pos] = 0
        self._drag_data[y_pos] = 0

    def drag(self, event):
        """Handle dragging of an object"""
        # compute how much the mouse has moved
        global TRANSLATE_FACTOR
        x_pos, y_pos = 0, 1
        delta_x = (event.x - self._drag_data[x_pos]) / TRANSLATE_FACTOR
        delta_y = (event.y - self._drag_data[y_pos]) / TRANSLATE_FACTOR

        # record the new translation value
        self._origin[x_pos] += delta_x
        self._origin[y_pos] += delta_y
        update_positions()

    def mouse_wheel(self, event):
        """Handle scrolling of mouse wheel."""
        global AXES_DIVISIONS

        zoom_delta = event.delta / 120
        if AXES_DIVISIONS == 2 and zoom_delta < 0:
            pass
        else:
            AXES_DIVISIONS += zoom_delta
        resize_canvas(canvas_circles.width, canvas_circles.height)


class draw_axes:
    """Draw the x-axis and y-axis within the provided canvas object."""

    def __init__(self, canvas_object, axesColour):
        """Draw the x and y axes on the canvas."""
        self.canvas = canvas_object
        x_pos, y_pos = self.canvas._origin

        self.xid = canvas_object.id.create_line(0, y_pos,
                                                self.canvas.width, y_pos,
                                                fill=axesColour)

        self.yid = canvas_object.id.create_line(x_pos, 0,
                                                x_pos, self.canvas.height,
                                                fill=axesColour)

    def move_axis(self):
        x_pos, y_pos = self.canvas._origin

        self.canvas.id.coords(self.xid, 0, y_pos, self.canvas.width, y_pos)
        self.canvas.id.coords(self.yid, x_pos, 0, x_pos, self.canvas.height)

    def update_colour(self, axes_colour):
        self.canvas.id.itemconfig(self.xid, fill=axes_colour)
        self.canvas.id.itemconfig(self.yid, fill=axes_colour)


class draw_circle:
    """Draw a circle object at with radius r at (xo, yo)."""

    @ staticmethod
    def point_on_circle(circle_instance, t):
        """Calculate cartesian co-ordinate of point on radius of circle.

        Will convert polar coordinates to cartesian coordinates and return
        (x, y) position of point for entered value of
        theta and self.radius.

        Omega is the angular frequency: w = 2*pi*f*
        Indicates how many cycles per second. Assumed equal to 1 i.e. f = 1/2pi
        phi specifies the start position of oscillatory cycle.
        """
        # Calculate co-ordinate on circle cirumference
        radius = circle_instance.radius
        frequency = circle_instance.frequency
        phi = circle_instance.phi
        x_centre = circle_instance.x_centre
        y_centre = circle_instance.y_centre
        a1, a2 = circle_instance.a1, circle_instance.a2

        x_temp = (radius * a1) * math.cos(2 * math.pi * frequency * a2 * t
                                          + phi)
        y_temp = (radius * a1) * math.sin(2 * math.pi * frequency * a2 * t
                                          + phi)

        # Add scale & off-set to account for circle origin
        x_edge = x_temp + x_centre
        y_edge = -y_temp + y_centre

        return (x_edge, y_edge)

    @ staticmethod
    def get_coords(x_centre, y_centre, radius, a1,
                   x_scale_factor, y_scale_factor, x_origin, y_origin):

        x1 = ((x_centre - radius * a1) * x_scale_factor + x_origin)
        y1 = ((y_centre - radius * a1) * y_scale_factor + y_origin)
        x2 = ((x_centre + radius * a1) * x_scale_factor + x_origin)
        y2 = ((y_centre + radius * a1) * y_scale_factor + y_origin)

        return x1, y1, x2, y2

    def __init__(self, canvas_object, x_centre, y_centre, radius, frequency,
                 phi, a1, a2, colour, start_time):
        self.canvas = canvas_object
        self.radius = radius
        self.x_centre = x_centre
        self.y_centre = -y_centre
        self.frequency = frequency
        self.phi = phi
        self.colour = colour
        self.a1 = a1
        self.a2 = a2
        self.x_edge, self.y_edge = self.point_on_circle(self, start_time)

        coords = self.get_coords(self.x_centre, self.y_centre,
                                 self.radius, self.a1,
                                 *self.canvas.get_canvas_scaleFactors(),
                                 *self.canvas._origin)

        self.id = canvas_object.id.create_oval(*coords, outline=self.colour)

    def move_circle(self, new_x_centre, new_y_centre, t):
        """Move circle to new coordinate."""
        self.x_centre = new_x_centre
        self.y_centre = -new_y_centre
        self.x_edge, self.y_edge = self.point_on_circle(self, t)

        coords = self.get_coords(self.x_centre, self.y_centre,
                                 self.radius, self.a1,
                                 *self.canvas.get_canvas_scaleFactors(),
                                 *self.canvas._origin)

        self.canvas.id.coords(self.id, *coords)

    def update_colour(self, circle_colour):
        self.colour = circle_colour
        self.canvas.id.itemconfig(self.id, outline=self.colour)


class draw_wiper:
    """Draw a line from centre of circle to point on circumference.

    or a circle of origin (xo, yo) and radius r, will draw a line from
    the centre to a pre-determined point on the circle's circumference.
    """
    @ staticmethod
    def get_coords(x_centre, y_centre, x_edge, y_edge,
                   x_scale_factor, y_scale_factor, x_origin, y_origin):
        x1 = x_centre
        y1 = y_centre
        x2 = x_edge
        y2 = y_edge
        coords = x1, y1, x2, y2

        canvas_x1 = x1 * x_scale_factor + x_origin
        canvas_y1 = y1 * y_scale_factor + y_origin
        canvas_x2 = x2 * x_scale_factor + x_origin
        canvas_y2 = y2 * y_scale_factor + y_origin
        canvas_coords = canvas_x1, canvas_y1, canvas_x2, canvas_y2

        return coords, canvas_coords

    def __init__(self, canvas_object, circle, colour):

        self.canvas = canvas_object
        self.circle = circle
        self.colour = colour

        coords, canvas_coords = self.get_coords(self.circle.x_centre,
                                                self.circle.y_centre,
                                                self.circle.x_edge,
                                                self.circle.y_edge,
                                                *self.canvas.
                                                get_canvas_scaleFactors(),
                                                *self.canvas._origin)

        self.x1, self.y1, self.x2, self.y2 = coords
        self.id = self.canvas.id.create_line(*canvas_coords,
                                             fill=self.colour)

    def move_wiper(self):
        """Updates position of wiper."""
        coords, canvas_coords = self.get_coords(self.circle.x_centre,
                                                self.circle.y_centre,
                                                self.circle.x_edge,
                                                self.circle.y_edge,
                                                *self.canvas.
                                                get_canvas_scaleFactors(),
                                                *self.canvas._origin)

        self.x1, self.y1, self.x2, self.y2 = coords
        self.canvas.id.coords(self.id, *canvas_coords)

    def update_colour(self, vector_colour):
        self.colour = vector_colour
        self.canvas.id.itemconfig(self.id, fill=self.colour)


class draw_arrow_head:
    """Create arrow tip for end of wiper."""

    @ staticmethod
    def get_triangle_coords(centre_coords, edge_coords):
        """Get coordiantes of isosceles triangle.

        Centre_coords is a tuple of the coordinates of the circle's
        centre (x0, y0).
        edge_coords is a tuple of the coordinates of point on circle's
        circumference.

        Uses ratio formula and midpoint formula.
        Let point R(x, y) be the point which divides PQ in the ratio K1:K2
        i.e. PR:PQ = K1:K2
        """
        # Determines ratio of line length : arrow height
        ratio_lineArrow = 0.175
        ratio_heightBase = 1.2

        # Using mid-point formula to get midpoint of base of triangle
        x1, y1 = centre_coords
        x2, y2 = edge_coords
        x3 = (x2 + (ratio_lineArrow * x1)) / (1 + ratio_lineArrow)
        y3 = (y2 + (ratio_lineArrow * y1)) / (1 + ratio_lineArrow)

        # Defining isoscles triangle base and height
        height = math.sqrt((x3 - x2)**2 + (y3 - y2)**2)
        base = height / math.sqrt(ratio_heightBase**2 - 0.25)

        # Defining slope and intercept of base which is perpendicular to height
        # Wiper is horizontal
        if abs(y2 - y1) < 1e-5:
            x4 = x5 = x3
            y4 = y3 + 0.5 * base
            y5 = y3 - 0.5 * base
        # Wiper is vertical
        elif abs(x2 - x1) < 1e-5:
            y5 = y4 = y3
            x4 = x3 + 0.5 * base
            x5 = x3 - 0.5 * base
        # Wiper has defined slope
        else:
            perp_slope = -(x2 - x1) / (y2 - y1)
            perp_intercept = y3 - (perp_slope * x3)

            # Defining quadratic equation terms x = (-b +- sqrt(b**2 - 4ac))/2a
            a = 1 + perp_slope**2
            b = 2 * (perp_slope * (perp_intercept - y3) - x3)
            c = x3**2 + (perp_intercept - y3)**2 - (base / 2)**2
            d = math.sqrt(b**2 - 4 * a * c)  # d = sqrt(b**2 - 4ac)

            x4 = (-b + d) / (2 * a)
            x5 = (-b - d) / (2 * a)
            y4 = perp_slope * x4 + perp_intercept
            y5 = perp_slope * x5 + perp_intercept

        return (x2, y2, x4, y4, x5, y5)

    @ staticmethod
    def get_coords(triangle_points,
                   x_scale_factor, y_scale_factor, x_origin, y_origin):
        x2, y2, x4, y4, x5, y5 = triangle_points
        x2_canvas = x2 * x_scale_factor + x_origin
        y2_canvas = y2 * y_scale_factor + y_origin
        x4_canvas = x4 * x_scale_factor + x_origin
        y4_canvas = y4 * y_scale_factor + y_origin
        x5_canvas = x5 * x_scale_factor + x_origin
        y5_canvas = y5 * y_scale_factor + y_origin

        return (x2_canvas, y2_canvas, x4_canvas, y4_canvas,
                x5_canvas, y5_canvas)

    def __init__(self, canvas_object, wiper, colour):
        self.canvas = canvas_object
        self.wiper = wiper
        self.colour = colour

        centre_point = (self.wiper.x1, self.wiper.y1)
        edge_point = (self.wiper.x2, self.wiper.y2)
        self.coords = self.get_triangle_coords(centre_point, edge_point)

        canvas_coords = self.get_coords(self.coords,
                                        *self.canvas.get_canvas_scaleFactors(),
                                        *self.canvas._origin)

        self.id = self.canvas.id.create_polygon(canvas_coords,
                                                fill=self.colour)

    def move_arrow(self):
        """Re-positions arrow head for updated positions."""
        centre_point = (self.wiper.x1, self.wiper.y1)
        edge_point = (self.wiper.x2, self.wiper.y2)
        self.coords = self.get_triangle_coords(centre_point, edge_point)

        canvas_coords = self.get_coords(self.coords,
                                        *self.canvas.get_canvas_scaleFactors(),
                                        *self.canvas._origin)

        self.canvas.id.coords(self.id, canvas_coords)

    def update_colour(self, vector_colour):
        self.colour = vector_colour
        self.canvas.id.itemconfig(self.id, fill=vector_colour)


class line_drawer:
    """Create line from end point on circle to edge of canvas."""

    @ staticmethod
    def get_coords(x1, y, x_scale_factor, y_scale_factor, x_origin, y_origin):
        x1_canvas = x1 * x_scale_factor + x_origin
        y_canvas = y * y_scale_factor + y_origin
        x2_canvas = wave_xStart * x_scale_factor + x_origin

        return x1_canvas, y_canvas, x2_canvas, y_canvas

    def __init__(self, canvas_object, wiper, colour):
        self.canvas = canvas_object
        self.wiper = wiper
        self.colour = colour

        coords = self.get_coords(self.wiper.x2, self.wiper.y2,
                                 *self.canvas.get_canvas_scaleFactors(),
                                 *self.canvas._origin)

        self.id = self.canvas.id.create_line(*coords, fill=self.colour)

    def move_line(self):
        """Move line."""
        coords = self.get_coords(self.wiper.x2, self.wiper.y2,
                                 *self.canvas.get_canvas_scaleFactors(),
                                 *self.canvas._origin)

        self.canvas.id.coords(self.id, *coords)

    def update_colour(self, vector_colour):
        self.colour = vector_colour
        self.canvas.id.itemconfig(self.id, fill=vector_colour)


class wave_handler:
    """Destroy and create square wave."""

    @ staticmethod
    def get_coords(x_point, y_point, radius,
                   x_scale_factor, y_scale_factor, x_origin, y_origin):

        x_canvas = x_point * x_scale_factor + x_origin
        y_canvas = y_point * y_scale_factor + y_origin

        x1 = x_canvas + radius
        y1 = y_canvas + radius
        x2 = x_canvas - radius
        y2 = y_canvas - radius

        return x1, y1, x2, y2

    def __init__(self, canvas_object, colour):
        self.canvas = canvas_object
        self.colour = colour
        self.point_ids = []
        self.y_points = []
        self.radius = 1.5

    def create_points(self, y_point):
        global TRANSLATE_FACTOR

        self.y_points.insert(0, y_point)
        coords = self.get_coords(wave_xStart, y_point, self.radius,
                                 *self.canvas.get_canvas_scaleFactors(),
                                 *self.canvas._origin)
        self.point_ids.insert(0,
                              self.canvas.id.create_oval(*coords,
                                                         fill=self.colour,
                                                         outline=self.colour))

        if len(self.point_ids) > NUMBER_POINTS:
            self.y_points.pop()
            self.canvas.id.delete(self.point_ids.pop())

    def move_points(self):
        for index, point in enumerate(self.point_ids):
            x_point = time_list[index] + wave_xStart
            y_point = self.y_points[index]

            coords = self.get_coords(x_point, y_point, self.radius,
                                     *self.canvas.get_canvas_scaleFactors(),
                                     *self.canvas._origin)

            self.canvas.id.coords(point, *coords)

    def update_colour(self, waveform_colour):
        self.colour = waveform_colour
        for point in self.point_ids:
            self.canvas.id.itemconfig(point, fill=self.colour)

# Delete
def get_wave_coeff(n, option):
    if option == 1:  # Square wave
        radius = math.pi / 4
        phi = 0
        a1 = (4 / math.pi) / (2 * n + 1)
        a2 = (2 * n + 1)
    elif option == 2:  # Triangular wave
        radius = math.pi**2 / 8
        phi = math.pi / 2
        a1 = (8 / math.pi**2) / (2 * n + 1)**2
        a2 = (2 * n + 1)
    else:  # Sawtooth wave
        radius = math.pi / 2
        phi = 0
        a1 = (2 / math.pi) / ((-1)**(n + 2) * (n + 1))
        a2 = n + 1

    return radius, phi, a1, a2


def create_objects(option, x_init, y_init, end, start=0, start_time=0):
    """Create additional objects when required."""
    global drawing_line

    for n in range(start, end):
        radius, phi, a1, a2 = get_wave_coeff(n, option=option)
        circles.append(draw_circle(canvas_circles, x_init, y_init,
                                   radius, frequency, phi, a1, a2,
                                   circle_colour, start_time))
        x_init, y_init = circles[n].x_edge, -circles[n].y_edge

    for circle in circles[start:]:
        wipers.append(draw_wiper(canvas_circles, circle, vector_colour))

    for wiper in wipers[start:NUMBER_ARROWS]:
        arrows.append(draw_arrow_head(canvas_circles, wiper, vector_colour))

    drawing_line = line_drawer(canvas_circles, wipers[-1], vector_colour)


def initiate_objects(option=1):
    """Initiate objects to be drawn on canvas."""

    global canvas_axes
    global waveform_axes
    global waveform

    x_origin, y_origin = canvas_circles._origin

    canvas_axes = draw_axes(canvas_circles, axes_colour)
    if clear_switch:
        create_objects(rbValue.get(), 0, 0, NUMBER_CIRCLES)
    else:
        create_objects(option, 0, 0, NUMBER_CIRCLES)
    waveform = wave_handler(canvas_circles, waveform_colour)
    # Saving initial point
    y_last = circles[-1].y_edge
    waveform.create_points(y_last)


class animation_handler:
    """Update and refresh canvas."""

    def __init__(self, canvas_object):
        self.canvas = canvas_object

    def move_objects(self):
        """Moves all objects."""
        new_x_centre, new_y_centre = 0, 0

        for circle in circles:
            circle.move_circle(new_x_centre, new_y_centre, time)
            new_x_centre = circle.x_edge
            new_y_centre = -circle.y_edge

        for wiper in wipers:
            wiper.move_wiper()

        for arrow in arrows:
            arrow.move_arrow()

        drawing_line.move_line()
        canvas_axes.move_axis()
        waveform.move_points()

    def update_wave(self):
        y_last = wipers[-1].y2
        waveform.create_points(y_last)

    def animate(self):
        """Refresh positions of all objects."""
        if start_switch:
            global time
            global TIME_INC
            time += TIME_INC
            self.move_objects()
            self.update_wave()
            self.canvas.id.after(REFRESH_TIME, self.animate)


def canvas_sizeChange(resize):
    """Detect a size change in the canvas."""
    resize_canvas(resize.width, resize.height)


def update_positions():
    for circle in circles:
        x_centre, y_centre = circle.x_centre, -circle.y_centre
        circle.move_circle(x_centre, y_centre, time)

    for wiper in wipers:
        wiper.move_wiper()

    for arrow in arrows:
        arrow.move_arrow()

    drawing_line.move_line()
    canvas_axes.move_axis()
    waveform.move_points()


def resize_canvas(width, height):
    """Track changes to the window size and re-size canvas automatically."""
    canvas_circles.update_size()
    canvas_axes.move_axis()
    update_positions()


def start():
    global start_switch
    global clear_switch
    if clear_switch:
        initiate_objects(option=rbValue.get())
        clear_switch = False
    if not start_switch:
        start_switch = True
        animation.animate()


def stop():
    global start_switch
    start_switch = False


def clear():
    global clear_switch
    global canvas_axes
    global waveform_axes

    clear_switch = True
    stop()
    canvas_circles.id.delete('all')
    circles.clear()
    wipers.clear()
    arrows.clear()
    # canvas_axes = draw_axes(canvas_circles, axes_colour)


def reset():
    """Reset colours & drawing options to initial defaults."""
    global time
    global clear_switch
    global canvas_circles
    global AXES_DIVISIONS

    update_bgc(colour='#19232D')
    update_axes_colour(colour='white')
    update_circle_colour(colour='#9F7B00')
    update_vector_colour(colour='#FFDE6F')
    update_waveform_colour(colour='#FFDE6F')

    time = 0
    circle_slider.set(3)
    frequency_slider.set(0.25)
    AXES_DIVISIONS = 8
    rbValue.set(1)
    canvas_circles._origin = [canvas_circles.width / 4,
                              canvas_circles.height / 2]
    canvas_circles.update_size()

    if not clear_switch:
        change_waveform()

    animation.move_objects()


def change_waveform():
    for n, circle in enumerate(circles):
        radius, phi, a1, a2 = get_wave_coeff(n, option=rbValue.get())
        circle.radius = radius
        circle.phi = phi
        circle.a1 = a1
        circle.a2 = a2

    update_positions()
    y_last = circles[-1].y_edge
    waveform.create_points(y_last)


def change_NoCircles(slider_value):
    global circles, wipers, arrows, drawing_line, NUMBER_CIRCLES

    number_circles = int(slider_value)
    x_init, y_init = circles[-1].x_edge, -circles[-1].y_edge

    old_noCircles = len(circles)

    if number_circles > old_noCircles:
        canvas_circles.id.delete(drawing_line.id)
        create_objects(rbValue.get(), x_init, y_init, number_circles,
                       start=old_noCircles, start_time=time)

    if number_circles < old_noCircles:
        for circle in circles[number_circles:]:
            canvas_circles.id.delete(circle.id)
        circles = circles[:number_circles]
        for wiper in wipers[number_circles:]:
            canvas_circles.id.delete(wiper.id)
        wipers = wipers[:number_circles]
        canvas_circles.id.delete(drawing_line.id)
        drawing_line = line_drawer(canvas_circles, wipers[-1], vector_colour)
        for arrow in arrows[number_circles:]:
            canvas_circles.id.delete(arrow.id)
        arrows = arrows[:number_circles]

    NUMBER_CIRCLES = number_circles


def change_frequency(slider_value):
    global frequency
    global time
    old_freq = frequency
    frequency = float(slider_value)
    for circle in circles:
        circle.frequency = frequency
    time = old_freq * time / frequency


def update_bgc(colour=None):
    """Update canvas background colour."""
    global canvas_colour
    if colour is None:
        canvas_colour = colorchooser.askcolor()[-1]
    else:
        canvas_colour = colour
    canvas_circles.update_colour(canvas_colour)
    bcg_colour_button.config(background=canvas_colour)


def update_axes_colour(colour=None):
    global axes_colour
    if colour is None:
        axes_colour = colorchooser.askcolor()[-1]
    else:
        axes_colour = colour
    canvas_axes.update_colour(axes_colour)
    axes_colour_button.config(background=axes_colour)


def update_circle_colour(colour=None):
    global circle_colour
    if colour is None:
        circle_colour = colorchooser.askcolor()[-1]
    else:
        circle_colour = colour
    for circle in circles:
        circle.update_colour(circle_colour)
    circle_colour_button.config(background=circle_colour)


def update_vector_colour(colour=None):
    global vector_colour
    if colour is None:
        vector_colour = colorchooser.askcolor()[-1]
    else:
        vector_colour = colour
    for wiper in wipers:
        wiper.update_colour(vector_colour)
    for arrow in arrows:
        arrow.update_colour(vector_colour)
    drawing_line.update_colour(vector_colour)
    vector_colour_button.config(background=vector_colour)


def update_waveform_colour(colour=None):
    global waveform_colour
    if colour is None:
        waveform_colour = colorchooser.askcolor()[-1]
    else:
        waveform_colour = colour
    waveform.update_colour(waveform_colour)
    wave_colour_button.config(background=waveform_colour)


# %%
# Global Variables
# Start, stop, clear switches
start_switch = False
clear_switch = False

# Initialising variables
circles = []
wipers = []
arrows = []
drawing_line = None
canvas_axes = None
waveform_axes = None
waveform = None

# Default Settings
# GUI Colours
NUMBER_CIRCLES = 3
NUMBER_ARROWS = 50
AXES_DIVISIONS = 8
wave_xStart = 4
frequency = 0.25
NUMBER_POINTS = 400
canvas_colour = '#19232D'
axes_colour = 'white'
circle_colour = '#9F7B00'
vector_colour = '#FFDE6F'
waveform_colour = '#FFDE6F'

# Time & refresh variables
TRANSLATE_FACTOR = 50
REFRESH_TIME = 17  # 18 # ms 17
TIME_INC = 0.017  # ms 0.015 for testing
time = 0
time_list = np.arange(TIME_INC,
                      TIME_INC * (NUMBER_POINTS + 1),
                      TIME_INC).tolist()

# %%
# Drawing on Tkinter
# creating main window
mainWindow = tkinter.Tk()
mainWindow.title('Fourier Series')
window_width, window_height,  = 1000, 650
mainWindow.geometry('{}x{}'.format(window_width, window_height))
mainWindow.wm_attributes('-topmost', 1)
mainWindow.minsize(window_width, window_height)
mainWindow.update()
# window_width = mainWindow.winfo_width()
# window_height = mainWindow.winfo_height()


# Option Menu
FONT1 = 'Helvetica 14'
FONT2 = 'Helvetica 11'

menu_frame = tkinter.Frame(mainWindow, relief='flat', padx=20, borderwidth=1)
menu_frame.pack(side='left', fill='y')
menu_frame.update()

# Drawing menu options
label = tkinter.Label(menu_frame, text="Options Menu",
                      font='Helvetica 20 bold')
label.grid(row=0, column=0)  # inserts the label

# Drawing Buttons
button_frame = tkinter.Frame(menu_frame)
button_frame.grid(row=1, column=0)
start_button = tkinter.Button(button_frame, text='Start', font=FONT1,
                              command=start)
start_button.grid(row=0, column=0, sticky='we')
stop_button = tkinter.Button(button_frame, text='Stop', font=FONT1,
                             command=stop)
stop_button.grid(row=0, column=1, sticky='we')
clear_button = tkinter.Button(button_frame, text='Clear', font=FONT1,
                              command=clear)
clear_button.grid(row=0, column=2, sticky='we')
reset_button = tkinter.Button(button_frame, text='Reset', font=FONT1,
                              command=reset)
reset_button.grid(row=1, column=2, sticky='we')

# Radio buttons to select wave form
Waveform_frame = tkinter.LabelFrame(menu_frame, text='Wave Form', font=FONT1)
Waveform_frame.grid(row=2, column=0, sticky='we')

# Initiate radio button variable
rbValue = tkinter.IntVar()
rbValue.set(1)  # Defualt value which is selected

radio_Square = tkinter.Radiobutton(Waveform_frame, text='Square', font=FONT2,
                                   value=1, variable=rbValue,
                                   command=change_waveform)
radio_Triangle = tkinter.Radiobutton(Waveform_frame, text='Triangle',
                                     font=FONT2,
                                     value=2, variable=rbValue,
                                     command=change_waveform)
radio_Sawtooth = tkinter.Radiobutton(Waveform_frame, text='Sawtooth',
                                     font=FONT2,
                                     value=3, variable=rbValue,
                                     command=change_waveform)
radio_Square.grid(row=0, column=1, sticky='w')
radio_Triangle.grid(row=1, column=1, sticky='w')
radio_Sawtooth.grid(row=2, column=1, sticky='w')

# SLiders
scale_frame = tkinter.LabelFrame(menu_frame, text='Drawing Options',
                                 font=FONT1)
scale_frame.grid(row=3, column=0, sticky='we')


circle_slider = tkinter.Scale(scale_frame, from_=1, to=200,
                              orient='horizontal',
                              command=change_NoCircles)
circle_slider.set(NUMBER_CIRCLES)
circle_slider.grid(row=0, column=0, sticky='we')
circle_slider_label = tkinter.Label(scale_frame, text='Number of Circles',
                                    font=FONT2)
circle_slider_label.grid(row=1, column=0, sticky='w')

mainWindow.update()
frequency_slider = tkinter.Scale(scale_frame, from_=0.01, to=3,
                                 digits=3, resolution=0.01,
                                 orient='horizontal',
                                 length=scale_frame.winfo_width(),
                                 command=change_frequency)
frequency_slider.set(frequency)
frequency_slider.grid(row=3, column=0, sticky='we')
frequency_slider_label = tkinter.Label(scale_frame, text='Frequency (Hz)',
                                       font=FONT2)
frequency_slider_label.grid(row=4, column=0, columnspan=3, sticky='w')

# Colour Options
colourOptions_frame = tkinter.LabelFrame(menu_frame, text='Colour Options',
                                         font=FONT1)
colourOptions_frame.grid(row=4, column=0, sticky='we')
# Background Colour
bcg_colour_button = tkinter.Button(colourOptions_frame,
                                   background=canvas_colour,
                                   width=2, height=1, command=update_bgc)
bcg_colour_button.grid(row=0, column=0, sticky='we')
bcg_color_label = tkinter.Label(colourOptions_frame, text='Background',
                                font=FONT2)
bcg_color_label.grid(row=0, column=1, sticky='w')
# Axes Colour
axes_colour_button = tkinter.Button(colourOptions_frame,
                                    background=axes_colour,
                                    comman=update_axes_colour)
axes_colour_button.grid(row=1, column=0, sticky='we')
axes_colour_label = tkinter.Label(colourOptions_frame, text='Axes', font=FONT2)
axes_colour_label.grid(row=1, column=1, sticky='w')
# Circle Colour
circle_colour_button = tkinter.Button(colourOptions_frame,
                                      background=circle_colour,
                                      command=update_circle_colour)
circle_colour_button.grid(row=2, column=0, sticky='we')
circle_colour_label = tkinter.Label(colourOptions_frame, text='Circles',
                                    font=FONT2)
circle_colour_label.grid(row=2, column=1, sticky='w')
# Vector Colour
vector_colour_button = tkinter.Button(colourOptions_frame,
                                      background=vector_colour,
                                      command=update_vector_colour)
vector_colour_button.grid(row=3, column=0, sticky='we')
vector_colour_label = tkinter.Label(colourOptions_frame, text='Vectors',
                                    font=FONT2)
vector_colour_label.grid(row=3, column=1, sticky='w')
# Wave Colour
wave_colour_button = tkinter.Button(colourOptions_frame,
                                    background=waveform_colour,
                                    command=update_waveform_colour)
wave_colour_button.grid(row=4, column=0, sticky='we')
wave_colour_label = tkinter.Label(colourOptions_frame, text='Waveform',
                                  font=FONT2)
wave_colour_label.grid(row=4, column=1, sticky='w')

menu_frame.rowconfigure((1, 2, 3, 4), pad=10)

# Drawing Canvas
mainWindow.update()
menu_width = menu_frame.winfo_width()
canvas_width = (window_width - menu_width)

canvas_circles = create_canvas(mainWindow, canvas_width, window_height,
                               canvas_colour)
canvas_circles.id.pack(side='left', fill="both", expand=True)
canvas_circles.id.bind("<Configure>", canvas_sizeChange)
mainWindow.update()

# Set the number of incremenets on the scale
if window_height > canvas_width:
    Y_AXIS_INC = window_height / canvas_width * AXES_DIVISIONS
    X_AXIS_INC = AXES_DIVISIONS
else:
    X_AXIS_INC = canvas_width / window_height * AXES_DIVISIONS
    Y_AXIS_INC = AXES_DIVISIONS

initiate_objects()

# Animation code
animation = animation_handler(canvas_circles)

animation.animate()
mainWindow.mainloop()
