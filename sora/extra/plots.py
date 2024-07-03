import astropy.units as u
import numpy as np
from scipy.signal import savgol_filter

__all__ = ['draw_ellipse']

def get_ellipse_cartesian_coordinates(theta, equatorial_radius, oblateness, 
                                      center_f, center_g, position_angle):
    """
    Calculate the Cartesian coordinates of points on an ellipse.

    Parameters:
    ----------
    theta : `float`
        The angle in radians.
    equatorial_radius  : `float`
        The equatorial radius of the ellipse.
    oblateness  : `float`
        The oblateness of the ellipse.
    center_f  : `float`
        The x-coordinate of the center of the ellipse.
    center_g  : `float`
        The y-coordinate of the center of the ellipse.
    position_angle  : `float`
        The position angle of the ellipse.

    Returns:
    ----------
    tuple : `float`
        A tuple containing the x and y coordinates of the points on the ellipse.
    """
    # Calculate x and y coordinates of the ellipse
    circle_x = equatorial_radius * np.cos(theta)
    circle_y = equatorial_radius * (1.0 - oblateness) * np.sin(theta)
    
    # Convert position angle to degrees
    pos_ang = position_angle * u.deg
    
    # Calculate final x and y positions of the ellipse
    x_pos = circle_x * np.cos(pos_ang) + circle_y * np.sin(pos_ang) + center_f
    y_pos = -circle_x * np.sin(pos_ang) + circle_y * np.cos(pos_ang) + center_g
    
    return x_pos, y_pos

    
def draw_ellipse(equatorial_radius, oblateness=0.0, center_f=0.0, center_g=0.0,
                 position_angle=0.0, center_dot=False, ax=None, 
                 draw_error_ellipses=False, **kwargs):
    """Plots an ellipse with the given input parameters.

    Parameters
    ----------
    equatorial_radius : `float`, `int`, default=0
        Semi-major axis of the ellipse.

    oblateness : `float`, `int`, default=0
        Oblateness of the ellipse.

    center_f : `float`, `int`, default=0
        Coordinate of the ellipse (abscissa).

    center_g : `float`, `int`, default=0
        Coordinate of the ellipse (ordinate).

    center_dot : `bool`, default=False
        If True, plots a dot at the center of the ellipse.

    position_angle : `float`, `int`, default= 0
        Pole position angle (Default=0.0).

    ax : `maptlotlib.pyplot.Axes`
        Axis where to plot ellipse.

    **kwargs
        All other parameters. They will be parsed directly by matplotlib.
    """
    import matplotlib.pyplot as plt

    equatorial_radius = np.array(equatorial_radius, ndmin=1)
    oblateness = np.array(oblateness, ndmin=1)
    center_f = np.array(center_f, ndmin=1)
    center_g = np.array(center_g, ndmin=1)
    position_angle = np.array(position_angle, ndmin=1)

    theta_size = 1800
    dtheta = 2*np.pi / theta_size
    theta = np.arange(-np.pi, np.pi, dtheta)
    
    polar_radius = np.full((theta_size, len(equatorial_radius)), np.nan)
    polar_theta  = np.full((theta_size, len(equatorial_radius)), np.nan)
    
    transformed_polar_radius = np.full((theta_size, len(equatorial_radius)), np.nan)
    transformed_polar_theta  = np.full((theta_size, len(equatorial_radius)), np.nan)
    
    ax = ax or plt.gca()

    if len(equatorial_radius) == 1:
        if 'color' not in kwargs:
            kwargs['color'] = 'black'
        if 'lw' not in kwargs:
            kwargs['lw'] = 2

        x_pos, y_pos = get_ellipse_cartesian_coordinates(theta, 
                                                         equatorial_radius, 
                                                         oblateness, center_f, 
                                                         center_g, 
                                                         position_angle)
        ax.plot(x_pos, y_pos, **kwargs)
    else:
        if 'color' not in kwargs:
            kwargs['color'] = 'gray'
        if 'lw' not in kwargs:
            kwargs['lw'] = 0.1
        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.5
        if 'zorder' not in kwargs:
            kwargs['zorder'] = 0.5
            
        for i in np.arange(len(equatorial_radius)):
            x_pos, y_pos = get_ellipse_cartesian_coordinates(theta, 
                                                             equatorial_radius[i], 
                                                             oblateness[i], 
                                                             center_f[i], 
                                                             center_g[i], 
                                                             position_angle[i])
            radius = np.sqrt(x_pos**2 + y_pos**2)
            theta_rot = np.arctan2(y_pos, x_pos)
            polar_radius[:,i], polar_theta[:, i] = radius, theta_rot
    
            # TODO: Remove the draw_error_ellipses once is validated
            if draw_error_ellipses:
                ax.plot(x_pos, y_pos, **kwargs)        
    
        
        row, col = np.shape(polar_theta)
        # Get the transformation indices (a sort of binning of the angles)
        indices = (((polar_theta + np.pi) / (2 * np.pi)) * theta_size).astype(int)
        
        # Use advanced indexing to assign values to new_r and new_t
        transformed_polar_radius[indices, np.arange(col)] = polar_radius
        transformed_polar_theta[indices, np.arange(col)] = polar_theta   
        # Repeat the first line at the end of each array
        transformed_polar_radius[0, :] = transformed_polar_radius[-1, :]
        transformed_polar_theta[0, :] = transformed_polar_theta[-1, :]
    
        # Get the extreme values of the radii
        transformed_polar_radius_max = np.nanmax(transformed_polar_radius, axis=1)
        transformed_polar_radius_min = np.nanmin(transformed_polar_radius, axis=1)
        transformed_polar_theta_mean = np.nanmean(transformed_polar_theta, axis=1)
        
        # smooth out high frequency artifacts
        transformed_polar_radius_max = savgol_filter(transformed_polar_radius_max,
        						 window_length=31, polyorder=4)
        transformed_polar_radius_min = savgol_filter(transformed_polar_radius_min, 
        						 window_length=31, polyorder=4)
        
        # transform the max and min circles back to cartesian
        x_max = transformed_polar_radius_max * np.cos(transformed_polar_theta_mean)
        y_max = transformed_polar_radius_max * np.sin(transformed_polar_theta_mean)
        x_min = transformed_polar_radius_min * np.cos(transformed_polar_theta_mean)
        y_min = transformed_polar_radius_min * np.sin(transformed_polar_theta_mean)
    
        x_fill = np.concatenate([x_min, np.flip(x_max)])
        y_fill = np.concatenate([y_min, np.flip(y_max)])
    
        # Draw the shade region between the circles
        # TODO: Remove the draw_error_ellipses once is validated
        if draw_error_ellipses:
            ax.fill(x_fill, y_fill, edgecolor=None, color='red', alpha=0.5, zorder= -1)
        else:
            ax.fill(x_fill, y_fill, edgecolor=None, **kwargs)
    
    if center_dot:
        kwargs.pop('lw')
        plt.plot(center_f, center_g, '.', **kwargs)
    plt.axis('equal')