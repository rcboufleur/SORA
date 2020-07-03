from astropy.coordinates import SkyCoord, EarthLocation, GCRS, AltAz
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.time import Time
import astropy.units as u
from astroquery.mpc import MPC
import numpy as np
from sora.config import test_attr, input_tests


def search_code_mpc():
    """ Reads the MPC Observer Database

    Return:
        observatories (dict): A python dictionaty with all the sites
            as an Astropy EarthLocation object
    """
    obs = MPC.get_observatory_codes()
    observatories = {}
    for line in obs:
        code = line['Code']
        lon = line['Longitude']*u.deg
        rcphi = line['cos']*6378.137*u.km
        rsphi = line['sin']*6378.137*u.km
        name = line['Name']
        site = EarthLocation.from_geocentric(rcphi*np.cos(lon), rcphi*np.sin(lon), rsphi)
        observatories[code] = (name, site)
    return observatories


class Observer():
    """Define the observer object
    """
    __names = []

    def __init__(self, **kwargs):
        """
        Parameters:
            name (str): Name for the Observer. (required)
                Each time an Observer object is defined, the name must be different.
            code (str): The IAU code to look for coordinates in MPC database
            site (EarthLocation): To provide an EarthLocation object.
            lon (str, float): The Longitude of the site in degrees.
            lat (str, float): The Latitude of the site in degrees.
            height (int, float): The height of the site in meters above see level.

        The user must provide one of the followings:
        Observer(code)
        Observer(name, code) if the user wants to use a different name from the MPC database
        Observer(name, site)
        Observer(name, lon, lat, height)
        """
        input_tests.check_kwargs(kwargs, allowed_kwargs=['code', 'height', 'lat', 'lon', 'name', 'site'])
        if 'code' in kwargs:
            self.code = kwargs['code']
            try:
                name, self.site = search_code_mpc()[self.code]
                self.__name = kwargs.get('name', name)
            except:
                raise ValueError('code {} could not be located in MPC database'.format(self.code))
        elif all(i in kwargs for i in ['name', 'site']):
            self.__name = kwargs['name']
            self.site = test_attr(kwargs['site'], EarthLocation, 'site')
        elif all(i in kwargs for i in ['name', 'lon', 'lat', 'height']):
            self.__name = kwargs['name']
            self.site = EarthLocation(kwargs['lon'], kwargs['lat'], kwargs['height'])
        else:
            raise ValueError('Input parameters could not be determined')
        if self.__name in self.__names:
            raise ValueError("name {} already defined for another Observer object. "
                             "Please choose a different one.".format(self.__name))
        self.__names.append(self.__name)

    def get_ksi_eta(self, time, star):
        """ Calculates relative position to star in the orthographic projection.

        Parameters:
        time (str, Time):Time from which to calculate the position.
            It can be a string in the format "yyyy-mm-dd hh:mm:ss.s" or an astropy Time object
        star (str, SkyCoord):The coordinate of the star in the same frame as the ephemeris.
            It can be a string in the format "hh mm ss.s +dd mm ss.ss" or an astropy SkyCoord object.

        Returns:
        ksi, eta (float): on-sky orthographic projection of the observer relative to a star
        """
        time = test_attr(time, Time, 'time')
        try:
            star = SkyCoord(star, unit=(u.hourangle, u.deg))
        except:
            raise ValueError('star is not an astropy object or a string in the format "hh mm ss.s +dd mm ss.ss"')

        itrs = self.site.get_itrs(obstime=time)
        gcrs = itrs.transform_to(GCRS(obstime=time))
        rz = rotation_matrix(star.ra, 'z')
        ry = rotation_matrix(-star.dec, 'y')

        cp = gcrs.cartesian.transform(rz).transform(ry)
        return cp.y.to(u.km).value, cp.z.to(u.km).value

    def sidereal_time(self, time, mode='local'):
        """Calculates the Apparent Sidereal Time at a certain time

        Parameters:
            time (str,Time): Time to calculate sidereal time.
            mode (str): if 'local' it calculates the sidereal time for
                the coordinates of this object. If 'greenwich', it
                calculates the Greenwich Apparent Sidereal Time.

        Returns:
            sidereal_time: An Astropy Longitude object with the ST.
        """
        # return local or greenwich sidereal time
        time = test_attr(time, Time, 'time')
        time.location = self.site
        if mode == 'local':
            return time.sidereal_time('apparent')
        elif mode == 'greenwich':
            return time.sidereal_time('apparent', 'greenwich')
        else:
            raise ValueError('mode must be "local" or "greenwich"')

    def altaz(self, time, coord):
        """Calculates the Altitude and Azimuth at a certain time for a coordinate

        Parameters:
            time (str,Time): Time to calculate sidereal time.
            coord (str, astropy.SkyCoord):Coordinate of the target ICRS.

        Returns:
            altitude (float): in degrees.
            azimuth (float): in degrees.
        """
        time = test_attr(time, Time, 'time')
        if type(coord) == str:
            coord = SkyCoord(coord, unit=(u.hourangle, u.deg))
        ephem_altaz = coord.transform_to(AltAz(obstime=time, location=self.site))
        return ephem_altaz.alt.deg, ephem_altaz.az.deg

    @property
    def name(self):
        return self.__name

    @property
    def lon(self):
        return self.site.lon

    @lon.setter
    def lon(self, lon):
        lat = self.site.lat
        height = self.site.height
        site = EarthLocation(lon, lat, height)
        self.site = site

    @property
    def lat(self):
        return self.site.lat

    @lat.setter
    def lat(self, lat):
        lon = self.site.lon
        height = self.site.height
        site = EarthLocation(lon, lat, height)
        self.site = site

    @property
    def height(self):
        return self.site.height

    @height.setter
    def height(self, height):
        lon = self.site.lon
        lat = self.site.lat
        site = EarthLocation(lon, lat, height)
        self.site = site

    def __str__(self):
        """String representation of the Observer class
        """
        out = ('Site: {}\n'
               'Geodetic coordinates: Lon: {}, Lat: {}, height: {:.3f}'.format(
                   self.name, self.site.lon.__str__(), self.site.lat.__str__(),
                   self.site.height.to(u.km))
               )
        return out

    def __del__(self):
        """ When this object is deleted, it removes the name from the Class name list.
        """
        try:
            self.__names.remove(self.__name)
        except:
            pass
