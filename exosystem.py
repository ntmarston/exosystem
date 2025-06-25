"""
Filename: exosystem.py
Author: Nicholas Marston
Date: 2025-06-17
Version: 0.1
Description:
    Tool for visualization and analysis of exoplanet systems, pulls data from the IPAC Exoplanet Archive and SIMBAD.
    Written for python version 3.13

License: GNU GPL 3.0
Contact: ntmarston@gmail.com
Dependencies: astropy.units, astropy.constants, astroquery, matplotlib, math, numpy

References:
[1]Howe AR, Becker JC, Stark CC, Adams FC (2025) Architecture Classification for Extrasolar Planetary Systems. AJ 169:149.
https://doi.org/10.3847/1538-3881/adabdb
"""
import astropy.units
#--> debug template
#todo remove debug
#print(f"Executing line: {inspect.currentframe().f_lineno}")

from astropy import units as u
from astropy.units import Quantity
from astropy.constants import R_earth, M_earth, R_sun, M_sun, G
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.simbad import Simbad
import matplotlib.pyplot as plt
from math import gcd, pi
import numpy as np
import inspect





class Planet:

    def __init__(self, name: str, mass=None, radius=None, orbital_period=None, density=None, system=None, discovery_method=None):
        """
        Constructor for Planet object. Radius and Mass assumed to be in Earth units, so mass=12 is read in as 12M_earth.
        ->All numerical arguments expect float or int, this constructor will convert them to instances of astropy.units

        :param name: Planet name
        :param mass: Mass of the planet in Earth Masses
        :param radius: Radius/Mean Radius of the planet in Earth Radii
        :param orbital_period: Orbital period (days)
        :param density: Bulk density of the planet in g/cm3
        :param system: Defaults to None. Set by the System class.
        :raises ValueError: raises an exception when neither radius nor mass are specified
        """
        #Note everything in Earth units unless otherwise noted
        self.name = name
        self.discovery_method = discovery_method
        self.mass = None
        self.radius = None
        self.orbital_period = None
        self.density = None
        self.system = None

        if radius is not None:
            if radius <= 0:
                raise ValueError("Radius must be a number greater than zero")
            self.radius = radius * u.earthRad

        if mass is not None:
            if mass <= 0:
                raise ValueError("Mass must be a number greater than zero")
            self.mass = mass * u.earthMass

        if orbital_period is not None:
            if orbital_period <= 0:
                raise ValueError("Orbital period must be a number greater than zero")
            self.orbital_period = orbital_period * u.d

        if density is not None:
            if density <= 0:
                raise ValueError("Density cannot be negative!")
            self.density = density * u.g / u.cm ** 3

        if system is not None:
            self.system = system[0]

        if self.radius is None and self.mass is None:
            raise ValueError("Planet needs either a mass or a radius!")

        # Automatic assignments
        if self.density is None and self.mass is not None and self.radius is not None:
            self.density = self.__pl_density_cgs(self.radius, self.mass)

        self.base_class = self.__get_base_class()
        self.spec_class = self.__get_special_class()


        #todo add literature list

    def __str__(self):
        """
        Overrides default tostring
        """
        out = (f"Name: {self.name}\n"
        f"Host/System name: {self.system}\n"
        f"Type: {self.base_class}\n" 
        f"Radius: {self.radius}\n" 
        f"Mass: {self.mass}\n" 
        f"Orbital Period: {self.orbital_period}\n" 
        f"Density: {self.density}\n" 
        f"Detected by: {self.discovery_method}")
        if not self.spec_class is None:
            out += f"Special Class: {[s for s in self.spec_class if self.spec_class[s] == True]}\n"

        return out


    def __get_base_class(self):
        """
        Internal method, classifies planets based on radius/mass into categories  vaguely based on the scheme used
        in [1]
        """
        #Prefers radius classification because it is more reliable, follows framework of [1]

        if (self.radius is None) and (self.mass is None):
            raise Exception("Cannot get base class for planet with no mass and no radius")
        base_class = None

        if isinstance(self.radius, u.Quantity):
            R_p = self.radius.value
        else:
            R_p = self.radius
        if isinstance(self.mass, u.Quantity):
            M_p = self.mass.value
        else:
            M_p = self.mass

        if not (self.radius is None):
            if R_p < 1:
                base_class = "sub-Earth"
            if 1 <= R_p < 1.75:
                base_class = "super-Earth"
            if 1.75 <= R_p < 3.5:
                base_class = "sub-Neptune"
            if 3.5 <= R_p < 6:
                base_class = "Neptune"
            if R_p >= 6:
                base_class = "Jovian"
                if M_p >= 4133:
                    base_class = "Brown Dwarf"

            return base_class

        if not (self.mass is None):
            if M_p < 1:
                base_class = "sub-Earth"
            if 1 <= M_p < 3.7:
                base_class = "super-Earth"
            if 3.7 <= M_p < 12:
                base_class = "sub-Neptune"
            if 12 <= M_p < 30:
                base_class = "Neptune"
            if 30 <= M_p < 4133:
                base_class = "Jovian"
            if M_p >= 4133:
                base_class = "Brown Dwarf"

            return base_class

    def __get_special_class(self):
        """
        Used to check if this planet meets the criteria for any sort of special classification.
        Currently only special classes are super-puff (ρ<0.3gcm3, M<=30Mearth),
        and near-super-puff (ρ<0.3gcm3, 30<=M<55Mearth). Returns None if no special classifcations match.
        """
        spec_attributes = {'Super-puff':False,
                            'near-Super-puff':False,
                            'Hot Jupiter':False,
                            'USP':False
                            }
        #todo make more than one special class acceptable
        sp_density_threshold = 0.3 * (u.g / u.cm**3)
        sp_mass_threshold = 30 * u.earthMass #not in accordance with [1]
        if not self.density is None:
            if self.density <= sp_density_threshold:
                if self.mass < sp_mass_threshold:
                    spec_attributes["Super-puff"] = True
                elif self.mass < (55 * u.earthMass):
                    spec_attributes["near-Super-puff"] = True

        if not self.orbital_period is None:
            if self.orbital_period.value < 1:
                spec_attributes["USP"] = True
            if self.orbital_period.value < 10 and self.base_class == "Jovian":
                spec_attributes["Hot Jupiter"] = True

        if len([s for s in spec_attributes if spec_attributes[s] == True]) == 0:
            return None
        else:
            return spec_attributes

    @staticmethod
    def __pl_density_cgs(radius, mass):
        """
        (Disclaimer: current docstring written by AI, will be re-done in future release)
        Calculate the bulk density of a planet in cgs units (g/cm³).

        Parameters
        ----------
        radius : float or `~astropy.units.Quantity`
            Planetary radius. If a float is provided, it is assumed to be in Earth radii.
            If a `Quantity` is provided, it must be compatible with length units.

        mass : float or `~astropy.units.Quantity`
            Planetary mass. If a float is provided, it is assumed to be in Earth masses.
            If a `Quantity` is provided, it must be compatible with mass units.

        Returns
        -------
        density : `~astropy.units.Quantity`
            Planetary bulk density in grams per cubic centimeter (g/cm³).

        Notes
        -----
        - If both inputs are floats, the function assumes they are in Earth units.
        - Automatically converts units to CGS (centimeters and grams) before computing.
        - Volume is computed assuming a spherical planet.
        """
        if not isinstance(radius, u.Quantity):
            radius = radius * u.R_earth  # assume in Earth radii
        if not isinstance(mass, u.Quantity):
            mass = mass * u.M_earth  # assume in Earth masses

        # Convert to cgs
        radius_cgs = radius.to(u.cm)
        mass_cgs = mass.to(u.g)

        V = (4 / 3) * pi * radius_cgs ** 3
        density = mass_cgs / V

        return density.to(u.g / u.cm ** 3)

class Star:

    def __init__(self, name: str, spec_type=None, t_eff=None, radius=None, mass=None, metallicity=None, met_ratio = '[Fe/H]', luminosity=None, density=None, age=None, reference=None):
        """
        Constructor for Star object. Radius and Mass assumed to be in solar units. Based on NASA Exoplanet Archive Stellar hosts table.
        ->All numerical arguments expect float or int, this constructor will convert them to instances of astropy.units

        :param name: Star name
        """
        #Note everything in Earth units unless otherwise noted
        self.name = name
        self.spec_type = spec_type  # Morgan-Keenan system

        self.t_eff = t_eff * u.K if t_eff is not None else None
        self.radius = radius * u.Rsun if radius is not None else None
        self.mass = mass * u.Msun if mass is not None else None
        self.metallicity = metallicity * u.dex if metallicity is not None else None
        self.metallicity_ratio = met_ratio #Assumes 'Fe/H' but is sometimes reported differently
        self.luminosity = (10 ** luminosity) * u.solLum if luminosity is not None else None
        self.density = density * u.g / (u.cm ** 3) if density is not None else None
        self.age = age * 1e9 * u.Gyr if age is not None else None  # Gyr to yr
        self.parameter_reference = reference if reference is not None else None

        self.id_map = {
            "Gaia DR2": None, "Gaia DR3": None, "TIC": None,
            "KIC": None, "HIP": None, "2MASS": None, "GALAH": None
        }


    def get_plot_color(self):
        self.Validate_units()
        print(self.t_eff)
        if self.t_eff >= 33000 * u.K: #O
            return '#92B5FF' #Blue
        if self.t_eff >= 10000 * u.K: #B
            return '#A2C0FF' #Blueish white
        if self.t_eff >= 7300 * u.K: #A
            return 'ivory' #White
        if self.t_eff >= 6000 * u.K: #F
            return 'lemonchiffon' #Yellowish white
        if self.t_eff >= 5300 * u.K: #G
            return 'gold'
        if self.t_eff >= 3900 * u.K: #K
            return '#FFDAB5' #pale orange
        if self.t_eff >= 2300 * u.K: #M
            return 'orangered'
        else:
            return 'lightgrey'


    def __str__(self):
        """
        Overrides default tostring
        """
        return (f"Name: {self.name}\n"
        f"Host/Effective Temp: {self.t_eff}\n"
        f"Type: {self.spec_type}\n" 
        f"Luminosity: {self.luminosity}\n" 
        f"Radius: {self.radius}\n" 
        f"Mass: {self.mass}\n" 
        f"Metallicity: {self.metallicity}\n" 
        f"Density: {self.density}\n" 
        f"age: {self.age}\n"
        f"Ref: {self.parameter_reference}\n")

    def Validate_units(self):
        # Temperature (Kelvin)
        if not self.t_eff is None:
            if not isinstance(self.t_eff, Quantity):
                print("Converted to K")
                self.t_eff = self.t_eff * u.K
                print(self.t_eff)
            elif self.t_eff.unit != u.K:
                self.t_eff = self.t_eff.value * u.K
                # Radius [solar radii]
            if self.radius is not None:
                if not isinstance(self.radius, Quantity):
                    self.radius = self.radius * u.R_sun
                elif self.radius.unit != u.R_sun:
                    self.radius = self.radius.to(u.R_sun)

                # Mass [solar masses]
            if self.mass is not None:
                if not isinstance(self.mass, Quantity):
                    self.mass = self.mass * u.M_sun
                elif self.mass.unit != u.M_sun:
                    self.mass = self.mass.to(u.M_sun)

                # Metallicity [dex] — dimensionless but treated as log-scaled
            if self.metallicity is not None:
                if not isinstance(self.metallicity, Quantity):
                    self.metallicity = self.metallicity * u.dex
                elif self.metallicity.unit != u.dex:
                    self.metallicity = self.metallicity.to(u.dex)

                # Luminosity [solar luminosities]
            if self.luminosity is not None:
                if not isinstance(self.luminosity, Quantity):
                    self.luminosity = self.luminosity * u.L_sun
                elif self.luminosity.unit != u.L_sun:
                    self.luminosity = self.luminosity.to(u.L_sun)

                # Density [g/cm^3]
            if self.density is not None:
                if not isinstance(self.density, Quantity):
                    self.density = self.density * (u.g / u.cm ** 3)
                elif not self.density.unit.is_equivalent(u.g / u.cm ** 3):
                    self.density = self.density.to(u.g / u.cm ** 3)

                # Age [Gyr]
            if self.age is not None:
                if not isinstance(self.age, Quantity):
                    self.age = self.age * u.yr
                elif self.age.unit != u.yr:
                    self.age = self.age.to(u.yr)


class System:

    def __init__(self, name, star=None, planets=None, binary=False, refs=None):
        if refs is None:
            refs = []
        if planets is None:
            planets = []
        self.name = name
        self.star = star
        self.planets = planets
        self.num_planets = len(planets)
        self.binary = binary
        self.ref_list = refs
        self.resonant_chain = self.find_resonant_pairs()
        if (not self.star is None) and (not self.star.luminosity is None) and (not self.star.mass is None) and (not self.star.t_eff is None):
            self.HZ_bounds = {0.1: self.find_hz(0.1), 1: self.find_hz(1), 5: self.find_hz(5)}
        else:
            self.HZ_bounds = None

    def __str__(self):
        """Default tostring method"""
        out = (f"{self.name}\n-------------------\n"
               f"Spectral type: {self.star.spec_type}\n"
               f"Planets({self.num_planets}):\n")
        for p in self.planets:
            out += f"{p.name} ({p.base_class}): R={p.radius}, M={p.mass}, P={p.orbital_period}\n"
            if not p.spec_class is None:
                out += f"-> {[s for s in p.spec_class if p.spec_class[s] == True]}\n"

        out += "Extended Analytics\n"
        out += "---------------------\n"
        out += "Mean Motion Resonance (MMR) Information:\n"
        for i, resonance_dict in self.resonant_chain.items():
            for pq, within in resonance_dict.items():
                out += f"Planet {i} and {i - 1} are within {(within * 100):.3f}% of {pq} resonance\n"
        if not self.HZ_bounds is None:
            out += (f"Estimated Habitable Zone Bounds:\n"
                    f"For 0.1 earthMass planet: [{str(self.HZ_bounds[0.1][0])}d - {str(self.HZ_bounds[0.1][1])}]\n"
                    f"For 1 earthMass planet: [{str(self.HZ_bounds[1][0])}d - {str(self.HZ_bounds[1][1])}]\n"
                    f"For 5 earthMass planet: [{str(self.HZ_bounds[5][0])}d - {str(self.HZ_bounds[5][1])}]\n"
                    f"Bounds are between point of runaway greenhouse and maximum greenhouse\n"
                    f"Calculations based on Kopparapu 2014 (https://arxiv.org/pdf/1404.5292)\n"
                    f"Code adapted from https://github.com/Eelt/HabitableZoneCalculator")
        return  out

    def find_resonant_pairs(self):
        """
        Identifies j:j-k mean motion resonance (MMR) pairs within the system. Only considers nearest-neighbor pairs.
        Will report resonant pairs up to 7:1. Resonant pairs are flagged if they are within 5% of a j:j-k MMR,
        defined as those which satisfy the condition:

        |Delta|=|(j - k) / j * (P_outer / P_inner) - 1| < 0.05


        :param self: System
        :returns: A nested dictionary, the outer having keys corresponding to planet pairs from shortest to longest
        orbital period, and the inner being a dictionary of resonances(keys) within 5% and the corresponding value of |Delta|
        """
        if self.planets is None:
            return None

        planets_sorted = sorted(self.planets, key=lambda p: p.orbital_period)
        resonant_pairs = {}
        for i in range(1, len(planets_sorted)):
            P_inner = planets_sorted[i - 1].orbital_period
            P_outer = planets_sorted[i].orbital_period
            resonance_matches = {}
            for j in range(1, 8):
                for k in range(1, j):
                    if gcd(j, k) >= 2: # Stop registering non-relatively-prime pairs
                        continue
                    DeltA = np.abs((j - k) / j * (P_outer / P_inner) - 1)
                    if DeltA < 0.05:
                        resonance_matches[f"{j}:{j - k}"] = float(DeltA)
            resonant_pairs[i] = resonance_matches

        return resonant_pairs



    def find_hz(self, pl_mass):
        """Adapted from Eelt...
        pre-condition: 2600-7200K, pl_masse in (0.1, 5)


        L: Luminosity (Lsun)
        a: Semi-major axis (AU)
        T_eff: Star temperature

        """
        L = self.star.luminosity.value
        T_eff = self.star.t_eff.value
        T_s = (T_eff - 5780)

        # A. Find insolation flux from semi-major axis
        def __insolation_flux_from_a(a):
            return ((1 / a) ** 2) * L

        # B. Find semimajor axis from effective solar flux
        # From Kopparapu et al. 2014. Equation 5, Section 3.1, Page 9
        def __dist_from_Seff(Seff):
            au = (L / Seff) ** 0.5
            return au

        def __orb_per_from_au(au):
            a = au * u.AU
            M = self.star.mass
            T = 2 * pi * np.sqrt((a.to(u.m) ** 3) / (G * M))

            return T.to(u.d).value

        # directly from Eelt
        def Kopparapu2014(SeffSUN, a, b, c, d, tS):
            return SeffSUN + a * tS + b * ((tS) ** 2) + c * ((tS) ** 3) + d * ((tS) ** 4)

        def __find_greenhouse_bounds(temp, zone, pl_mass=5):
            a, b, c, d, Seff_solar = 0, 0, 0, 0, 0
            match zone:
                case "rg":  # runaway greenhouse
                    match pl_mass:
                        case 0.1:
                            Seff_solar = 0.99
                        case 1:
                            Seff_solar = 1.107
                        case 5:
                            Seff_solar = 1.188
                    a = 1.332 * (10 ** -4)
                    b = 1.580 * (10 ** -8)
                    c = -8.308 * (10 ** -12)
                    d = -1.931 * (10 ** -15)
                    return Kopparapu2014(Seff_solar, a, b, c, d, T_s)

                case "mg":
                    Seff_solar = 0.356
                    a = 6.171 * (10 ** -5)
                    b = 1.689 * (10 ** -9)
                    c = -3.198 * (10 ** -12)
                    d = -5.575 * (10 ** -16)
                    return Kopparapu2014(Seff_solar, a, b, c, d, T_s)

                case _:
                    raise ValueError("Invalid Zone")

        runawayGreenhouse = __find_greenhouse_bounds(T_eff, "rg", pl_mass)
        maximumGreenhouse = __find_greenhouse_bounds(T_eff, "mg", pl_mass)

        bounds_au = [__dist_from_Seff(runawayGreenhouse), __dist_from_Seff(maximumGreenhouse)]
        bounds_orbper = [__orb_per_from_au(bounds_au[0]), __orb_per_from_au(bounds_au[1])]

        return bounds_orbper

    #---Plotting methods-----#
    def __set_up_axes(self, y_pos, x_bounds):
        label = self.name
        fig, ax = plt.subplots(figsize=(10, 1.5))
        ax.axhspan(0.999 * y_pos, 1.001 * y_pos, xmin=-0.5, xmax=10, color="slategray")

        ax.set_yticks([y_pos])
        ax.set_yticklabels([label])
        ax.set_ylim(0.5, 1.5)
        ax.set_xlim(x_bounds[0], x_bounds[1])
        ax.set_xscale("log")

        return fig, ax

    def __plt_mark_HZ(self, ax, pl_masse=5, y_pos=1):
        lower_bound = self.HZ_bounds[5][0]
        upper_bound = self.HZ_bounds[5][1]
        ax.plot([lower_bound, upper_bound], [0.5, 0.5], color="springgreen",
                label=f"{5}earthMass HZ", linewidth=10)
        ax.annotate("HZ Estimate", [lower_bound, 0.5], textcoords='offset pixels', xytext=(0, -20), fontsize='small')

    def plot(self, labels = False):
        """
        Plots a representation of the system using matplotlib. Each base class of planet (e.g. sub-Earth, super-Earth...)
        is plotted as a different size and different color. Super-puffs are plotted in the style of [1]

        :param labels: Default=False. If set to True, will label the planets with the last letter of the planet name
        :returns: None
        """
        system = self.planets
        system = sorted(system, key=lambda p: p.radius, reverse=True)

        periods = [p.orbital_period.value for p in system]
        xmin = min(periods) * 0.8
        xmax = max(periods) * 1.2

        y_pos = 1
        fig, ax = self.__set_up_axes(y_pos, [xmin, xmax])

        #Plot star and HZ if stellar parameters available
        if not self.star is None:
            ax.scatter(xmin, 1, s=5000, color=self.star.get_plot_color())
            if not self.HZ_bounds is None:
                self.__plt_mark_HZ(ax, pl_masse=5, y_pos=y_pos)
        #plot planets
        for p in system:
            if p.orbital_period.value is None:
                print(f"Cannot plot {p.name}: No valid orbital period")

            planet_label = p.name.strip()[-1] #gets the planet letter for labeling
            if self.name == "Solar System":
                planet_label = p.name[0]

            #Defaults to get rid of "may not have been initialized" warning
            plot_size = 10
            plot_color = 'red'
            #Match case to determine plot appearance
            match p.base_class:
                case "sub-Earth":
                    plot_size = 100
                    plot_color = 'skyblue'
                case "super-Earth":
                    plot_size = 300
                    plot_color = 'darkgreen'
                case "sub-Neptune":
                    plot_size = 400
                    plot_color = 'blue'
                case "Neptune":
                    plot_size = 500
                    plot_color = 'cadetblue'
                case "Jovian":
                    plot_size = 1100
                    plot_color = 'goldenrod'
                case "Brown Dwarf":
                    plot_size = 4900
                    plot_color = 'saddlebrown'
                case None:
                    print(f"Default parameter set does not include mass or radius measurements for {p.name}!\n Future updates will hopefully handle this!")
                    if labels:
                        ax.scatter(p.orbital_period, y_pos, s=500, c='darkgrey')
                        ax.scatter(p.orbital_period, y_pos, s=150, c='maroon', marker=f"${planet_label}$")
                    else:
                        ax.scatter(p.orbital_period, y_pos, s=500, c='darkgrey')
                        ax.scatter(p.orbital_period, y_pos, s=190, c='maroon', marker="$?$")

                    continue

            #add planet
            ax.scatter(p.orbital_period, y_pos, s=plot_size, c=plot_color)


            # Over plot special planets
            if not p.spec_class is None:
                if p.spec_class["near-Super-puff"] == True or p.spec_class["Super-puff"] == True:
                    ax.scatter(p.orbital_period, y_pos, s=plot_size * 0.7, c='white')
                    ax.scatter(p.orbital_period, y_pos, s=plot_size * 0.33, c=plot_color)

            if labels:
                ax.scatter(p.orbital_period, y_pos, s=0.1*plot_size, c='black', marker=f"${planet_label}$")



        plt.show()

class Queries:

    def __init__(self, database="Exoplanet Archive"):
        self.queries = None
        db_lookup = {
            "exoplanetarchive": NasaExoplanetArchive,
            "ea": NasaExoplanetArchive,
            "ipac": NasaExoplanetArchive,
            "simbad": Simbad
        }
        key = str(database).lower().replace(" ", "")
        if key in db_lookup or database == 1:
            if key == "simbad":
                print("simbad")
                self.queries = db_lookup["simbad"]
            else:
                print("IPAC/EA")
                self.queries = db_lookup.get(key, NasaExoplanetArchive)


    @staticmethod
    def get_planet(planet_name):
        """Builds Planet object by querying NASA Exoplanet Archive, uses default parameter set if
        either pl_rade or pl_bmasse are not null

        :param planet_name: The planet name, as listed on Exoplanet Archive (pl_name)
        :returns Planet: Planet object
        """
        t = NasaExoplanetArchive.query_criteria(table="ps", where=f"pl_name='{planet_name}' and default_flag=1")
        #if default param set has no mass and radius, try a different one
        if (not t[0]['pl_rade'] > 0) and not (t[0]['pl_bmasse'] > 0):
            t2 = NasaExoplanetArchive.query_criteria(table="ps", where=f"pl_name='{planet_name}' and not pl_rade is null")
            print(f"Default parameter set ({t['pl_refname'].value[0].split()[2]}) does not contain either radius or mass measurements. Using non-default set ({t2[0]['pl_refname'].split()[2]})")
            t = t2[0]
        pl_mass = t['pl_bmasse'][0] #store the "best mass estimate"
        pl_name = t['pl_name'][0]
        pl_rade = t['pl_rade'][0]
        pl_system = t['hostname'][0]
        pl_orbper = t['pl_orbper'][0]
        pl_density = t['pl_dens'][0]
        param_refname = t['pl_refname'][0]
        pl_disco_method = t['discoverymethod'][0]

        try:
            pl_mass = pl_mass.value
        except Exception as e:
            pl_mass = None
        try:
            pl_rade = pl_rade.value
        except Exception as e:
            pl_rade = None
        try:
            pl_density = pl_density.value
        except Exception as e:
            pl_density = None
        try:
            pl_orbper = pl_orbper.value
        except Exception as e:
            pl_orbper = None

        pl = Planet(name=pl_name, mass=pl_mass, radius=pl_rade, orbital_period=pl_orbper, density=pl_density, system=pl_system, discovery_method=pl_disco_method)
        try:
            param_refname = param_refname.value[0].split()[2]
            return pl, param_refname
        except Exception as e:
            return pl, None

    @staticmethod
    def get_star(star_name):
        """Only looks at the first entry"""
        table = NasaExoplanetArchive.query_criteria(table="stellarhosts", where=f"hostname='{star_name}' and st_teff IS NOT null")
        if len(table) == 0:
            raise ValueError("Star name not found in IPAC/Exoplanet Archive, or star does not have a t_eff")


        t_eff = float(table[0]['st_teff'].value)
        radius = float(table[0]['st_rad'].value) if not np.isnan(table[0]['st_rad']) else None
        mass = float(table[0]['st_mass'].value) if not np.isnan(table[0]['st_mass']) else None
        metallicity = float(table[0]['st_met'].value) if not np.isnan(table[0]['st_met']) else None
        met_ratio = table[0]['st_metratio'] if not np.isnan(table[0]['st_met']) else None
        luminosity = float(table[0]['st_lum'].value) if not np.isnan(table[0]['st_lum']) else None
        density = float(table[0]['st_dens'].value) if not np.isnan(table[0]['st_dens']) else None
        age = float(table[0]['st_age'].value) if not np.isnan(table[0]['st_age']) else None
        spec_type = table[0]['st_spectype'] if len(table[0]['st_spectype']) > 0 else None
        parameter_reference = table[0]['st_refname']

        return Star(name=star_name, spec_type = spec_type, t_eff = t_eff, radius = radius, mass = mass, metallicity = metallicity, met_ratio=met_ratio, luminosity = luminosity, density = density, age = age, reference = parameter_reference)

    @staticmethod
    def build_system(system_name):
        """Builds System object by querying NASA Exoplanet Archive, uses get_planet()

        :param system_name: The system name, as listed on Exoplanet Archive ('hostname')
        :returns System: System object
        """
        planet_array = []
        planet_param_refs = []
        result = NasaExoplanetArchive.query_criteria(table="ps", where=f"hostname='{system_name}' and default_flag=1")
        for index in range(0, len(result)):
            pl_name = result[index]['pl_name']
            pl, ref = Queries.get_planet(pl_name)
            planet_array.append(pl)
            planet_param_refs.append(ref)
        try:
            star = Queries.get_star(system_name)
        except ValueError:
            star = None
        #stellar parameters
        system = System(system_name,planets=planet_array, star=star, refs=planet_param_refs)
        return system



    def get_object_references(self):
        """To be implemented"""
        pass


#PREFABS
#-Solar System
Sun = Star(name="Sun", spec_type="G2V", mass=1, radius=1, density=1.41, t_eff=5778, metallicity=0, luminosity=0, age=4.603)
Mercury = Planet(name = "Mercury", mass=0.0553, radius=0.383, orbital_period=88.0, density=5.427)
Venus = Planet(name = "Venus", mass=0.815, radius=0.949, orbital_period=224.7, density=5.243)
Earth = Planet(name = "Earth", mass=1, radius=1, orbital_period=365.2, density=5.52)
Mars = Planet(name = "Mars", mass=0.107, radius=0.532, orbital_period=687.0, density=3.9335)
Jupiter = Planet(name = "Jupiter", mass=317.8, radius=11.21, orbital_period=4331, density=1.326)
Saturn = Planet(name = "Saturn", mass=95.2, radius=9.45, orbital_period=10747, density=0.69)
Uranus = Planet(name = "Uranus", mass=14.5, radius=4.01, orbital_period=30589, density=1.27)
Neptune = Planet(name = "Neptune", mass=17.1, radius=3.88/2, orbital_period=59800, density=1.638)
Pluto = Planet(name = "Pluto?", mass=0.0022, radius=0.187/2, orbital_period=90560, density=1.853)
solar_system = System("Solar System", star=Sun, planets=[Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto], binary=False)
