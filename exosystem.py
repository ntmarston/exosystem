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

from astropy import units as u
from astropy.constants import R_earth, M_earth
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.simbad import Simbad
import matplotlib.pyplot as plt
from math import gcd, pi
import numpy as np

class Planet:

    def __init__(self, name: str, mass=-1.0, radius=-1.0, orbital_period=-1.0, density=-1.0, system=None):
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
        self.radius = radius * u.earthRad
        self.mass = mass * u.earthMass
        self.orbital_period = orbital_period * u.d
        self.density = density * u.g / (u.cm ** 3)
        self.system = system
        if not self.system is None:
            self.system = system[0]


        if mass < 0:
            self.mass = None
        if radius < 0:
            self.radius = None
        if orbital_period < 0:
            self.orbital_period = None
        if density < 0:
            self.density = None

        if (self.radius is None) and (self.mass is None):
            raise ValueError("Planet needs either a mass or a radius!")

        #Automatic assignments
        if (self.density is None) and not (self.mass is None or self.radius is None):
            self.density = self.__pl_density_cgs(self.radius, self.mass) #g/cm3
        self.base_class = self.__get_base_class()
        self.spec_class = self.__get_special_class()


        #todo add literature list

    def __str__(self):
        """
        Overrides default tostring
        """
        return (f"Name: {self.name}\n"
        f"Host/System name: {self.system}\n"
        f"Type: {self.base_class}\n" 
        f"Radius: {self.radius}\n" 
        f"Mass: {self.mass}\n" 
        f"Orbital Period: {self.orbital_period}\n" 
        f"Density: {self.density}\n" 
        f"Special Class: {self.spec_class}\n")

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

        sp_density_threshold = 0.3 * (u.g / u.cm**3)
        sp_mass_threshold = 30 * u.earthMass #not in accordance with [1]
        if self.density is None:
            return None

        if self.density <= sp_density_threshold:
            if self.mass < sp_mass_threshold:
                return "super-puff"
            elif self.mass < (55 * u.earthMass):
                return "near-super-puff"

        return None

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
            radius = radius * R_earth  # assume in Earth radii
        if not isinstance(mass, u.Quantity):
            mass = mass * M_earth  # assume in Earth masses

        # Convert to cgs
        radius_cgs = radius.to(u.cm)
        mass_cgs = mass.to(u.g)

        V = (4 / 3) * pi * radius_cgs ** 3
        density = mass_cgs / V

        return density.to(u.g / u.cm ** 3)



class System:

    def __init__(self, name, spectral_type, planets=[], binary=False, refs = []):
        self.name = name
        self.spectral_type = spectral_type
        self.planets = planets
        self.num_planets = len(planets)
        self.binary = binary
        self.ref_list = refs
        self.resonant_chain = self.find_resonant_pairs()

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

                # print(f"{j}:{j-k} -> {DeltA}")

    def __str__(self):
        """Default tostring method"""
        out = (f"{self.name}\n-------------------\n"
               f"Spectral type (todo:fix): {self.spectral_type}\n"
               f"Planets({self.num_planets}):\n")
        for p in self.planets:
            out += f"{p.name} ({p.base_class}): R={p.radius}, M={p.mass}, P={p.orbital_period}\n"
            if not p.spec_class is None:
                out += f"-> {p.spec_class}\n"

        out += "Extended Analytics\n"
        out += "---------------------\n"
        out += "Mean Motion Resonance (MMR) Information:\n"
        for i, resonance_dict in self.resonant_chain.items():
            for pq, within in resonance_dict.items():
                out += f"Planet {i} and {i - 1} are within {(within * 100):.3f}% of {pq} resonance\n"
        return  out


    def plot(self, labels = False):
        """
        Plots a representation of the system using matplotlib. Each base class of planet (e.g. sub-Earth, super-Earth...)
        is plotted as a different size and different color. Super-puffs are plotted in the style of [1]

        :param labels: Default=False. If set to True, will label the planets with the last letter of the planet name
        :returns: None
        """
        system = self.planets
        system = sorted(system, key=lambda p: p.radius, reverse=True)
        fig, ax = plt.subplots(figsize=(10, 1.5))

        periods = [p.orbital_period.value for p in system]

        label = self.name

        y_pos = 1
        ax.axhspan(0.999 * y_pos, 1.001 * y_pos, xmin=0, xmax=10, color='lightgrey')
        for p in system:
            if p.orbital_period.value is None:
                print(f"Cannot plot {p.name}: No valid orbital period")

            planet_label = p.name.strip()[-1] #gets the planet letter for labeling
            if self.name == "Solar System":
                planet_label = p.name[0]
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
                    plot_size = 5000
                    plot_color = 'saddlebrown'
                case None:
                    print(f"Default parameter set does not include mass or radius measurements for {p.name}!\n Future updates will hopefully handle this!")
                    ax.scatter(p.orbital_period, y_pos, s=500, c='darkgrey')
                    ax.scatter(p.orbital_period, y_pos, s=190, c='maroon', marker="$?$")
                    continue



            ax.scatter(p.orbital_period, y_pos, s=plot_size, c=plot_color)
            if labels:
                ax.scatter(p.orbital_period, y_pos, s=0.1*plot_size, c='black', marker=f"${planet_label}$")
            # Plot these differently
            if p.spec_class == "near-super-puff" or p.spec_class == "super-puff":
                ax.scatter(p.orbital_period, y_pos, s=plot_size * 0.7, c='white')
                ax.scatter(p.orbital_period, y_pos, s=plot_size * 0.33, c=plot_color)

        ax.set_yticks([y_pos])
        ax.set_yticklabels([label])
        ax.set_ylim(0.5, 1.5)
        ax.set_xlim(min(periods) * 0.8, max(periods) * 1.2)
        ax.set_xscale("log")
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
        pl_mass = t['pl_bmasse'] #store the "best mass estimate"
        pl_name = t['pl_name'][0]
        masslim_flag = t['pl_bmasselim']
        pl_rade = t['pl_rade']
        radlim_flag = t['pl_radelim']
        pl_system = t['hostname']
        pl_orbper = t['pl_orbper']
        pl_density = t['pl_dens']
        param_refname = t['pl_refname']
        pl_disco_method = t['discoverymethod']

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
            pl_orbper = pl_orbper.value[0]
        except Exception as e:
            pl_orbper = None

        pl = Planet(name=pl_name, mass=pl_mass, radius=pl_rade, orbital_period=pl_orbper, density=pl_density, system=pl_system)
        try:
            param_refname = param_refname.value[0].split()[2]
            return pl, param_refname
        except Exception as e:
            return pl, None

        #candidate flag
        #everything else into an array?
        pass

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
        system = System(system_name, spectral_type=result['st_spectype'][0],planets=planet_array, refs=planet_param_refs)
        return system




    def get_object_references(self):
        """To be implemented"""
        pass


#PREFABS
#-Solar System
Mercury = Planet(name = "Mercury", mass=0.0553, radius=0.383, orbital_period=88.0, density=5.427)
Venus = Planet(name = "Venus", mass=0.815, radius=0.949, orbital_period=224.7, density=5.243)
Earth = Planet(name = "Earth", mass=1, radius=1, orbital_period=365.2, density=5.52)
Mars = Planet(name = "Mars", mass=0.107, radius=0.532, orbital_period=687.0, density=3.9335)
Jupiter = Planet(name = "Jupiter", mass=317.8, radius=11.21, orbital_period=4331, density=1.326)
Saturn = Planet(name = "Saturn", mass=95.2, radius=9.45, orbital_period=10747, density=0.69)
Uranus = Planet(name = "Uranus", mass=14.5, radius=4.01, orbital_period=30589, density=1.27)
Neptune = Planet(name = "Neptune", mass=17.1, radius=3.88/2, orbital_period=59800, density=1.638)
Pluto = Planet(name = "Pluto?", mass=0.0022, radius=0.187/2, orbital_period=90560, density=1.853)
solar_system = System("Solar System", spectral_type="G2V", planets=[Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto], binary=False)