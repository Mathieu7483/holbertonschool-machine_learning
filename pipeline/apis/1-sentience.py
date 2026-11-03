#!/usr/bin/env python3
"""
Module to query SWAPI and return the names
of the home planets of all sentient species.
"""
import requests


def sentientPlanets():
    """Returns a list of names of the home planets of all sentient species."""
    url = "https://swapi-api.hbtn.io/api/species/"
    sentient_planets = []

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            break

        data = response.json()
        for species in data.get("results", []):
            classification = species.get("classification", "").lower()
            designation = species.get("designation", "").lower()

            # Check if species is sentient
            if "sentient" in classification or "sentient" in designation:
                homeworld_url = species.get("homeworld")
                if homeworld_url:
                    planet_response = requests.get(homeworld_url)
                    if planet_response.status_code == 200:
                        planet_data = planet_response.json()
                        planet_name = planet_data.get("name")
                        if planet_name and planet_name not in sentient_planets:
                            sentient_planets.append(planet_name)

        url = data.get("next")

    return sentient_planets
