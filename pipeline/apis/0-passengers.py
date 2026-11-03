#!/usr/bin/env python3
"""
Module to query SWAPI and filter starships by passenger capacity.
"""
import requests


def availableShips(passengerCount):
    """Returns a list of starship names that can hold at least

    the specified number of passengers.
    """
    url = "https://swapi-api.hbtn.io/api/starships/"
    eligible_ships = []

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            break

        data = response.json()
        for ship in data.get("results", []):
            passengers = ship.get("passengers")
            if passengers and passengers not in ("n/a", "unknown"):
                try:
                    capacity = int(passengers.replace(",", ""))
                    if capacity >= passengerCount:
                        eligible_ships.append(ship.get("name"))
                except ValueError:
                    pass

        url = data.get("next")

    return eligible_ships
