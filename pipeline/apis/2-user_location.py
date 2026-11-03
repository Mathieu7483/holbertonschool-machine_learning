#!/usr/bin/env python3
"""Prints the location of a GitHub user"""
import requests
import sys
import time


if __name__ == '__main__':
    url = sys.argv[1]
    res = requests.get(url)

    if res.status_code == 200:
        print(res.json().get('location'))
    elif res.status_code == 404:
        print('Not found')
    elif res.status_code == 403:
        reset = int(res.headers.get('X-Ratelimit-Reset'))
        minutes = int((reset - time.time()) / 60)
        print('Reset in {} min'.format(minutes))
