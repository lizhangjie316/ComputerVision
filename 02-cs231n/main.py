# -*- encoding: utf-8 -*-
"""
@File    : main.py
@Time    : 2019/8/2 19:33
@Author  : Keen
@Software: PyCharm
"""

import requests

response = requests.get("https://httpbin.org/ip")
print('Your IP is {0}'.format(response.json()['origin']))
