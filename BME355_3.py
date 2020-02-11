"""
BME 355 - Assignment 1 Q3
BME355_3.py
Sarah Schwartzel    20710946
Emily Bauer         20727725
2020/02/10
"""

# c is CO2
# phi is constant rate of respiration
# dt is change in time
# alpha is constant
# r is respiration rate (L/s)
# theta is const

c = phi*dt - alpha*r
rMax = c/(theta+c)