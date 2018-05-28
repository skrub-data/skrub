"""
Basic dirty_cat example: manipulating and looking at data
=======================================================

"""

#########################################################################
# A first step: looking at our data
# ----------------------------------
#
import os
import pandas as pd
from dirty_cat.datasets.fetching import fetch_employee_salaries
import matplotlib.pyplot as plt

data_path = fetch_employee_salaries()
data_path = os.path.join(data_path, 'rows.csv')
df = pd.read_csv(data_path)
print(df.head(n=5))

#########################################################################
# Let's look at the the distribution of each column
print(df.columns)
print(df.nunique())

#########################################################################
# Some numerical columns (Gross pay, etc..) and some obvious categorical
# columns such as full_name
# of course have many different values. but it is also the case
# for other categorical columns  such as Employee position title

sorted_values=df['Employee Position Title'].sort_values().unique()
for i in range(5):
    print(sorted_values[i]+'\n')


#########################################################################
# Here we go! See how there are 3 kinds of Accountant/Auditor? I,II,and III. 
# Using simple one-hot encoding will create orthogonal features, whereas
# it is clear that those 3 terms have a lot in common. That's where our 
# encoders get into play
