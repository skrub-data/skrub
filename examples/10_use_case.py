"""
Usecase: developing locally, and avoiding to repeat code in production
=======================================================================

We will imagine a use case here and try to use skrub to answer it.
For now, we start with a simple "flat" use case.
Next, we can imagine that the data scientist has to predict fraud in retail. The target
is at the basket level, while most information is at the items level.

"""

# %%
###############################################################################
# A typical use case
# ------------------
#
# As a data scientist, I'm given a project where I have to predict if an email is fishy.
# I am developing and testing my models locally: in a notebook or a python script.
# Once I'm happy, I want to deploy my model.
# In this use case, the model is used in a microservice, and the model has to return the
# score for each row. The microservice receives a payload in a json.
# To avoid having to recode the pipeline once the model is validated into the
# microservice, which is both error-prone and troublesome, I would like to have the
# skrub pipeline ready, and just have to load it.
# It is also helpful, because it forces me to know before hand what kind of data I will
# or can have at the entrance of the microservice, and avoid to build a model based on
# information that is not accessible yet, in this part of the product pipeline.
# For instance, in my use case, I want to detect a spam before it reaches the receiver
# mailbox. Therefore, I cannot use a feature which would use it the receiver opens the
# email.

# %%
###############################################################################
# Let's create the data set
# ------------------
#
# Given my use case, the data set is not a dataframe, but a list of dict.
# We are going to generate a fully random data set. We will not have a look at the
# quality of the prediction, we want to focus on the pipeline construction.

import random
import string
import uuid
from datetime import datetime, timedelta

import numpy as np


def generate_id():
    return str(uuid.uuid4())


def generate_email():
    length = random.randint(5, 10)
    username = "".join(random.choice(string.ascii_lowercase) for _ in range(length))
    domain = ["google", "yahoo", "whatever"]
    tld = ["fr", "en", "com", "net"]
    return f"{username}@{random.choice(domain)}.{random.choice(tld)}"


def generate_datetime():
    random_seconds = random.randint(0, int(timedelta(days=2).total_seconds()))
    random_datetime = datetime.now() - timedelta(seconds=random_seconds)
    return random_datetime


def generate_text(min_str_length, max_str_length):
    random_length = random.randint(min_str_length, max_str_length)
    random_text = "".join(
        random.choice(string.ascii_letters + string.digits + string.punctuation)
        for _ in range(random_length)
    )
    return random_text


# %%
n_samples = 1000

X = [
    {
        "id": generate_id(),
        "sender": generate_email(),
        "title": generate_text(max_str_length=10, min_str_length=2),
        "content": generate_text(max_str_length=100, min_str_length=10),
        "date": generate_datetime(),
        "cc_emails": [generate_email() for _ in range(random.randint(0, 5))],
    }
    for _ in range(n_samples)
]


# generate array of 1 and 0 to represent the target variable
y = np.random.binomial(n=1, p=0.9, size=[n_samples])

# %%
# Let's start our skrub DataPlan by indicating what are the features and the target
# variable.
import skrub

X = skrub.X(X)
y = skrub.y(y)

# %%
# The variable X for now is a list of dicts. It's not something that an estimator can
# handle directly.
# Let's convert it to a pandas DataFrame.
import pandas as pd

df = X.skb.apply_func(pd.DataFrame)

# %%
# For this example, we will use a strong baseline, with a tabular learner.
learner = skrub.tabular_learner("classification")

# We can now apply the learner to the data.
# The pipeline is fitted when applying the learner to the data.
predictions = df.skb.apply(learner, y=y)

predictions.skb.draw_graph()

# %%
# To end the explorative work, I can save the model to a file somewhere.
import joblib

with open("learner.pkl", "wb") as f:
    joblib.dump(predictions.skb.get_pipeline(fitted=True), f, protocol=5)

# %%
# In my microservice, I receive a payload in json format.
X_input = {
    "id": generate_id(),
    "sender": generate_email(),
    "title": generate_text(max_str_length=10, min_str_length=2),
    "content": generate_text(max_str_length=100, min_str_length=10),
    "date": generate_datetime(),
    "cc_emails": [generate_email() for _ in range(random.randint(0, 5))],
}

# I just have to load the learner and use it to predict the score for this input.
with open("learner.pkl", "rb") as f:
    loaded_learner = joblib.load(f)

prediction = loaded_learner.predict({"X": [X_input]})
prediction

# %%
###############################################################################
# Conclusion
# ----------
#
# Thanks to skrub pipeline, I have the insurance that all the transformations and
# preprocessing done when developing the models are similar to the ones done in
# production.
# The code in production becomes very lightweight, and it's straightforward to deploy.
