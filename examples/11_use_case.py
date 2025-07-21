"""
Usecase: developing locally, and avoiding to repeat code in production
======================================================================

"""

# %%
# As a team of data scientists, we are given a project where we have to predict if an
# email is fishy.
# We am developing and testing our models locally: in a notebook or a python script.
# Once we are happy, we want to deploy our model.
#
# In this use case, every time the email provider receives an email, before actually
# displaying it into the receiver mailbox, they want to check if it's a spam or not.
# To do this, they want to use a machine learning model, contained into a microservice.
# The microservice receives a payload in a json, and returns a score between 0 and 1,
# depending on how likely it is that the email is a spam.
#
# To avoid having to recode the pipeline once the model is validated into the
# microservice, which is both error-prone and troublesome, we would like to be
# able to load an object that executes the same operations as the pipeline:
# the skrub learner can help with this.
# Working in this way is also helpful,
# because it forces us to know beforehand what kind of data we have at the entrance
# of the microservice, and avoids building a model based on information that is not
# accessible yet in this part of the product pipeline. For instance, in this use case,
# we want to detect a spam email before it reaches the receiver mailbox. Therefore, we
# cannot use a feature which is available only when the receiver opens the email.
#
# We will not have a look at the
# quality of the prediction, since this example focuses on the pipeline construction.

# %%
# Generating the training data
# ----------------------------
# In this section, we define a few functions that help us with generating the
# training data in dictionary form. We are going to generate a fully random data set.
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
# We generate 1000 training samples and store them in a list of dictionaries:

n_samples = 1000

# %%
# In this use case, the emails to be tested when the model is put in production
# are not contained in a dataframe, but in a json. As a result, our training data
# should also be contained in a list of dictionaries.

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
y = np.random.binomial(n=1, p=0.9, size=n_samples)

# %%
# Building the DataOps plan
# -------------------------
# Let's start our DataOps plan by indicating what are the features and the target
# variable.
import skrub

X = skrub.X(X)
y = skrub.y(y)

# %%
# The variable X for now is a list of dicts. It's not something that an estimator can
# handle directly.
# Let's convert it to a pandas DataFrame using :func:`~skrub.DataOp.skb.apply_func`.
import pandas as pd

df = X.skb.apply_func(pd.DataFrame)

# %%
# For this example, we will use a strong baseline, with skrub's
# :func:`~skrub.tabular_pipeline()`.
tab_pipeline = skrub.tabular_pipeline("classification")

# We can now apply the predictive model to the data.
# The DataOps plan is ready after applying the model to the data.
predictions = df.skb.apply(tab_pipeline, y=y)

# We can then explore the full plan:
predictions.skb.draw_graph()

# %%
# To end the explorative work, we need to build the learner, fit it, and save it to a
# file.
# Passing ``fitted=True`` to the :meth:`.skb.make_learner() <DataOp.skb.make_learner>`
# function makes it so that the learner is fitted on the data that has been passed to
# the variables of the DataOps plan.
import joblib

with open("learner.pkl", "wb") as f:
    learner = predictions.skb.make_learner(fitted=True)
    joblib.dump(learner, f)

# %%
# Production phase
# ----------------
#
# In our microservice, we receive a payload in json format.
X_input = {
    "id": generate_id(),
    "sender": generate_email(),
    "title": generate_text(max_str_length=10, min_str_length=2),
    "content": generate_text(max_str_length=100, min_str_length=10),
    "date": generate_datetime(),
    "cc_emails": [generate_email() for _ in range(random.randint(0, 5))],
}

# We just have to load the learner and use it to predict the score for this input.
with open("learner.pkl", "rb") as f:
    loaded_learner = joblib.load(f)
# ``X_input`` must be passed as a list so that it can be parsed correctly as a dataframe
# by Pandas.
prediction = loaded_learner.predict({"X": [X_input]})
prediction

# %%
# Conclusion
# ----------
#
# Thanks to the skrub DataOps and learner, we are assured that all the transformations
# and preprocessing done during model development are exactly the same that are done in
# production.
# It becomes easy and straightforward to deploy.
