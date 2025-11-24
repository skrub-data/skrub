"""
Use case: developing locally and deploying to production
=======================================================
"""

# %%
# As a team of data scientists, we are tasked with a project to predict whether an email
# is potentially malicious (i.e., spam or phishing). We develop and test our models
# locally, either in a Jupyter notebook or within a Python script. Once we are satisfied
# with the model's performance, we move on to deploying it.
#
# In this use case, every time the email provider receives a new email, they want to
# verify whether it is spam before displaying it in the recipient's inbox. To achieve
# this, they plan to integrate a machine learning model within a microservice. This
# microservice will accept an email's data as a JSON payload and return a score between
# 0 and 1, indicating the likelihood that the email is spam.
#
# To avoid rewriting the entire data pipeline when moving from model validation to
# production deployment, which is both error-prone and inefficient, we prefer to load an
# object that encapsulates the same processing pipeline used during model development.
# This is where the :class:`~skrub.SkrubLearner` can help.
#
# Adopting this workflow also has the benefit of forcing us to clearly define the type
# of data that will be available at the input of the microservice. It helps ensure we
# build models that rely only on information accessible at this specific point in the
# product pipeline. For example, since we want to detect spam before the email reaches
# the recipient's inbox, we cannot use features that are only available after the
# recipient opens the email.
#
# Since this example is focused on the pipeline construction itself, we won't look at
# our model performance.

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
# are not contained in a dataframe, but in a JSON. As a result, our training data
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
# Let's start our DataOps plan by indicating what the features and the target
# variables are.
import skrub

X = skrub.X(X)
y = skrub.y(y)

# %%
# The variable X is currently a list of dictionaries, which estimators cannot
# handle directly. Let's convert it to a pandas DataFrame using
# :func:`~skrub.DataOp.skb.apply_func`.
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
# Passing ``fitted=True`` to the :func:`~skrub.DataOp.skb.make_learner`
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
# In our microservice, we receive a payload in JSON format.
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
# Thanks to the skrub DataOps and learner, we ensure that all the transformations
# and preprocessing done during model development are exactly the same as those done in
# production. This makes deployment straightforward and reduces the risk of errors
# when moving from development to production environments.
