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
# I am developing and testing my models locally, whether in a notebook or in a script.
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


n_samples = 1000

X_dict = [
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

# generate array of 1 and 0
y_np = np.random.binomial(n=1, p=0.9, size=[n_samples])

# %%
import skrub

X = skrub.X(X_dict)
y = skrub.y(y_np)

# %%
import pandas as pd

# %%
vectorizer = skrub.TableVectorizer()
# %%i
df = X.skb.apply_func(pd.DataFrame)
full_data_ops = df.skb.apply(vectorizer, y=y)

# %%
# is that how we are supposed to do it?
# it's how I found it in https://skrub-data.org/dev/reference/generated/skrub.Expr.skb.apply.html#skrub.Expr.skb.apply,
# but it seems very heavy to me.
# Why do I have to regive X, it's supposed to have been defined before?
# Apparently not, because it's buggy.
# full_data_ops.skb.get_pipeline().fit({"X": X_dict})

# %%
# Other test
trained_pipeline = full_data_ops.skb.get_pipeline().fit({"X": X_dict, "y": y_np})
# %%
import pickle

saved_model = pickle.dumps(trained_pipeline)

# %%
# in my microservice:
X_input = {
    "id": generate_id(),
    "sender": generate_email(),
    "title": generate_text(max_str_length=10, min_str_length=2),
    "content": generate_text(max_str_length=100, min_str_length=10),
    "date": generate_datetime(),
    "cc_emails": [generate_email() for _ in range(random.randint(0, 5))],
}

loaded_model = pickle.loads(saved_model)
# here I'm quite puzzled, because none of two options work.
# while it's what is described in
# https://skrub-data.org/dev/auto_examples/expressions/10_expressions_intro.html
loaded_model.predict({"X": X_input})
loaded_model.skb.predict({"X": X_input})
# %%
