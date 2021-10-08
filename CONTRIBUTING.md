# Contributing to dirty_cat

First off, thanks for taking the time to contribute !

The following is a set of guidelines for contributing to 
[dirty_cat](https://github.com/dirty-cat/dirty_cat).

<hr />

## Table of contents

[I don't want to read the whole thing I just have a question !](#i-dont-want-to-read-the-whole-thing-i-just-have-a-question-)

[What should I know before I get started ?](#what-should-i-know-before-i-get-started-)

[How can I contribute ?](#how-can-i-contribute-)  
- [Reporting bugs](#reporting-bugs)  
- [Suggesting enhancements](#suggesting-enhancements)

<hr />

## I don't want to read the whole thing I just have a question !

We use GitHub Discussions for general chat and Q&As.
[Check it out !](https://github.com/dirty-cat/dirty_cat/discussions)

## What should I know before I get started ?

If you want to truly understand what are the incentives behind dirty_cat,
and if scientific literature doesn't scare you, it is advised to read
the two papers 
[Similarity encoding for learning with dirty categorical variables](https://hal.inria.fr/hal-01806175) 
and 
[Encoding high-cardinality string categorical variables](https://hal.inria.fr/hal-02171256v4).

## How can I contribute ?

### Reporting bugs

Even if we unit-test our code, using the library is the best way to discover
new bugs and limitations.  

If you stumble upon one, please
- [Check if a similar or identical issue already exists](https://github.com/dirty-cat/dirty_cat/issues?q=is%3Aissue)
- If yes:
  - **The issue is still open** : leave the emote :+1: on the original message, 
    which will let us know there are several users affected by this issue
  - **The issue has been closed** : 
    - **It has been closed by a merged pull request** 
      (1) update your dirty_cat version, or 
      (2) the fix has not been released in a version yet
    - **Otherwise** there might be a `wontfix` label, 
      and / or a reason at the bottom of the conversation
- If not
  - [File a new issue](https://github.com/dirty-cat/dirty_cat/issues/new)
    (see following section)


#### How do I submit a (good) bug report ?

To solve your issue as soon as possible, explain the problem and include 
additional details to help maintainers easily reproduce the problem:

- **Use a clear and descriptive title** which identifies the problem
- **Describe the result you expected**
- **Add additional details to your description problem** such as situations 
  where the bug should have appeared but didn't
- **Include a snippet of code that reproduces the error**, 
  as it allows maintainers to reproduce it in a matter of seconds !
- **Specify versions** of Python, dirty_cat, and other dependencies which might 
  be linked to the issue (e.g., scikit-learn, numpy, pandas, etc.). 
  You can get these versions with ``pip3 freeze`` (``pip freeze`` for Windows)

Of course, some of these bullet points might not apply depending on the kind
of error you're submitting.

### Suggesting enhancements

This section will guide you through submitting a new enhancement for dirty_cat,
whether it is a small fix or a new feature.

First, you should 
[check if the feature has not already been proposed or implemented](https://github.com/dirty-cat/dirty_cat/pulls?q=is%3Apr).

If not, the next thing you should do, before writing any code, 
is to [submit a new issue](https://github.com/dirty-cat/dirty_cat/issues/new) 
proposing the change.

#### How do I submit a (good) enhancement proposal ?

- **Use a clear and descriptive title**
- **Provide a quick explanation of the goal of this enhancement**
- **Provide a step-by-step description of the suggested enhancement** 
  with as many details as possible
- **If it exists elsewhere, link resources**

Of course, some of these bullet points might not apply depending on the kind
of enhancement you're submitting.

##### If the enhancement is validated

Let maintainers know whether :
- **You will write the code and submit a PR**. Writing the feature yourself 
  is the fastest way to getting it implemented in the library, 
  and we'll help in that process if guidance is needed !
- **You won't be able to write the code**, in which case a developer interested 
  in the feature can start working on it. Note however that maintainers are 
  **volunteers**, and therefore cannot guarantee how much time it will take 
  to implement the change.

##### If the enhancement is refused

There are specific incentives behind dirty_cat. While most enhancement ideas 
are good, they don't always fit in the context of the library.

If you'd like to implement your idea regardless, we'd be very glad if you create 
a new package that builds on top of dirty_cat !  
In some cases, we might even feature it on the official repository !
