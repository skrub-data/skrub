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

> Add social / mailing list / other direct means of asking simple questions
> 
> Also FAQs

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
- Search for your issue on Internet (especially StackOverflow)
- [Check if a similar or identical issue already exist on the repo](https://github.com/dirty-cat/dirty_cat/issues?q=is%3Aissue)
- If yes:
  - **If the issue is still open** : leave the emote :+1: on the original message, 
    which will let us know there are several users affected by this issue
  - **If the issue has been closed** : 
    - **If it has been closed by a merged pull request** 
      (1) update your dirty_cat version, or 
      (2) the fix has not been released in a version yet.
    - **Otherwise** there might be a `wontfix` label, 
      and / or a reason at the bottom of the conversation.
- If not
  - [File a new issue](https://github.com/dirty-cat/dirty_cat/issues/new)
    (see following section)


#### How do I submit a (good) bug report ?

For your issue to be solved as soon as possible,
explain the problem and include additional details 
to help maintainers reproduce the problem:

- **Use a clear and descriptive title** for the issue to identify the problem
- **Describe the result you expected**
- **Describe the problem in finer details**
- **Include a snippet of code that reproduces the error** ; it's a must, 
  as it allows maintainers to reproduce it in a matter of seconds !
- **Specify versions** of Python, dirty_cat, and other dependencies which might 
  be linked to the issue (e.g., scikit-learn, numpy, pandas, etc.). 
  You can get these versions with ``pip3 freeze`` (``pip freeze`` for Windows)

Of course, some of these bullet points might not apply depending on the kind
of error you're submitting.

### Suggesting enhancements

This section will guide you through submitting a new enhancement for dirty_cat,
whether it is a small fix to a big new feature.

First, you should 
[check if the feature has not already been proposed](https://github.com/dirty-cat/dirty_cat/pulls?q=is%3Apr).

If not, the next thing you should do, before writing any code, 
is to [submit a new issue](https://github.com/dirty-cat/dirty_cat/issues/new) 
proposing the change.

#### How do I submit a (good) enhancement proposal ?

- **Use a clear and descriptive title** for the issue to identify the suggestion
- **Provide a quick explanation of the goal of this enhancement**
- **Provide a step-by-step description of the suggested enhancement** 
  in as many details as possible
- **If it exists elsewhere, link resources**

Of course, some of these bullet points might not apply depending on the kind
of enhancement you're submitting.

##### If maintainers validate the enhancement

You can :
- **Write the code and submit a PR** and let maintainers know, 
  so that they don't start working on the feature for nothing
- **Let maintainers know you won't be able to write the code**, 
  so that they can start working on it

Know however that maintainers are **volunteers**, 
and therefore cannot guarantee how much time it will take 
to implement the enhancement.

Writing the feature yourself is the fastest way of getting it implemented 
in the library, and we'll help you as much as we can in that process !

##### If the maintainers refuse the enhancement

There are specific incentives behind dirty_cat. While most enhancement ideas 
are good, they don't always fit in the context of the library.

If you'd like to implement your idea regardless, we'd be very happy if you create 
a new package that builds on top of dirty_cat !  
In some cases, we might even feature it on the official repository !
