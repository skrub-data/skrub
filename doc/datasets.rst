.. _datasets:

========
Datasets
========

.. currentmodule:: skrub

skrub allows you to fetch different datasets used in
examples and benchmarks.

It consists of various methods to get generated, embeddings and `real world` data.

Real world datasets
-------------------

From openml.org:

* :func:`datasets.fetch_employee_salaries`
* :func:`datasets.fetch_road_safety`
* :func:`datasets.fetch_medical_charge`
* :func:`datasets.fetch_midwest_survey`
* :func:`datasets.fetch_open_payments`
* :func:`datasets.fetch_traffic_violations`
* :func:`datasets.fetch_drug_directory`

From other sources:

* :func:`datasets.fetch_world_bank_indicator`
* :func:`datasets.fetch_figshare`

Generated dataset
-----------------

* :func:`datasets.make_deduplication_data`

Embeddings
-----------

* :func:`datasets.fetch_ken_embeddings`
* :func:`datasets.fetch_ken_types`
* :func:`datasets.fetch_ken_table_aliases`
