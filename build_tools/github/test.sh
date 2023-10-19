#!/bin/bash -x

pytest --pyargs skrub --cov=skrub -n auto --doctest-modules


sqlite3 ./.coverage 'with pat(pat) as (select "site-packages/") update file set path=substring(path, instr(path, pat) + length(pat)) from pat;'
