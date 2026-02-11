# TODO: Fix Failing Doctests in skrub

## Step 1: Fix Docstrings (Complete)

- [x] Fix SkrubLearner docstring in skrub/_data_ops/_estimator.py
- [x] Fix ParamSearch docstring in skrub/_data_ops/_estimator.py
- [x] Fix OptunaParamSearch docstring in skrub/_data_ops/_optuna.py
- [x] Fix Drop class docstring in skrub/_select_cols.py

## Step 2: Update DOCSTRING_TEMP_IGNORE_SET (Complete)

- [x] Remove fixed entries from DOCSTRING_TEMP_IGNORE_SET in skrub/tests/test_docstrings.py

## Step 3: Verify Fixes (Complete)

- [x] Run the doctest suite to ensure fixes pass

## Summary of Changes

### Files Modified:

1. **skrub/_data_ops/_estimator.py**
   - Added `Parameters` section to `SkrubLearner` class docstring
   - Added `Parameters` section to `ParamSearch` class docstring

2. **skrub/_data_ops/_optuna.py**
   - Added comprehensive `Parameters` section to `OptunaParamSearch` class docstring

3. **skrub/_select_cols.py**
   - Added class docstring to `Drop` class
   - Added docstrings to `fit_transform` and `transform` methods

4. **skrub/tests/test_docstrings.py**
   - Removed `"skrub._data_ops._estimator"` from DOCSTRING_TEMP_IGNORE_SET
   - Removed `"skrub._data_ops._optuna"` from DOCSTRING_TEMP_IGNORE_SET
   - Removed `"skrub._select_cols.Drop"` from DOCSTRING_TEMP_IGNORE_SET

All docstring validations now pass for the fixed classes.
