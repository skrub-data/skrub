# Analysis of Selector Function Docstrings in `_selectors.py`

## Overview
This document analyzes the docstrings of all selector functions in `skrub/selectors/_selectors.py` to identify gaps that make them harder for users and AI agents to learn and understand.

**File analyzed:** `skrub/selectors/_selectors.py` (738 lines)

**Selectors reviewed:** 12 functions across 3 categories
- Name-based: `glob()`, `regex()`
- Data-type based: `numeric()`, `integer()`, `float()`, `has_dtype()`, `any_date()`, `categorical()`, `string()`, `object()`, `boolean()`
- Content/property based: `cardinality_below()`, `has_nulls()`

---

## Summary of Findings

### Overall Assessment: **Good but Inconsistent**

**Strengths:**
- All functions have comprehensive docstrings with examples
- Most docstrings include "See Also" sections linking related selectors
- Examples are runnable and demonstrate both success and edge cases
- Parameter descriptions are clear

**Weaknesses:**
- Inconsistent structure and depth across functions
- Missing high-level "Purpose" or "When to use" guidance
- Some selectors lack important context about their behavior
- Limited cross-referencing between related selectors
- No clear guidance on operator combinations with these selectors
- Some examples are verbose without being clarifying
- Missing information about performance implications

---

## Detailed Analysis by Selector

### 1. Name-Based Selectors

#### `glob(pattern)`
**Location:** Lines 29-71

**Current State:**
- ✅ Clear one-liner: "Select columns by name with Unix shell style 'glob' pattern"
- ✅ Parameter documentation with pattern syntax
- ✅ See Also section linking to `regex()`
- ✅ Simple, focused examples

**Issues:**
- ❌ No mention of **when to use glob vs regex**
- ❌ No example showing what *doesn't* match (could clarify edge cases)
- ❌ No mention that this is case-sensitive
- ❌ Missing context: "Use this for simple wildcard patterns; use regex for complex matching"

**Suggested Improvements:**
1. Add note: "Case-sensitive matching" or "Use `glob()` for simple wildcard matching, `regex()` for complex patterns"
2. Add example showing non-matches to clarify behavior
3. Add "When to use" guidance comparing glob vs regex

---

#### `regex(pattern, flags=0)`
**Location:** Lines 78-148

**Current State:**
- ✅ Thorough explanation with link to Python `re` module documentation
- ✅ Shows `re.match()` semantics clearly
- ✅ Good edge case examples (matching at beginning, using `$` for end-of-string)
- ✅ Flag usage explained with 3 equivalent approaches
- ✅ See Also section

**Issues:**
- ⚠️ Docstring is quite long (70 lines) for learning purposes
- ❌ No explicit comparison with `glob()` in the docstring body
- ❌ Missing: "When to use this" — when is regex needed instead of glob?
- ❌ Could add: common regex patterns for column names

**Suggested Improvements:**
1. Add opening paragraph: "Use for complex name patterns; prefer `glob()` for simple wildcards"
2. Add subsection "When to use regex" with examples
3. Consider adding common patterns (e.g., "Select columns with numbers: `regex(r'[0-9]+$')`")

---

### 2. Data-Type Based Selectors

#### `numeric()`
**Location:** Lines 156-215

**Current State:**
- ✅ Clear: "Select columns that have a numeric data type"
- ✅ Explicitly states: "not Boolean columns"
- ✅ Good example showing dtype variety
- ✅ Shows how to combine with `boolean()` to include booleans
- ✅ See Also section

**Issues:**
- ⚠️ Docstring is verbose (60 lines) with large DataFrame repr
- ❌ No mention of **why** booleans are excluded (important design choice!)
- ❌ Missing: How this interacts with `integer()` and `float()`
- ❌ No guidance: "Use this for machine learning when you want to scale numeric features"

**Suggested Improvements:**
1. Add explanation: "Booleans are excluded because they often need different handling than numeric data"
2. Add note on relationship to `integer()`, `float()`: "numeric() ≈ integer() | float()"
3. Add use case example in docstring

---

#### `integer()`
**Location:** Lines 218-276

**Current State:**
- ✅ Clear distinction from other numeric types
- ✅ Explicitly excludes booleans
- ✅ Similar structure to `numeric()`

**Issues:**
- ⚠️ Docstring length matches `numeric()` but less essential information
- ❌ Same issue: doesn't explain *why* booleans excluded
- ❌ Missing: "Use when you specifically need integer features"
- ❌ No mention that `integer() | float() ≈ numeric()` (helps users understand relationships)

**Suggested Improvements:**
1. Add opening note: "Use `integer()` when you need exclusively integer-typed columns (excluding floats and booleans)"
2. Add clarification: Integer + Float ≈ Numeric
3. Shorten DataFrame example (too much vertical space)

---

#### `float()`
**Location:** Lines 279-328

**Current State:**
- ✅ Brief and focused
- ✅ Clear difference from `integer()`
- ✅ See Also section

**Issues:**
- ⚠️ Much shorter than `integer()` and `numeric()` despite same importance
- ❌ No guidance on use cases
- ❌ Missing relationship to `numeric()`

**Suggested Improvements:**
1. Add opening guidance: "Use when you specifically need floating-point columns"
2. Consistency: Either shorten all numeric selectors or expand this one
3. Add note: "float() selects floating-point dtypes (float32, float64, etc.)"

---

#### `has_dtype(*dtypes)`
**Location:** Lines 335-377

**Current State:**
- ✅ Clear explanation of "hands-off approach"
- ✅ Good guidance: "get the dtype from an existing column"
- ✅ Shows multiple dtype example

**Issues:**
- ❌ **Confusing name** — users might confuse with "has this dtype" vs "check if dtype exists"
- ❌ Missing: When would you use this instead of `numeric()`, `string()`, etc.?
- ❌ No guidance: "Use only when dtype is truly non-standard or library-specific"
- ❌ Missing: "This is an advanced selector for edge cases"

**Suggested Improvements:**
1. Add opening paragraph: "Advanced selector for matching specific dtypes not covered by other selectors (e.g., custom dtypes, list columns)"
2. Add use case: "Use when working with specialized dtypes like pandas ListDtype or polars Object"
3. Add warning: "Not recommended for standard types (use `numeric()`, `string()`, etc. instead)"

---

#### `any_date()`
**Location:** Lines 380-415

**Current State:**
- ✅ Clear one-liner
- ✅ Good example with different date types
- ✅ Shows timezone handling

**Issues:**
- ❌ Missing: Difference between Date and Datetime (some libraries distinguish them)
- ❌ No guidance: "Use when you need date/datetime preprocessing"
- ❌ Missing: Behavior across different libraries (pandas datetime64 vs polars Date/Datetime)

**Suggested Improvements:**
1. Add note: "Selects both Date and Datetime columns (including timezone-aware)"
2. Add use case: "Use when you need to apply date-specific transformations"
3. Add library note: "Behavior may differ between pandas and polars"

---

#### `categorical()`
**Location:** Lines 418-454

**Current State:**
- ✅ Clear definition
- ✅ Cross-references `string()`
- ✅ Good example

**Issues:**
- ❌ Very brief (37 lines) — could expand on when to use
- ❌ Missing: Relationship to `string()` — both are text types, why different?
- ❌ No guidance on categorical encoding

**Suggested Improvements:**
1. Add opening: "Selects columns with explicitly categorical dtype (pandas Categorical or polars Enum)"
2. Add note: "Different from `string()` — categorical has a fixed set of categories"
3. Add use case: "Use to identify columns ready for one-hot encoding"

---

#### `string()`
**Location:** Lines 457-521

**Current State:**
- ✅ Excellent Notes section explaining pandas version differences
- ✅ Detailed examples showing object vs string dtypes
- ✅ Shows how to combine with `categorical()`

**Issues:**
- ⚠️ Complex due to pandas version differences (but necessary)
- ❌ Missing: Relationship to `object()` selector — both select some object columns, why?
- ❌ Missing: Use case guidance

**Suggested Improvements:**
1. Add opening use case: "Use to find columns containing text data for encoding or NLP processing"
2. Add comparison with `object()` in docstring
3. Add note about performance: "May be slower than `object()` on mixed-type columns"

---

#### `object()`
**Location:** Lines 524-564

**Current State:**
- ✅ Clear explanation of pandas version differences
- ✅ Distinguishes from `string()`
- ✅ Good example

**Issues:**
- ❌ No guidance: When would you use `object()` over `string()`?
- ❌ Missing: This selector is "lower-level" than `string()` — could be clearer
- ❌ No mention of mixed-type columns

**Suggested Improvements:**
1. Add opening: "Use for columns with the 'object' dtype, which may contain mixed types"
2. Add warning: "Prefer `string()` for text columns; use `object()` only when you specifically need mixed-type columns"
3. Add clarification: "object() is broader and less semantic than string()"

---

#### `boolean()`
**Location:** Lines 567-606

**Current State:**
- ✅ Clear and focused
- ✅ See Also section
- ✅ Good example

**Issues:**
- ❌ Very brief without context
- ❌ Missing: Relationship to other selectors (e.g., why excluded from `numeric()`?)
- ❌ No use case guidance

**Suggested Improvements:**
1. Add note: "Selects boolean-typed columns (bool, bool_, boolean dtypes)"
2. Add guidance: "Use to separate boolean features from numeric ones (different encoding strategies)"
3. Explain exclusion: "Excluded from `numeric()` because boolean features often require different preprocessing"

---

### 3. Content/Property-Based Selectors

#### `cardinality_below(threshold)`
**Location:** Lines 622-676

**Current State:**
- ✅ Clear definition with "strictly below" clarification
- ✅ Important note: "Null values do not count"
- ✅ Good examples with multiple thresholds
- ✅ Shows inversion: `~s.cardinality_below(3)` to get high-cardinality columns

**Issues:**
- ⚠️ Long docstring (54 lines) but well-justified
- ❌ Missing: Performance note — this requires counting unique values (might be slow on large datasets)
- ❌ Missing: Use case — when would you want low-cardinality columns?
- ❌ No guidance: "Use to find categorical/discrete features vs continuous"

**Suggested Improvements:**
1. Add use case: "Use to identify categorical/low-cardinality features for encoding or feature selection"
2. Add performance note: "Requires computing unique value counts; may be slow on large datasets"
3. Add example: "Use `~s.cardinality_below(100)` to select high-cardinality features"

---

#### `has_nulls(proportion=0.0)`
**Location:** Lines 687-738

**Current State:**
- ✅ Clear with default behavior
- ✅ Good parameter explanation
- ✅ Multiple examples showing different proportions
- ✅ See Also section with related tools

**Issues:**
- ⚠️ Docstring could be slightly condensed
- ❌ Missing: Relationship to `DropUninformative` is mentioned in See Also but not explained
- ❌ Missing: Use case — when would you want columns with nulls?
- ❌ Missing: Note about NaN vs None vs null across libraries

**Suggested Improvements:**
1. Add use case: "Use to identify columns needing imputation or to drop columns with excessive missing data"
2. Add clarification: "Null values include NaN, None, NA depending on dataframe library"
3. Add note: "Default `proportion=0.0` selects any column with at least one null value"
4. Add example: "Use `~s.has_nulls(0.5)` to drop columns with >50% missing values"

---

## Cross-Cutting Issues

### Issue 1: Missing "When to Use" Guidance
**Affected selectors:** All

**Problem:** Users don't understand the purpose and ideal use cases for each selector.

**Example:** Someone might use `glob('*_id')` when they should use `cardinality_below(100)` to find ID-like columns.

**Impact on agents:** AI agents need explicit guidance on selector choice.

**Suggested fix:** Add opening guidance paragraph to each docstring:
```python
def selector_name():
    """
    Brief one-liner.

    **Use this when:** [specific use case explanation]

    [Rest of docstring...]
    """
```

---

### Issue 2: Inconsistent Docstring Length and Structure
**Affected selectors:** All

**Problem:** Some docstrings are 60+ lines (numeric, integer, regex) while others are 30 lines (float, categorical, boolean), making some harder to scan and others feel underdeveloped.

**Examples:**
- `numeric()`: 60 lines
- `float()`: 50 lines
- `boolean()`: 40 lines
- `categorical()`: 37 lines

**Impact:** Users might think less-documented selectors are less important.

**Suggested fix:** Standardize structure while respecting content importance. Core selectors (numeric, string, categorical) should be similar length.

---

### Issue 3: Missing Relationships Between Selectors
**Affected selectors:** Data-type based

**Problem:** Users don't understand that:
- `numeric() ≈ integer() | float()`
- `string()` ⊂ `object()` (sometimes)
- `categorical()` vs `string()` (different dtypes for text)
- `object()` is broader and less semantic

**Impact on agents:** Makes it harder for agents to suggest correct selector combinations.

**Suggested fix:** Add "Relationships" subsections to docstrings or cross-reference more explicitly.

---

### Issue 4: No Operator Combination Guidance
**Affected selectors:** All

**Problem:** Docstrings show examples with single selectors but rarely show combinations, even when combinations are the main use case.

**Examples:**
- `s.numeric() | s.categorical()` to select all "feature-like" columns
- `s.string() | s.categorical()` to find text columns for encoding
- `s.cardinality_below(10) & s.numeric()` to find discrete numeric features

**Suggested fix:** Add "Combination examples" section to key selectors showing common multi-selector patterns.

---

### Issue 5: Missing Performance/Behavior Notes
**Affected selectors:**
- `cardinality_below()` — requires computing unique counts
- `has_nulls()` with proportion — requires scanning entire column
- `any_date()` — behavior varies by library

**Problem:** Users might not realize some selectors are expensive on large datasets.

**Suggested fix:** Add "Performance" or "Notes" section where relevant.

---

### Issue 6: Limited Cross-Referencing
**Affected selectors:** All

**Problem:** See Also sections are minimal. Users need to understand which selectors work together.

**Examples:**
- `numeric()` should mention `cardinality_below()` for numeric discretization
- `string()` should mention `filter()` for custom text patterns
- `has_nulls()` should mention `DropUninformative` more prominently

**Suggested fix:** Expand See Also sections with more connections and brief explanations.

---

### Issue 7: No Filter/Filter_Names Cross-Reference
**Affected selectors:** All

**Problem:** Docstrings don't mention that `filter()` and `filter_names()` provide custom alternatives.

**Suggested fix:** Add note at end of each docstring: "For custom criteria not covered here, see `filter()` and `filter_names()`"

---

## Recommended Implementation Priority

### High Priority (Impact & Effort Balance)
1. **Add "Use this when" paragraphs** to all docstrings — helps users understand purpose
2. **Standardize structure** across all docstrings — improves scannability
3. **Add operator combination examples** — shows practical usage patterns
4. **Fix relationships** between related selectors — clarifies mental model

### Medium Priority
5. Add performance notes to expensive selectors
6. Expand See Also sections with brief explanations
7. Add cross-references to `filter()` and `filter_names()`

### Low Priority (But Valuable)
8. Consolidate verbose examples (e.g., reduce DataFrame repr size)
9. Add "Common patterns" subsections
10. Add library-specific behavior notes (pandas vs polars)

---

## Example Structure for Improved Docstring

```python
def selector_name(param):
    """
    One-liner description.

    **Use this when:**
    - Specific use case 1
    - Specific use case 2

    This selector selects columns based on [criterion]. [Additional context].

    Parameters
    ----------
    param : type
        Description.

    See Also
    --------
    related_selector : Brief explanation of relationship

    Notes
    -----
    Any library-specific behavior or performance considerations.

    Examples
    --------
    [Basic example]

    [Common combination example]

    [Edge case or advanced example]
    """
```

---

## Integration with Overall Documentation

These selector docstring improvements should be paired with the user guide improvements recommended in the main documentation plan:

1. **User Guide "Decision Tree"** → Relies on clear docstrings to guide users
2. **"When to Use" Guidance** → Docstrings should provide the same info
3. **Common Patterns Section** → Should reference these docstring examples
4. **Operator Combination Examples** → Docstrings should show same patterns

**Synergy:** Improved docstrings make doctest examples more discoverable, and user guide provides broader context.
