# Comprehensive Plan: Improving Selector Documentation for User and Agent Learning

**Status:** Planning Phase (Read-Only)
**Priority:** High
**Scope:** Documentation improvements only (no code changes)
**Goal:** Make selectors easier to learn for both users and AI agents

---

## Executive Summary

This document outlines a comprehensive plan to improve selector documentation across three areas:

1. **Base Class Documentation** (`skrub/selectors/_base.py`)
2. **Selector Function Docstrings** (`skrub/selectors/_selectors.py`)
3. **User Guide Pages** (`doc/modules/multi_column_operations/`)

**Current State:** Documentation is comprehensive but fragmented, inconsistent, and lacks clear learning progressions.

**Desired State:** Cohesive, progressive documentation that helps users understand selectors conceptually and choose the right selector for their task.

---

## Document Organization

This plan includes:

1. **Section 1:** Analysis Summary & Key Issues
2. **Section 2:** Base Class (`Selector`) Documentation Improvements
3. **Section 3:** Function Docstring (`_selectors.py`) Improvements
4. **Section 4:** User Guide (`selectors.rst`, etc.) Improvements
5. **Section 5:** Implementation Timeline & Effort Estimates
6. **Section 6:** Cross-Document Consistency Guidelines

---

## Section 1: Analysis Summary & Key Issues

### Current Documentation Landscape

**Files Analyzed:**
- `skrub/selectors/_base.py` (726 lines) - Core class definitions
- `skrub/selectors/_selectors.py` (738 lines) - 12 selector functions
- `doc/modules/multi_column_operations/selectors.rst` (235 lines) - Main user guide
- `doc/modules/multi_column_operations/type_of_selectors.rst` (99 lines) - Types reference
- `doc/modules/multi_column_operations/advanced_selectors.rst` (126 lines) - Advanced guide

**Total Documentation:** ~1,900 lines across 5 files

### Identified Issues (Categorized by Type)

#### Issue Class 1: Missing Conceptual Guidance
- **Locations:** All files
- **Problem:** Users don't understand *why* selectors exist or when to use which one
- **Impact:** High - affects fundamental understanding
- **Manifestation:**
  - No "When to use" guidance in docstrings
  - No comparison between similar selectors (glob vs regex, string vs categorical)
  - No explanation of design choices (e.g., why booleans excluded from numeric)

#### Issue Class 2: Inconsistent Structure & Depth
- **Locations:** `_selectors.py` docstrings (12 functions)
- **Problem:** 37-70 line docstrings with different structures
- **Impact:** Medium - affects scannability and perceived importance
- **Manifestation:**
  - `numeric()` = 60 lines, `boolean()` = 40 lines
  - Some have "Use this when" sections, others don't
  - Examples vary in complexity and relevance

#### Issue Class 3: Missing Operator Combination Guidance
- **Locations:** `_base.py` class docstring, user guides
- **Problem:** Examples show single selectors; combinations underexplored
- **Impact:** High - combinations are the main power of selectors
- **Manifestation:**
  - Few examples with `|`, `&`, `-`, `^`, `~` operators
  - No guidance on short-circuit evaluation
  - No pattern library for common combinations

#### Issue Class 4: Incomplete Cross-Referencing
- **Locations:** All docstrings
- **Problem:** Relationships between selectors not documented
- **Impact:** Medium - affects discovery and understanding
- **Manifestation:**
  - `numeric() ≈ integer() | float()` not documented
  - `string()` ⊂ `object()` relationship unclear
  - `filter()` and `filter_names()` not mentioned in type-based selectors

#### Issue Class 5: Missing "How Selectors Work" Explanation
- **Locations:** User guides (`selectors.rst`)
- **Problem:** No explanation of selector execution flow
- **Impact:** High - affects mental model
- **Manifestation:**
  - Why `.expand()` exists not explained
  - When selectors are evaluated (at fit time vs transform time) unclear
  - Delayed selection benefit not concrete

#### Issue Class 6: Performance/Behavior Information Scattered
- **Locations:** Various docstrings
- **Problem:** Important context about behavior/performance not consistently documented
- **Impact:** Medium - affects production usage
- **Manifestation:**
  - `cardinality_below()` can be slow on large data (not documented)
  - `has_nulls()` behavior with NaN/None/NA not explained
  - Library-specific differences (pandas vs polars) underexplored

#### Issue Class 7: "Decision Tree" Missing
- **Locations:** Entire documentation
- **Problem:** Users must read all docs to find the right selector
- **Impact:** High - affects time-to-productivity
- **Manifestation:**
  - No "If I want to... then I should use..." guidance
  - No visual hierarchy of selector types
  - No quick reference comparing all selectors

---

## Section 2: Base Class Documentation Improvements

### Current State
**File:** `skrub/selectors/_base.py` (lines 354-360)

```python
class Selector:
    """Generic selector type, that returns set columns when applied.

    This class is not meant to be instantiated manually, ``Selector``
    objects are created by calling one of the selector builders such
    as :meth:`skrub.selectors.all()` or :meth:`skrub.selectors.make_selector()`.
    """
```

**Issues:**
- One paragraph; no explanation of what a Selector *is*
- No mention of key methods or their purposes
- No explanation of the matching logic
- "Defining new selectors" content (lines 1-74) is hidden in module docstring

### Proposed Improvements

#### 2.1 Expand `Selector` Class Docstring

**Location:** `skrub/selectors/_base.py`, line 354-360

**Changes:**
Replace the single paragraph with comprehensive docstring including:

1. **What is a Selector?**
   - Definition: "A reusable rule for selecting columns from a dataframe"
   - Key insight: "Selectors are evaluated lazily, allowing rules to be defined before data is available"

2. **Visual Representation** (text-based)
   ```
   Input dataframe → Selector.expand(df) → List of column names
                  → Selector evaluation on each column
   ```

3. **Key Methods**
   - `expand(df)` - Get list of matching column names
   - `expand_index(df)` - Get indices of matching columns
   - `_matches(col)` - (Internal) Check if column should be selected

4. **How Selectors Work**
   - "For each column in the dataframe, the selector evaluates `_matches(column)`"
   - "Returns `True` to select the column, `False` to exclude it"

5. **When to Use `.expand()` vs Using Selectors Directly**
   - Use `.expand()` for exploratory work or when you just want column names
   - Use selectors in transformers (ApplyToCols, DropCols, SelectCols) for pipelines
   - Use selectors in DataOps for composable data transformations

6. **Operator Support**
   - "Selectors can be combined with operators: `|` (OR), `&` (AND), `-` (except), `^` (XOR), `~` (NOT)"
   - "These operators work with other selectors, strings, and lists of column names"

7. **Design Notes**
   - "No direct instantiation - use selector builder functions"
   - "See `skrub.selectors` module for available selectors"

**Estimated length:** 60-80 lines

#### 2.2 Enhance Method Docstrings

**Location:** `skrub/selectors/_base.py`, lines 362-451

**Changes to `_matches(self, col)`:**
- Add docstring explaining it must be overridden by subclasses
- Explain parameter (a column object) and return value (bool)
- Note about implementation in subclasses

**Changes to `expand(self, df)`:**
- Add note about when users would call this vs framework calling it
- Clarify that it uses `_matches()` internally
- Add relationship to `expand_index()`

**Changes to `expand_index(self, df)`:**
- Explain when you'd use indices vs column names
- Show relationship to `expand()`

#### 2.3 Remove/Relocate "Defining New Selectors" Section

**Location:** `skrub/selectors/_base.py`, lines 1-74

**Decision Needed:** Since scope is "users shouldn't write custom selectors yet"

**Options:**
1. **Remove entirely** - Keep for future when custom selectors are supported
2. **Move to comments** - As internal documentation for developers
3. **Archive separately** - Create separate "Advanced: Custom Selectors" (future-proof)

**Recommendation:** Option 2 (move to comments with forward-looking note)

---

## Section 3: Selector Function Docstring Improvements

### Current State
12 selector functions in `skrub/selectors/_selectors.py`

**Categories:**
- **Name-based (2):** `glob()`, `regex()`
- **Data-type (7):** `numeric()`, `integer()`, `float()`, `has_dtype()`, `any_date()`, `categorical()`, `string()`, `object()`, `boolean()`
- **Content/property (2):** `cardinality_below()`, `has_nulls()`

### Issues by Selector

See `SELECTOR_DOCSTRING_ANALYSIS.md` for detailed issues.

**Summary of issues:**
- ❌ Missing "When to use" guidance (all 12)
- ❌ Inconsistent docstring length/structure (all 12)
- ❌ Missing selector relationships (data-type group especially)
- ❌ No operator combination examples (all 12)
- ❌ Missing performance notes (cardinality_below, has_nulls)
- ❌ Limited cross-referencing (all 12)
- ❌ No filter/filter_names mention (all 12)

### Proposed Improvements

#### 3.1 Standardize Docstring Structure

**Target Structure (all 12 functions):**

```python
def selector_name(param):
    """
    Brief one-liner description.

    **When to use:**
    Use this selector when you need to [specific use case].
    This is useful for [practical example].

    Description paragraph explaining what the selector does and any important behavior.

    Parameters
    ----------
    param : type
        Description of parameter.

    See Also
    --------
    related_selector : Brief explanation (e.g., "Use for X; prefer this for Y")
    filter : For custom criteria not covered by built-in selectors
    filter_names : For custom name-based criteria

    Notes
    -----
    Any important behavior, library-specific differences, or performance implications.

    Examples
    --------
    Basic usage example.

    Combination examples (showing with operators):
    >>> s.selector_name() | s.other_selector()

    Edge cases or advanced usage.
    """
```

**Key additions:**
- "When to use:" section (required for all)
- Operator combination examples (required for most)
- Performance/behavior notes (required for expensive ones)
- Expanded "See Also" with brief explanations
- Consistent examples structure

#### 3.2 Specific Improvements by Selector

**Name-Based Selectors:**

| Selector | Key Changes |
|----------|------------|
| `glob()` | Add "When to use" (simple wildcards), comparison with regex, case-sensitivity note |
| `regex()` | Add "When to use" (complex patterns), comparison with glob, common patterns examples |

**Data-Type Selectors:**

| Selector | Key Changes |
|----------|------------|
| `numeric()` | Add explanation of why booleans excluded, relationship to integer/float |
| `integer()` | Add "When to use", clarify integer() \| float() ≈ numeric() |
| `float()` | Expand to match integer() length, add "When to use" |
| `has_dtype()` | Add warning "advanced/rare use", explain when to use vs other selectors |
| `any_date()` | Add library note (pandas vs polars date handling) |
| `categorical()` | Add "When to use" (categorical columns), relationship to string() |
| `string()` | Add "When to use" (text columns), comparison with object() |
| `object()` | Add warning to prefer string(), clarify broader/less semantic |
| `boolean()` | Expand, add "When to use", explain exclusion from numeric() |

**Content/Property Selectors:**

| Selector | Key Changes |
|----------|------------|
| `cardinality_below()` | Add "When to use" (low-cardinality features), PERFORMANCE NOTE (expensive) |
| `has_nulls()` | Add "When to use" (nulls handling), clarify NaN/None/NA behavior |

#### 3.3 Add "Common Combinations" Subsection

**For key selectors, add examples:**

```python
# numeric()
>>> s.select(df, s.numeric())  # Select all numeric columns for scaling
>>> s.select(df, s.numeric() & ~s.cardinality_below(100))  # Numeric features only (no IDs)

# string()
>>> s.select(df, s.string() | s.categorical())  # All text-like columns for encoding

# cardinality_below()
>>> s.select(df, s.cardinality_below(10) & s.numeric())  # Discrete numeric features
```

#### 3.4 Add Cross-References to filter/filter_names

**Location:** End of each docstring

**Addition:**
```python
Notes
-----
For selection criteria not covered by this selector,
see :func:`filter` (for column-based criteria) or
:func:`filter_names` (for name-based criteria).
```

---

## Section 4: User Guide Improvements

### Current Files & Structure

**File 1: `selectors.rst` (235 lines)**
- Introduction to selectors
- Type of selectors overview
- Combining selectors section
- Visualizing selectors
- Using with transformers

**File 2: `type_of_selectors.rst` (99 lines)**
- Lists selector categories (dtypes, content, names)
- Shows category breakdown

**File 3: `advanced_selectors.rst` (126 lines)**
- filter() and filter_names() explanation
- Custom criteria examples
- Selecting by null values

### Issues in Current User Guide

1. **No Conceptual "How Selectors Work" Section**
   - No explanation of selector execution flow
   - No mental model building
   - Jumps to usage without foundation

2. **No "Decision Tree" or "Choosing a Selector" Section**
   - Users must read all guides to find right selector
   - No quick reference navigation
   - Hard for agents to follow

3. **filter() and filter_names() Under-Positioned**
   - Treated as "advanced" when they're alternatives for custom logic
   - Could be more discoverable
   - Missing comparison section

4. **Limited "Common Patterns" Examples**
   - Docstrings show single selector usage
   - Real-world patterns not documented
   - Combinations not emphasized

5. **No Cross-Links Between Guide Sections**
   - Relationship between base class and guides unclear
   - Function docstrings not referenced
   - No navigation hints

### Proposed Improvements

#### 4.1 Add "How Selectors Work" Section

**Location:** `selectors.rst` (new subsection after intro, before "Type of selectors")

**Content:**
- **What is a Selector?** - Definition and conceptual model
  - "A reusable rule for picking columns from a dataframe"
  - Benefits: expressiveness, lazy evaluation, delayed selection

- **Selector Execution Flow** (text/ascii diagram)
  ```
  Dataframe Input
       ↓
  Selector Rule
       ↓
  For Each Column → Evaluate Rule → Include or Exclude
       ↓
  Column Names
  ```

- **Four Ways to Use Selectors**
  1. Direct selection with `s.select(df, selector)`
  2. In transformers: `ApplyToCols(transformer, cols=selector)`
  3. In DataOps: `skrub.X(df).skb.apply(..., cols=selector)`
  4. Manual expansion: `selector.expand(df)` or `selector.expand_index(df)`

- **Why Delayed Selection Matters**
  - Real example: Can't instantiate ApplyToCols with hardcoded column names when doing train/test split
  - Selectors solve this by providing rules instead of values

**Estimated length:** 50-70 lines

#### 4.2 Add "Choosing a Selector" Decision Guide

**Location:** `type_of_selectors.rst` (expand significantly)

**Content Structure:**
"What do you want to select by?" flowchart in text form:

```
Column Name
├─ Exact match → cols('name1', 'name2')
├─ Wildcard pattern → glob('*_mm', 'col_[0-9]')
├─ Complex pattern → regex('^col_[0-9]+_final$')
└─ Custom rule on name → filter_names(lambda n: len(n) > 10)

Data Type
├─ Numeric (int or float) → numeric()
├─ Integer only → integer()
├─ Float only → float()
├─ String/text → string()
├─ Categorical → categorical()
├─ Date/datetime → any_date()
├─ Boolean → boolean()
└─ Other object → object()

Data Properties
├─ Unique values (cardinality) → cardinality_below(threshold)
├─ Missing values (nulls) → has_nulls(proportion=0.0)
└─ Custom rule on data → filter(lambda col: col.mean() > 50)

Multiple Criteria
├─ Any of these → selector1 | selector2
├─ All of these → selector1 & selector2
├─ All except → selector1 - selector2
└─ NOT → ~selector
```

**Estimated length:** 60-80 lines

#### 4.3 Expand "Combining Selectors" Section

**Location:** `selectors.rst` - enhance existing section

**Additions:**
1. **Operator Explanation Table**
   | Operator | Meaning | Example |
   |----------|---------|---------|
   | `\|` | Union (OR) | `s.numeric() \| s.boolean()` |
   | `&` | Intersection (AND) | `s.numeric() & s.cardinality_below(10)` |
   | `-` | Difference (except) | `s.all() - s.glob('*_id')` |
   | `^` | XOR | `s.numeric() ^ s.integer()` |
   | `~` | Inversion (NOT) | `~s.string()` |

2. **Short-Circuit Evaluation Note**
   - "Some operators short-circuit: `s.all() | expensive_selector()` won't evaluate the expensive one"
   - "This matters for performance with expensive selectors like `cardinality_below()`"

3. **Common Combination Patterns** (subsection)
   ```python
   # Pattern 1: Numeric scaling
   s.numeric() & ~s.boolean()

   # Pattern 2: Text encoding
   s.string() | s.categorical()

   # Pattern 3: Discrete numeric features
   s.cardinality_below(10) & s.numeric()

   # Pattern 4: Drop low-information columns
   s.all() - (s.has_nulls(0.5) | s.cardinality_below(2))
   ```

**Estimated additions:** 40-60 lines

#### 4.4 Create "Common Patterns" Section

**Location:** New subsection in `selectors.rst` (after "Visualizing selectors")

**Content:** Real-world use cases with complete examples

```python
# Pattern 1: Preprocessing Pipeline
# Select all columns and apply type-specific transformers
ApplyToCols(StandardScaler(), cols=s.numeric()).fit_transform(df)
ApplyToCols(OneHotEncoder(), cols=s.categorical()).fit_transform(df)

# Pattern 2: Feature Selection
# Select low-cardinality numeric features (likely discrete)
discrete_numeric = s.cardinality_below(10) & s.numeric()
SelectCols(discrete_numeric).fit_transform(df)

# Pattern 3: Data Quality Checks
# Drop columns with excessive missing data
high_null_cols = s.has_nulls(proportion=0.5)
DropCols(high_null_cols).fit_transform(df)

# Pattern 4: Type-Based Processing
# Identify all columns needing text processing
text_cols = s.string() | s.categorical()
s.select(df, text_cols)

# Pattern 5: ID Column Removal
# Remove columns that look like IDs (low cardinality + high uniqueness)
id_cols = s.glob('*_id') | s.glob('id_*')
DropCols(id_cols).fit_transform(df)
```

**Estimated length:** 40-60 lines

#### 4.5 Reposition filter() and filter_names()

**Current Location:** `advanced_selectors.rst` (positioned as "advanced")

**Change:**
Move discussion to main `selectors.rst` guide, after "Type of selectors" section

**New Title:** "Custom Selection Criteria with filter() and filter_names()"

**Structure:**
1. **When to use filter() vs filter_names()**
   - Side-by-side comparison with examples
   - filter(): "Use when you need to check column values/statistics"
   - filter_names(): "Use when you only need to check column names"

2. **filter() Examples**
   - Column mean > threshold
   - Contains specific value
   - All non-null (custom check)

3. **filter_names() Examples**
   - Starts/ends with pattern
   - Contains substring
   - Length-based selection

4. **Pickling Consideration**
   - Brief note: use importable functions, not lambdas
   - Reference to docstring for details

**Estimated length:** 50-80 lines

#### 4.6 Add "Relationships Between Selectors" Section

**Location:** `type_of_selectors.rst` (new subsection after category list)

**Content:** Explains interconnections:

```
Mathematical Relationships:
- numeric() = integer() | float()
- ~s.string() includes all non-string types
- s.all() - s.cardinality_below(1) removes constant columns

Type Hierarchy:
- object() ⊃ string() (object is broader)
- categorical() ≠ string() (different dtypes, both text)

Usage Hierarchy:
- Use specific (numeric, string) before general (object)
- Use dtype selectors before content selectors
- Use filter() as fallback for custom criteria
```

**Estimated length:** 30-40 lines

---

## Section 5: Implementation Timeline & Effort Estimates

### Phase 1: Foundation (High Priority, Low Risk)
**Duration:** 1-2 days
**Effort:** ~8-10 hours

**Tasks:**
1. Expand `Selector` base class docstring
2. Create "How Selectors Work" user guide section
3. Create "Choosing a Selector" decision tree

**Output:** Users understand selector concepts before learning specific selectors

### Phase 2: Standardization (High Priority, Medium Risk)
**Duration:** 2-3 days
**Effort:** ~12-16 hours

**Tasks:**
1. Standardize all 12 selector docstrings (from `_selectors.py`)
2. Add "When to use:" to each
3. Add operator combination examples to key selectors
4. Expand See Also sections

**Output:** Consistent, learnable docstrings across all selectors

**Risk:** Must maintain doctest compatibility; changes should be additive

### Phase 3: Integration (Medium Priority, Medium Risk)
**Duration:** 1-2 days
**Effort:** ~8-10 hours

**Tasks:**
1. Add "Combining Selectors" enhancements (tables, patterns)
2. Reposition filter/filter_names() to main guide
3. Add "Common Patterns" section to guides
4. Add "Relationships Between Selectors" section

**Output:** Users can see connections between selectors

**Risk:** Potential duplication between docstrings and guides (mitigate with clear cross-references)

### Phase 4: Polish (Low Priority, Low Risk)
**Duration:** 1 day
**Effort:** ~4-6 hours

**Tasks:**
1. Cross-link all documents
2. Review for consistency
3. Add index/TOC to guide pages
4. Final doctest runs

**Output:** Polished, cross-referenced documentation

### Total Effort Estimate
- **Low estimate:** 30-35 hours
- **High estimate:** 40-50 hours
- **Most likely:** 35-40 hours
- **Timeline:** 1-2 weeks (depending on review/feedback cycle)

---

## Section 6: Cross-Document Consistency Guidelines

### Rule 1: Terminology
**Consistent terms across all docs:**
- "Selector" (not "selector function")
- "Matching" (when selector checks if column fits criteria)
- "Expanding" (when selector returns column names)
- "Combining" (when using operators)

### Rule 2: Example Datasets
**Use same example df across:**
- Base class docstring
- Function docstrings
- User guide

Example:
```python
df = pd.DataFrame({
    "height_mm": [297.0, 420.0],
    "width_mm": [210.0, 297.0],
    "kind": ["A4", "A3"],
    "ID": [4, 3],
})
```

**Benefit:** Users see consistent examples everywhere

### Rule 3: Cross-References
**Pattern:**
- Docstrings reference user guide sections via `:ref:` directives
- User guides reference docstrings via `:func:`, `:class:` roles
- Example cross-link format:
  ```rst
  See :ref:`selector_decision_tree` for help choosing between selectors.
  For custom criteria, see :func:`skrub.selectors.filter`.
  ```

### Rule 4: "When to Use" Guidance
**Format (consistent everywhere):**
```
**When to use:**
- Use when you need to [specific criterion]
- This is useful for [practical scenario]
- Choose this over [alternative] when [distinction]
```

### Rule 5: See Also Sections
**Format (docstrings):**
```python
See Also
--------
related_selector : Brief explanation (one line)
    - "Use for simpler patterns"
    - "Try this if X is complex"
filter : For custom criteria beyond built-in selectors
```

### Rule 6: Example Structure
**All examples follow pattern:**
1. Import statement
2. Data setup
3. Selector definition
4. Application
5. Result/output

**Benefit:** Clear, predictable examples

### Rule 7: Library-Specific Notes
**Format:**
```python
Notes
-----
Behavior may differ between pandas and polars:
- Pandas: [behavior]
- Polars: [behavior]
See library documentation for details.
```

### Rule 8: Performance Notes
**Format:**
```python
Notes
-----
Performance Consideration: This selector requires [operation].
On large datasets (>1M rows), this may be slow.
Consider using simpler selectors when possible.
```

---

## Section 7: Recommended Implementation Order

### Optimal Sequence (for clarity + minimal conflicts)

1. **Step 1:** Expand `Selector` base class docstring
   - Foundation for all else
   - No conflicts

2. **Step 2:** Add "How Selectors Work" to user guide
   - Provides context for upcoming docstring changes
   - No conflicts

3. **Step 3:** Add "Choosing a Selector" decision tree
   - Helps users understand structure
   - Supports Step 4

4. **Step 4:** Standardize all 12 selector docstrings
   - Large task; use decision tree as template
   - Could be parallelized

5. **Step 5:** Enhance "Combining Selectors" user guide section
   - Uses docstring examples
   - Should come after Step 4

6. **Step 6:** Add "Common Patterns" section
   - Builds on standardized docstrings
   - Final reference layer

7. **Step 7:** Add "Relationships Between Selectors" section
   - Helps users understand connections
   - Last conceptual piece

8. **Step 8:** Cross-link all documents
   - Final polish
   - Connects everything together

---

## Section 8: Verification & Testing

### Doctest Verification
- Run `pytest --doctest-modules` on `_selectors.py`
- Ensure all examples still pass
- Verify no new syntax errors

### User Guide Verification
- Build docs with Sphinx: `make html`
- Check for broken references (`:ref:`, `:func:`, `:class:`)
- Verify code blocks render correctly

### Cross-Reference Check
- Verify all `:ref:` links are valid
- Verify all `:func:` links exist
- Check for circular references

### Learning Path Verification
1. New user reads "How Selectors Work" → understands concepts
2. User looks at "Choosing a Selector" → finds right selector
3. User reads selector docstring → learns specific usage
4. User looks at "Common Patterns" → applies to real use case

### AI Agent Testing
- Test with multiple model sizes (Claude 3.5 Sonnet, other models)
- Verify selector choice recommendations
- Check that agents understand relationships

---

## Section 9: Future Enhancements (Out of Scope)

These are valuable but beyond current scope:

1. **Interactive Selector Chooser**
   - Web tool to guide selector selection
   - Could be added to skrub website

2. **Video Tutorials**
   - "Choosing a Selector" video
   - "Common Patterns" walkthrough

3. **Jupyter Notebook Tutorial**
   - Executable examples
   - Interactive exploration

4. **Selector Gallery**
   - Real-world examples from real datasets
   - Before/after with different selectors

5. **Performance Benchmarks**
   - Show which selectors are fastest
   - Guidance for large datasets

6. **Custom Selector Guide**
   - Guide for creating custom selectors
   - Currently in scope restriction; future feature

---

## Questions for User Input

Before implementation begins, please provide input on:

1. **Docstring Structure:**
   - Proposed structure is `[One-liner] → [When to use] → [Description] → [Examples] → [Notes]`
   - Acceptable? Any changes?

2. **User Guide Organization:**
   - Should "How Selectors Work" be in `selectors.rst` or separate file?
   - Should "Decision Tree" be in `type_of_selectors.rst` or separate?

3. **Cross-Referencing Depth:**
   - Add cross-links between every related selector?
   - Or keep minimal to avoid documentation bloat?

4. **Example Dataset:**
   - Use current `height_mm`/`width_mm` dataset everywhere?
   - Or vary by selector type for realism?

5. **Implementation Parallelization:**
   - Can multiple people work on different selector docstrings simultaneously?
   - Who will coordinate consistency checks?

6. **Timeline Preference:**
   - Implement all at once (1-2 weeks)?
   - Phased approach (1-2 per week)?
   - Specific deadline?

---

## Related Documents

This plan is accompanied by:
- `SELECTOR_DOCSTRING_ANALYSIS.md` - Detailed analysis of all 12 selectors
- `SELECTOR_DOCUMENTATION_PLAN.md` - This document

---

## Conclusion

This comprehensive plan provides a roadmap for improving selector documentation across three layers:

1. **Base class** - Conceptual foundation
2. **Function docstrings** - Specific guidance for each selector
3. **User guides** - Learning progressions and patterns

By addressing the seven identified issue classes through structured improvements, we will create a cohesive learning experience that helps both users and AI agents understand selectors deeply and choose the right tool for their task.

**Next Step:** User reviews questions in Section 9 and confirms approach before implementation begins.
