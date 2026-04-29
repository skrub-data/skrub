# {{ summary.title if summary.title else "DataFrame Report" }}

**Module:** {{ summary.dataframe_module }}
**Shape:** {{ summary.n_rows }} rows × {{ summary.n_columns }} columns

---

## Columns

{% for col in summary.columns %}
### `{{ col.name }}`

| Property | Value |
|---|---|
| Position | {{ col.position }} |
| Type | {{ col.dtype }} |
| Unique values | {{ col.n_unique }} ({{ "%.1f" | format(col.unique_proportion * 100) }}%) |
| Nulls | {{ col.null_count }} ({{ "%.2f" | format(col.null_proportion * 100) }}%) — {{ col.nulls_level }} |
| High cardinality | {{ col.is_high_cardinality }} |
{% if col.value_is_constant %}| Constant value | true |
{% endif %}
{% if col.value_counts %}

**Most frequent values:**

| Value | Count |
|---|---|
{% for value, count in col.value_counts %}| {{ value }} | {{ count }} |
{% endfor %}
{% endif %}
{% if col.mean is defined and col.mean is not none %}

**Numeric statistics:**

| Statistic | Value |
|---|---|
| Mean | {{ "%.4f" | format(col.mean) }} |
| Std dev | {{ "%.4f" | format(col.standard_deviation) }} |
| IQR | {{ col.inter_quartile_range }} |
| Min | {{ col.quantiles[0.0] }} |
| 25th pct | {{ col.quantiles[0.25] }} |
| Median | {{ col.quantiles[0.5] }} |
| 75th pct | {{ col.quantiles[0.75] }} |
| Max | {{ col.quantiles[1.0] }} |
{% endif %}
{% endfor %}

---

## Associations (Cramér's V)

{% if summary.top_associations %}
| Left column | Right column | Cramér's V |
|---|---|---|
{% for assoc in summary.top_associations %}| {{ assoc.left_column_name }} | {{ assoc.right_column_name }} | {{ "%.4f" | format(assoc.cramer_v) }} |
{% endfor %}
{% else %}
Associations were not computed.
{% endif %}
