# {{ summary.title if summary.title else "DataFrame Report" }}

**Module:** {{ summary.dataframe_module }}
**Shape:** {{ summary.n_rows }} rows × {{ summary.n_columns }} columns

---

## Columns

| Position | Column | Type | Unique | Nulls | High Card | Constant |
|---|---|---|---|---|---|---|
{% for col in summary.columns %}| {{ col.position }} | {% if col.nulls_level == 'high' %}**`{{ col.name }}`**{% else %}`{{ col.name }}`{% endif %} | {{ col.dtype }} | {{ col.n_unique }} ({{ "%.1f" | format(col.unique_proportion * 100) }}%) | {{ col.null_count }} ({{ "%.2f" | format(col.null_proportion * 100) }}%) | {{ col.is_high_cardinality }} | {{ col.value_is_constant }} |
{% endfor %}

{% for col in summary.columns %}
{% if col.value_counts or (col.mean is defined and col.mean is not none) %}

### `{{ col.name }}` — Details
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
{% endif %}
{% endfor %}

---

## Associations (Cramér's V)

{% if summary.top_associations %}
| Left column | Right column | Cramér's V |
|---|---|---|
{% for assoc in summary.top_associations %}| {{ assoc.left_column_name }} | {{ assoc.right_column_name }} | {% if assoc.cramer_v > 0.9 %}**{{ "%.4f" | format(assoc.cramer_v) }}**{% else %}{{ "%.4f" | format(assoc.cramer_v) }}{% endif %} |
{% endfor %}
{% else %}
Associations were not computed.
{% endif %}
