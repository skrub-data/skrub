"""Quick script to check numpydoc validation for specific methods."""
import sys

try:
    from numpydoc.validate import validate
except ImportError:
    print("numpydoc not installed. Please install it first.")
    sys.exit(1)

def check_docstring(import_path):
    """Check a docstring and print errors."""
    print(f"\n{'='*80}")
    print(f"Checking: {import_path}")
    print('='*80)
    
    res = validate(import_path)
    
    print(f"\nDocstring:\n{res['docstring']}\n")
    
    if res['errors']:
        print("Errors found:")
        for code, message in res['errors']:
            print(f"  - {code}: {message}")
    else:
        print("✓ No errors found!")
    
    return res['errors']

if __name__ == "__main__":
    paths = [
        "skrub._table_vectorizer.TableVectorizer.fit",
        "skrub._table_vectorizer.TableVectorizer.set_params",
        "skrub._data_ops._estimator.SkrubLearner.truncated_after",
    ]
    
    all_errors = {}
    for path in paths:
        errors = check_docstring(path)
        if errors:
            all_errors[path] = errors
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    if all_errors:
        print(f"\nFound errors in {len(all_errors)} docstrings:")
        for path, errors in all_errors.items():
            print(f"\n{path}:")
            for code, message in errors:
                print(f"  - {code}: {message}")
    else:
        print("\n✓ All docstrings pass validation!")

