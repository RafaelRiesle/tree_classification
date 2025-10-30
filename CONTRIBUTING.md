# Contributing to Tree Classification AWP2

## Code Style and Naming Conventions

We follow [PEP 8](https://peps.python.org/pep-0008/) Python style guide with some specific project conventions:

| Element                    | Convention                  | Examples                         |
|----------------------------|-----------------------------|---------------------------------|
| Folders (Packages/Modules) | `lowercase_with_underscores` | `data_loader`, `models`, `utils` |
| Python files             | `lowercase_with_underscores.py` | `preprocessing.py`, `math_utils.py` |
| Classes                 | `CamelCase`                 | `TreeClassifier`, `DatasetLoader` |
| Functions & variables    | `lowercase_with_underscores` | `load_data()`, `split_dataset()`  |

### Additional guidelines:

- Each class should preferably be placed in its own file if it represents a standalone concept.
- Group related helper functions logically (e.g., `utils.py`, `metrics.py`).
- Avoid generic or unclear names such as `stuff.py`, `helpers.py`, or `MyClass.py`.
- Keep code readable and well-documented.

---
