# 🎲 SDG-PGMs: Probabilistic Graphical Models for Synthetic Data Generation

## Overview

SDG-PGMs is a Python framework for building **Probabilistic Graphical Models (PGMs)** that generate synthetic data with realistic, statistically-grounded relationships between attributes. It extends [pgmpy](https://pgmpy.org/) with support for cascaded Bayesian networks, arbitrary post-processing between stages, and integration with [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner), an open-source [compound-AI](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/) framework for synthetic data generation that leverages multiple models (including non-LLM models like PGMs), validators, and tools together.

PGMs excel at capturing the kinds of complex conditional dependencies that naive random sampling or LLM-only generation cannot: how age correlates with occupation, how geography influences education, how department affects seniority and compensation. Any domain where structured relationships between entities matter -- people, products, transactions, devices, or proprietary business objects -- can be modeled as a PGM. All that is needed is probability distributions over the attributes and their relationships.

When PGMs are built from **aggregate statistical distributions** (counts, crosstabs, census tables) rather than individual records, the generated data is sampled from learned distributions rather than copied from source rows. This makes PGMs a natural fit for generating realistic datasets that mirror sensitive or proprietary distributions while reducing exposure of the underlying data -- though formal privacy guarantees (e.g., differential privacy; see also [Safe Synthesizer](https://github.com/NVIDIA-NeMo/Safe-Synthesizer)) require additional measures beyond what this framework provides.

## 🌍 Showcase: Nemotron-Personas

Together with Data Designer, SDG-PGMs helps power the [Nemotron-Personas HF collection](https://huggingface.co/collections/nvidia/nemotron-personas) -- multilingual, region-specific synthetic persona datasets for sovereign AI development. The [USA dataset](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA) alone contains 6M personas grounded in US Census data, with realistic demographic correlations across age, sex, geography, education, marital status, and 560+ occupations.

Personas are just one application. The same PGM-based approach works for any domain where structured relationships between attributes matter -- from customer profiles to financial transactions to device telemetry.

## ✨ Key Features

- **Cascaded PGMs**: Build multiple interconnected Bayesian networks that generate data in stages
- **Flexible Post-processing**: Apply custom transformations between PGM stages
- **Evidence-based Sampling**: Generate data conditioned on specific requirements
- **Rejection Sampling**: Automatically handle complex constraints with adaptive batch sizing
- **PyTorch Backend**: Leverage GPU acceleration for efficient sampling
- **Data Designer Integration**: Plug any `PGMGenerator` subclass into [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner) as a column generator in a [compound-AI](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/) pipeline

## 📦 Installation

```bash
git clone https://github.com/NVIDIA-NeMo/SDG-PGMs.git
cd SDG-PGMs
uv sync
```

## 🚀 Quick Start

The framework ships as a base layer. To generate data, implement a `PGMGenerator` subclass (see [Writing Your Own Generator](#writing-your-own-pgm-generator) below):

```python
from my_generators import MyPersonGenerator

generator = MyPersonGenerator()

# Generate 1000 synthetic records
samples = generator.generate_samples(1000)

# Generate with constraints (rejection sampling)
samples = generator.generate_samples(
    1000,
    evidence={"region": "West"},
)
```

## 🏗️ Architecture

### Core Components

1. **`PGMGenerator`** (`pgms.generators.base.pgm_generator`)
   - Abstract base class that all generators inherit from
   - Handles PGM construction, sampling, and post-processing
   - Provides rejection sampling for complex constraints

2. **`TarFileMixin`** (`pgms.data_ingest.data_banks`)
   - Mixin for Pydantic data models that backs them by a tar archive of parquet files
   - Supports local paths and remote URIs (S3, GCS, HTTP) via `smart-open`

3. **Cascaded Models**
   - Multiple Bayesian networks can be chained together
   - Each stage can condition on outputs from previous stages
   - Post-processing functions transform data between stages

## ✏️ Writing Your Own PGM Generator

To create a custom PGM generator, extend `PGMGenerator` and implement its abstract methods. The pattern is:

1. Collect statistical distributions (census data, surveys, etc.) as count DataFrames
2. Define a Pydantic data model backed by `TarFileMixin`
3. Implement the four abstract methods
4. Optionally add post-processing steps

### Step 1: Collect and Prepare Your Statistical Data

Each distribution should be a DataFrame with variable columns and a `"count"` column. Here is a simple example to illustrate the pattern:

```python
import pandas as pd

# Age and sex distribution
age_sex_dist = pd.DataFrame({
    "age_group": ["0-17", "0-17", "18-34", "18-34", "35-64", "35-64"],
    "sex":       ["M",    "F",    "M",     "F",     "M",     "F"],
    "count":     [1200,   1150,   2500,    2400,    3200,    3100],
})

# Occupation by education
occupation_dist = pd.DataFrame({
    "education":  ["high_school", "high_school", "bachelors", "bachelors"],
    "occupation": ["technician",  "manager",     "technician","manager"],
    "count":      [500,           100,           200,         800],
})
```

Use pandas categorical types for discrete variables, and reuse the same category dtype
across DataFrames that share a variable -- this keeps CPD construction efficient.

### Step 2: Define Your Data Model

```python
from pydantic import BaseModel, ConfigDict
from pgms.data_ingest.data_banks import TarFileMixin
import pandas as pd

class MyData(BaseModel, TarFileMixin):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    age_sex_dist: pd.DataFrame
    occupation_dist: pd.DataFrame
    # ... more distributions
```

`TarFileMixin` adds `from_tarfile(path, subpath_map)` and `to_tarfile(path, subpath_map)`
class/instance methods so your data model can be serialized to and loaded from a tar archive.

### Step 3: Create Your Generator Class

```python
from pgms.generators.base.pgm_generator import PGMGenerator, Edge

class MyGenerator(PGMGenerator):
    def __init__(self, data_path=None):
        super().__init__(data_path=data_path)
```

### Step 4: Implement Required Methods

#### `get_data()` -- Load your statistical data

```python
def get_data(self, data_path=None):
    if data_path:
        return MyData.from_tarfile(data_path, FILE_MAPPINGS)
    return MyData(
        age_sex_dist=pd.DataFrame(...),
        occupation_dist=pd.DataFrame(...),
    )
```

#### `get_variables()` -- Declare variable domains

```python
def get_variables(self, data: MyData) -> dict[str, list[str]]:
    return {
        "age_group": ["0-17", "18-34", "35-64", "65+"],
        "sex":       list(data.age_sex_dist["sex"].cat.categories),
        "occupation": list(data.occupation_dist["occupation"].cat.categories),
    }
```

#### `get_edges()` -- Define the PGM structure

```python
def get_edges(self, *args, **kwargs) -> list[list[Edge]]:
    return [
        # Stage 1
        [
            Edge(start="age_group", end="sex"),
            Edge(start="age_group", end="occupation"),
        ],
        # Stage 2 -- can reference stage-1 outputs
        [
            Edge(start="occupation", end="income_bracket"),
        ],
    ]
```

#### `get_cpds()` -- Create Conditional Probability Distributions

```python
from pgmpy import config
from pgmpy.factors.discrete import TabularCPD
from pgms.generators.base.utils import tensor_normalize

def get_cpds(self, data: MyData) -> list[dict[str, TabularCPD]]:
    cpd_age = TabularCPD(
        variable="age_group",
        variable_card=4,
        values=tensor_normalize(
            data.age_sex_dist.groupby("age_group")["count"].sum().values[:, None],
            config.get_dtype(),
            config.get_device(),
        ),
        state_names={"age_group": self.variables["age_group"]},
    )
    cpd_occ = self.fill_na_and_gen_cpd(
        counts=data.occupation_dist,
        variable_name="occupation",
        evidence=["education"],
    )
    return [
        {"age_group": cpd_age, "occupation": cpd_occ},  # Stage 1 CPDs
        # Stage 2 CPDs defined similarly ...
    ]
```

### Step 5: Optional Methods

#### `get_latents()` -- Variables used internally but excluded from output

```python
def get_latents(self) -> list[str]:
    return ["age_group"]  # Collapsed into "age" during post-processing
```

#### `get_mappings()` -- Lookup tables for post-processing

```python
def get_mappings(self, data: MyData) -> dict[str, pd.DataFrame]:
    return {"region_lookup": data.region_lookup}
```

#### `get_postprocessing_steps()` -- Deterministic transforms after each stage

```python
from collections.abc import Callable

def get_postprocessing_steps(self) -> list[list[Callable] | None] | None:
    return [
        [self.add_full_name],   # After stage 1
        [self.add_derived_ids], # After stage 2
    ]

def add_full_name(self, samples: pd.DataFrame) -> pd.DataFrame:
    samples["full_name"] = samples["first_name"] + " " + samples["last_name"]
    return samples
```

## 🎨🔌 Data Designer Integration

In a [Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner) [compound-AI](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/) pipeline, PGMs handle structured, statistically-grounded attribute generation -- the kinds of correlated fields that LLMs alone struggle to produce faithfully -- while LLMs handle free-text generation, and validators ensure quality. Data Designer's plugin system ties them all together into a single pipeline.

SDG-PGMs ships with a ready-made plugin that exposes any `PGMGenerator` subclass as a Data Designer column generator. Pass the dotted class path of your generator to the column config:

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner
from data_designer_plugins.pgm_generator_plugin import PGMGeneratorPluginConfig

config_builder = dd.DataDesignerConfigBuilder()
config_builder.add_column(
    PGMGeneratorPluginConfig(
        name="person",
        generator_class="my_generators.MyPersonGenerator",
        evidence={"region": "West"},
    )
)

data_designer = DataDesigner()
results = data_designer.preview(config_builder)
```

## 💡 Best Practices

### Data Preparation

- Use categorical types for discrete variables; share the same dtype across DataFrames for the same variable
- Ensure count data covers all relevant variable combinations
- Add small pseudocounts (e.g. `1e-10`) via `fill_na_and_gen_cpd()` to avoid zero probabilities

### Model Design

- **Start simple**: Begin with a few variables and expand
- **Use latent variables**: Hide intermediate computations not needed in final output
- **Cascade wisely**: Group tightly correlated variables in the same stage

### Performance

- **Batch sampling**: Generate many samples at once rather than one at a time
- **GPU acceleration**: PyTorch tensors are used internally; CUDA is leveraged automatically if available

## 🔍 Troubleshooting

### Rejection Sampling Timeout

Evidence is too restrictive -- the combination may be rare or absent in your distributions. Check that the evidence values exist in your count data, or consider relaxing constraints.

### Zero Probability Errors

Add pseudocounts via the `pseudocount` argument of `fill_na_and_gen_cpd()`, or check for missing variable combinations in your source data.

### Memory Issues with Large Distributions

Use sparse matrix representations, process in batches, or reduce the granularity of high-cardinality variables.

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on setup, development workflow, and submitting pull requests.

## 📄 License

Apache 2.0 -- see [LICENSE](LICENSE) for details.
