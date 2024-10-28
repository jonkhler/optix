# Optix 🔍

A functional lensing library for JAX/Equinox, providing a way to focus on and modify nested values within PyTree structures. Optix generates the same HLO code as direct access, ensuring zero overhead.

## Features

- Type-safe lenses for any JAX PyTree structure
- Zero runtime overhead (generates identical HLO code)
- Intuitive API for accessing and modifying nested values
- Complete static typing support

## Example

```python
from optix import focus
import jax.numpy as jnp

# Create a nested PyTree structure
data = MyStruct(
    x=jnp.array([1.0, 2.0]),
    nested=NestedStruct(y=jnp.array(3.0))
)

# Focus on and modify a nested value
result = focus(data).at(lambda x: x.nested.y).apply(jnp.square)
>>> MyStruct(
>>>     x=Array([1., 2.], dtype=float32),
>>>     nested=NestedStruct(
>>>         y=Array(9., dtype=float32)
>>>     )
>>> )
```

## Installation

```bash
pip install jax-optix
```

## License

MIT License

## Credits

Special thanks to [Patrick Kidger](https://kidger.site/) for providing helpful hints and the [Equinox](https://github.com/patrick-kidger/equinox) library, which this project builds upon.
