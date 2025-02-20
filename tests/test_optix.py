import re

import equinox as eqx
import jax
import jax.numpy as jnp

from optix import focus


class Foo(eqx.Module):
    a: jax.Array
    b: str = eqx.field(static=True)


class Bar(eqx.Module):
    x: jax.Array
    foo: Foo


def test_indexing():
    bar = Bar(
        x=jnp.array([1.0, 2.0, 3.0]), foo=Foo(a=jnp.array([1.0, 2.0, 3.0]), b="hello")
    )
    assert jnp.allclose(
        focus(bar).at(lambda bar: bar.foo.a).get(), jnp.array([1.0, 2.0, 3.0])
    )
    assert jax.tree.map(
        lambda x, y: jnp.allclose(x, y),
        focus(bar).at(lambda bar: bar.foo.a).apply(jnp.cos),
        Bar(
            x=jnp.array([1.0, 2.0, 3.0]),
            foo=Foo(a=jnp.cos(jnp.array([1.0, 2.0, 3.0])), b="hello"),
        ),
    )
    assert focus(bar).at(lambda bar: bar.foo.a).at[1].get() == 2.0
    jax.tree.map(
        lambda x, y: jnp.allclose(x, y),
        focus(bar).at(lambda bar: bar.foo.a).at[1].apply(jnp.cos),
        Bar(
            x=jnp.array([1.0, 2.0, 3.0]),
            foo=Foo(a=jnp.array([1.0, jnp.cos(2.0), 3.0]), b="hello"),
        ),
    )
    assert jnp.allclose(
        focus(bar.foo.a).at(lambda x: x).at[1].apply(jnp.cos),
        jnp.array([1.0, jnp.cos(2.0), 3.0]),
    )


def test_correct_hlo():
    @jax.jit
    def test_verbose(bar: Bar):
        return bar.x, jnp.cos(bar.foo.a)

    @jax.jit
    def test_lens(bar: Bar):
        return focus(bar).at(lambda x: x.foo.a).apply(jnp.cos)

    bar = Bar(
        x=jnp.array([1.0, 2.0, 3.0]), foo=Foo(a=jnp.array([1.0, 2.0, 3.0]), b="hello")
    )

    hlo_verbose = test_verbose.lower(bar).compile().as_text()
    hlo_lens = test_lens.lower(bar).compile().as_text()

    # remove module name
    hlo_verbose = re.sub(r"HloModule .*?, ", "", hlo_verbose)
    hlo_lens = re.sub(r"HloModule .*?, ", "", hlo_lens)

    # remove metadata
    hlo_verbose = re.sub(r"metadata=\{.*\}", "", hlo_verbose)
    hlo_lens = re.sub(r"metadata=\{.*\}", "", hlo_lens)

    assert hlo_verbose == hlo_lens
