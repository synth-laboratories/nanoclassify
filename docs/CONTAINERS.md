# Container references

NanoClassify should use existing task containers as behavioral and protocol
references. Do not copy a container wholesale until its ownership, license, and
intended reuse boundary are understood.

## Confirmed local reference

### Banking77

- Repository: `/Users/joshuapurtell/GitHub/evals`
- Image source: `/Users/joshuapurtell/GitHub/evals/containers/images/banking77`
- Task implementation:
  `/Users/joshuapurtell/GitHub/evals/containers/images/banking77/banking77_classify`
- Domain material: `/Users/joshuapurtell/GitHub/evals/domains/banking77`
- Nonsensitive example:
  `/Users/joshuapurtell/GitHub/evals/containers/nonsensitive/arbitrary/banking77`

Treat the container's current API and scoring behavior as the reference until a
versioned NanoClassify adapter contract is written. Record exact source revision
and dirty-input digests when deriving behavior from a local checkout.

## Reference slots to resolve

The initial filesystem inventory did not locate authoritative local container
directories for:

- ChemProt;
- DDI2013;
- LexGLUE SCOTUS.

Before implementing those adapters, resolve their canonical repositories,
container revisions, dataset licenses, task schemas, and held-out split rules.
Do not infer these from benchmark names alone.

## Common adapter boundary

Each task adapter should eventually expose the same conceptual operations:

```text
info        task identity, revision, labels, and metric contract
sample      immutable example ID plus model-visible input
predict     exact label or typed invalid output
score       per-example correctness and aggregate metrics
receipt     split, model, candidate, usage, and artifact provenance
```

The concrete transport can follow the existing Synth container protocol. The
shared NanoClassify layer should normalize evidence, not erase task-specific
semantics.
