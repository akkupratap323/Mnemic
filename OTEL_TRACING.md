# OpenTelemetry Tracing in Mnemic

Mnemic supports OpenTelemetry distributed tracing. Tracing is optional - without a tracer, operations use no-op implementations with zero overhead.

## Installation

```bash
uv add opentelemetry-sdk
```

## Basic Usage

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from mnemic import Mnemic

# Set up OpenTelemetry
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# Get tracer and pass to Mnemic
tracer = trace.get_tracer(__name__)
mnemic = Mnemic(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    tracer=tracer,
    trace_span_prefix="myapp.mnemic"  # Optional, defaults to "mnemic"
)
```

## With Kuzu (In-Memory)

```python
from mnemic.driver.kuzu_driver import KuzuDriver

kuzu_driver = KuzuDriver()
mnemic = Mnemic(graph_driver=kuzu_driver, tracer=tracer)
```

## Example

See `examples/opentelemetry/` for a complete working example with stdout tracing

