from phoenix.otel import register

tracer_provider = register(
    project_name="my-camel-agents"
)