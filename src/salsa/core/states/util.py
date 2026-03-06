from pydantic import BaseModel

def vector_dim(obj: BaseModel) -> int:
    dim = 0
    for name, field in obj.model_fields.items():
        value = getattr(obj, name)

        if isinstance(value, (float, int, bool)):
            dim += 1

        elif isinstance(value, list):
            dim += sum(1 for x in value if isinstance(x, (float, int, bool)))

        elif isinstance(value, dict):
            dim += sum(1 for x in value.values() if isinstance(x, (float, int, bool)))

        elif isinstance(value, BaseModel):
            dim += vector_dim(value)

    return dim
