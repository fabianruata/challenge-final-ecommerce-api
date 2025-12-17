from pydantic import BaseModel, Field

class ProductInput(BaseModel):
    codigo: str = Field(..., description="Código único del producto")
    imagen: str
    descripcion: str
    caracteristicas: str
    precio_venta: float


class CustomerQuestion(BaseModel):
    telefono: str
    nombre_apellido: str
    pregunta: str
