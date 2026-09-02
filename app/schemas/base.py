"""
Base schema classes and shared configurations.
"""

from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    """
    Base schema with common configuration for all schemas.

    - populate_by_name: allow both alias and field name
    - str_strip_whitespace: auto-strip whitespace from strings (so a
      whitespace-only question fails min_length validation)
    - use_enum_values: serialize enums by value
    """

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        use_enum_values=True,
    )


class MessageResponse(BaseModel):
    """Simple message response for operations without data."""

    success: bool = True
    message: str
