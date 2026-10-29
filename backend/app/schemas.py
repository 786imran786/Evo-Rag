from pydantic import BaseModel, Field, HttpUrl
from typing import Optional

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=12)
    adaptive: bool = True

class UrlRequest(BaseModel):
    url: HttpUrl

class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: int = Field(ge=1, le=5)
    comment: Optional[str] = None
