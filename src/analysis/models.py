from pydantic import BaseModel, field_validator
from datetime import date

class BaseArticle(BaseModel):
    date: date
    headline: str | None = ""
    article: str | None = ""
    link: str | None = ""

    @field_validator("headline", "article", "link", mode="before")
    @classmethod
    def none_to_empty(cls, v): return v or ""

class Article(BaseArticle):
    short_headline: str | None = ""
    short_text: str | None = ""

class Mail(BaseArticle): pass
class Paper(BaseArticle): pass
