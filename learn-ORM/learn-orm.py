from sqlalchemy import (create_engine,
                        MetaData,
                        Table,
                        Column,
                        ForeignKey,
                        Index, )

from sqlalchemy import (ForeignKeyConstraint,
                        PrimaryKeyConstraint,
                        UniqueConstraint,
                        CheckConstraint, )

from sqlalchemy.types import *

from sqlalchemy.ext.declarative import declarative_base




def _declarative_constructor(self, **kwargs):
    """A simple constructor that allows initialization from kwargs.

    Sets attributes on the constructed instance using the names and
    values in ``kwargs``.

    Only keys that are present as
    attributes of the instance's class are allowed. These could be,
    for example, any mapped columns or relationships.
    """
    cls_ = type(self)
    for k in kwargs:
        if not hasattr(cls_, k):
            raise TypeError(
                "%r is an invalid keyword argument for %s" %
                (k, cls_.__name__))
        setattr(self, k, kwargs[k])


_declarative_constructor.__name__ = '__init__'

v = Column(Integer(), primary_key=True)
k = 'colname'

Base = declarative_base(constructor=False)
Base1 = declarative_base()


class a(Base1):
    __tablename__ = 'abc'
    exec(k+'=v')


Base1.metadata.tables
Base1.metadata.clear()
