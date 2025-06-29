from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

# User 테이블
class User(Base):
    __tablename__ = "User"
    UserID = Column(String(256), primary_key=True)  # varchar(max)
    PassWord = Column(String)
    UserName = Column(String)

    bookmarks = relationship("BookMark", back_populates="user")
    study_records = relationship("StudyRecord", back_populates="user")

# Word 테이블
class Word(Base):
    __tablename__ = "Word"
    WID = Column(Integer, primary_key=True)
    Word = Column(String)

    animations = relationship("Animation", back_populates="word")
    bookmarks = relationship("BookMark", back_populates="word")
    study_words = relationship("StudyWord", back_populates="word")

# Animation 테이블
class Animation(Base):
    __tablename__ = "Animation"
    AID = Column(Integer, primary_key=True)
    WID = Column(Integer, ForeignKey("Word.WID"))
    AnimePath = Column(String)

    word = relationship("Word", back_populates="animations")

# Study 테이블
class Study(Base):
    __tablename__ = "Study"
    SID = Column(Integer, primary_key=True)
    Study_Course = Column(String)

    study_words = relationship("StudyWord", back_populates="study")
    study_records = relationship("StudyRecord", back_populates="study")

# Study_word 테이블
class StudyWord(Base):
    __tablename__ = "Study_word"
    SID = Column(Integer, ForeignKey("Study.SID"), primary_key=True)
    WID = Column(Integer, ForeignKey("Word.WID"), primary_key=True)
    Index = Column(Integer)

    study = relationship("Study", back_populates="study_words")
    word = relationship("Word", back_populates="study_words")

# Study_records 테이블
class StudyRecord(Base):
    __tablename__ = "Study_records"
    UserID = Column(String(256), ForeignKey("User.UserID"), primary_key=True)
    SID = Column(Integer, ForeignKey("Study.SID"), primary_key=True)
    Study_Date = Column(DateTime, primary_key=True)
    # Complate = Column(Boolean)

    user = relationship("User", back_populates="study_records")
    study = relationship("Study", back_populates="study_records")

# BookMark 테이블
class BookMark(Base):
    __tablename__ = "BookMark"
    BID = Column(Integer, primary_key=True)
    UserID = Column(String(256), ForeignKey("User.UserID"))
    WID = Column(Integer, ForeignKey("Word.WID"))

    user = relationship("User", back_populates="bookmarks")
    word = relationship("Word", back_populates="bookmarks")

# Token 테이블
class Token(Base):
    __tablename__ = "Token"
    TID = Column(Integer, primary_key=True, autoincrement=True)
    UserID = Column(String(256), ForeignKey("User.UserID"))
    Refresh_token = Column(String(1024), unique=True)
    Expires = Column(DateTime)

    user = relationship("User", backref="token")
