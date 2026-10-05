from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import User
from app.schemas import ConsentIn, RegisterIn, TokenOut, UserOut
from app.security import create_access_token, get_current_user, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])
log = logging.getLogger(__name__)


def _user_out(u: User) -> UserOut:
    return UserOut(id=u.id, alias=u.alias, email=u.email, is_admin=u.is_admin,
                   consent_to_research=u.consent_to_research, created_at=u.created_at)


@router.post("/register", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
def register(body: RegisterIn, db: Session = Depends(get_db)) -> TokenOut:
    email = body.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")
    first_user = (db.scalar(select(func.count()).select_from(User)) or 0) == 0
    user = User(alias=body.alias.strip(), email=email, password_hash=hash_password(body.password), is_admin=first_user)
    db.add(user)
    db.commit()
    log.info("user registered id=%s admin=%s", user.id, user.is_admin)
    return TokenOut(access_token=create_access_token(user.id))


@router.post("/token", response_model=TokenOut)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)) -> TokenOut:
    user = db.scalar(select(User).where(User.email == form.username.lower()))
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    return TokenOut(access_token=create_access_token(user.id))


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> UserOut:
    return _user_out(user)


@router.patch("/me/consent", response_model=UserOut)
def set_consent(body: ConsentIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> UserOut:
    user.consent_to_research = body.consent_to_research
    db.commit()
    return _user_out(user)


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
def delete_account(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> None:
    db.delete(user)
    db.commit()
    log.info("user deleted id=%s", user.id)
