from fastapi import APIRouter
from .endpoints import register_routes

from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, HTTPException

# Create API router
router = APIRouter(
    prefix="/analysis",
    tags=["analysis"],
    responses={404: {"description": "Not found"}},
)