async def get_db_pool():
    """Database pool dependency - to be imported from main app"""
    from main import get_db_pool
    return await get_db_pool()