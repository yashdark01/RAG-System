"""
Base repository with common CRUD operations.
"""

from typing import TypeVar, Generic, Type, Optional, List, Dict, Any
import uuid
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func
from sqlalchemy.sql import Select

from app.database.models import Base
from app.logging.logger import get_logger

logger = get_logger(__name__)

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """
    Base repository with generic CRUD operations.
    
    Usage:
        class UserRepository(BaseRepository[User]):
            def __init__(self, session: AsyncSession):
                super().__init__(User, session)
    """
    
    def __init__(self, model: Type[ModelType], session: AsyncSession):
        """
        Initialize repository.
        
        Args:
            model: SQLAlchemy model class
            session: Database session
        """
        self.model = model
        self.session = session
    
    async def create(self, **kwargs) -> ModelType:
        """
        Create a new record.
        
        Args:
            **kwargs: Model fields
            
        Returns:
            Created model instance
        """
        instance = self.model(**kwargs)
        self.session.add(instance)
        await self.session.flush()
        await self.session.refresh(instance)
        return instance
    
    async def get_by_id(self, id: uuid.UUID) -> Optional[ModelType]:
        """
        Get record by ID.
        
        Args:
            id: Record ID
            
        Returns:
            Model instance or None
        """
        query = select(self.model).where(self.model.id == id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: Optional[str] = None,
        order_desc: bool = True
    ) -> List[ModelType]:
        """
        Get all records with pagination.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records
            order_by: Column to order by
            order_desc: Order descending if True
            
        Returns:
            List of model instances
        """
        query = select(self.model)
        
        if order_by:
            column = getattr(self.model, order_by, None)
            if column is not None:
                if order_desc:
                    query = query.order_by(column.desc())
                else:
                    query = query.order_by(column.asc())
        
        query = query.offset(skip).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_by_field(
        self,
        field: str,
        value: Any,
        first: bool = False
    ) -> Optional[ModelType] | List[ModelType]:
        """
        Get records by field value.
        
        Args:
            field: Column name
            value: Field value
            first: Return only first result if True
            
        Returns:
            Model instance(s)
        """
        column = getattr(self.model, field)
        query = select(self.model).where(column == value)
        
        result = await self.session.execute(query)
        
        if first:
            return result.scalar_one_or_none()
        
        return list(result.scalars().all())
    
    async def update(
        self,
        id: uuid.UUID,
        **kwargs
    ) -> Optional[ModelType]:
        """
        Update a record.
        
        Args:
            id: Record ID
            **kwargs: Fields to update
            
        Returns:
            Updated model instance or None
        """
        # Add updated_at if model has this field
        if hasattr(self.model, 'updated_at'):
            kwargs['updated_at'] = datetime.utcnow()
        
        query = (
            update(self.model)
            .where(self.model.id == id)
            .values(**kwargs)
            .returning(self.model)
        )
        
        result = await self.session.execute(query)
        await self.session.flush()
        
        updated = result.scalar_one_or_none()
        return updated
    
    async def delete(self, id: uuid.UUID) -> bool:
        """
        Delete a record.
        
        Args:
            id: Record ID
            
        Returns:
            True if deleted
        """
        query = delete(self.model).where(self.model.id == id)
        result = await self.session.execute(query)
        await self.session.flush()
        
        return result.rowcount > 0
    
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count records.
        
        Args:
            filters: Optional field filters
            
        Returns:
            Record count
        """
        query = select(func.count()).select_from(self.model)
        
        if filters:
            for field, value in filters.items():
                column = getattr(self.model, field)
                query = query.where(column == value)
        
        result = await self.session.execute(query)
        return result.scalar_one()
    
    async def exists(self, id: uuid.UUID) -> bool:
        """
        Check if record exists.
        
        Args:
            id: Record ID
            
        Returns:
            True if exists
        """
        query = select(self.model.id).where(self.model.id == id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none() is not None
    
    async def bulk_create(self, items: List[Dict[str, Any]]) -> List[ModelType]:
        """
        Create multiple records.
        
        Args:
            items: List of field dictionaries
            
        Returns:
            List of created model instances
        """
        instances = [self.model(**item) for item in items]
        self.session.add_all(instances)
        await self.session.flush()
        
        for instance in instances:
            await self.session.refresh(instance)
        
        return instances
    
    async def bulk_update(
        self,
        ids: List[uuid.UUID],
        values: Dict[str, Any]
    ) -> int:
        """
        Update multiple records.
        
        Args:
            ids: List of record IDs
            values: Fields to update
            
        Returns:
            Number of updated records
        """
        if hasattr(self.model, 'updated_at'):
            values['updated_at'] = datetime.utcnow()
        
        query = (
            update(self.model)
            .where(self.model.id.in_(ids))
            .values(**values)
        )
        
        result = await self.session.execute(query)
        await self.session.flush()
        
        return result.rowcount
    
    async def bulk_delete(self, ids: List[uuid.UUID]) -> int:
        """
        Delete multiple records.
        
        Args:
            ids: List of record IDs
            
        Returns:
            Number of deleted records
        """
        query = delete(self.model).where(self.model.id.in_(ids))
        result = await self.session.execute(query)
        await self.session.flush()
        
        return result.rowcount
    
    async def search(
        self,
        search_fields: List[str],
        search_term: str,
        limit: int = 10
    ) -> List[ModelType]:
        """
        Search records by text fields.
        
        Args:
            search_fields: List of column names to search
            search_term: Search term
            limit: Maximum results
            
        Returns:
            List of matching model instances
        """
        conditions = []
        
        for field in search_fields:
            column = getattr(self.model, field)
            conditions.append(column.ilike(f"%{search_term}%"))
        
        from sqlalchemy import or_
        query = (
            select(self.model)
            .where(or_(*conditions))
            .limit(limit)
        )
        
        result = await self.session.execute(query)
        return list(result.scalars().all())