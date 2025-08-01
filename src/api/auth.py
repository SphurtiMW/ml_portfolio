"""
Authentication and Authorization Module

This module provides:
- JWT token authentication
- Role-based access control
- User management
- Security utilities
- Token validation and refresh
"""

import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from passlib.context import CryptContext
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from ..core.config import settings
from ..core.logger import get_logger
from ..core.exceptions import AuthenticationError, AuthorizationError

logger = get_logger(__name__)

# Password encryption context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token scheme
security = HTTPBearer()


class User(BaseModel):
    """User model"""
    user_id: str
    username: str
    email: Optional[str] = None
    roles: List[str] = []
    permissions: List[str] = []
    is_active: bool = True
    created_at: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_123",
                "username": "john_doe",
                "email": "john@example.com",
                "roles": ["analyst", "forecaster"],
                "permissions": ["predict", "upload_data"],
                "is_active": True,
                "created_at": "2023-01-01T00:00:00"
            }
        }


class TokenData(BaseModel):
    """Token payload data"""
    sub: str  # subject (user_id)
    exp: datetime  # expiration
    iat: datetime  # issued at
    roles: List[str] = []
    permissions: List[str] = []


class AuthManager:
    """Authentication and authorization manager"""
    
    def __init__(self):
        self.secret_key = settings.api.secret_key
        self.algorithm = settings.api.algorithm
        self.access_token_expire_minutes = settings.api.access_token_expire_minutes
        
        # Mock user database (in production, use real database)
        self._users_db = {
            "demo_user": {
                "user_id": "user_demo",
                "username": "demo_user",
                "email": "demo@example.com",
                "hashed_password": self.get_password_hash("demo_password"),
                "roles": ["analyst", "forecaster"],
                "permissions": ["predict", "upload_data", "train_model"],
                "is_active": True,
                "created_at": datetime.utcnow()
            },
            "admin": {
                "user_id": "user_admin",
                "username": "admin",
                "email": "admin@example.com",
                "hashed_password": self.get_password_hash("admin_password"),
                "roles": ["admin", "analyst", "forecaster"],
                "permissions": ["*"],  # All permissions
                "is_active": True,
                "created_at": datetime.utcnow()
            }
        }
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        return self._users_db.get(username)
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with username and password"""
        user = self.get_user(username)
        if not user:
            logger.warning("Authentication failed: user not found", username=username)
            return None
        
        if not self.verify_password(password, user["hashed_password"]):
            logger.warning("Authentication failed: invalid password", username=username)
            return None
        
        if not user["is_active"]:
            logger.warning("Authentication failed: user inactive", username=username)
            return None
        
        logger.info("User authenticated successfully", username=username, user_id=user["user_id"])
        return user
    
    def create_access_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT access token"""
        expires_delta = timedelta(minutes=self.access_token_expire_minutes)
        expire = datetime.utcnow() + expires_delta
        
        token_data = {
            "sub": user_data["user_id"],
            "exp": expire,
            "iat": datetime.utcnow(),
            "username": user_data["username"],
            "roles": user_data.get("roles", []),
            "permissions": user_data.get("permissions", [])
        }
        
        encoded_jwt = jwt.encode(token_data, self.secret_key, algorithm=self.algorithm)
        
        logger.info("Access token created", 
                   user_id=user_data["user_id"],
                   expires_at=expire.isoformat())
        
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check expiration
            exp = datetime.fromtimestamp(payload.get("exp", 0))
            if datetime.utcnow() > exp:
                logger.warning("Token verification failed: token expired")
                return None
            
            token_data = TokenData(
                sub=payload.get("sub"),
                exp=exp,
                iat=datetime.fromtimestamp(payload.get("iat", 0)),
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", [])
            )
            
            return token_data
        
        except jwt.PyJWTError as e:
            logger.warning("Token verification failed", error=str(e))
            return None
    
    def check_permission(self, user_permissions: List[str], required_permission: str) -> bool:
        """Check if user has required permission"""
        # Admin has all permissions
        if "*" in user_permissions:
            return True
        
        return required_permission in user_permissions
    
    def check_role(self, user_roles: List[str], required_role: str) -> bool:
        """Check if user has required role"""
        return required_role in user_roles


# Global auth manager instance
auth_manager = AuthManager()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Get current authenticated user from JWT token
    
    Args:
        credentials: HTTP Bearer credentials
    
    Returns:
        User information dictionary
    
    Raises:
        HTTPException: If authentication fails
    """
    try:
        token = credentials.credentials
        token_data = auth_manager.verify_token(token)
        
        if token_data is None:
            raise AuthenticationError("Invalid token")
        
        # Get user data (in production, query from database)
        user = next(
            (user for user in auth_manager._users_db.values() 
             if user["user_id"] == token_data.sub),
            None
        )
        
        if user is None:
            raise AuthenticationError("User not found")
        
        if not user["is_active"]:
            raise AuthenticationError("User account is inactive")
        
        # Return user info
        return {
            "user_id": user["user_id"],
            "username": user["username"],
            "email": user.get("email"),
            "roles": user.get("roles", []),
            "permissions": user.get("permissions", [])
        }
        
    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error("Authentication error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_permission(permission: str):
    """
    Decorator to require specific permission
    
    Args:
        permission: Required permission
    
    Returns:
        Dependency function
    """
    def permission_checker(current_user: Dict = Depends(get_current_user)) -> Dict[str, Any]:
        user_permissions = current_user.get("permissions", [])
        
        if not auth_manager.check_permission(user_permissions, permission):
            logger.warning("Permission denied",
                          user_id=current_user.get("user_id"),
                          required_permission=permission,
                          user_permissions=user_permissions)
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        
        return current_user
    
    return permission_checker


def require_role(role: str):
    """
    Decorator to require specific role
    
    Args:
        role: Required role
    
    Returns:
        Dependency function
    """
    def role_checker(current_user: Dict = Depends(get_current_user)) -> Dict[str, Any]:
        user_roles = current_user.get("roles", [])
        
        if not auth_manager.check_role(user_roles, role):
            logger.warning("Role requirement not met",
                          user_id=current_user.get("user_id"),
                          required_role=role,
                          user_roles=user_roles)
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' required"
            )
        
        return current_user
    
    return role_checker


# Optional authentication (allows both authenticated and anonymous users)
async def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))) -> Optional[Dict[str, Any]]:
    """
    Get current user if authenticated, otherwise return None
    
    Args:
        credentials: Optional HTTP Bearer credentials
    
    Returns:
        User information dictionary or None
    """
    if credentials is None:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


# Utility functions for testing and demo
def create_demo_token() -> str:
    """Create a demo token for testing"""
    demo_user = auth_manager._users_db["demo_user"]
    return auth_manager.create_access_token(demo_user)


def create_admin_token() -> str:
    """Create an admin token for testing"""
    admin_user = auth_manager._users_db["admin"]
    return auth_manager.create_access_token(admin_user)


# Export main components
__all__ = [
    "AuthManager",
    "User",
    "TokenData",
    "auth_manager",
    "get_current_user",
    "get_current_user_optional",
    "require_permission",
    "require_role",
    "create_demo_token",
    "create_admin_token"
]