import os
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler


class DatabaseConfig:
    """Database configuration management"""

    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "")

        self.environments = {
            "causal": {
                "name": os.getenv("NEO4J_DATABASE_causal", "causal"),
                "description": "Main causal relationship database",
                "max_connections": int(os.getenv("NEO4J_MAX_CONNECTIONS_causal", "50"))
            },
            "neo4j": {
                "name": os.getenv("NEO4J_DATABASE_NEO4J", "neo4j"),
                "description": "Default knowledge graph for vector search",
                "max_connections": int(os.getenv("NEO4J_MAX_CONNECTIONS_causal_1", "20"))
            },
            "knowledge": {
                "name": os.getenv("NEO4J_DATABASE_KNOWLEDGE", "knowledge"),
                "description": "General medical knowledge database",
                "max_connections": int(os.getenv("NEO4J_MAX_CONNECTIONS_KNOWLEDGE", "50"))
            }
        }

        self.current_database = os.getenv("NEO4J_CURRENT_DATABASE", "causal")

    def get_config(self, database: Optional[str] = None) -> Dict:
        db_name = database or self.current_database
        if db_name not in self.environments:
            raise ValueError(f"Invalid database name: {db_name}")

        env = self.environments[db_name]
        return {
            "uri": self.uri,
            "username": self.username,
            "password": self.password,
            "database": env["name"],
            "max_connections": env["max_connections"]
        }

    def set_database(self, database: str) -> None:
        if database not in self.environments:
            raise ValueError(f"Invalid database. Must be one of: {', '.join(self.environments.keys())}")
        self.current_database = database


class Config:
    _instance = None
    logger_instances = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize configuration"""
        # Load environment variables
        load_dotenv()

        # Set project root first
        self.project_root = self._get_project_root()

        # Create basic paths without logging
        self.paths = {
            "data": self.project_root / "data",
            "cache": self.project_root / "cache",
            "output": self.project_root / "output",
            "logs": self.project_root / "logs",
        }

        # Create necessary directories
        self.paths["logs"].mkdir(parents=True, exist_ok=True)
        self.paths["cache"].mkdir(parents=True, exist_ok=True)

        # Now setup logger after paths are created
        self.logger = self._setup_logger("causal_graphrag")

        # Now we can log directory status
        if not self.paths["data"].exists():
            self.logger.warning(f"Data directory not found at {self.paths['data']}")
        if not self.paths["output"].exists():
            self.logger.warning(f"Output directory not found at {self.paths['output']}")

        # Load remaining configuration
        self._load_config()

        # Log initialization
        self.logger.info("Configuration initialized")
        self.logger.debug(f"Project root: {self.project_root}")
        self.logger.debug(f"Current database: {self.db.current_database}")

    def _get_project_root(self) -> Path:
        """Determine project root directory"""
        if project_root := os.getenv("PROJECT_ROOT"):
            return Path(project_root)
        return Path(__file__).parent.parent

    def _setup_logger(self, name: str) -> logging.Logger:
        """Setup logging configuration for a specific name"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Only add handlers if they haven't been added yet
        if not logger.handlers:
            # Create handlers
            console_handler = logging.StreamHandler()
            file_handler = RotatingFileHandler(
                self.paths["logs"] / "causal_graphrag.log",
                maxBytes=1024 * 1024,  # 1MB
                backupCount=5
            )

            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)

            # Add handlers
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

        return logger

    def _load_config(self) -> None:
        """Load configuration components"""
        # Initialize database configuration
        self.db = DatabaseConfig()

        # OpenAI configuration
        self.openai = {
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0")),
        }

        self.cohere = {
            "api_key": os.getenv("COHERE_API_KEY", ""),
            "model": "embed-english-v3.0",  # Default model
        }

    def get_db_config(self, database: Optional[str] = None) -> Dict:
        return self.db.get_config(database)

    def set_database(self, database: str) -> None:
        self.db.set_database(database)
        self.logger.info(f"Switched to database: {database}")

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance for the given name"""
        if name not in self.logger_instances:
            logger_name = f"causal_graphrag.{name}"
            self.logger_instances[name] = self._setup_logger(logger_name)
        return self.logger_instances[name]


# Global configuration instance
config = Config()