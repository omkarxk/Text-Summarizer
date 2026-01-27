"""
Configuration Settings
All adjustable settings for the application
"""

import os


class Config:
    """Base configuration settings"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')
    DEBUG = False
    
    # Server settings
    HOST = '0.0.0.0'
    PORT = 5001
    
    # Model settings
    MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"
    
    # Summary size options
    # Format: (sentences_to_extract, max_output_tokens, min_output_tokens)
    SUMMARY_SIZES = {
        "very_small": (2, 50, 20),
        "small": (3, 80, 25),
        "medium": (4, 120, 35),
        "large": (5, 180, 50),
        "very_large": (6, 250, 80)
    }
    
    # Context level adjustments (added to sentence count)
    CONTEXT_LEVELS = {
        "simple": -1,
        "balanced": 0,
        "detailed": 1
    }
    
    # Input validation
    MIN_INPUT_LENGTH = 50  # Minimum characters required


class DevelopmentConfig(Config):
    """Development settings"""
    DEBUG = True


class ProductionConfig(Config):
    """Production settings"""
    DEBUG = False


# Easy access to configurations
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
