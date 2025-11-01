"""
Smart Fitness Pod - Machine Learning Module
P2025-26 Project: Real-time exercise classification and repetition counting.
"""

__version__ = "0.1.0"
__author__ = "Erick LI"

from fitness_ai.models.exercise_classifier import ExerciseClassifier
from fitness_ai.models.repetition_counter import RepetitionCounter
from fitness_ai.data.preprocessor import IMUDataPreprocessor

__all__ = [
    "ExerciseClassifier",
    "RepetitionCounter", 
    "IMUDataPreprocessor",
]