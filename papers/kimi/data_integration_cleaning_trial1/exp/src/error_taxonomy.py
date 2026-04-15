"""
Error Taxonomy for CESF
Defines 8 error types across 3 dimensions: syntactic, structural, semantic
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional
import re


class ErrorType(ABC):
    """Base class for all error types"""
    
    def __init__(self, name: str, dimension: str):
        self.name = name
        self.dimension = dimension
    
    @abstractmethod
    def generate(self, value: Any, context: Dict, rng: np.random.Generator) -> Any:
        """Generate an error from the original value"""
        pass
    
    def __repr__(self):
        return f"{self.dimension}/{self.name}"


# ==================== SYNTACTIC ERRORS ====================

class TypoError(ErrorType):
    """Keyboard-adjacent typos and character mutations"""
    
    # Keyboard adjacency mapping (QWERTY)
    KEYBOARD_ADJACENT = {
        'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx',
        'e': 'wsdr', 'f': 'dcvgtr', 'g': 'fvbhty', 'h': 'gbnjuy',
        'i': 'ujko', 'j': 'hnmkui', 'k': 'jmlio', 'l': 'kop',
        'm': 'njk', 'n': 'bhjm', 'o': 'iklp', 'p': 'ol',
        'q': 'wa', 'r': 'edft', 's': 'wedxza', 't': 'rfgy',
        'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc',
        'y': 'tghu', 'z': 'asx',
        '0': '19', '1': '02', '2': '13', '3': '24', '4': '35',
        '5': '46', '6': '57', '7': '68', '8': '79', '9': '80'
    }
    
    def __init__(self):
        super().__init__("typo", "syntactic")
    
    def generate(self, value: Any, context: Dict, rng: np.random.Generator) -> Any:
        if not isinstance(value, str) or len(value) == 0:
            return value
        
        value_str = str(value).lower()
        if len(value_str) == 0:
            return value
            
        # Select random position
        pos = rng.integers(0, len(value_str))
        char = value_str[pos]
        
        # Try keyboard-adjacent typo
        if char in self.KEYBOARD_ADJACENT and rng.random() < 0.7:
            adjacent = self.KEYBOARD_ADJACENT[char]
            new_char = adjacent[rng.integers(0, len(adjacent))]
            result = value_str[:pos] + new_char + value_str[pos+1:]
        else:
            # Random character substitution
            all_chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
            new_char = all_chars[rng.integers(0, len(all_chars))]
            result = value_str[:pos] + new_char + value_str[pos+1:]
        
        return result


class FormattingError(ErrorType):
    """Case errors, punctuation errors, format violations"""
    
    def __init__(self):
        super().__init__("formatting", "syntactic")
    
    def generate(self, value: Any, context: Dict, rng: np.random.Generator) -> Any:
        if not isinstance(value, str):
            return value
        
        error_type = rng.choice(['case', 'punctuation', 'spacing'])
        
        if error_type == 'case':
            # Random case changes
            return ''.join(c.upper() if rng.random() < 0.3 else c.lower() 
                          for c in str(value))
        elif error_type == 'punctuation':
            # Add/remove punctuation
            if rng.random() < 0.5 and len(value) > 0:
                # Remove punctuation
                return re.sub(r'[^\w\s]', '', value)
            else:
                # Add random punctuation
                punct = rng.choice(['.', ',', '!', '?', '-', '_'])
                pos = rng.integers(0, len(value) + 1) if len(value) > 0 else 0
                return value[:pos] + punct + value[pos:]
        else:  # spacing
            # Add extra spaces or remove spaces
            if ' ' in value and rng.random() < 0.5:
                return value.replace(' ', '')
            else:
                pos = rng.integers(0, len(value) + 1) if len(value) > 0 else 0
                return value[:pos] + ' ' + value[pos:]


class WhitespaceError(ErrorType):
    """Leading/trailing whitespace, multiple spaces, tab issues"""
    
    def __init__(self):
        super().__init__("whitespace", "syntactic")
    
    def generate(self, value: Any, context: Dict, rng: np.random.Generator) -> Any:
        if not isinstance(value, str):
            value = str(value)
        
        error_type = rng.choice(['leading', 'trailing', 'double', 'tab'])
        
        if error_type == 'leading':
            spaces = ' ' * rng.integers(1, 4)
            return spaces + value
        elif error_type == 'trailing':
            spaces = ' ' * rng.integers(1, 4)
            return value + spaces
        elif error_type == 'double':
            return value.replace(' ', '  ')
        else:  # tab
            pos = rng.integers(0, len(value) + 1) if len(value) > 0 else 0
            return value[:pos] + '\t' + value[pos:]


# ==================== STRUCTURAL ERRORS ====================

class FDViolationError(ErrorType):
    """Functional Dependency violations (e.g., ZIP -> City)"""
    
    def __init__(self):
        super().__init__("fd_violation", "structural")
    
    def generate(self, value: Any, context: Dict, rng: np.random.Generator) -> Any:
        # For FD violations, we need to change a value that breaks an FD
        # Context should contain FD info: {'fd': {'determinant': val}, 'current_dependent': val}
        if 'fd_violation_options' in context:
            options = context['fd_violation_options']
            return rng.choice(options)
        
        # Default: return a different value of same type
        if isinstance(value, str):
            # Add suffix to break FD
            return value + '_err'
        else:
            return value + 1 if isinstance(value, (int, float)) else value


class DCViolationError(ErrorType):
    """Denial Constraint violations"""
    
    def __init__(self):
        super().__init__("dc_violation", "structural")
    
    def generate(self, value: Any, context: Dict, rng: np.random.Generator) -> Any:
        # DC violations require violating a predicate
        # e.g., t1.Price > t2.Price for same flight
        if 'dc_violation_value' in context:
            return context['dc_violation_value']
        
        # Default: negate or modify
        if isinstance(value, (int, float)):
            return -value if value > 0 else abs(value) + 1
        elif isinstance(value, str):
            return 'INVALID_' + str(value)
        return value


class KeyViolationError(ErrorType):
    """Primary key violations (duplicates)"""
    
    def __init__(self):
        super().__init__("key_violation", "structural")
    
    def generate(self, value: Any, context: Dict, rng: np.random.Generator) -> Any:
        # Key violations: duplicate an existing key
        if 'existing_keys' in context:
            existing = context['existing_keys']
            return rng.choice(list(existing))
        
        # Default: append duplicate marker
        return str(value) + '_DUP'


# ==================== SEMANTIC ERRORS ====================

class OutlierError(ErrorType):
    """Statistical outliers using domain statistics"""
    
    def __init__(self):
        super().__init__("outlier", "semantic")
    
    def generate(self, value: Any, context: Dict, rng: np.random.Generator) -> Any:
        if 'column_stats' in context:
            stats = context['column_stats']
            mean = stats.get('mean', 0)
            std = stats.get('std', 1)
            
            # Generate outlier: mean +/- (3-5)*std
            direction = rng.choice([-1, 1])
            multiplier = rng.uniform(3.0, 5.0)
            outlier = mean + direction * multiplier * std
            
            if isinstance(value, int):
                return int(outlier)
            return outlier
        
        # Default: multiply by large factor
        try:
            num_val = float(value)
            return num_val * rng.uniform(10, 100)
        except:
            return str(value) + '_OUTLIER'


class ImplausibleValueError(ErrorType):
    """Domain violations (impossible values)"""
    
    IMPLAUSIBLE_VALUES = {
        'age': [-5, 150, 200, 999],
        'date': ['0000-00-00', '9999-99-99', '2025-25-25'],
        'zip': ['00000', '99999', 'XXXXX'],
        'price': [-1, -100, 9999999],
        'hours': [-1, 168, 999],
        'default': ['INVALID', 'N/A', 'NULL', 'NONE', 'XXX']
    }
    
    def __init__(self):
        super().__init__("implausible", "semantic")
    
    def generate(self, value: Any, context: Dict, rng: np.random.Generator) -> Any:
        # Detect column type from context
        col_name = context.get('column_name', '').lower()
        
        # Match column name to domain
        matched_domain = None
        for domain in self.IMPLAUSIBLE_VALUES:
            if domain in col_name:
                matched_domain = domain
                break
        
        if matched_domain:
            implausible = self.IMPLAUSIBLE_VALUES[matched_domain]
            return rng.choice(implausible)
        
        # Try to determine from value type
        if isinstance(value, (int, float)):
            return rng.choice([-999, -1, 0, 999999])
        else:
            return rng.choice(self.IMPLAUSIBLE_VALUES['default'])


# ==================== ERROR TAXONOMY ====================

class ErrorTaxonomy:
    """Complete error taxonomy with 8 error types"""
    
    ERROR_TYPES = {
        # Syntactic
        'typo': TypoError(),
        'formatting': FormattingError(),
        'whitespace': WhitespaceError(),
        # Structural
        'fd_violation': FDViolationError(),
        'dc_violation': DCViolationError(),
        'key_violation': KeyViolationError(),
        # Semantic
        'outlier': OutlierError(),
        'implausible': ImplausibleValueError()
    }
    
    DIMENSIONS = {
        'syntactic': ['typo', 'formatting', 'whitespace'],
        'structural': ['fd_violation', 'dc_violation', 'key_violation'],
        'semantic': ['outlier', 'implausible']
    }
    
    @classmethod
    def get_error_type(cls, name: str) -> ErrorType:
        return cls.ERROR_TYPES.get(name)
    
    @classmethod
    def get_all_types(cls) -> List[str]:
        return list(cls.ERROR_TYPES.keys())
    
    @classmethod
    def get_types_by_dimension(cls, dimension: str) -> List[str]:
        return cls.DIMENSIONS.get(dimension, [])
    
    @classmethod
    def get_dimension(cls, error_type: str) -> str:
        for dim, types in cls.DIMENSIONS.items():
            if error_type in types:
                return dim
        return "unknown"
