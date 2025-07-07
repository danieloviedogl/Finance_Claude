import numpy as np
import pandas as pd

def to_serializable(value: any) -> any:
    """Converts numpy types to standard Python types for JSON compatibility."""
    if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
        return int(value)
    elif isinstance(value, (np.float64, np.float32, np.float16)):
        if pd.isna(value):
            return None
        elif np.isinf(value):
            return None
        return float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, np.ndarray):
        return to_serializable(value.tolist())
    elif isinstance(value, (pd.Timestamp, pd.Period)):
        return value.isoformat()
    elif isinstance(value, list):
        return [to_serializable(item) for item in value]
    elif isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    return value
