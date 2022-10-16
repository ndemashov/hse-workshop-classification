import pandas as pd

def add_early_wakeup(df: pd.DataFrame) -> pd.DataFrame:
    def _get__wakeup_time(time: str) -> int:
        return int(time.split(':')[0])
    df['Раннее пробуждение'] = df['Время пробуждения'].apply(lambda x: 1 if _get__wakeup_time(x) < 7 else 0)
    return df