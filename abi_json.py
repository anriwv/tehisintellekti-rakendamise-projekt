import pandas as pd
import json


def parse_json_safe(json_str):
    """
    Teisendab JSON-stringi turvaliselt Pythoni objektiks (list või dict).
    Käsitleb tühje väärtusi (NaN, None) ja katkist JSON-it, tagastades vea korral None.
    """
    if pd.isna(json_str) or json_str == '': 
        return None
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None


def extract_eeldusained(json_str):
    """
    Eraldab eeldusainete koodid JSON-struktuurist.
    Tagastab komadega eraldatud unikaalsed ainekoodid tähestikulises järjekorras.
    """
    data = parse_json_safe(json_str)
    if not data:
        return None
    
    codes = [
        item.get('code')
        for item in data
        if isinstance(item, dict) and item.get('code')
    ]
    
    return ", ".join(sorted(set(codes))) if codes else None


def extract_keel(json_str):
    """
    Eraldab õppekeele nimetuse (eesti keeles).
    Tagastab komadega eraldatud unikaalsed keeled.
    """
    data = parse_json_safe(json_str)
    if not data:
        return None
    
    languages = [
        item.get('et')
        for item in data
        if isinstance(item, dict) and item.get('et')
    ]
    
    return ", ".join(sorted(set(languages))) if languages else None


def extract_oppeaste(json_str):
    """
    Eraldab õppeastme nimetuse (eesti keeles).
    Tagastab komadega eraldatud unikaalsed õppeastmed.
    """
    data = parse_json_safe(json_str)
    if not data:
        return None
    
    levels = [
        item.get('et')
        for item in data
        if isinstance(item, dict) and item.get('et')
    ]
    
    return ", ".join(sorted(set(levels))) if levels else None
