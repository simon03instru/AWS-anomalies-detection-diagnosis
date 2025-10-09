from typing import List, Dict, Any, Optional
from datetime import datetime
import requests


def get_weather_param(
    latitude: float,
    longitude: float,
    start_time: str,
    end_time: str,
    parameters: List[str]
) -> Dict[str, Any]:
    """
    Get specified weather parameters for a given location and time range.
    
    Args:
        latitude: The latitude of the location (e.g., 52.52)
        longitude: The longitude of the location (e.g., 13.41)
        start_time: The start time in ISO format (e.g., '2025-07-25T03:00:00Z')
        end_time: The end time in ISO format (e.g., '2025-07-25T03:15:00Z')
        parameters: A list of weather parameters to retrieve 
                   (e.g., ['temperature_2m', 'precipitation', 'relative_humidity_2m'])
    
    Returns:
        A dictionary containing the requested weather parameters and their values for the time range.
        Returns error message if request fails.
    
    Example:
        >>> get_weather_param(
        ...     latitude=52.52,
        ...     longitude=13.41,
        ...     start_time='2025-07-25T03:00:00Z',
        ...     end_time='2025-07-25T03:15:00Z',
        ...     parameters=['temperature_2m', 'precipitation']
        ... )
        {
            'times': ['2025-07-25T03:00', '2025-07-25T03:15'],
            'temperature_2m': [18.5, 18.3],
            'precipitation': [0.0, 0.0]
        }
    """
    # Validate inputs
    if not all([latitude, longitude, start_time, end_time, parameters]):
        return {"error": "Missing required parameters: latitude, longitude, start_time, end_time, or parameters list."}
    
    # Parse start and end times
    try:
        dt_start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        dt_end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        
        start_date_str = dt_start.strftime("%Y-%m-%d")
        end_date_str = dt_end.strftime("%Y-%m-%d")
        
        start_time_str = dt_start.strftime("%Y-%m-%dT%H:%M")
        end_time_str = dt_end.strftime("%Y-%m-%dT%H:%M")
    except ValueError as e:
        return {"error": f"Invalid time format: {str(e)}"}
    
    # Prepare API request
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "minutely_15": ",".join(parameters),
        "timezone": "UTC"
    }
    
    # Send request to Open-Meteo
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return {"error": f"API request failed with status {response.status_code}"}
        
        data = response.json()
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}
    
    # Extract data
    minutely_data = data.get("minutely_15", {})
    all_times = minutely_data.get("time", [])
    
    if not all_times:
        return {"error": "No data returned from API"}
    
    # Filter data for the requested time range
    try:
        start_index = all_times.index(start_time_str)
        end_index = all_times.index(end_time_str)
    except ValueError:
        return {"error": f"Requested time range not found in API response. Available times: {all_times[:5]}..."}
    
    # Build result with time range
    result = {
        "times": all_times[start_index:end_index + 1]
    }
    
    # Extract parameter values for the time range
    for param in parameters:
        values = minutely_data.get(param)
        if values and len(values) > end_index:
            result[param] = values[start_index:end_index + 1]
        else:
            result[param] = None
    
    return result