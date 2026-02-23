import pandas
import requests

class api_call:
    def fetch_psm4_data(self):
             
        url = "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-conus-v4-0-0-download.json?api_key=GGkBibjKJxaCaiQ8giyDyYvvd3luSAJM7HEJhvDB"
        payload = "api_key={{API_KEY}}&wkt=POINT(-90.727 43.156)&" \
            "attributes=wind_speed,ghi,dni,dhi,air_temperature,surface_albedo,total_precipitable_water"\
            "&names=2024&utc=false&leap_day=true"\
            "&email=wsbzan@gmail.com&interval=15&utc=false&reason=Academic"
        headers = {
            'content-type': "application/x-www-form-urlencoded",
            'cache-control': "no-cache"
        }
        response = requests.request("POST", url, data=payload, headers=headers)
        return response.text
    
    def fetch_weatherbit_data(self):
        url = "https://api.weatherbit.io/v2.0/history/subhourly?"\
        "lat=43.156&lon=-90.727&start_date=2024-01-01&end_date=2024-01-31&"\
        "tz=local&key=bcd834a25045449f93bae6b3e585e2f6"
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            # Convert the JSON response to a Python dictionary/list
            data = response.json()
        else:
            print(f"Failed to fetch data: Status code {response.status_code}")
            data = None
        return data