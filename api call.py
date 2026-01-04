import pandas
import requests

url = "https://api.weatherbit.io/v2.0/history/subhourly?"+\
        "lat=42.3876&lon=-71.0995&start_date=2024-01-01&end_date=2024-01-31&"+\
        "tz=local&key=3c3de281bc4542d69ee808dae914a42c"

response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Convert the JSON response to a Python dictionary/list
    data = response.json()
else:
    print(f"Failed to fetch data: Status code {response.status_code}")
    data = None

df = pandas.json_normalize(data, 'data',["city_id","city_name","country_code","lat","lon","state_code","station_id","timezone"])
print (df)
df.to_csv("sample data.csv")
