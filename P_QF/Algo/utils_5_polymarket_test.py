import requests

def check_geoblock():
    response = requests.get("https://polymarket.com/api/geoblock")
    return response.json()

# Usage
geo = check_geoblock()

# Print full response to see all fields
print(f"Blocked: {geo['blocked']}")
print(f"IP: {geo['ip']}")
print(f"Country: {geo['country']}")
print(f"Region: {geo['region']}")

if geo["blocked"]:
    print(f"Trading not available in {geo['country']} - {geo['region']}")
else:
    print("Trading available")