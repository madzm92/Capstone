import os
import base64
import hashlib
import requests
import pandas as pd
from dotenv import load_dotenv

# Download nameplate data from MassDOT Go API

load_dotenv('.env')

USERNAME = os.getenv("MASSDOT_USERNAME")
PASSWORD = os.getenv("MASSDOT_PASSWORD")
SECRET_KEY = os.getenv("MASSDOT_SECRET_KEY")


BASE_URL = "https://data-api.massgotime.com"
INVENTORY_ENDPOINTS = [
    "networks",
    "nodes",
    "links",
    "routes",
    "device-sites",
    "signs",
    "solar-controllers",
    "detectors"
]

def get_bearer_token():
    """Authenticate and return a Bearer token."""
    basic_auth_str = f"{USERNAME}:{PASSWORD}"
    basic_auth_encoded = base64.b64encode(basic_auth_str.encode()).decode()

    headers = {
        "Authorization": f"Basic {basic_auth_encoded}",
        "Accept": "application/json"
    }

    response = requests.get(f"{BASE_URL}/auth-token", headers=headers, timeout=5)
    response.raise_for_status()
    token = response.json()["token"]

    hash_input = f"{SECRET_KEY}:{token}"
    sha256_hash = hashlib.sha256(hash_input.encode()).hexdigest()
    bearer_string = f"{USERNAME}:{sha256_hash}"
    bearer_token = base64.b64encode(bearer_string.encode()).decode()

    return f"Bearer {bearer_token}"

def download_routes(token):
    """Download and flatten the /routes endpoint with custom logic."""
    url = f"{BASE_URL}/v1/inventory/routes"
    headers = {
        "Authorization": token,
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()

    org_info = data.get("organization-information", {})
    org_details = org_info.get("organization-contact-details", {})
    update_time = data.get("extended-properties", {}).get("inventory-update-time")

    rows = []
    for item in data.get("routes-inventory-list", []):
        ext = item.get("extended-properties", {})
        row = {
            "network-id": item.get("network-id"),
            "route-id": item.get("route-id"),
            "route-name": item.get("route-name"),
            "route-length": item.get("route-length"),
            "route-link-id-list": ", ".join(item.get("route-link-id-list", [])),
            "route-node-id-list": ", ".join(item.get("route-node-id-list", [])),
            "is-enabled": item.get("is-enabled"),
            "last-update-time": item.get("last-update-time"),
            "associated-sign-id": ext.get("associated-sign-id"),
            "organization-id": org_info.get("organization-id"),
            "organization-name": org_info.get("organization-name"),
            "contact-id": org_details.get("contact-id"),
            "person-name": org_details.get("person-name"),
            "email-address": org_details.get("email-address"),
            "phone-number": org_details.get("phone-number"),
            "inventory-update-time": update_time
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("routes.csv", index=False)
    print(f"Saved {len(df)} records to routes.csv")

def download_device_sites(token):
    """Download and flatten the /device-sites endpoint with corrected structure."""
    url = f"{BASE_URL}/v1/inventory/device-sites"
    headers = {
        "Authorization": token,
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()

    org_info = data.get("organization-information", {})
    org_details = org_info.get("organization-contact-details", {})
    update_time = data.get("extended-properties", {}).get("inventory-update-time")

    rows = []
    for site in data.get("device-site-inventory-list", []):
        devices = site.get("device-list", [])
        row = {
            "device-site-id": site.get("device-site-id"),
            "device-site-name": site.get("device-site-name"),
            "network-id": site.get("network-id"),
            "description": site.get("description"),
            "roadway-name": site.get("roadway-name"),
            "roadway-direction": site.get("roadway-direction"),
            "mile-marker": site.get("mile-marker"),
            "latitude": site.get("geo-location", {}).get("latitude"),
            "longitude": site.get("geo-location", {}).get("longitude"),
            "device-id-list": ", ".join(d.get("device-id") for d in devices),
            "device-type-list": ", ".join(d.get("device-type") for d in devices),
            "last-update-time": site.get("last-update-time"),
            "organization-id": org_info.get("organization-id"),
            "organization-name": org_info.get("organization-name"),
            "contact-id": org_details.get("contact-id"),
            "person-name": org_details.get("person-name"),
            "email-address": org_details.get("email-address"),
            "phone-number": org_details.get("phone-number"),
            "inventory-update-time": update_time
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("device-sites.csv", index=False)
    print(f"Saved {len(df)} records to device-sites.csv")

def download_signs(token):
    """Download and flatten the /signs endpoint into a CSV."""
    url = f"{BASE_URL}/v1/inventory/signs"
    headers = {
        "Authorization": token,
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()

    top_org_info = data.get("organization-information", {})
    top_org_contact = top_org_info.get("organization-contact-details", {})
    inventory_update_time = data.get("extended-properties", {}).get("inventory-update-time")

    rows = []
    for sign in data.get("sign-inventory-list", []):
        header = sign.get("device-inventory-header", {})
        org_info = header.get("organization-information", {})
        contact = org_info.get("organization-contact-details", {})
        location = header.get("device-location", {})
        sign_ext = sign.get("extended-properties", {})
        hybrid = sign_ext.get("hybrid-sign-details", {})

        row = {
            "device-id": header.get("device-id"),
            "device-name": header.get("device-name"),
            "device-description": header.get("device-description"),
            "latitude": location.get("latitude"),
            "longitude": location.get("longitude"),
            "last-update-time": header.get("last-update-time"),
            "device-site-id": header.get("extended-properties", {}).get("device-site-id"),
            "device-type": header.get("extended-properties", {}).get("device-type"),
            "is-enabled": header.get("extended-properties", {}).get("is-enabled"),
            "dms-sign-type": sign.get("dms-sign-type"),
            "signTechnology": sign.get("signTechnology"),
            "signHeightPixels": sign.get("signHeightPixels"),
            "signWidthPixels": sign.get("signWidthPixels"),
            "charHeightPixels": sign.get("charHeightPixels"),
            "charWidthPixels": sign.get("charWidthPixels"),
            "enclosure-type": sign_ext.get("enclosure-type"),
            "enclosure-position": hybrid.get("enclosure-position"),
            "collocated-sign-id-list": ", ".join(hybrid.get("collocated-sign-id-list", [])),
            "static-text-location": hybrid.get("static-text-location"),
            "static-text-multi-string": hybrid.get("static-text-multi-string"),
            "organization-id": org_info.get("organization-id"),
            "organization-name": org_info.get("organization-name"),
            "contact-id": contact.get("contact-id"),
            "person-name": contact.get("person-name"),
            "email-address": contact.get("email-address"),
            "phone-number": contact.get("phone-number"),
            "inventory-update-time": inventory_update_time
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("signs.csv", index=False)
    print(f"Saved {len(df)} records to signs.csv")

def download_solar_controllers(token):
    """Download and flatten the /solar-controllers endpoint into a CSV."""
    url = f"{BASE_URL}/v1/inventory/solar-controllers"
    headers = {
        "Authorization": token,
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()

    top_org_info = data.get("organization-information", {})
    top_org_contact = top_org_info.get("organization-contact-details", {})
    inventory_update_time = data.get("extended-properties", {}).get("inventory-update-time")

    rows = []
    for item in data.get("solar-controller-inventory-list", []):
        header = item.get("device-inventory-header", {})
        org_info = header.get("organization-information", {})
        contact = org_info.get("organization-contact-details", {})
        location = header.get("device-location", {})
        ext_props = header.get("extended-properties", {})

        row = {
            "device-id": header.get("device-id"),
            "device-name": header.get("device-name"),
            "device-description": header.get("device-description"),
            "latitude": location.get("latitude"),
            "longitude": location.get("longitude"),
            "last-update-time": header.get("last-update-time"),
            "device-site-id": ext_props.get("device-site-id"),
            "device-type": ext_props.get("device-type"),
            "is-enabled": ext_props.get("is-enabled"),
            "organization-id": org_info.get("organization-id"),
            "organization-name": org_info.get("organization-name"),
            "contact-id": contact.get("contact-id"),
            "person-name": contact.get("person-name"),
            "email-address": contact.get("email-address"),
            "phone-number": contact.get("phone-number"),
            "inventory-update-time": inventory_update_time
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("solar_controllers.csv", index=False)
    print(f"Saved {len(df)} records to solar_controllers.csv")

def download_detectors(token):
    """Download and flatten the /detectors endpoint into a CSV."""
    url = f"{BASE_URL}/v1/inventory/detectors"
    headers = {
        "Authorization": token,
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()

    top_org_info = data.get("organization-information", {})
    top_org_contact = top_org_info.get("organization-contact-details", {})
    inventory_update_time = data.get("extended-properties", {}).get("inventory-update-time")

    rows = []
    for item in data.get("detector-inventory-list", []):
        header = item.get("device-inventory-header", {})
        org_info = header.get("organization-information", {})
        contact = org_info.get("organization-contact-details", {})
        location = header.get("device-location", {})
        header_ext = header.get("extended-properties", {})
        item_ext = item.get("extended-properties", {})

        row = {
            "device-id": header.get("device-id"),
            "device-name": header.get("device-name"),
            "device-description": header.get("device-description"),
            "latitude": location.get("latitude"),
            "longitude": location.get("longitude"),
            "last-update-time": header.get("last-update-time"),
            "device-site-id": header_ext.get("device-site-id"),
            "device-type": header_ext.get("device-type"),
            "is-enabled": header_ext.get("is-enabled"),
            "detector-type": item.get("detector-type"),
            "detector-technology": item_ext.get("detector-technology"),
            "organization-id": org_info.get("organization-id"),
            "organization-name": org_info.get("organization-name"),
            "contact-id": contact.get("contact-id"),
            "person-name": contact.get("person-name"),
            "email-address": contact.get("email-address"),
            "phone-number": contact.get("phone-number"),
            "inventory-update-time": inventory_update_time
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("detectors.csv", index=False)
    print(f"Saved {len(df)} records to detectors.csv")

def download_links(token):
    """Download and flatten the /links endpoint with custom logic."""
    url = f"{BASE_URL}/v1/inventory/links"
    headers = {
        "Authorization": token,
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()

    org_info = data.get("organization-information", {})
    org_details = org_info.get("organization-contact-details", {})
    update_time = data.get("extended-properties", {}).get("inventory-update-time")

    rows = []
    for item in data.get("link-inventory-list", []):
        begin_loc = item.get("link-begin-node-location", {})
        end_loc = item.get("link-end-node-location", {})
        ext = item.get("extended-properties", {})

        row = {
            "network-id": item.get("network-id"),
            "link-id": item.get("link-id"),
            "link-name": item.get("link-name"),
            "link-type": item.get("link-type"),
            "link-length": item.get("link-length"),
            "link-begin-node-id": item.get("link-begin-node-id"),
            "begin-latitude": begin_loc.get("latitude"),
            "begin-longitude": begin_loc.get("longitude"),
            "link-end-node-id": item.get("link-end-node-id"),
            "end-latitude": end_loc.get("latitude"),
            "end-longitude": end_loc.get("longitude"),
            "is-enabled": item.get("is-enabled"),
            "last-update-time": item.get("last-update-time"),
            "roadway-name": ext.get("roadway-name"),
            "roadway-direction": ext.get("roadway-direction"),
            "free-flow-travel-time": ext.get("free-flow-travel-time"),
            "geometry": str(ext.get("geometry")),
            "organization-id": org_info.get("organization-id"),
            "organization-name": org_info.get("organization-name"),
            "contact-id": org_details.get("contact-id"),
            "person-name": org_details.get("person-name"),
            "email-address": org_details.get("email-address"),
            "phone-number": org_details.get("phone-number"),
            "inventory-update-time": update_time
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("links.csv", index=False)
    print(f"Saved {len(df)} records to links.csv")


def download_nodes(token):
    """Download and flatten the /nodes endpoint with custom logic."""
    url = f"{BASE_URL}/v1/inventory/nodes"
    headers = {
        "Authorization": token,
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()

    org_info = data.get("organization-information", {})
    org_details = org_info.get("organization-contact-details", {})
    update_time = data.get("extended-properties", {}).get("inventory-update-time")

    rows = []
    for item in data.get("node-inventory-list", []):
        loc = item.get("node-location", {})
        row = {
            "network-id": item.get("network-id"),
            "node-id": item.get("node-id"),
            "node-name": item.get("node-name"),
            "node-link-number": item.get("node-link-number"),
            "latitude": loc.get("latitude"),
            "longitude": loc.get("longitude"),
            "is-enabled": item.get("is-enabled"),
            "last-update-time": item.get("last-update-time"),
            "organization-id": org_info.get("organization-id"),
            "organization-name": org_info.get("organization-name"),
            "contact-id": org_details.get("contact-id"),
            "person-name": org_details.get("person-name"),
            "email-address": org_details.get("email-address"),
            "phone-number": org_details.get("phone-number"),
            "inventory-update-time": update_time
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("nodes.csv", index=False)
    print(f"Saved {len(df)} records to nodes.csv")


def download_networks(token):
    """Download and flatten the /networks endpoint with custom logic."""
    url = f"{BASE_URL}/v1/inventory/networks"
    headers = {
        "Authorization": token,
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()

    org_info = data.get("organization-information", {})
    org_details = org_info.get("organization-contact-details", {})
    update_time = data.get("extended-properties", {}).get("inventory-update-time")

    rows = []
    for item in data.get("network-inventory-list", []):
        row = {
            "network-id": item.get("network-id"),
            "network-name": item.get("network-name"),
            "description": item.get("description"),
            "organization-id": org_info.get("organization-id"),
            "organization-name": org_info.get("organization-name"),
            "contact-id": org_details.get("contact-id"),
            "person-name": org_details.get("person-name"),
            "email-address": org_details.get("email-address"),
            "phone-number": org_details.get("phone-number"),
            "inventory-update-time": update_time
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("networks.csv", index=False)
    print(f"Saved {len(df)} records to networks.csv")

def download_generic(endpoint, token):
    """Download and save other endpoints."""
    url = f"{BASE_URL}/v1/inventory/{endpoint}"
    headers = {
        "Authorization": token,
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()

    df = pd.json_normalize(data)
    filename = f"{endpoint.replace('/', '_')}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} records to {filename}")

def main():
    try:
        token = get_bearer_token()
        print("Successfully obtained Bearer token.")

        download_networks(token)
        download_nodes(token)
        download_links(token)
        download_routes(token)
        download_device_sites(token)
        download_signs(token)
        download_solar_controllers(token)
        download_detectors(token)

        print("All data downloaded and saved successfully.")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
