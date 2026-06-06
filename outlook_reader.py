import msal
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────
CLIENT_ID = os.getenv("OUTLOOK_CLIENT_ID")   # from .env file
TENANT_ID = os.getenv("OUTLOOK_TENANT_ID")   # from .env file
SCOPES = [
    "Mail.Read",
    "MailboxSettings.Read",
    "User.Read",
    # NOTE: Do not add 'offline_access', 'openid', or 'profile' here.
    # MSAL adds these reserved OIDC scopes automatically.
]
GRAPH_BASE = "https://graph.microsoft.com/v1.0"
TOKEN_CACHE_FILE = "token_cache.json"        # saves your login so you don't re-auth every time

# ── TOKEN CACHE (persists login between runs) ──────────────────
cache = msal.SerializableTokenCache()
if os.path.exists(TOKEN_CACHE_FILE):
    cache.deserialize(open(TOKEN_CACHE_FILE, "r").read())

app = msal.PublicClientApplication(
    client_id=CLIENT_ID,
    authority=f"https://login.microsoftonline.com/{TENANT_ID}",
    token_cache=cache
)

# ── AUTHENTICATE ───────────────────────────────────────────────
def get_token():
    accounts = app.get_accounts()
    result = None

    if accounts:
        # silently refresh from cache
        result = app.acquire_token_silent(SCOPES, account=accounts[0])

    if not result:
        # First time: open browser for login
        flow = app.initiate_device_flow(scopes=SCOPES)
        print(f"\n👉 {flow['message']}\n")  # prints a URL + code to enter
        result = app.acquire_token_by_device_flow(flow)

    # Save updated cache
    with open(TOKEN_CACHE_FILE, "w") as f:
        f.write(cache.serialize())

    if "access_token" in result:
        return result["access_token"]
    else:
        raise Exception(f"Auth failed: {result.get('error_description')}")

# ── GRAPH API HELPERS ──────────────────────────────────────────
def graph_get(token, endpoint, params=None):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{GRAPH_BASE}{endpoint}", headers=headers, params=params)
    response.raise_for_status()
    return response.json()

# ── FETCH EMAILS ───────────────────────────────────────────────
def get_emails(token, folder="inbox", top=25):
    params = {
        "$top": top,
        "$orderby": "receivedDateTime DESC",
        "$select": "subject,from,receivedDateTime,isRead,bodyPreview,categories"
    }
    data = graph_get(token, f"/me/mailFolders/{folder}/messages", params)
    return data.get("value", [])

# ── FETCH FOLDERS/LABELS ───────────────────────────────────────
def get_folders(token):
    data = graph_get(token, "/me/mailFolders")
    return data.get("value", [])

# ── FETCH CATEGORIES (Labels) ─────────────────────────────────
def get_categories(token):
    data = graph_get(token, "/me/outlook/masterCategories")
    return data.get("value", [])

# ── MAIN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    token = get_token()

    print("\n📁 YOUR MAIL FOLDERS:")
    print("─" * 50)
    folders = get_folders(token)
    for f in folders:
        print(f"  [{f['id'][:8]}...]  {f['displayName']}  ({f['totalItemCount']} items, {f['unreadItemCount']} unread)")

    print("\n🏷️  YOUR CATEGORIES (Labels):")
    print("─" * 50)
    categories = get_categories(token)
    for c in categories:
        print(f"  {c['displayName']}  (color: {c.get('color', 'none')})")

    print("\n📧 LATEST 25 EMAILS (Inbox):")
    print("─" * 50)
    emails = get_emails(token, folder="inbox", top=25)
    for i, email in enumerate(emails, 1):
        read_status = "✅" if email["isRead"] else "🔵"
        sender = email["from"]["emailAddress"]["name"]
        subject = email["subject"]
        date = email["receivedDateTime"][:10]
        preview = email["bodyPreview"][:80].replace("\n", " ")
        labels = ", ".join(email.get("categories", [])) or "—"
        print(f"\n{i}. {read_status} [{date}] {sender}")
        print(f"   Subject : {subject}")
        print(f"   Labels  : {labels}")
        print(f"   Preview : {preview}...")