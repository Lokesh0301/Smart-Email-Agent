from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from gmail_auth import authenticate_gmail

def fetch_recent_emails(max_results=5):
    """Fetches and displays the user's recent emails.
    """
    try:
        # Authenticate using the function in gmail_auth.py
        creds = authenticate_gmail()
        service = build("gmail", "v1", credentials=creds)

        print(f"Fetching your top {max_results} recent emails...")
        # Get list of messages
        results = service.users().messages().list(userId="me", maxResults=max_results).execute()
        messages = results.get("messages", [])

        if not messages:
            print("No messages found.")
            return

        print("\n" + "="*60)
        print(f"{"RECENT EMAILS":^60}")
        print("="*60)

        for index, msg in enumerate(messages, 1):
            # Fetch message metadata (Subject, From, Date, and Snippet)
            msg_detail = service.users().messages().get(
                userId="me", 
                id=msg["id"], 
                format="metadata", 
                metadataHeaders=["Subject", "From", "Date"]
            ).execute()
            
            headers = msg_detail.get("payload", {}).get("headers", [])
            
            # Extract specific header values
            subject = "No Subject"
            sender = "Unknown Sender"
            date = "Unknown Date"
            for header in headers:
                if header["name"].lower() == "subject":
                    subject = header["value"]
                elif header["name"].lower() == "from":
                    sender = header["value"]
                elif header["name"].lower() == "date":
                    date = header["value"]
            
            snippet = msg_detail.get("snippet", "")
            
            print(f"\n[{index}] FROM:   {sender}")
            print(f"    DATE:   {date}")
            print(f"    SUBJ:   {subject}")
            print(f"    BODY:   {snippet}...")
            print("-"*60)

    except HttpError as error:
        print(f"An error occurred: {error}")

if __name__ == "__main__":
    fetch_recent_emails()
